"""Единый обучающий цикл для CartPole и LunarLander."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import trange

from .agents import make_agent
from .analysis import analyze_episodes, plot_training_history
from .checkpoints import save_checkpoint
from .config import TrainConfig
from .envs import make_env
from .evaluation import evaluate
from .interaction import play_and_record
from .losses import compute_td_loss, compute_td_loss_double
from .replay import ReplayBuffer
from .utils import linear_decay, resolve_device, set_seed, write_json
from .video import record_video


def _assert_seed_separation(config: TrainConfig) -> None:
    validation = set(range(
        config.validation_seed, config.validation_seed + config.validation_games
    ))
    final = set(range(config.final_seed, config.final_seed + config.final_games))
    # HELD OUT SEEDS VERIFIED / HELD OUT LUNAR SEEDS VERIFIED
    if validation & final:
        raise ValueError("Validation и final seed не должны пересекаться")


def train(config: TrainConfig, record_final_video: bool = True) -> dict:
    config.validate()
    _assert_seed_separation(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    train_env = make_env(config.env_name)
    eval_env = make_env(config.env_name)
    probe_env = make_env(config.env_name)
    try:
        state_shape = train_env.observation_space.shape
        n_actions = train_env.action_space.n
        if train_env.spec.max_episode_steps != config.max_episode_steps:
            raise ValueError("max_episode_steps не совпадает со спецификацией среды")

        # SAME AGENT VERIFIED: LunarLander использует те же классы и TD-target.
        agent = make_agent(
            config.agent, state_shape, n_actions,
            epsilon=config.initial_epsilon, hidden_dim=config.hidden_dim,
        ).to(device)
        target_network = make_agent(
            config.agent, state_shape, n_actions,
            epsilon=0.0, hidden_dim=config.hidden_dim,
        ).to(device)
        target_network.load_state_dict(agent.state_dict())
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
        replay = ReplayBuffer(config.replay_size)

        state, _ = train_env.reset(seed=config.seed)
        agent.epsilon = 1.0
        while len(replay) < config.initial_fill:
            _, state = play_and_record(
                state, agent, train_env, replay, n_steps=config.collect_steps
            )
        print(f"[{config.name}] replay buffer: {len(replay)}, device: {device}")

        history = {
            key: [] for key in (
                "loss_steps", "td_loss", "grad_norm", "eval_steps",
                "validation_mean", "validation_std", "validation_p10",
                "selection_score", "initial_state_value",
            )
        }
        best_selection = -np.inf
        best_state = copy.deepcopy(agent.state_dict())
        best_step = 0
        best_validation = {}
        strong_streak = 0
        executed_steps = 0

        progress = trange(1, config.total_steps + 1, desc=config.name)
        for step in progress:
            agent.epsilon = linear_decay(
                config.initial_epsilon, config.final_epsilon,
                step, config.decay_steps,
            )
            _, state = play_and_record(state, agent, train_env, replay, n_steps=1)
            batch = replay.sample(config.batch_size)
            loss_function = compute_td_loss_double if config.double else compute_td_loss
            loss_kwargs = {"loss_kind": config.loss_kind} if config.double else {}
            loss = loss_function(
                *batch, agent, target_network, gamma=config.gamma,
                device=device, **loss_kwargs,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                agent.parameters(), config.max_grad_norm
            )
            optimizer.step()
            executed_steps = step

            if step % config.loss_freq == 0:
                history["loss_steps"].append(step)
                history["td_loss"].append(float(loss.detach().cpu()))
                history["grad_norm"].append(float(grad_norm))
            if step % config.target_update_freq == 0:
                target_network.load_state_dict(agent.state_dict())
            if step % config.eval_freq != 0:
                continue

            scores = evaluate(
                eval_env, agent, n_games=config.validation_games,
                greedy=True, t_max=config.max_episode_steps,
                seed=config.validation_seed, return_rewards=True,
            )
            mean_score = float(scores.mean())
            std_score = float(scores.std())
            p10_score = float(np.percentile(scores, 10))
            selection = mean_score - config.validation_std_penalty * std_score
            probe_state, _ = probe_env.reset(seed=20_000)
            initial_value = float(agent.get_qvalues([probe_state]).max())

            history["eval_steps"].append(step)
            history["validation_mean"].append(mean_score)
            history["validation_std"].append(std_score)
            history["validation_p10"].append(p10_score)
            history["selection_score"].append(selection)
            history["initial_state_value"].append(initial_value)

            if selection > best_selection:
                best_selection = selection
                best_step = step
                best_state = copy.deepcopy(agent.state_dict())
                best_validation = {
                    "mean": mean_score, "std": std_score, "p10": p10_score,
                    "selection_score": selection,
                }
                # ROBUST CHECKPOINT VERIFIED
                save_checkpoint(
                    output_dir / "best.pt", agent, config, step, best_validation
                )

            is_strong = (
                mean_score >= config.validation_reward_threshold
                and p10_score >= config.validation_p10_threshold
            )
            strong_streak = strong_streak + 1 if is_strong else 0
            progress.set_postfix(
                reward=f"{mean_score:.1f}", p10=f"{p10_score:.1f}",
                epsilon=f"{agent.epsilon:.3f}",
            )
            print(
                f"[{config.name}] step={step}, validation={mean_score:.1f}"
                f"±{std_score:.1f}, p10={p10_score:.1f}, best_step={best_step}"
            )
            if (
                config.early_stopping
                and strong_streak >= config.required_strong_evaluations
            ):
                print(f"[{config.name}] ранняя остановка: устойчивый результат")
                break

        # BEST CHECKPOINT RESTORED
        agent.load_state_dict(best_state)
        target_network.load_state_dict(best_state)
        save_checkpoint(output_dir / "best.pt", agent, config, best_step, best_validation)
    finally:
        train_env.close()
        eval_env.close()
        probe_env.close()

    final_env = make_env(config.env_name)
    try:
        # THIRTY SEEDS VERIFIED / SINGLE EPISODE REJECTED
        final_scores = evaluate(
            final_env, agent, n_games=config.final_games, greedy=True,
            t_max=config.max_episode_steps, seed=config.final_seed,
            return_rewards=True,
        )
    finally:
        final_env.close()
    np.save(output_dir / "final_scores.npy", final_scores)
    write_json(output_dir / "history.json", history)
    plot_training_history(history, output_dir / "training_curves.png")

    final_summary = {
        "mean": float(final_scores.mean()),
        "std": float(final_scores.std()),
        "median": float(np.median(final_scores)),
        "p10": float(np.percentile(final_scores, 10)),
        "min": float(final_scores.min()),
        "max": float(final_scores.max()),
        "games": int(config.final_games),
        "seed_start": int(config.final_seed),
        "threshold": float(config.final_threshold),
        "threshold_met": bool(final_scores.mean() >= config.final_threshold),
        "share_above_threshold": float(
            (final_scores >= config.final_threshold).mean()
        ),
    }
    summary = {
        "name": config.name,
        "config": config.to_dict(),
        "device": str(device),
        "executed_steps": int(executed_steps),
        "best_step": int(best_step),
        "best_validation": best_validation,
        "final": final_summary,
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    write_json(output_dir / "summary.json", summary)

    if config.analysis_episodes > 0:
        summary["analysis"] = analyze_episodes(
            agent, config, output_dir / "analysis"
        )
    if record_final_video:
        video_path, video_score = record_video(
            config.env_name, agent, output_dir / "videos",
            config.video_seed, config.max_episode_steps,
            name_prefix=config.name,
        )
        summary["video"] = {"path": str(video_path), "reward": video_score}
    summary["elapsed_seconds"] = float(time.perf_counter() - started_at)
    write_json(output_dir / "summary.json", summary)
    print(
        f"[{config.name}] FINAL mean={final_summary['mean']:.2f}, "
        f"std={final_summary['std']:.2f}, p10={final_summary['p10']:.2f}, "
        f"min={final_summary['min']:.2f}, max={final_summary['max']:.2f}"
    )
    return summary
