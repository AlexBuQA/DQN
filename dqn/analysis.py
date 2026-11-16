"""Графики обучения, калибровка ценности и траектории эпизодов."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .envs import make_env
from .evaluation import get_discounted_returns, play_and_log_episode
from .utils import write_json


def plot_training_history(history: dict, output_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    eval_steps = history["eval_steps"]
    if eval_steps:
        axes[0, 0].errorbar(
            eval_steps, history["validation_mean"],
            yerr=history["validation_std"], marker="o", capsize=3,
            label="mean ± std",
        )
        axes[0, 0].plot(eval_steps, history["validation_p10"], label="p10")
        axes[0, 0].legend()
        axes[1, 0].plot(eval_steps, history["initial_state_value"])
    if history["loss_steps"]:
        axes[0, 1].plot(history["loss_steps"], history["td_loss"])
        axes[1, 1].plot(history["loss_steps"], history["grad_norm"])
    axes[0, 0].set(title="Validation reward", xlabel="Шаг")
    axes[0, 1].set(title="TD-loss", xlabel="Шаг")
    axes[1, 0].set(title="V начального состояния", xlabel="Шаг")
    axes[1, 1].set(title="Норма градиента", xlabel="Шаг")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _trajectory_plot(records: list[dict], output_path: Path) -> None:
    n_episodes = len(records)
    n_cols = min(3, n_episodes)
    n_rows = math.ceil(n_episodes / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    scatter = None
    for episode_idx, record in enumerate(records):
        axis = axes.flat[episode_idx]
        positions = record["states"][:, :2]
        steps = np.arange(len(positions))
        axis.plot(positions[:, 0], positions[:, 1], color="0.75", linewidth=1)
        scatter = axis.scatter(
            positions[:, 0], positions[:, 1], c=steps, cmap="viridis", s=14
        )
        axis.scatter(*positions[0], marker="o", s=70, color="tab:green", label="старт")
        axis.scatter(*positions[-1], marker="X", s=80, color="tab:red", label="финиш")
        axis.set_title(
            f"Эпизод {episode_idx + 1}: return={record['rewards'].sum():.1f}, "
            f"шагов={len(positions)}"
        )
        axis.set(xlabel="state[0]", ylabel="state[1]")
        axis.grid(alpha=0.25)
    for axis in axes.flat[n_episodes:]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Шаг эпизода", shrink=0.75)
    fig.suptitle("Траектории независимых эпизодов", fontsize=14)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_episodes(agent, config, output_dir: str | Path) -> dict:
    """Анализирует эпизоды и сравнивает V(s) с MC return."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    env = make_env(config.env_name)
    try:
        for episode_idx in range(config.analysis_episodes):
            records.append(play_and_log_episode(
                env, agent, seed=config.analysis_seed + episode_idx,
                max_steps=config.max_episode_steps,
            ))
    finally:
        env.close()

    all_mc, all_values = [], []
    episode_summaries = []
    for episode_idx, record in enumerate(records, start=1):
        bootstrap_value = 0.0
        if record["truncated"][-1] and not record["terminated"][-1]:
            # TIME LIMIT BIAS CHECKED: учитываем ценность после TimeLimit.
            bootstrap_value = float(agent.get_qvalues(
                [record["next_states"][-1]]
            ).max())
        mc_return = get_discounted_returns(
            record["rewards"], config.gamma, bootstrap_value=bootstrap_value
        )
        values = record["qvalues"].max(axis=1)
        all_mc.extend(mc_return)
        all_values.extend(values)
        episode_summaries.append({
            "episode": episode_idx,
            "seed": config.analysis_seed + episode_idx - 1,
            "reward": float(record["rewards"].sum()),
            "steps": int(len(record["rewards"])),
            "terminated": bool(record["terminated"][-1]),
            "truncated": bool(record["truncated"][-1]),
        })

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(mc_return, label="Monte-Carlo return")
        axes[0].plot(values, label="Agent V")
        axes[0].set(title=f"Эпизод {episode_idx}: ценность", xlabel="Шаг")
        axes[0].legend()
        axes[1].plot(record["rewards"])
        axes[1].set(title=f"Эпизод {episode_idx}: reward", xlabel="Шаг")
        for axis in axes:
            axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"episode_{episode_idx}.png", dpi=150)
        plt.close(fig)

    _trajectory_plot(records, output_dir / "trajectories.png")
    mc_values = np.asarray(all_mc, dtype=np.float64)
    agent_values = np.asarray(all_values, dtype=np.float64)
    mae = float(np.mean(np.abs(agent_values - mc_values)))
    bias = float(np.mean(agent_values - mc_values))
    if np.std(mc_values) > 0 and np.std(agent_values) > 0:
        correlation = float(np.corrcoef(mc_values, agent_values)[0, 1])
    else:
        correlation = None

    low = float(min(mc_values.min(), agent_values.min()))
    high = float(max(mc_values.max(), agent_values.max()))
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(mc_values, agent_values, alpha=0.25, s=12)
    axis.plot([low, high], [low, high], "k--", label="x=y")
    # CALIBRATION AXES MATCHED
    axis.set(
        xlim=(low, high), ylim=(low, high),
        xlabel="Monte-Carlo return", ylabel="Agent V",
        title="Калибровка ценности состояния",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.savefig(output_dir / "value_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "episodes": episode_summaries,
        "mae": mae,
        "bias_agent_minus_mc": bias,
        "pearson_correlation": correlation,
        "truncation_bootstrap": True,
    }
    write_json(output_dir / "analysis.json", summary)
    return summary
