#!/usr/bin/env python
"""Независимая проверка сохранённого best.pt на новых seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dqn.checkpoints import load_checkpoint
from dqn.envs import make_env
from dqn.evaluation import evaluate
from dqn.utils import resolve_device
from dqn.video import record_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка checkpoint DQN")
    parser.add_argument("checkpoint", help="Путь к best.pt")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=500_000,
                        help="Новый начальный seed, не использованный при обучении")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-folder", default="runs/evaluation_video")
    args = parser.parse_args()
    if args.games < 2:
        raise ValueError("Для оценки устойчивости используйте минимум 2 эпизода")

    device = resolve_device(args.device)
    agent, config, payload = load_checkpoint(args.checkpoint, device=device)
    env = make_env(config.env_name)
    try:
        scores = evaluate(
            env, agent, n_games=args.games, greedy=True,
            t_max=config.max_episode_steps, seed=args.seed,
            return_rewards=True,
        )
    finally:
        env.close()
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_step": payload["step"],
        "seed_start": args.seed,
        "games": args.games,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "median": float(np.median(scores)),
        "p10": float(np.percentile(scores, 10)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    if args.video:
        path, reward = record_video(
            config.env_name, agent, args.video_folder,
            args.seed + args.games + 1, config.max_episode_steps,
            "independent-evaluation",
        )
        result["video"] = {"path": str(path), "reward": reward}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
