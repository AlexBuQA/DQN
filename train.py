#!/usr/bin/env python
"""CLI обучения: python train.py --config configs/cartpole_dqn.json"""

from __future__ import annotations

import argparse
import json

from dqn.config import TrainConfig
from dqn.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение DQN-агента")
    parser.add_argument("--config", required=True, help="Путь к JSON-конфигурации")
    parser.add_argument("--output-dir", help="Переопределить папку результатов")
    parser.add_argument("--total-steps", type=int, help="Сократить/изменить бюджет")
    parser.add_argument("--eval-freq", type=int, help="Частота validation")
    parser.add_argument("--seed", type=int, help="Переопределить seed обучения")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda или cuda:0")
    parser.add_argument("--skip-video", action="store_true", help="Не записывать MP4")
    parser.add_argument("--skip-analysis", action="store_true", help="Не строить анализ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_json(args.config)
    overrides = {
        "output_dir": args.output_dir,
        "total_steps": args.total_steps,
        "eval_freq": args.eval_freq,
        "seed": args.seed,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    if args.skip_analysis:
        config.analysis_episodes = 0
    config.validate()
    summary = train(config, record_final_video=not args.skip_video)
    print(json.dumps(summary["final"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
