"""Запись одного наглядного эпизода восстановленной политики."""

from __future__ import annotations

from pathlib import Path

from gymnasium.wrappers import RecordVideo

from .envs import make_env
from .evaluation import evaluate


def record_video(
    env_name: str,
    agent,
    video_folder: str | Path,
    seed: int,
    max_steps: int,
    name_prefix: str,
) -> tuple[Path, float]:
    video_folder = Path(video_folder)
    video_folder.mkdir(parents=True, exist_ok=True)
    video_env = RecordVideo(
        make_env(env_name, render_mode="rgb_array"),
        video_folder=str(video_folder),
        episode_trigger=lambda episode_number: episode_number == 0,
        name_prefix=name_prefix,
        disable_logger=True,
    )
    try:
        # VIDEO IS NOT EVALUATION
        score = evaluate(
            video_env, agent, n_games=1, greedy=True,
            t_max=max_steps, seed=seed,
        )
    finally:
        video_env.close()
    paths = sorted(
        video_folder.glob(f"{name_prefix}*.mp4"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise RuntimeError("Видео эпизода не было создано")
    return paths[-1], float(score)
