#!/usr/bin/env python
"""Диагностика локального окружения перед тестами и обучением."""

import platform
import sys

import gymnasium as gym
import imageio_ffmpeg
import matplotlib
import numpy as np
import torch

from dqn.envs import make_env


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}")
    print(f"Gymnasium: {gym.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"FFmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
    for env_name in ("CartPole-v1", "LunarLander-v3"):
        env = make_env(env_name)
        try:
            state, _ = env.reset(seed=0)
            print(f"{env_name}: observation={state.shape}, actions={env.action_space.n} — OK")
        finally:
            env.close()


if __name__ == "__main__":
    main()
