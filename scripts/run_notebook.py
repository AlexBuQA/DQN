#!/usr/bin/env python
"""Последовательно выполняет сдаваемый ноутбук в текущем окружении."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "MEPHI_RL26_hw2_DQN_Бужор_Александра.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный Run All ноутбука DQN")
    parser.add_argument("notebook", nargs="?", default=str(NOTEBOOK))
    args = parser.parse_args()
    notebook_path = Path(args.notebook).resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=7200,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    finally:
        # Сохраняем outputs и traceback даже при сбое, чтобы упростить диагностику.
        nbformat.write(notebook, notebook_path)
    print(f"Выполненный ноутбук сохранён: {notebook_path}")


if __name__ == "__main__":
    main()
