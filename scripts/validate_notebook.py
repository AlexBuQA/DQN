#!/usr/bin/env python
"""Проверяет полноту, порядок выполнения и отсутствие ошибок в ноутбуке."""

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "MEPHI_RL26_hw2_DQN_Бужор_Александра.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка выполненного ноутбука")
    parser.add_argument("notebook", nargs="?", default=str(NOTEBOOK))
    args = parser.parse_args()
    notebook = nbformat.read(Path(args.notebook), as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    expected_count = 1
    errors = []
    for index, cell in enumerate(code_cells):
        if "### ВАШ КОД" in cell.source:
            errors.append(f"Кодовая ячейка {index}: остался маркер заглушки")
        if cell.execution_count is None:
            errors.append(f"Кодовая ячейка {index}: не выполнена")
        elif cell.execution_count != expected_count:
            errors.append(
                f"Кодовая ячейка {index}: execution_count={cell.execution_count}, "
                f"ожидался {expected_count}"
            )
        else:
            expected_count += 1
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    f"Кодовая ячейка {index}: {output.get('ename')}: {output.get('evalue')}"
                )
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(code_cells)} кодовых ячеек выполнены последовательно, ошибок нет")


if __name__ == "__main__":
    main()
