"""Validation: extract the grid from the model's reply and compare it."""

import json
import re


def extract_grid(text: str) -> list[list[int]] | None:
    """Return the grid from the last ```json block (or last [[...]] found)."""
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = blocks[::-1] or re.findall(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)[::-1]
    for c in candidates:
        try:
            grid = json.loads(c.strip())
        except json.JSONDecodeError:
            continue
        if (
            isinstance(grid, list) and grid
            and all(isinstance(r, list) and r for r in grid)
            and all(isinstance(v, int) for r in grid for v in r)
        ):
            return grid
    return None


def is_correct(predicted: list[list[int]] | None, expected: list[list[int]]) -> bool:
    """Exact match: same shape and same values."""
    return predicted == expected
