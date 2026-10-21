"""Prompt construction: turns an ARC task into text for the LLM."""

SYSTEM_PROMPT = (
    "You are an expert at solving ARC-AGI puzzles. Each puzzle has a few "
    "example pairs (input grid -> output grid) that share one hidden "
    "transformation rule. Infer the rule, then apply it to the test input."
)

ANSWER_FORMAT = (
    "Think carefully about the rule. Then give the final output grid as a "
    "JSON list of lists of integers (0-9), inside a ```json code block. "
    "Put nothing else inside the code block."
)


def grid_to_text(grid: list[list[int]]) -> str:
    """One row per line, digits separated by spaces."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def build_prompt(task: dict, test_index: int = 0) -> str:
    """Build the user prompt: train examples, then the test input."""
    parts = []
    for i, pair in enumerate(task["train"], 1):
        parts.append(f"Example {i}\nInput:\n{grid_to_text(pair['input'])}")
        parts.append(f"Output:\n{grid_to_text(pair['output'])}")
    test_input = task["test"][test_index]["input"]
    parts.append(f"Test input:\n{grid_to_text(test_input)}")
    parts.append(ANSWER_FORMAT)
    return "\n\n".join(parts)
