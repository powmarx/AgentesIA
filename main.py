"""ARC-AGI-1 solver.

Usage:
    uv run main.py 3aa6fb7a
    uv run main.py 3aa6fb7a --reasoning low
    uv run main.py 3aa6fb7a --model gemini-3.5-flash
    uv run main.py --file tasks.txt --wait 30      # seconds between API calls
    uv run main.py --file tasks.txt --label my_experiment
    uv run main.py --file tasks.txt --label cmp --reasoning minimal low medium high

With --file, each finished test is appended to results/live_<label>_<level>.jsonl
(the Streamlit dashboard reads these files while the run is in progress, and its
Flashback tab replays them later; each line keeps the predicted grid) and a
summary JSON is saved at the end of each reasoning level.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from llm import GOOGLE_MODELS, MODELS, NO_MINIMAL, REASONING_LEVELS, ask
from prompt import SYSTEM_PROMPT, build_prompt
from validator import extract_grid, is_correct

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"  # default; changed with --results-dir
WAIT_SECONDS = 10.0  # default; changed with --wait
_last_call_end: float | None = None


def wait_before_call() -> None:
    """Sleep so that API calls are at least WAIT_SECONDS apart."""
    if _last_call_end is not None:
        remaining = WAIT_SECONDS - (time.monotonic() - _last_call_end)
        if remaining > 0:
            time.sleep(remaining)


def solve_task(task_id: str, provider: str, reasoning: str, model: str, live_path: Path | None) -> dict:
    """Solve every test input of one task; the task counts only if all are right."""
    global _last_call_end
    task = json.loads((DATA_DIR / f"{task_id}.json").read_text())
    n_tests = len(task["test"])
    tests = []
    for i, test in enumerate(task["test"]):
        wait_before_call()
        start = time.monotonic()
        error = None
        predicted = None
        tokens = (0, 0, 0)
        try:
            result = ask(SYSTEM_PROMPT, build_prompt(task, i), provider, reasoning, model)
            predicted = extract_grid(result.text)
            tokens = (result.input_tokens, result.reasoning_tokens, result.answer_tokens)
        except Exception as e:  # a failed call counts as wrong; keep going
            error = str(e)
        finally:
            _last_call_end = time.monotonic()
        ok = error is None and is_correct(predicted, test["output"])
        seconds = round(time.monotonic() - start, 1)
        print(
            f"[{task_id}#{i}] reasoning={reasoning} input={tokens[0]} "
            f"reasoning_tokens={tokens[1]} answer={tokens[2]} "
            f"-> {'CORRECT' if ok else 'WRONG'}" + (f" ERROR: {error}" if error else "")
        )
        row = {
            "test_index": i,
            "correct": ok,
            "input_tokens": tokens[0],
            "reasoning_tokens": tokens[1],
            "answer_tokens": tokens[2],
            "seconds": seconds,
            "predicted": predicted,
            "error": error,
        }
        tests.append(row)
        if live_path:
            event = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "task_id": task_id,
                "n_tests": n_tests,
                "reasoning": reasoning,
                "model": model,
                **row,
            }
            with live_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
    return {"task_id": task_id, "solved": all(t["correct"] for t in tests), "tests": tests}


def read_task_ids(path: str) -> list[str]:
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def run_experiment(ids: list[str], provider: str, reasoning: str, model: str, label: str,
                   save: bool) -> None:
    """Run all tasks at one reasoning level."""
    live_path = None
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        live_path = RESULTS_DIR / f"live_{label or 'sem_rotulo'}_{reasoning}.jsonl"
        live_path.write_text("")  # start fresh

    results = [solve_task(t, provider, reasoning, model, live_path) for t in ids]
    solved = sum(r["solved"] for r in results)
    print(f"\n[{model} / {reasoning}] Solved {solved}/{len(results)}\n")

    if save:
        tests = [t for r in results for t in r["tests"]]
        summary = {
            "label": label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "provider": provider,
            "model": model,
            "reasoning": reasoning,
            "wait_seconds": WAIT_SECONDS,
            "tasks_solved": solved,
            "tasks_total": len(results),
            "input_tokens": sum(t["input_tokens"] for t in tests),
            "reasoning_tokens": sum(t["reasoning_tokens"] for t in tests),
            "answer_tokens": sum(t["answer_tokens"] for t in tests),
            "results": results,
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"summary_{label + '_' if label else ''}{reasoning}_{stamp}.json"
        (RESULTS_DIR / name).write_text(json.dumps(summary, indent=2))
        print(f"Summary saved to {RESULTS_DIR / name}")


def main() -> None:
    global WAIT_SECONDS, RESULTS_DIR
    p = argparse.ArgumentParser(description="Simple ARC-AGI-1 solver")
    p.add_argument("task", nargs="?", help="a single task id, e.g. 3aa6fb7a")
    p.add_argument("--file", help="text file with one task id per line")
    p.add_argument("--label", default="", help="experiment label")
    p.add_argument("--provider", choices=list(MODELS), default="google")
    p.add_argument("--model", help="model id (default: the provider's default model). "
                   f"Google options: {', '.join(GOOGLE_MODELS)}")
    p.add_argument("--reasoning", nargs="+", choices=REASONING_LEVELS, default=["high"],
                   help="one or more reasoning levels, run in order (default: high)")
    p.add_argument("--wait", type=float, default=WAIT_SECONDS, metavar="SECONDS",
                   help="seconds to wait between API calls (default: %(default)s)")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR, metavar="DIR",
                   help="where to write results (default: results/)")
    p.add_argument("--limit", type=int, help="only use the first N tasks of --file")
    args = p.parse_args()
    if args.wait < 0:
        p.error("--wait must be >= 0")
    WAIT_SECONDS = args.wait
    RESULTS_DIR = args.results_dir

    if bool(args.task) == bool(args.file):
        p.error("give either a task id or --file")

    model = args.model or MODELS[args.provider]
    if "minimal" in args.reasoning and model in NO_MINIMAL:
        p.error(f"{model} does not support the minimal reasoning level")

    ids = read_task_ids(args.file) if args.file else [args.task]
    if args.limit:
        ids = ids[: args.limit]
    for level in args.reasoning:
        run_experiment(ids, args.provider, level, model, args.label, save=bool(args.file))


if __name__ == "__main__":
    main()
