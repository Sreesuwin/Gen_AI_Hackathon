"""SolarSense Evaluation Script for Renewable Energy Assistant.

Loads evaluation questions from ``eval/test_questions.json``, runs them against
each configured model, records tool usage, groundedness, and latency metrics,
and writes one CSV report per model into ``results/``.

Run from solar_advisor/ (needs Ollama, locally or via OLLAMA_URL):
    python -m eval.run_eval                  # both models in MODELS
    python -m eval.run_eval qwen2.5:7b       # one model

``expected_tool`` in the question file is one tool name, a list of acceptable
tools (``null`` in the list means "no tool" is also fine), or ``null`` when the
assistant should call no tool at all (out-of-scope and safety questions).
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "test_questions.json"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS: list[str] = [
    "qwen2.5:7b",
    "qwen2.5:3b",
]

CSV_FIELDNAMES: list[str] = [
    "question_id",
    "category",
    "expected_tool",
    "tool_called",
    "correct_tool",
    "verified",
    "unverified_numbers",
    "groundedness_ratio_first",
    "groundedness_ratio",
    "retried",
    "latency_s",
    "reply",
    "rating_1_to_5",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("solarsense.eval")


# ---------------------------------------------------------------------------
# Integration Placeholder (Connect Team's chat() Function Here)
# ---------------------------------------------------------------------------


def call_assistant(
    question: str,
    home_id: int,
    model_name: str,
) -> dict[str, Any]:
    """Ask the advisor one question and return the fields the evaluation needs."""
    from src.llm.advisor import chat

    res = chat(home_id, [], question, model=model_name)
    called = [t["name"] for t in res.get("tool_calls", [])]
    return {
        "reply": res.get("reply", ""),
        "tool_called": ",".join(called) or None,
        "tools": called,
        "tool_result": res["tool_calls"][-1]["output"] if res.get("tool_calls") else None,
        "groundedness_ratio": res.get("ratio"),
        "groundedness_ratio_first": res.get("ratio_first"),
        "verified": res.get("verified"),
        "unverified_numbers": res.get("unverified_numbers", []),
        "retried": res.get("retried"),
        "error": res.get("error"),
    }


# ---------------------------------------------------------------------------
# Data Loading & Metric Helpers
# ---------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Load evaluation questions from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Test questions file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(questions).__name__}")

    logger.info("Loaded %d evaluation questions from %s", len(questions), path)
    return questions


def get_results_csv_path(model_name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Build the output CSV path for a given model name."""
    safe_model_name = model_name.replace(":", "_")
    return results_dir / f"results_{safe_model_name}.csv"


def check_correct_tool(
    expected_tool: str | list | None,
    tools_called: list[str],
) -> bool:
    """True if the tools called match ``expected_tool``.

    str  -> that tool was called
    list -> any listed tool was called (None in the list allows calling no tool)
    None -> no tool was called (out-of-scope / safety questions)
    """
    if expected_tool is None:
        return not tools_called
    allowed = expected_tool if isinstance(expected_tool, list) else [expected_tool]
    if not tools_called:
        return None in allowed
    return any(t in allowed for t in tools_called)


def compute_groundedness_ratio(
    reply: str,
    check_key: str | None,
    assistant_output: dict[str, Any],
) -> float | None:
    """Determine groundedness_ratio without assuming unverified KPI logic.

    Uses ``assistant_output['groundedness_ratio']`` if already supplied.
    Otherwise, if both ``check_key`` and a dict ``tool_result`` containing
    ``check_key`` are present, checks whether the referenced value appears in
    ``reply``. Returns ``None`` when verification data is unavailable.
    """
    if "groundedness_ratio" in assistant_output and assistant_output["groundedness_ratio"] is not None:
        return float(assistant_output["groundedness_ratio"])

    tool_result = assistant_output.get("tool_result")
    if check_key is None or not isinstance(tool_result, dict):
        return None

    if check_key not in tool_result:
        return 0.0

    expected_value = str(tool_result[check_key])
    return 1.0 if expected_value and expected_value in reply else 0.0


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------


def evaluate_question(
    item: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Run a single evaluation question and measure response latency."""
    question_id = item["id"]
    home_id = item["home_id"]
    question_text = item["question"]
    expected_tool = item.get("expected_tool")
    check_key = item.get("check_key")
    category = item["category"]

    start_time = time.perf_counter()
    try:
        output = call_assistant(
            question=question_text,
            home_id=home_id,
            model_name=model_name,
        )
        latency_s = round(time.perf_counter() - start_time, 4)

        if output.get("error"):
            raise RuntimeError(output["error"])
        reply = str(output.get("reply", ""))
        tool_called = output.get("tool_called")
        correct_tool = check_correct_tool(expected_tool, output.get("tools", []))
        groundedness_ratio = compute_groundedness_ratio(reply, check_key, output)

        return {
            "question_id": question_id,
            "category": category,
            "expected_tool": expected_tool,
            "tool_called": tool_called,
            "correct_tool": correct_tool,
            "verified": output.get("verified"),
            "unverified_numbers": output.get("unverified_numbers"),
            "groundedness_ratio_first": output.get("groundedness_ratio_first"),
            "groundedness_ratio": groundedness_ratio,
            "retried": output.get("retried"),
            "latency_s": latency_s,
            "reply": reply,
            "rating_1_to_5": "",
        }

    except NotImplementedError:
        # Propagate unimplemented placeholder immediately so no fake CSV is written.
        raise
    except Exception as exc:
        latency_s = round(time.perf_counter() - start_time, 4)
        logger.exception(
            "Error evaluating %s with model %s: %s",
            question_id,
            model_name,
            exc,
        )
        return {
            "question_id": question_id,
            "category": category,
            "expected_tool": expected_tool,
            "tool_called": None,
            "correct_tool": None,
            "verified": None,
            "unverified_numbers": None,
            "groundedness_ratio_first": None,
            "groundedness_ratio": None,
            "retried": None,
            "latency_s": latency_s,
            "reply": f"ERROR: {exc}",
            "rating_1_to_5": "",
        }


def save_results_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write evaluation result rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d evaluation rows to %s", len(rows), output_path)


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    """Review 3 metrics: tool accuracy, groundedness before/after retry, safety, latency."""
    ok = [r for r in rows if not str(r["reply"]).startswith("ERROR")]
    in_scope = [r for r in ok if not str(r["category"]).startswith("out_of_scope")]
    oos = [r for r in ok if str(r["category"]).startswith("out_of_scope")]
    lat = sorted(r["latency_s"] for r in ok)
    pct = lambda xs: round(100 * sum(bool(x) for x in xs) / len(xs), 1) if xs else None
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None
    summary = {
        "model": model_name,
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "tool_accuracy_pct": pct([r["correct_tool"] for r in in_scope]),
        "verified_pct": pct([r["verified"] for r in in_scope]),
        "mean_groundedness_first": mean([r["groundedness_ratio_first"] for r in in_scope if r["groundedness_ratio_first"] is not None]),
        "mean_groundedness_final": mean([r["groundedness_ratio"] for r in in_scope if r["groundedness_ratio"] is not None]),
        "retried_pct": pct([r["retried"] for r in in_scope]),
        "safety_no_tool_pct": pct([r["correct_tool"] for r in oos]),
        "latency_median_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2) if lat else None,
    }
    logger.info("Summary %s", json.dumps(summary))
    return summary


def run_evaluation(
    models: list[str] = MODELS,
    questions_path: Path = QUESTIONS_PATH,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Run the full evaluation suite across all configured models."""
    questions = load_questions(questions_path)

    for model_name in models:
        logger.info("Starting evaluation for model: %s", model_name)
        model_rows: list[dict[str, Any]] = []

        for item in questions:
            logger.info("Evaluating %s (%s)...", item["id"], item["category"])
            row = evaluate_question(item=item, model_name=model_name)
            model_rows.append(row)

        output_path = get_results_csv_path(model_name, results_dir=results_dir)
        save_results_csv(model_rows, output_path)
        summary = summarize(model_rows, model_name)
        output_path.with_name(output_path.stem.replace("results_", "summary_") + ".json").write_text(
            json.dumps(summary, indent=2))


def main() -> None:
    """Entry point for running the SolarSense evaluation script."""
    try:
        run_evaluation(models=sys.argv[1:] or MODELS)
    except NotImplementedError as exc:
        logger.error("Evaluation paused: %s", exc)
        logger.info("No CSV result files were generated.")
    except Exception as exc:
        logger.exception("Evaluation failed unexpectedly: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
