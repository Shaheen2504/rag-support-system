"""
Guardrail false-positive sanity check.

Runs normal customer-support questions through the input scanners
(PromptInjection, Toxicity, TokenLimit) and hand-written polite FAQ answers
through the output scanners (LanguageSame, Relevance, Toxicity), using the same
node functions as the graph. Every item is legitimate, so any block is a false
positive. No LLM calls and no random selection.

    uv run python -m src.evaluation.evaluate_guardrails [dataset.json]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.config import settings
from src.graph.answer_check_node import (
    check_language_same,
    check_relevance,
    check_toxicity,
)
from src.graph.question_check_node import (
    scan_prompt_injection,
    scan_token_limit,
    scan_toxicity,
)

DATASET_PATH = settings.BASE_DIR / "evaluation_data" / "guardrail_eval.json"

# Each check returns {"<status_key>": [1]} when it blocks, [0] when it passes.
INPUT_CHECKS = {
    "PromptInjection": scan_prompt_injection,
    "Toxicity": scan_toxicity,
    "TokenLimit": scan_token_limit,
}
OUTPUT_CHECKS = {
    "LanguageSame": check_language_same,
    "Relevance": check_relevance,
    "Toxicity": check_toxicity,
}


def blocked_by(checks: dict, state: dict, status_key: str) -> list[str]:
    """Names of the scanners that block this state."""
    return [name for name, check in checks.items() if check(state)[status_key] == [1]]


def run(dataset_path: Path) -> tuple[list[dict], list[dict], dict]:
    data = json.loads(dataset_path.read_text())
    input_rows = []
    for ex in data["questions"]:
        blockers = blocked_by(
            INPUT_CHECKS, {"question": ex["question"]}, "question_status"
        )
        input_rows.append({**ex, "passed": not blockers, "blocked_by": blockers})
        logger.info(f"input  {'PASS' if not blockers else blockers} | {ex['question']}")
    output_rows = []
    for ex in data["answers"]:
        state = {"question": ex["question"], "llm_output": ex["answer"]}
        blockers = blocked_by(OUTPUT_CHECKS, state, "answer_status")
        output_rows.append({**ex, "passed": not blockers, "blocked_by": blockers})
        logger.info(f"output {'PASS' if not blockers else blockers} | {ex['question']}")
    return input_rows, output_rows, summarize(input_rows, output_rows)


def stage_summary(rows: list[dict], scanners: dict) -> dict:
    blocked = [r for r in rows if not r["passed"]]
    return {
        "total": len(rows),
        "passed": len(rows) - len(blocked),
        "blocked": len(blocked),
        "false_positive_rate": round(len(blocked) / len(rows), 4),
        "blocks_per_scanner": {
            name: sum(name in r["blocked_by"] for r in rows) for name in scanners
        },
        "false_positives": [{k: r[k] for k in ["id", "blocked_by"]} for r in blocked],
    }


def summarize(input_rows: list[dict], output_rows: list[dict]) -> dict:
    return {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "input_guardrails": stage_summary(input_rows, INPUT_CHECKS),
        "output_guardrails": stage_summary(output_rows, OUTPUT_CHECKS),
    }


if __name__ == "__main__":
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_PATH
    input_rows, output_rows, summary = run(dataset)
    out_dir = Path(settings.EVALUATION_OUTPUT_DIR)
    (out_dir / "guardrail_eval_results.json").write_text(
        json.dumps({"questions": input_rows, "answers": output_rows}, indent=2)
    )
    (out_dir / "guardrail_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
