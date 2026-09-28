"""
Evaluate the production intent router on a labeled question set.

Runs every question through `router_node` sequentially (Groq rate limits),
retrying with exponential backoff, and writes per-question predictions and a
summary (accuracy, per-intent accuracy, confusion matrix, order-ID extraction).

    uv run python -m src.evaluation.evaluate_router [dataset.json]
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.config import settings
from src.graph.router_node import router_node

DATASET_PATH = settings.BASE_DIR / "evaluation_data" / "router_eval.json"
INTENTS = ["FAQ", "ORDER", "REFUND", "OFF_TOPIC"]
PAUSE_SECONDS = 2  # between requests, to stay under the tokens-per-minute limit
MAX_ATTEMPTS = 5


def route_with_retry(question: str) -> dict:
    """Call the router; back off 5s, 10s, 20s, ... on any API error."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return router_node({"question": question})
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 5 * 2 ** (attempt - 1)
            logger.warning(f"Attempt {attempt} failed ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)


def run(dataset_path: Path) -> tuple[list[dict], dict]:
    examples = json.loads(dataset_path.read_text())
    rows = []
    for i, ex in enumerate(examples, 1):
        start = time.perf_counter()
        try:
            result = route_with_retry(ex["question"])
            pred_intent, pred_order_id, error = result["intent"], result["order_id"], None
        except Exception as e:
            pred_intent, pred_order_id, error = None, None, f"{type(e).__name__}: {e}"
        rows.append(
            {
                **ex,
                "pred_intent": pred_intent,
                "pred_order_id": pred_order_id,
                "intent_correct": pred_intent == ex["intent"],
                "order_id_correct": pred_order_id == ex["order_id"],
                "latency_s": round(time.perf_counter() - start, 2),
                "error": error,
            }
        )
        logger.info(f"[{i}/{len(examples)}] {ex['intent']} -> {pred_intent} | {ex['question']}")
        time.sleep(PAUSE_SECONDS)
    return rows, summarize(rows)


def summarize(rows: list[dict]) -> dict:
    """Metrics over all rows; API errors count as wrong, not skipped."""
    confusion = {t: {p: 0 for p in INTENTS + ["ERROR"]} for t in INTENTS}
    for r in rows:
        confusion[r["intent"]][r["pred_intent"] or "ERROR"] += 1

    with_id = [r for r in rows if r["order_id"] is not None]
    without_id = [r for r in rows if r["order_id"] is None]
    return {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "llm_model": settings.LLM_MODEL,
        "n": len(rows),
        "errors": sum(r["error"] is not None for r in rows),
        "accuracy": round(sum(r["intent_correct"] for r in rows) / len(rows), 4),
        "per_intent_accuracy": {
            t: round(confusion[t][t] / sum(confusion[t].values()), 4) for t in INTENTS
        },
        "confusion_matrix": confusion,  # rows = true intent, columns = predicted
        "order_id_extraction": {
            "questions_with_id": len(with_id),
            "exact_match": sum(r["order_id_correct"] for r in with_id),
            "questions_without_id": len(without_id),
            "spurious_id_extracted": sum(r["pred_order_id"] is not None for r in without_id),
        },
        "misclassified": [
            {k: r[k] for k in ["id", "question", "intent", "pred_intent", "error"]}
            for r in rows
            if not r["intent_correct"]
        ],
        "order_id_mismatches": [
            {k: r[k] for k in ["id", "question", "order_id", "pred_order_id"]}
            for r in rows
            if not r["order_id_correct"]
        ],
    }


if __name__ == "__main__":
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_PATH
    rows, summary = run(dataset)
    out_dir = Path(settings.EVALUATION_OUTPUT_DIR)
    (out_dir / "router_eval_predictions.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "router_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "misclassified"}, indent=2))
