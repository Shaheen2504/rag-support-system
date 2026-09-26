"""
Compare retrieval modes on the held-out test split (never indexed).

A hit means a retrieved Q&A pair has the same intent as the test question.

    uv run python -m src.evaluation.evaluate_retrieval [sample_size]
"""

import sys
import time

import polars as pl

from src.config import settings
from src.graph.utils import load_faiss_index

MODES = ["faiss", "hybrid", "hybrid_rerank"]


def evaluate(mode: str, test_df: pl.DataFrame) -> dict:
    retriever = load_faiss_index(mode)
    hit1 = hit5 = mrr = 0.0
    start = time.perf_counter()
    for row in test_df.iter_rows(named=True):
        intents = [d.metadata["intent"] for d in retriever.invoke(row["question"])]
        if row["intent"] in intents:
            rank = intents.index(row["intent"]) + 1
            hit1 += rank == 1
            hit5 += 1
            mrr += 1 / rank
    n = test_df.height
    return {
        "mode": mode,
        "n": n,
        "intent_hit@1": round(hit1 / n, 4),
        "intent_hit@5": round(hit5 / n, 4),
        "mrr@5": round(mrr / n, 4),
        "ms_per_query": round((time.perf_counter() - start) / n * 1000, 1),
    }


if __name__ == "__main__":
    test_df = pl.read_csv(settings.TEST_DATA_PATH)
    if len(sys.argv) > 1:
        test_df = test_df.sample(n=int(sys.argv[1]), seed=settings.EVALUATION_RANDOM_SEED)
    results = pl.DataFrame([evaluate(mode, test_df) for mode in MODES])
    print(results)
    results.write_csv(f"{settings.EVALUATION_OUTPUT_DIR}/retrieval_modes.csv")
