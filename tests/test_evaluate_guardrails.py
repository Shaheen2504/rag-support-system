from src.evaluation.evaluate_guardrails import INPUT_CHECKS, stage_summary


def row(id, blocked_by):
    return {"id": id, "passed": not blocked_by, "blocked_by": blocked_by}


def test_stage_summary_counts_false_positives_per_scanner():
    s = stage_summary(
        [
            row("a", []),
            row("b", ["PromptInjection"]),
            row("c", []),
            row("d", ["PromptInjection", "Toxicity"]),
        ],
        INPUT_CHECKS,
    )
    assert (s["total"], s["passed"], s["blocked"]) == (4, 2, 2)
    assert s["false_positive_rate"] == 0.5
    assert s["blocks_per_scanner"] == {
        "PromptInjection": 2,
        "Toxicity": 1,
        "TokenLimit": 0,
    }
    assert [fp["id"] for fp in s["false_positives"]] == ["b", "d"]
