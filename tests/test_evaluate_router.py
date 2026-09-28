from src.evaluation.evaluate_router import summarize


def row(intent, pred, order_id=None, pred_order_id=None, error=None):
    return {
        "id": "x", "question": "q", "intent": intent, "pred_intent": pred,
        "order_id": order_id, "pred_order_id": pred_order_id,
        "intent_correct": pred == intent, "order_id_correct": pred_order_id == order_id,
        "error": error,
    }


def test_summarize_counts_errors_as_wrong():
    s = summarize([
        row("FAQ", "FAQ"),
        row("ORDER", "ORDER", 1042, 1042),
        row("REFUND", "ORDER", 1043, None),
        row("OFF_TOPIC", None, error="RateLimitError"),
    ])
    assert s["accuracy"] == 0.5
    assert s["errors"] == 1
    assert s["confusion_matrix"]["REFUND"]["ORDER"] == 1
    assert s["confusion_matrix"]["OFF_TOPIC"]["ERROR"] == 1
    assert s["per_intent_accuracy"]["ORDER"] == 1.0
    assert s["order_id_extraction"]["exact_match"] == 1
    assert s["order_id_extraction"]["questions_with_id"] == 2
