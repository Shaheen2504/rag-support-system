"""Output checks on FAQ answers, run with the real LLM Guard scanners (models load once)."""

import pytest

from src.graph.answer_check_node import (
    answer_check_node,
    check_language_same,
    check_relevance,
    check_toxicity,
)

QUESTION = "How do I cancel my order?"


def run_checks(answer: str, question: str = QUESTION) -> dict:
    """Mirror the graph: parallel checks append to answer_status, then the check node decides."""
    state = {"question": question, "llm_output": answer, "answer_status": []}
    for check in (check_language_same, check_relevance, check_toxicity):
        state["answer_status"] += check(state)["answer_status"]
    return answer_check_node(state)


def test_neutral_answer_passes():
    answer = (
        "To cancel your order, go to Order History, select the order and click Cancel."
    )
    result = run_checks(answer)
    assert result == {"llm_output": answer, "answer_valid": True}


@pytest.mark.parametrize(
    "answer",
    [
        "I'm sorry for the inconvenience. Unfortunately your order was delayed, "
        "but we will fix this problem. To cancel it, go to Order History and click Cancel.",
        "To cancel your order, go to Order History, select the order and click Cancel. "
        "If you have any problem, contact support.",
        "I'm really sorry to hear about your bad experience. I understand your frustration "
        "and apologize for the trouble. You can cancel the order from Order History.",
    ],
)
def test_apologetic_answer_passes(answer):
    assert run_checks(answer)["answer_valid"] is True


def test_toxic_answer_fails():
    # On-topic and in English, so only the toxicity check can reject it.
    answer = (
        "Cancel your order in Order History, you stupid idiot. "
        "Stop wasting my time with dumb questions."
    )
    state = {"question": QUESTION, "llm_output": answer}
    assert check_toxicity(state)["answer_status"] == [1]
    result = run_checks(answer)
    assert result["answer_valid"] is False
    assert result["llm_output"] == "Answer failed checks, please try again."


def test_answer_in_other_language_fails():
    answer = (
        "Para cancelar su pedido, vaya al historial de pedidos y haga clic en cancelar."
    )
    result = run_checks(answer)
    assert result["answer_valid"] is False
    assert result["llm_output"] == "Answer failed checks, please try again."


def test_irrelevant_answer_fails():
    result = run_checks("The Eiffel Tower is in Paris and was built in 1889.")
    assert result["answer_valid"] is False
