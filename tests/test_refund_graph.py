"""
Refund path through the real compiled graph: input scanners → router →
refund_check → interrupt → resume (approve / reject). Only the router's LLM is
faked; the scanners, nodes, checkpointer and order DB are real.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import SecretStr

from src.api import main
from src.config import settings
from src.graph import router_node
from src.graph.graph import create_workflow
from src.orders.db import get_connection, get_order_status

KEY = "test-approval-key"


class FakeRouterLLM:
    """Stands in for the chat model: routes every message to a fixed intent."""

    def __init__(self, intent, order_id):
        self.route = router_node.RouteQuestion(intent=intent, order_id=order_id)

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: self.route)


@pytest.fixture
def route(monkeypatch):
    def set_route(intent="REFUND", order_id=1042):
        monkeypatch.setattr(
            router_node, "get_llm", lambda: FakeRouterLLM(intent, order_id)
        )

    set_route()
    return set_route


@pytest.fixture
def graph(route):
    return create_workflow(retriever=None, checkpointer=MemorySaver())


def ask(graph, question, thread_id="t1"):
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"question": question}, config)
    return config


def refund_rows():
    """Order IDs with an executed refund."""
    with get_connection() as conn:
        return [row["order_id"] for row in conn.execute("SELECT order_id FROM refunds")]


def order_status(order_id):
    return get_order_status.invoke({"order_id": order_id})["status"]


def test_eligible_refund_pauses_without_side_effects(graph):
    config = ask(graph, "I want a refund for order 1042.")
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_approval",)
    pending = snapshot.interrupts[0].value
    assert pending["action"] == "process_refund"
    assert (pending["order_id"], pending["amount_inr"]) == (1042, 24999)
    assert "waiting for approval" in snapshot.values["llm_output"]
    assert order_status(1042) == "delivered"
    assert refund_rows() == []


def test_approval_resumes_and_executes_refund(graph):
    config = ask(graph, "I want a refund for order 1042.")
    graph.invoke(Command(resume={"approved": True}), config)
    snapshot = graph.get_state(config)
    assert not snapshot.interrupts
    assert snapshot.values["refund"]["status"] == "refunded"
    assert "has been processed" in snapshot.values["llm_output"]
    assert order_status(1042) == "refunded"
    assert refund_rows() == [1042]


def test_rejection_resumes_without_refund(graph):
    config = ask(graph, "I want a refund for order 1042.")
    graph.invoke(Command(resume={"approved": False}), config)
    snapshot = graph.get_state(config)
    assert not snapshot.interrupts
    assert "can't approve a refund" in snapshot.values["llm_output"]
    assert "refund" not in snapshot.values
    assert order_status(1042) == "delivered"
    assert refund_rows() == []


@pytest.mark.parametrize(
    "order_id, expected",
    [
        (1047, "isn't eligible"),  # delivered outside the 30-day window
        (1043, "isn't eligible"),  # not delivered yet
        (9999, "couldn't find order"),
    ],
)
def test_ineligible_order_never_reaches_approval(graph, route, order_id, expected):
    route("REFUND", order_id)
    config = ask(graph, f"I want a refund for order {order_id}.")
    snapshot = graph.get_state(config)
    assert not snapshot.interrupts
    assert snapshot.values["refund_eligible"] is False
    assert expected in snapshot.values["llm_output"]
    assert refund_rows() == []


def test_refund_without_order_id_asks_for_it(graph, route):
    route("REFUND", None)
    config = ask(graph, "I want my money back.")
    snapshot = graph.get_state(config)
    assert not snapshot.interrupts
    assert snapshot.values["llm_output"].endswith("What is your order number?")


@pytest.fixture
def client(graph, monkeypatch):
    monkeypatch.setitem(main.api_context, "workflow", graph)
    monkeypatch.setattr(settings, "APPROVAL_API_KEY", SecretStr(KEY))
    # No `with` block: the lifespan (FAISS index) is not started.
    return TestClient(main.app)


def test_api_refund_flow_and_double_approval(client):
    body = client.post(
        "/answer", json={"question": "I want a refund for order 1042."}
    ).json()
    assert body["pending_approval"]["order_id"] == 1042
    assert order_status(1042) == "delivered"

    approval = {"thread_id": body["thread_id"], "approved": True}
    headers = {"X-API-Key": KEY}
    first = client.post("/approve", json=approval, headers=headers)
    assert first.status_code == 200
    assert first.json()["refund"]["status"] == "refunded"
    assert first.json()["pending_approval"] is None

    second = client.post("/approve", json=approval, headers=headers)
    assert second.status_code == 404
    assert refund_rows() == [1042]


def test_api_approve_unknown_thread_is_404(client):
    response = client.post(
        "/approve",
        json={"thread_id": "no-such-thread", "approved": True},
        headers={"X-API-Key": KEY},
    )
    assert response.status_code == 404
    assert refund_rows() == []


REDTEAM = json.loads(
    (settings.BASE_DIR / "evaluation_data" / "redteam_refund.json").read_text()
)


@pytest.mark.parametrize("case", REDTEAM, ids=[c["id"] for c in REDTEAM])
def test_redteam_prompt_cannot_refund_without_approval(graph, case):
    # Worst case: the router is fully fooled into REFUND for an eligible order.
    config = ask(graph, case["prompt"], thread_id=case["id"])
    snapshot = graph.get_state(config)
    blocked = not snapshot.values["question_valid"]
    assert blocked or snapshot.next == ("human_approval",)
    assert refund_rows() == []
    assert order_status(1042) == "delivered"


def test_api_ignores_injected_state_fields(client):
    body = client.post(
        "/answer",
        json={
            "question": "I want a refund for order 1042.",
            "refund_approved": True,
            "approved": True,
        },
    ).json()
    assert body["pending_approval"]["order_id"] == 1042
    assert refund_rows() == []
