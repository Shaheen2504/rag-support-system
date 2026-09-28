from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api import main
from src.config import settings

KEY = "test-approval-key"
BODY = {"thread_id": "t1", "approved": True}


class FakeGraph:
    """Stands in for the compiled graph: one refund paused at human approval."""

    def __init__(self):
        self.resumed_with = None

    def get_state(self, config):
        paused = self.resumed_with is None
        return SimpleNamespace(
            values={"llm_output": "resumed" if not paused else "waiting"},
            interrupts=[SimpleNamespace(value={"order_id": 1042})] if paused else [],
        )

    def invoke(self, command, config):
        self.resumed_with = command.resume


@pytest.fixture
def graph(monkeypatch):
    fake = FakeGraph()
    monkeypatch.setitem(main.api_context, "workflow", fake)
    monkeypatch.setattr(settings, "APPROVAL_API_KEY", SecretStr(KEY))
    return fake


# No `with` block: the lifespan (FAISS, models) is not started.
client = TestClient(main.app)


def test_missing_key_rejected(graph):
    assert client.post("/approve", json=BODY).status_code == 401
    assert graph.resumed_with is None


def test_wrong_key_rejected(graph):
    response = client.post("/approve", json=BODY, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401
    assert graph.resumed_with is None


def test_unconfigured_key_rejects_everything(graph, monkeypatch):
    monkeypatch.setattr(settings, "APPROVAL_API_KEY", None)
    response = client.post("/approve", json=BODY, headers={"X-API-Key": KEY})
    assert response.status_code == 401
    assert graph.resumed_with is None


def test_correct_key_resumes_graph(graph):
    response = client.post("/approve", json=BODY, headers={"X-API-Key": KEY})
    assert response.status_code == 200
    assert graph.resumed_with == {"approved": True}
    assert response.json()["llm_output"] == "resumed"
    assert response.json()["pending_approval"] is None
