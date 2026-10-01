"""Tests for the /api/predict endpoint with a stubbed router (no model load).

uv run pytest test_server.py
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import server as server_module


@pytest.fixture()
def client(monkeypatch):
    """TestClient with a stub router that echoes the request it received."""

    class StubRouter:
        def predict(self, state, questions, **kwargs):
            self.last_call = {"state": state, "questions": questions, "kwargs": kwargs}
            return {
                "answers": {
                    qid: {"type": q["type"], "confidence": 0.9}
                    for qid, q in questions.items()
                },
                "routing": {"model": kwargs.get("model", "english")},
            }

    stub = StubRouter()
    monkeypatch.setattr(server_module._RouterState, "router", stub)
    # Deliberately NOT a `with` block: entering the TestClient context would run
    # the app lifespan and replace the stub with the real (slow) router.
    yield TestClient(server_module.app), stub


VALID_CHOICE = {
    "type": "choice",
    "instructions": "Which department?",
    "criteria": {"billing": "payments", "other": "everything else"},
}
VALID_NOUL = {"type": "noul", "instructions": "Churn risk?"}


def test_predict_success(client):
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"subject": "Duplicate charge"},
        "questions": {"department": VALID_CHOICE, "churn_risk": VALID_NOUL},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["routing"]["model"] == "english"  # auto -> no override
    assert set(body["result"]["answers"]) == {"department", "churn_risk"}
    assert body["latency_ms"] >= 0
    assert stub.last_call["state"] == {"subject": "Duplicate charge"}


def test_predict_forwards_model_override(client):
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
        "model": "typed-decisions",
    })
    assert response.status_code == 200
    assert stub.last_call["kwargs"] == {"model": "typed-decisions"}


def test_predict_strips_criteria_for_noul(client):
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
    })
    assert response.status_code == 200
    assert stub.last_call["questions"]["q"] == {"type": "noul", "instructions": "Churn risk?"}


def test_rejects_choice_without_criteria(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {}}},
    })
    assert response.status_code == 422


def test_rejects_score_without_criteria(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": {"type": "score", "instructions": "how bad", "criteria": []}},
    })
    assert response.status_code == 422


def test_rejects_noul_with_criteria(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": {"type": "noul", "instructions": "yes?", "criteria": ["x"]}},
    })
    assert response.status_code == 422


def test_rejects_unknown_question_type(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": {"type": "essay", "instructions": "write"}},
    })
    assert response.status_code == 422


def test_rejects_empty_state_and_questions(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={"state": {}, "questions": {"q": VALID_NOUL}})
    assert response.status_code == 422
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"}, "questions": {}})
    assert response.status_code == 422


def test_rejects_unknown_model(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
        "model": "gpt-4",
    })
    assert response.status_code == 422


def test_health_ready(client):
    test_client, _ = client
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_index_served(client):
    test_client, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert "CJet web UI" in response.text


def test_concurrent_predictions_are_serialized(client, monkeypatch):
    """router.predict() must never run concurrently (MPS/Metal crashes if it does)."""
    import threading

    test_client, stub = client
    overlap = []

    def slow_predict(state, questions, **kwargs):
        inside = {"entered": True}
        overlap.append(inside)
        time.sleep(0.05)
        overlap.remove(inside)
        return {"answers": {}, "routing": {}}

    monkeypatch.setattr(stub, "predict", slow_predict)
    threads = [
        threading.Thread(
            target=test_client.post,
            args=("/api/predict",),
            kwargs={"json": {"state": {"body": "hi"}, "questions": {"q": VALID_NOUL}}},
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlap, "router.predict ran concurrently"