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


# ---- jev ---------------------------------------------------------------------
# These stub jev.predict, so the suite needs neither a network nor an API key.

JEV_REPLY = {
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "assign": {
            "type": "choice",
            "choice": "bus_b",
            "probabilities": {"bus_a": 0.3, "bus_b": 0.55, "bus_c": 0.15},
            "confidence": 0.42,
        }
    },
    "usage": {"input_tokens": 373, "output_tokens": 42},
    "cost_usd": 1.5666e-05,
}


@pytest.fixture()
def jev_stub(monkeypatch):
    """Stub the Jev client and hand back the calls it received."""
    calls = []

    def fake_predict(state, questions, **kwargs):
        calls.append({"state": state, "questions": questions, **kwargs})
        return dict(JEV_REPLY)

    monkeypatch.setattr(server_module.jev, "predict", fake_predict)
    return calls


def test_jev_branch_returns_laya_shaped_result(client, jev_stub):
    """Jev answers through the same envelope, so no client needs a branch."""
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "jev",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["answers"]["assign"]["choice"] == "bus_b"
    assert body["result"]["cost_usd"] == 1.5666e-05
    assert body["routing"]["model"] == "jev"
    # The laya router is never touched on this path.
    assert not hasattr(stub, "last_call")
    assert jev_stub[0]["state"] == {"body": "hi"}
    assert jev_stub[0]["questions"]["assign"]["criteria"] == VALID_CHOICE["criteria"]


def test_jev_answer_has_no_act_probability(client, jev_stub):
    """Jev omits action.act_probability; clients must tolerate its absence."""
    test_client, _ = client
    body = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "jev",
    }).json()
    answer = body["result"]["answers"]["assign"]
    assert "action" not in answer
    # fleet.html reads it as (answer.action || {}).act_probability ?? 1
    assert (answer.get("action") or {}).get("act_probability", 1) == 1


def test_jev_works_while_the_laya_router_is_missing(client, jev_stub, monkeypatch):
    """Jev never touches the local model, so it must not 503 on a cold router."""
    test_client, _ = client
    monkeypatch.setattr(server_module._RouterState, "router", None)
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "jev",
    })
    assert response.status_code == 200


def test_jev_does_not_take_the_predict_lock(client, jev_stub, monkeypatch):
    """The lock exists for MPS/Metal; a ~1s network call must not serialise on it."""
    import threading

    test_client, _ = client
    overlap = []

    def slow_predict(state, questions, **kwargs):
        overlap.append(1)
        time.sleep(0.05)
        overlap.remove(1)
        return dict(JEV_REPLY)

    monkeypatch.setattr(server_module.jev, "predict", slow_predict)
    threads = [
        threading.Thread(
            target=test_client.post,
            args=("/api/predict",),
            kwargs={"json": {"state": {"body": "hi"},
                             "questions": {"assign": VALID_CHOICE},
                             "model": "jev"}},
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlap, "jev.predict ran while holding predict_lock"


def test_jev_error_becomes_a_502_with_the_upstream_body(client, monkeypatch):
    test_client, _ = client

    def boom(state, questions, **kwargs):
        raise server_module.jev.JevError("Jev request failed (402)", status=402,
                                         body='{"error":{"message":"insufficient credit"}}')

    monkeypatch.setattr(server_module.jev, "predict", boom)
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "jev",
    })
    assert response.status_code == 502
    assert "insufficient credit" in response.json()["detail"]


def test_jev_rejects_unknown_model_still(client, jev_stub):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
        "model": "gpt-4",
    })
    assert response.status_code == 422


def test_jev_requires_a_key(monkeypatch):
    """A missing key must name the variable rather than fail opaquely."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(server_module.jev.JevError, match="OPENROUTER_API_KEY"):
        server_module.jev.api_key()