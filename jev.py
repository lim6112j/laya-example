"""Jev (TypeSafe) decision client, shaped to look exactly like laya.

Jev and laya share an interface: a typed question — `{type, instructions, criteria}` —
asked over a state, answering with `{type, choice, probabilities, confidence}`. This module
exists so that swap is invisible to everything downstream: `predict()` returns the same
envelope `Agent.system_one()` returns, so the browser and the A/B harness read
`result.answers.<qid>.choice` either way and neither has a branch on who answered.

Two deliberate differences, both handled here rather than in the callers:

- Jev returns `usage.cost` in USD; laya returns no cost at all. Carried through as
  `cost_usd` (and inside `usage`) so a comparison can quote it. laya's marginal cost is
  $0 — it runs locally — so the number is only ever meaningful next to that.
- Jev has no `act_probability`. The confidence gate in fleet.html already defaults that
  field to 1, so its absence is not a special case.

Uses stdlib urllib rather than adding an HTTP dependency for one POST. The key is read
from the environment and never leaves the server process.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping

JEV_MODEL = "typesafe/jev-1.13"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
KEY_ENV = "OPENROUTER_API_KEY"
# Jev is a network round trip plus inference; 30s is generous but bounded, so a hung
# upstream surfaces as an error the UI can show rather than a request that never returns.
DEFAULT_TIMEOUT = 30.0


class JevError(RuntimeError):
    """An upstream failure, carrying enough of the response to diagnose it."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def api_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise JevError(
            f"{KEY_ENV} is not set — Jev is reached through OpenRouter and needs a key. "
            f"Export it in the server's environment; it is never sent to the browser."
        )
    return key


def predict(
    state: Mapping[str, Any],
    questions: Mapping[str, Dict[str, Any]],
    model: str = JEV_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Ask Jev, and return laya's result envelope.

    Returns:
        {"model", "provider", "answers", "usage", "cost_usd"} — the same shape
        `laya.Agent.system_one` produces, plus the cost Jev reports.
    """
    payload = {"model": model, "state": dict(state), "questions": dict(questions)}
    request = urllib.request.Request(
        DECISIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise JevError(f"Jev request failed ({exc.code})", status=exc.code, body=detail) from exc
    except urllib.error.URLError as exc:
        raise JevError(f"could not reach OpenRouter: {exc.reason}") from exc

    answers = body.get("answers") or {}
    if not answers:
        raise JevError("Jev returned no answers", body=json.dumps(body)[:500])
    usage = body.get("usage") or {}
    return {
        # Jev pins a dated snapshot (typesafe/jev-1.13-20260917); keep it, it is the
        # only way to tell which weights answered.
        "model": body.get("model", model),
        "provider": body.get("provider", "TypeSafe"),
        "answers": answers,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "cost_usd": usage.get("cost", 0.0),
    }
