"""FastAPI server exposing the laya router to the web UI.

Serves static/index.html and a POST /api/predict endpoint that takes
{state, questions, model} and returns the laya prediction result. The router
is preloaded once at startup (see laya_startup.py and README "Startup
performance").

Run with: uv run uvicorn server:app

`model="jev"` swaps in TypeSafe's Jev through OpenRouter instead. Jev shares laya's
typed-question interface and its answer shape (see jev.py), so this endpoint's contract
and every client are unchanged — the only visible differences are latency, a reported
cost, and the absence of laya's `act_probability`.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from laya_startup import build_router

import jev

QuestionType = Literal["choice", "score", "noul"]
ModelOverride = Literal["auto", "english", "multilingual", "typed-decisions", "jev"]

STATIC_DIR = Path(__file__).parent / "static"


class Question(BaseModel):
    """One typed question, matching laya's question schema.

    choice -> criteria is a {label: description} dict; score -> criteria is a
    [labels] list; noul takes no criteria.
    """

    type: QuestionType
    instructions: str = Field(min_length=1)
    criteria: Any = None

    @field_validator("criteria")
    @classmethod
    def check_criteria(cls, value: Any, info) -> Any:
        qtype = info.data.get("type")
        if qtype == "choice":
            if not isinstance(value, dict) or not value:
                raise ValueError("choice questions need a non-empty criteria object")
            if not all(isinstance(k, str) and k for k in value):
                raise ValueError("choice criteria keys must be non-empty strings")
        elif qtype == "score":
            if not isinstance(value, list) or not value:
                raise ValueError("score questions need a non-empty criteria array")
        elif value is not None:
            raise ValueError("noul questions take no criteria")
        return value


class PredictRequest(BaseModel):
    state: dict[str, Any] = Field(min_length=1)
    questions: dict[str, Question] = Field(min_length=1)
    model: ModelOverride = "auto"


class _RouterState:
    router: Any = None
    # FastAPI runs sync endpoints in a threadpool, so concurrent requests can
    # call router.predict() at once — which crashes MPS/Metal ("a command
    # encoder is already encoding to this command buffer"). Serialize them.
    predict_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _RouterState.router = build_router()
    yield


app = FastAPI(title="CJet web UI", lifespan=lifespan)


def _stringify_state(state: dict[str, Any]) -> dict[str, str]:
    """laya accepts arbitrary state values; coerce them to strings for the API contract."""
    return {k: str(v) for k, v in state.items()}


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    questions = {
        name: q.model_dump(exclude_none=True) for name, q in request.questions.items()
    }
    start = time.perf_counter()

    # Jev is a network call that never touches the local model, so it deliberately does
    # NOT take predict_lock (that lock exists because concurrent encoder calls crash
    # MPS/Metal) and deliberately does not 503 when the laya router is still loading —
    # it does not need the router, only the key.
    if request.model == "jev":
        try:
            result = jev.predict(_stringify_state(request.state), questions)
        except jev.JevError as exc:
            raise HTTPException(
                status_code=502 if exc.status else 500,
                detail=f"jev failed: {exc}" + (f" — {exc.body[:400]}" if exc.body else ""),
            ) from exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "result": result,
            "latency_ms": latency_ms,
            "routing": {"model": "jev", "repo": jev.JEV_MODEL, "reason": "explicit model=jev"},
        }

    router = _RouterState.router
    if router is None:
        raise HTTPException(status_code=503, detail="router is still loading")
    try:
        with _RouterState.predict_lock:
            result = router.predict(
                _stringify_state(request.state),
                questions,
                **({"model": request.model} if request.model != "auto" else {}),
            )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    return {"result": result, "latency_ms": latency_ms}


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ready": _RouterState.router is not None}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")