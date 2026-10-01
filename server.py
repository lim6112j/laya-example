"""FastAPI server exposing the laya router to the web UI.

Serves static/index.html and a POST /api/predict endpoint that takes
{state, questions, model} and returns the laya prediction result. The router
is preloaded once at startup (see laya_startup.py and README "Startup
performance").

Run with: uv run uvicorn server:app
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

QuestionType = Literal["choice", "score", "noul"]
ModelOverride = Literal["auto", "english", "multilingual", "typed-decisions"]

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


app = FastAPI(title="laya-example web UI", lifespan=lifespan)


def _stringify_state(state: dict[str, Any]) -> dict[str, str]:
    """laya accepts arbitrary state values; coerce them to strings for the API contract."""
    return {k: str(v) for k, v in state.items()}


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    router = _RouterState.router
    if router is None:
        raise HTTPException(status_code=503, detail="router is still loading")
    questions = {
        name: q.model_dump(exclude_none=True) for name, q in request.questions.items()
    }
    try:
        start = time.perf_counter()
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