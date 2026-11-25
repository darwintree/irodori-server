"""Genie-compatible FastAPI server backed by Irodori-TTS MeanFlow."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from queue import Full
from typing import AsyncIterator
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from wrapper import IrodoriWrapper

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class CreateTaskRequest(BaseModel):
    character_name: str
    reference_audio_id: str = ""
    reference_audio_text: str = ""
    text: str
    # Optional Irodori-TTS extensions (backward-compatible with genie-server clients)
    caption: str | None = None
    num_steps: int | None = Field(default=None, ge=1, le=40)
    seed: int | None = None
    duration_scale: float | None = Field(default=None, gt=0.0, le=3.0)


class CreateTaskResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    status: str
    pending: int
    save_path: str | None = None
    save_path_compressed: str | None = None
    wav_expires_at: str | None = None
    ogg_expires_at: str | None = None
    error: str | None = None


def _client_ip(request: Request) -> str:
    forwarded_ip = request.headers.get("x-client-ip")
    if forwarded_ip:
        return forwarded_ip
    return request.client.host if request.client else "unknown"


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BASE_STATIC_URL = os.getenv("BASE_STATIC_URL", "").rstrip("/")
if not BASE_STATIC_URL:
    raise ValueError("BASE_STATIC_URL must be set in the environment.")

TASK_RATE_LIMIT = os.getenv("TASK_RATE_LIMIT", "100/day").strip() or "100/day"
PRELOAD_ON_STARTUP = _get_env_bool("PRELOAD_ON_STARTUP", True)

wrapper = IrodoriWrapper()
limiter = Limiter(key_func=_client_ip, headers_enabled=True)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if PRELOAD_ON_STARTUP:
        logger.info("Preloading Irodori-TTS model on startup...")
        wrapper.preload()
    yield


app = FastAPI(title="Irodori TTS Server", version="0.1.0", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/tasks", response_model=CreateTaskResponse)
@limiter.limit(TASK_RATE_LIMIT)
def create_task(
    request: Request,
    response: Response,
    body: CreateTaskRequest,
) -> CreateTaskResponse:
    """Create a background TTS synthesis task."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    try:
        task_id = wrapper.create_tts_task(
            character_name=body.character_name,
            reference_audio_id=body.reference_audio_id,
            reference_audio_text=body.reference_audio_text,
            text=body.text,
            caption=body.caption,
            num_steps=body.num_steps,
            seed=body.seed,
            duration_scale=body.duration_scale,
        )
    except Full as exc:
        raise HTTPException(
            status_code=503,
            detail="task queue is full",
            headers={"Retry-After": "60"},
        ) from exc
    return CreateTaskResponse(task_id=task_id)


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Query the current status of a TTS task."""
    status = wrapper.get_task_status(task_id)
    if status["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail={
                "status": status["status"],
                "pending": status["pending"],
            },
        )
    if status.get("save_path"):
        status["save_path"] = (
            f"{BASE_STATIC_URL}/{quote(status['save_path'], safe='/')}"
        )
    if status.get("save_path_compressed"):
        status["save_path_compressed"] = (
            f"{BASE_STATIC_URL}/{quote(status['save_path_compressed'], safe='/')}"
        )
    return TaskStatusResponse(**status)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health endpoint for readiness probes."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=12451, reload=True)
