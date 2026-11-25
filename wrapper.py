"""Async task queue and TTS orchestration wrapper (GenieWrapper-compatible)."""

from __future__ import annotations

from collections import deque
import logging
import os
from pathlib import Path
from queue import Full
import threading
import time
from typing import Any, Deque, Dict, Literal, Optional, TypedDict, cast

from dotenv import load_dotenv
from pydub import AudioSegment

from downloader import download_and_convert_m4a_to_wav
from engine import EngineConfig, IrodoriEngine
from r2_storage import R2Storage
from speakers import resolve_speaker_embed, sanitize_task_prefix

logger = logging.getLogger(__name__)

TaskState = Literal["pending", "running", "completed", "failed"]
TaskQueryState = Literal["pending", "running", "completed", "failed", "not_found"]


class TaskRecord(TypedDict):
    task_id: str
    character_name: str
    reference_audio_id: str
    reference_audio_text: str
    text: str
    caption: Optional[str]
    num_steps: Optional[int]
    seed: Optional[int]
    duration_scale: Optional[float]
    wav_path: str
    ogg_path: str
    wav_key: Optional[str]
    ogg_key: Optional[str]
    wav_expires_at: Optional[str]
    ogg_expires_at: Optional[str]
    status: TaskState
    error: Optional[str]


class TaskStatus(TypedDict, total=False):
    status: TaskQueryState
    pending: int
    save_path: str
    save_path_compressed: str
    wav_expires_at: str
    ogg_expires_at: str
    error: Optional[str]


load_dotenv()


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _build_engine_config_from_env() -> EngineConfig:
    base_dir = Path(__file__).resolve().parent
    default_ckpt = base_dir.parent / "outputs/sc29_v1_meanflow_clarity_fix/model_meanflow_s8000_clarity_fix.safetensors"
    default_repo = base_dir.parent / "Irodori-TTS"

    ckpt_raw = os.getenv("MODEL_CHECKPOINT", "").strip()
    checkpoint = Path(ckpt_raw) if ckpt_raw else default_ckpt

    repo_raw = os.getenv("IRODORI_REPO_PATH", "").strip()
    irodori_repo = Path(repo_raw) if repo_raw else (default_repo if default_repo.is_dir() else None)

    return EngineConfig(
        checkpoint=checkpoint,
        codec_repo=os.getenv("CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim").strip(),
        irodori_repo_path=irodori_repo,
        model_device=os.getenv("MODEL_DEVICE", "cpu").strip(),
        model_precision=os.getenv("MODEL_PRECISION", "auto").strip(),
        codec_device=os.getenv("CODEC_DEVICE", "cpu").strip(),
        codec_precision=os.getenv("CODEC_PRECISION", "fp32").strip(),
        cpu_threads=_get_env_int("CPU_THREADS", 8),
        default_num_steps=_get_positive_env_int("DEFAULT_NUM_STEPS", 6),
        max_num_steps=_get_positive_env_int("MAX_NUM_STEPS", 16),
        enable_watermark=_get_env_bool("ENABLE_WATERMARK", False),
    )


class IrodoriWrapper:
    """GenieWrapper-compatible async TTS queue backed by Irodori-TTS MeanFlow."""

    def __init__(
        self,
        storage: R2Storage | None = None,
        engine: IrodoriEngine | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        default_speakers = base_dir.parent / "outputs/sc29_v1_speakers_s8000"
        speakers_raw = os.getenv("SPEAKERS_DIR", "").strip()
        self.speakers_dir = Path(speakers_raw) if speakers_raw else default_speakers
        self.tmp_reference_dir = Path(os.getenv("TMP_REFERENCE_DIR", "./tmp_references"))
        self.reference_resource_server = os.getenv("REFERENCE_RESOURCE_SERVER", "").strip().rstrip("/")
        self.reference_download_timeout = _get_env_float("REFERENCE_DOWNLOAD_TIMEOUT_SECONDS", 15.0)
        self.fallback_to_embed_on_ref_error = _get_env_bool("FALLBACK_TO_EMBED_ON_REF_ERROR", True)
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        self._storage = storage if storage is not None else R2Storage()
        self._engine = engine if engine is not None else IrodoriEngine(_build_engine_config_from_env())

        self.tmp_reference_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._cleanup_interval_seconds = _get_env_int("CLEANUP_INTERVAL_SECONDS", 60 * 60)
        self._cleanup_age_seconds = _get_env_int("CLEANUP_AGE_SECONDS", 24 * 60 * 60)
        self._max_pending_tasks = _get_positive_env_int("MAX_PENDING_TASKS", 20)
        self._max_task_history = _get_positive_env_int("MAX_TASK_HISTORY", 1000)

        self._tasks: Dict[str, TaskRecord] = {}
        self._task_order: Deque[str] = deque()
        self._queue: Deque[str] = deque()
        self._lock: threading.Lock = threading.Lock()
        self._condition: threading.Condition = threading.Condition(self._lock)

        self._worker: threading.Thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._cleanup_worker: threading.Thread = threading.Thread(
            target=self._cleanup_worker_loop,
            daemon=True,
        )
        self._cleanup_worker.start()

    def preload(self) -> None:
        """Eagerly load Irodori weights and codec during server startup."""
        self._engine.load()

    def _get_resource_url(self, resource_audio_id: str) -> str:
        if not self.reference_resource_server:
            raise ValueError("REFERENCE_RESOURCE_SERVER is not configured.")
        normalized = resource_audio_id.strip().lstrip("/")
        return f"{self.reference_resource_server}/{normalized}.m4a"

    def _get_reference_audio_path(self, resource_audio_id: str) -> str:
        normalized_resource = resource_audio_id.strip().lstrip("/")
        if not normalized_resource:
            raise ValueError("reference_audio_id is empty.")
        rel_parts = Path(normalized_resource).parts
        if any(part in {"", ".", ".."} for part in rel_parts):
            raise ValueError(f"Invalid reference_audio_id: {resource_audio_id!r}")

        audio_path = (self.tmp_reference_dir / f"{normalized_resource}.wav").resolve()
        base_dir = self.tmp_reference_dir.resolve()
        if base_dir not in audio_path.parents:
            raise ValueError(f"reference_audio_id escapes tmp_reference_dir: {resource_audio_id!r}")

        if audio_path.is_file():
            return str(audio_path)

        url = self._get_resource_url(normalized_resource)
        download_and_convert_m4a_to_wav(
            url,
            str(audio_path),
            timeout=self.reference_download_timeout,
        )
        return str(audio_path)

    def _get_speaker_embed_path(self, character_name: str) -> str | None:
        path = resolve_speaker_embed(character_name, self.speakers_dir)
        return str(path) if path is not None else None

    def _get_task_id(self, character_name: str) -> str:
        prefix = sanitize_task_prefix(character_name)
        random_suffix = os.urandom(4).hex()
        timestamp = int(time.time())
        return f"{prefix}_{timestamp}_{random_suffix}"

    def _get_wav_path(self, task_id: str) -> str:
        return str(self.output_dir / f"{task_id}.wav")

    def _get_ogg_path(self, task_id: str) -> str:
        return str(self.output_dir / f"{task_id}.ogg")

    def _compress_wav_to_ogg(self, wav_path: str, ogg_path: str) -> None:
        audio = AudioSegment.from_file(wav_path, format="wav")
        audio.export(ogg_path, format="ogg", codec="libopus", bitrate="48k")

    def _cleanup_old_outputs(self) -> None:
        now = time.time()
        for path in self.output_dir.iterdir():
            if path.suffix not in {".wav", ".ogg"}:
                continue
            try:
                if now - path.stat().st_mtime >= self._cleanup_age_seconds:
                    path.unlink()
            except FileNotFoundError:
                continue

    def _cleanup_worker_loop(self) -> None:
        while True:
            try:
                self._cleanup_old_outputs()
            except Exception:
                logger.exception("Failed during periodic output cleanup.")
            time.sleep(self._cleanup_interval_seconds)

    def _prune_task_history_locked(self) -> None:
        while len(self._tasks) > self._max_task_history and self._task_order:
            oldest_id = self._task_order[0]
            oldest = self._tasks.get(oldest_id)
            if oldest is None:
                self._task_order.popleft()
                continue
            if oldest["status"] in {"pending", "running"}:
                break
            self._task_order.popleft()
            self._tasks.pop(oldest_id, None)

    def _synthesize_task_audio(self, task: TaskRecord) -> dict[str, Any]:
        ref_id = task["reference_audio_id"].strip()
        common_kwargs: dict[str, Any] = {
            "text": task["text"],
            "save_path": task["wav_path"],
            "caption": task.get("caption"),
            "num_steps": task.get("num_steps"),
            "seed": task.get("seed"),
            "duration_scale": task.get("duration_scale"),
        }

        # Strategy: ref_wav (reference_audio_id) preferred, 49 KB ref_embed fallback.
        if ref_id:
            try:
                ref_wav_path = self._get_reference_audio_path(ref_id)
                return self._engine.synthesize(ref_wav=ref_wav_path, **common_kwargs)
            except Exception as exc:
                embed_path = (
                    self._get_speaker_embed_path(task["character_name"])
                    if self.fallback_to_embed_on_ref_error
                    else None
                )
                if embed_path is None:
                    raise
                logger.warning(
                    "Task %s ref_wav failed (%s); falling back to ref_embed=%s",
                    task["task_id"],
                    exc,
                    embed_path,
                )
                return self._engine.synthesize(ref_embed=embed_path, **common_kwargs)

        embed_path = self._get_speaker_embed_path(task["character_name"])
        if embed_path is None:
            raise FileNotFoundError(
                f"No speaker embedding found for character_name={task['character_name']!r} "
                f"under {self.speakers_dir}"
            )
        return self._engine.synthesize(ref_embed=embed_path, **common_kwargs)

    def _process_task(self, task: TaskRecord) -> None:
        meta = self._synthesize_task_audio(task)
        logger.info(
            "Task %s synthesized: mode=%s steps=%s audio=%.2fs infer=%.2fs rtf=%.2fx",
            task["task_id"],
            meta.get("mode"),
            meta.get("num_steps"),
            meta.get("audio_sec", 0.0),
            meta.get("infer_sec", 0.0),
            meta.get("rtf", 0.0),
        )
        self._compress_wav_to_ogg(task["wav_path"], task["ogg_path"])
        (
            task["wav_key"],
            task["ogg_key"],
            task["wav_expires_at"],
            task["ogg_expires_at"],
        ) = self._storage.upload_audio(
            task["task_id"],
            task["wav_path"],
            task["ogg_path"],
        )
        for path in (task["wav_path"], task["ogg_path"]):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                # The periodic cleanup worker will retry stale local outputs.
                pass

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                task_id = self._queue.popleft()
                task = self._tasks.get(task_id)
                if not task:
                    continue
                task["status"] = "running"
            try:
                self._process_task(task)
                with self._lock:
                    task["status"] = "completed"
                    self._prune_task_history_locked()
            except Exception as exc:
                logger.exception("Task %s failed", task_id)
                with self._lock:
                    task["status"] = "failed"
                    task["error"] = str(exc)
                    self._prune_task_history_locked()

    def create_tts_task(
        self,
        character_name: str,
        reference_audio_id: str = "",
        reference_audio_text: str = "",
        text: str = "",
        *,
        caption: str | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        duration_scale: float | None = None,
    ) -> str:
        task_id = self._get_task_id(character_name)
        wav_path = self._get_wav_path(task_id)
        ogg_path = self._get_ogg_path(task_id)
        task: TaskRecord = {
            "task_id": task_id,
            "character_name": character_name,
            "reference_audio_id": reference_audio_id,
            "reference_audio_text": reference_audio_text,
            "text": text,
            "caption": caption,
            "num_steps": num_steps,
            "seed": seed,
            "duration_scale": duration_scale,
            "wav_path": wav_path,
            "ogg_path": ogg_path,
            "wav_key": None,
            "ogg_key": None,
            "wav_expires_at": None,
            "ogg_expires_at": None,
            "status": "pending",
            "error": None,
        }
        with self._condition:
            if len(self._queue) >= self._max_pending_tasks:
                raise Full
            self._tasks[task_id] = task
            self._task_order.append(task_id)
            self._queue.append(task_id)
            self._prune_task_history_locked()
            self._condition.notify()
        return task_id

    def get_task_status(self, task_id: str) -> TaskStatus:
        with self._lock:
            task = self._tasks.get(task_id)
            pending = len(self._queue)
            if not task:
                not_found: TaskStatus = {
                    "status": "not_found",
                    "pending": pending,
                }
                return not_found
            response: TaskStatus = {
                "status": task["status"],
                "pending": pending,
                "error": task.get("error"),
            }
            if task["status"] == "completed":
                response["save_path"] = cast(str, task["wav_key"])
                response["save_path_compressed"] = cast(str, task["ogg_key"])
                if task["wav_expires_at"]:
                    response["wav_expires_at"] = task["wav_expires_at"]
                if task["ogg_expires_at"]:
                    response["ogg_expires_at"] = task["ogg_expires_at"]
            return response
