"""Irodori-TTS MeanFlow CPU inference engine with BF16 support and watermark toggle."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def _cpu_supports_fast_bf16() -> bool:
    """Detect hardware BF16 ISA extensions (AMX-BF16 / AVX-512 BF16) on Linux x86_64."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return False
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    for line in text.splitlines():
        if line.startswith("flags"):
            tokens = set(line.split())
            return bool(tokens & {"amx_bf16", "avx512_bf16"})
    return False


def resolve_effective_precision(precision: str, device: str) -> str:
    """Resolve 'auto' / 'bf16' / 'fp32' into a concrete runtime precision string."""
    mode = str(precision or "auto").strip().lower()
    dev = str(device or "cpu").strip().lower()
    if mode == "auto":
        if dev == "cuda":
            return "bf16"
        if dev == "cpu":
            return "bf16" if _cpu_supports_fast_bf16() else "fp32"
        return "fp32"
    if mode not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported MODEL_PRECISION={precision!r}; expected auto, fp32, or bf16.")
    return mode


class _NoOpWatermarker:
    """Lightweight placeholder used when ENABLE_WATERMARK=false to skip loading SilentCipher."""

    def __init__(self, *, device: str, model_type: str = "44.1k") -> None:
        del device, model_type
        self.model = None

    @property
    def ready(self) -> bool:
        return False


@contextmanager
def _disabled_watermarker(runtime_module: Any) -> Iterator[None]:
    original = runtime_module.SilentCipherWatermarker
    runtime_module.SilentCipherWatermarker = _NoOpWatermarker
    try:
        yield
    finally:
        runtime_module.SilentCipherWatermarker = original


@dataclass(frozen=True)
class EngineConfig:
    checkpoint: Path
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    irodori_repo_path: Path | None = None
    model_device: str = "cpu"
    model_precision: str = "auto"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    cpu_threads: int = 8
    default_num_steps: int = 6
    max_num_steps: int = 16
    enable_watermark: bool = False


class IrodoriEngine:
    """Thread-safe single-model Irodori-TTS MeanFlow inference engine."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._runtime: Any = None
        self._effective_precision: str | None = None
        self._ensure_irodori_importable()

    def _ensure_irodori_importable(self) -> None:
        repo = self.config.irodori_repo_path
        if repo is not None and repo.is_dir():
            repo_str = str(repo.resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def effective_precision(self) -> str:
        if self._effective_precision is not None:
            return self._effective_precision
        return resolve_effective_precision(self.config.model_precision, self.config.model_device)

    def load(self) -> None:
        """Load model weights and DACVAE codec into memory (idempotent)."""
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("IRODORI_SDPA_CUDNN", "0")

        import torch
        import irodori_tts.inference_runtime as runtime_mod
        from irodori_tts.inference_runtime import (
            InferenceRuntime,
            RuntimeKey,
            _move_inference_module,
        )

        if self.config.model_device == "cpu" and self.config.cpu_threads > 0:
            max_cpus = os.cpu_count() or self.config.cpu_threads
            threads = max(1, min(max_cpus, self.config.cpu_threads))
            torch.set_num_threads(threads)
            logger.info("Configured PyTorch CPU threads=%d (os.cpu_count=%s)", threads, max_cpus)

        ckpt = self.config.checkpoint
        if not ckpt.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

        effective_precision = resolve_effective_precision(
            self.config.model_precision,
            self.config.model_device,
        )
        # Upstream InferenceRuntime.from_key rejects precision='bf16' on CPU.
        # Load as fp32 first, then safely cast non-complex floating parameters/buffers
        # via _move_inference_module (which preserves complex64 RoPE caches).
        load_precision = (
            "fp32"
            if self.config.model_device == "cpu" and effective_precision == "bf16"
            else effective_precision
        )

        key = RuntimeKey(
            checkpoint=str(ckpt),
            model_device=self.config.model_device,
            codec_repo=self.config.codec_repo,
            model_precision=load_precision,
            codec_device=self.config.codec_device,
            codec_precision=self.config.codec_precision,
            compile_model=False,
        )

        t0 = time.perf_counter()
        logger.info(
            "Loading Irodori runtime checkpoint=%s device=%s precision=%s watermark=%s",
            ckpt,
            self.config.model_device,
            effective_precision,
            self.config.enable_watermark,
        )
        if self.config.enable_watermark:
            rt = InferenceRuntime.from_key(key)
        else:
            with _disabled_watermarker(runtime_mod):
                rt = InferenceRuntime.from_key(key)
            rt.watermarker.model = None

        if self.config.model_device == "cpu" and effective_precision == "bf16":
            rt.model = _move_inference_module(
                rt.model,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )
            rt._model_dtype = torch.bfloat16

        elapsed = time.perf_counter() - t0
        logger.info(
            "Loaded Irodori runtime in %.2fs (model_dtype=%s, codec_dtype=%s, watermark=%s)",
            elapsed,
            next(rt.model.parameters()).dtype,
            rt.codec.dtype,
            rt.watermarker.ready,
        )
        self._runtime = rt
        self._effective_precision = effective_precision
        return rt

    def synthesize(
        self,
        *,
        text: str,
        save_path: str,
        ref_wav: str | None = None,
        ref_embed: str | None = None,
        caption: str | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        duration_scale: float | None = None,
    ) -> dict[str, Any]:
        """Synthesize a single utterance (no automatic long-text splitting) and write WAV."""
        text_clean = str(text or "").strip()
        if not text_clean:
            raise ValueError("text must be a non-empty string.")
        if bool(ref_wav) == bool(ref_embed):
            raise ValueError("Exactly one of ref_wav or ref_embed must be provided.")

        steps = self.config.default_num_steps if num_steps is None else int(num_steps)
        if steps < 1 or steps > self.config.max_num_steps:
            raise ValueError(
                f"num_steps must be between 1 and {self.config.max_num_steps}, got {steps}."
            )

        scale = 1.0 if duration_scale is None else float(duration_scale)
        if scale <= 0.0:
            raise ValueError(f"duration_scale must be > 0, got {scale}.")

        caption_clean = str(caption).strip() if caption is not None else ""

        with self._lock:
            rt = self._load_locked()
            from irodori_tts.inference_runtime import SamplingRequest, save_wav

            req = SamplingRequest(
                text=text_clean,
                caption=caption_clean or None,
                ref_wav=ref_wav,
                ref_embed=ref_embed,
                num_steps=steps,
                seed=seed,
                duration_scale=scale,
            )
            result = rt.synthesize(req, log_fn=None)
            save_wav(save_path, result.audio, result.sample_rate)

        audio_sec = float(result.audio.shape[-1]) / float(result.sample_rate)
        infer_sec = float(result.total_to_decode)
        return {
            "sample_rate": int(result.sample_rate),
            "audio_sec": round(audio_sec, 3),
            "infer_sec": round(infer_sec, 3),
            "rtf": round(infer_sec / max(1e-6, audio_sec), 3),
            "num_steps": steps,
            "used_seed": int(result.used_seed),
            "mode": "ref_wav" if ref_wav else "ref_embed",
        }
