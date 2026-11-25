from collections import deque
from pathlib import Path
from queue import Full
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, patch

from engine import resolve_effective_precision
from r2_storage import R2Storage
from wrapper import IrodoriWrapper, TaskRecord


def _make_bare_wrapper(directory: str) -> IrodoriWrapper:
    wrapper = object.__new__(IrodoriWrapper)
    wrapper.speakers_dir = Path(directory) / "speakers"
    wrapper.speakers_dir.mkdir(parents=True, exist_ok=True)
    wrapper.tmp_reference_dir = Path(directory) / "tmp_refs"
    wrapper.tmp_reference_dir.mkdir(parents=True, exist_ok=True)
    wrapper.output_dir = Path(directory) / "output"
    wrapper.output_dir.mkdir(parents=True, exist_ok=True)
    wrapper.reference_resource_server = "https://ref.example.test/events"
    wrapper.reference_download_timeout = 10.0
    wrapper.fallback_to_embed_on_ref_error = True
    wrapper._storage = Mock()
    wrapper._storage.upload_audio.return_value = (
        "wav/task.wav",
        "ogg/task.ogg",
        "2026-09-12T00:00:00+00:00",
        "2026-09-28T00:00:00+00:00",
    )
    wrapper._engine = Mock()
    wrapper._engine.synthesize.return_value = {
        "sample_rate": 48000,
        "audio_sec": 3.2,
        "infer_sec": 1.1,
        "rtf": 0.34,
        "num_steps": 6,
        "used_seed": 42,
        "mode": "ref_wav",
    }
    return wrapper


def _make_task(
    directory: str,
    *,
    character_name: str = "はづき",
    reference_audio_id: str = "special/123",
    caption: str | None = None,
) -> TaskRecord:
    return {
        "task_id": "task",
        "character_name": character_name,
        "reference_audio_id": reference_audio_id,
        "reference_audio_text": "ignored transcript",
        "text": "こんにちは、プロデューサーさん。",
        "caption": caption,
        "num_steps": None,
        "seed": None,
        "duration_scale": None,
        "wav_path": str(Path(directory) / "task.wav"),
        "ogg_path": str(Path(directory) / "task.ogg"),
        "wav_key": None,
        "ogg_key": None,
        "wav_expires_at": None,
        "ogg_expires_at": None,
        "status": "running",
        "error": None,
    }


class CompressionTest(unittest.TestCase):
    @patch("wrapper.AudioSegment.from_file")
    def test_compresses_to_ogg_opus(self, from_file: Mock) -> None:
        wrapper = object.__new__(IrodoriWrapper)

        wrapper._compress_wav_to_ogg("task.wav", "task.ogg")

        from_file.assert_called_once_with("task.wav", format="wav")
        from_file.return_value.export.assert_called_once_with(
            "task.ogg",
            format="ogg",
            codec="libopus",
            bitrate="48k",
        )


def _fake_synth(mode: str):
    def _run(**kw: object) -> dict[str, object]:
        Path(str(kw["save_path"])).write_bytes(b"wav")
        return {"mode": mode, "num_steps": 6, "audio_sec": 3.0, "infer_sec": 1.0, "rtf": 0.33}
    return _run


class ProcessTaskTest(unittest.TestCase):
    def test_prefers_ref_wav_when_reference_audio_id_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = _make_bare_wrapper(directory)
            embed = wrapper.speakers_dir / "hazuki.speaker.safetensors"
            embed.write_bytes(b"embed")
            wrapper._get_reference_audio_path = Mock(return_value=str(Path(directory) / "ref.wav"))
            wrapper._compress_wav_to_ogg = Mock(
                side_effect=lambda _wav, ogg: Path(ogg).write_bytes(b"ogg")
            )
            task = _make_task(directory)
            wrapper._engine.synthesize.side_effect = _fake_synth("ref_wav")

            wrapper._process_task(task)

            wrapper._engine.synthesize.assert_called_once_with(
                ref_wav=str(Path(directory) / "ref.wav"),
                text=task["text"],
                save_path=task["wav_path"],
                caption=None,
                num_steps=None,
                seed=None,
                duration_scale=None,
            )
            self.assertEqual(
                wrapper._storage.upload_audio.call_args,
                call("task", task["wav_path"], task["ogg_path"]),
            )
            self.assertEqual(task["wav_key"], "wav/task.wav")
            self.assertEqual(task["ogg_key"], "ogg/task.ogg")
            self.assertFalse(Path(task["wav_path"]).exists())
            self.assertFalse(Path(task["ogg_path"]).exists())

    def test_falls_back_to_embed_when_reference_audio_id_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = _make_bare_wrapper(directory)
            embed = wrapper.speakers_dir / "hazuki.speaker.safetensors"
            embed.write_bytes(b"embed")
            wrapper._compress_wav_to_ogg = Mock(
                side_effect=lambda _wav, ogg: Path(ogg).write_bytes(b"ogg")
            )
            task = _make_task(directory, reference_audio_id="   ", caption="優しい声")
            wrapper._engine.synthesize.side_effect = _fake_synth("ref_embed")

            wrapper._process_task(task)

            wrapper._engine.synthesize.assert_called_once_with(
                ref_embed=str(embed),
                text=task["text"],
                save_path=task["wav_path"],
                caption="優しい声",
                num_steps=None,
                seed=None,
                duration_scale=None,
            )

    def test_falls_back_to_embed_when_ref_wav_download_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = _make_bare_wrapper(directory)
            embed = wrapper.speakers_dir / "hazuki.speaker.safetensors"
            embed.write_bytes(b"embed")
            wrapper._get_reference_audio_path = Mock(side_effect=RuntimeError("404 not found"))
            wrapper._compress_wav_to_ogg = Mock(
                side_effect=lambda _wav, ogg: Path(ogg).write_bytes(b"ogg")
            )
            task = _make_task(directory, reference_audio_id="missing/clip")
            wrapper._engine.synthesize.side_effect = _fake_synth("ref_embed")

            wrapper._process_task(task)

            wrapper._engine.synthesize.assert_called_once_with(
                ref_embed=str(embed),
                text=task["text"],
                save_path=task["wav_path"],
                caption=None,
                num_steps=None,
                seed=None,
                duration_scale=None,
            )

    def test_raises_when_ref_wav_fails_and_fallback_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = _make_bare_wrapper(directory)
            wrapper.fallback_to_embed_on_ref_error = False
            (wrapper.speakers_dir / "hazuki.speaker.safetensors").write_bytes(b"embed")
            wrapper._get_reference_audio_path = Mock(side_effect=RuntimeError("network error"))
            task = _make_task(directory, reference_audio_id="missing/clip")

            with self.assertRaises(RuntimeError):
                wrapper._process_task(task)

    def test_rejects_path_traversal_in_reference_audio_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = _make_bare_wrapper(directory)
            with self.assertRaises(ValueError):
                wrapper._get_reference_audio_path("../../etc/passwd")


class TaskQueueTest(unittest.TestCase):
    def test_rejects_tasks_when_pending_queue_is_full(self) -> None:
        wrapper = object.__new__(IrodoriWrapper)
        wrapper.output_dir = Path("output")
        wrapper._tasks = {}
        wrapper._task_order = deque()
        wrapper._queue = deque(["existing"])
        wrapper._lock = threading.Lock()
        wrapper._condition = threading.Condition(wrapper._lock)
        wrapper._max_pending_tasks = 1
        wrapper._max_task_history = 10

        with self.assertRaises(Full):
            wrapper.create_tts_task("hazuki", "reference", "reference text", "text")

        self.assertEqual(list(wrapper._queue), ["existing"])
        self.assertEqual(wrapper._tasks, {})


class EnginePrecisionTest(unittest.TestCase):
    @patch("engine._cpu_supports_fast_bf16", return_value=True)
    def test_auto_precision_selects_bf16_on_fast_cpu(self, _mock: Mock) -> None:
        self.assertEqual(resolve_effective_precision("auto", "cpu"), "bf16")

    @patch("engine._cpu_supports_fast_bf16", return_value=False)
    def test_auto_precision_selects_fp32_without_bf16_isa(self, _mock: Mock) -> None:
        self.assertEqual(resolve_effective_precision("auto", "cpu"), "fp32")

    def test_explicit_precision_is_respected(self) -> None:
        self.assertEqual(resolve_effective_precision("bf16", "cpu"), "bf16")
        self.assertEqual(resolve_effective_precision("fp32", "cpu"), "fp32")


class R2StorageTest(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "R2_ACCOUNT_ID": "account",
            "R2_BUCKET": "audio",
            "R2_ACCESS_KEY_ID": "key",
            "R2_SECRET_ACCESS_KEY": "secret",
        },
    )
    @patch("r2_storage.boto3.client")
    def test_upload_audio_sets_keys_and_content_types(self, boto_client: Mock) -> None:
        client = boto_client.return_value
        client.head_object.side_effect = [
            {
                "Expiration": 'expiry-date="Sat, 12 Sep 2026 00:00:00 GMT", rule-id="expire-wav"'
            },
            {
                "Expiration": 'expiry-date="Mon, 28 Sep 2026 00:00:00 GMT", rule-id="expire-ogg"'
            },
        ]
        storage = R2Storage()

        result = storage.upload_audio("task", "task.wav", "task.ogg")

        boto_client.assert_called_once_with(
            service_name="s3",
            endpoint_url="https://account.r2.cloudflarestorage.com",
            aws_access_key_id="key",
            aws_secret_access_key="secret",
            region_name="auto",
        )
        self.assertEqual(
            result,
            (
                "wav/task.wav",
                "ogg/task.ogg",
                "2026-09-12T00:00:00+00:00",
                "2026-09-28T00:00:00+00:00",
            ),
        )


if __name__ == "__main__":
    unittest.main()
