import importlib
import os
from queue import Full
import sys
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class ServerCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = Mock()
        cls.backend.create_tts_task.return_value = "hazuki_1700000000_deadbeef"
        sys.modules.pop("server", None)
        with (
            patch.dict(
                os.environ,
                {
                    "BASE_STATIC_URL": "https://example.test/audio",
                    "PRELOAD_ON_STARTUP": "",
                    "TASK_RATE_LIMIT": "100/day",
                },
            ),
            patch("wrapper.IrodoriWrapper", return_value=cls.backend),
        ):
            cls.server = importlib.import_module("server")
        cls.client = TestClient(cls.server.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def setUp(self) -> None:
        self.backend.create_tts_task.reset_mock(return_value=True, side_effect=True)
        self.backend.create_tts_task.return_value = "hazuki_1700000000_deadbeef"
        self.backend.get_task_status.reset_mock(return_value=True, side_effect=True)

    def test_preloads_model_on_startup_by_default(self) -> None:
        self.assertTrue(self.server.PRELOAD_ON_STARTUP)
        self.backend.preload.assert_called_once_with()

    def test_skips_preload_when_env_disabled(self) -> None:
        disabled_backend = Mock()
        with (
            patch.dict(
                os.environ,
                {
                    "BASE_STATIC_URL": "https://example.test/audio",
                    "PRELOAD_ON_STARTUP": "false",
                    "TASK_RATE_LIMIT": "100/day",
                },
            ),
            patch("wrapper.IrodoriWrapper", return_value=disabled_backend),
        ):
            sys.modules.pop("server", None)
            try:
                disabled_server = importlib.import_module("server")
                with TestClient(disabled_server.app):
                    pass
            finally:
                sys.modules["server"] = self.server

        self.assertFalse(disabled_server.PRELOAD_ON_STARTUP)
        disabled_backend.preload.assert_not_called()

    @staticmethod
    def _genie_payload() -> dict[str, str]:
        return {
            "character_name": "はづき",
            "reference_audio_id": "special_communications/490300191/4903001910010",
            "reference_audio_text": "お誕生日おめでとうございます、プロデューサーさん♪",
            "text": "テスト音声です。",
        }

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_accepts_genie_payload_and_optional_extensions(self) -> None:
        resp = self.client.post(
            "/tasks",
            json=self._genie_payload(),
            headers={"x-client-ip": "203.0.113.1"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"task_id": "hazuki_1700000000_deadbeef"})

        resp2 = self.client.post(
            "/tasks",
            json={
                "character_name": "hazuki",
                "text": "埋め込みフォールバックのテストです。",
                "caption": "優しい声で話している。",
                "num_steps": 6,
                "seed": 42,
            },
            headers={"x-client-ip": "203.0.113.2"},
        )
        self.assertEqual(resp2.status_code, 200)
        self.backend.create_tts_task.assert_called_with(
            character_name="hazuki",
            reference_audio_id="",
            reference_audio_text="",
            text="埋め込みフォールバックのテストです。",
            caption="優しい声で話している。",
            num_steps=6,
            seed=42,
            duration_scale=None,
        )

    def test_limits_task_creation_to_100_per_ip_per_day(self) -> None:
        headers = {"x-client-ip": "203.0.113.10"}
        for _ in range(100):
            response = self.client.post("/tasks", json=self._genie_payload(), headers=headers)
            self.assertEqual(response.status_code, 200)

        response = self.client.post("/tasks", json=self._genie_payload(), headers=headers)
        self.assertEqual(response.status_code, 429)
        self.assertIn("retry-after", response.headers)

    def test_returns_503_when_task_queue_is_full(self) -> None:
        self.backend.create_tts_task.side_effect = Full

        response = self.client.post(
            "/tasks",
            json=self._genie_payload(),
            headers={"x-client-ip": "203.0.113.11"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["retry-after"], "60")

    def test_returns_404_when_task_not_found(self) -> None:
        self.backend.get_task_status.return_value = {"status": "not_found", "pending": 3}

        response = self.client.get("/tasks/unknown")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": {"status": "not_found", "pending": 3}})

    def test_returns_audio_expirations(self) -> None:
        self.backend.get_task_status.return_value = {
            "status": "completed",
            "pending": 0,
            "save_path": "wav/task.wav",
            "save_path_compressed": "ogg/task.ogg",
            "wav_expires_at": "2026-09-12T00:00:00+00:00",
            "ogg_expires_at": "2026-09-28T00:00:00+00:00",
        }

        response = self.client.get("/tasks/task")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "completed",
                "pending": 0,
                "save_path": "https://example.test/audio/wav/task.wav",
                "save_path_compressed": "https://example.test/audio/ogg/task.ogg",
                "wav_expires_at": "2026-09-12T00:00:00+00:00",
                "ogg_expires_at": "2026-09-28T00:00:00+00:00",
                "error": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
