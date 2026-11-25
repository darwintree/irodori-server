from pathlib import Path
import tempfile
import unittest

from speakers import (
    SPEAKER_CATALOG,
    resolve_speaker_embed,
    resolve_speaker_id,
    sanitize_task_prefix,
)


class SpeakerResolutionTest(unittest.TestCase):
    def test_catalog_has_all_29_characters(self) -> None:
        self.assertEqual(len(SPEAKER_CATALOG), 29)
        self.assertEqual(len({entry.speaker_id for entry in SPEAKER_CATALOG}), 29)

    def test_resolves_english_japanese_and_chinese_aliases(self) -> None:
        self.assertEqual(resolve_speaker_id("hazuki"), "hazuki")
        self.assertEqual(resolve_speaker_id("HAZUKI"), "hazuki")
        self.assertEqual(resolve_speaker_id("はづき"), "hazuki")
        self.assertEqual(resolve_speaker_id("七草はづき"), "hazuki")
        self.assertEqual(resolve_speaker_id("七草叶月"), "hazuki")
        self.assertEqual(resolve_speaker_id("叶月"), "hazuki")
        self.assertEqual(resolve_speaker_id("櫻木真乃"), "mano")
        self.assertEqual(resolve_speaker_id("桜木真乃"), "mano")
        self.assertEqual(resolve_speaker_id("冬優子"), "fuyuko")
        self.assertEqual(resolve_speaker_id("樋口円香"), "madoka")
        self.assertIsNone(resolve_speaker_id("unknown_character"))

    def test_resolves_embed_path_and_blocks_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            speakers_dir = Path(directory)
            hazuki_file = speakers_dir / "hazuki.speaker.safetensors"
            hazuki_file.write_bytes(b"embed")
            custom_file = speakers_dir / "custom_npc.speaker.safetensors"
            custom_file.write_bytes(b"custom")

            self.assertEqual(resolve_speaker_embed("はづき", speakers_dir), hazuki_file)
            self.assertEqual(resolve_speaker_embed("custom_npc", speakers_dir), custom_file)
            self.assertIsNone(resolve_speaker_embed("../hazuki", speakers_dir))
            self.assertIsNone(resolve_speaker_embed("missing", speakers_dir))

    def test_sanitize_task_prefix_is_r2_safe(self) -> None:
        self.assertEqual(sanitize_task_prefix("はづき"), "hazuki")
        self.assertEqual(sanitize_task_prefix("custom/name..x"), "custom_name_x")
        self.assertEqual(sanitize_task_prefix("///"), "tts")


if __name__ == "__main__":
    unittest.main()
