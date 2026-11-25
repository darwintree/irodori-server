"""Download reference M4A audio and convert to 48 kHz mono PCM WAV."""

from __future__ import annotations

import io
import os
from pathlib import Path

from pydub import AudioSegment
import requests


def download_and_convert_m4a_to_wav(
    url: str,
    output_filename: str,
    *,
    timeout: float = 15.0,
    target_sample_rate: int = 48000,
) -> None:
    """Fetch a remote M4A clip and atomically write a 48 kHz mono WAV file."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    m4a_data = io.BytesIO(response.content)
    audio = AudioSegment.from_file(m4a_data, format="m4a")
    audio = audio.set_channels(1).set_frame_rate(target_sample_rate)

    out_path = Path(output_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        audio.export(str(tmp_path), format="wav")
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
