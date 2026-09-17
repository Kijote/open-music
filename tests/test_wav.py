from pathlib import Path

import numpy as np

from open_music.wav import read_wav, write_wav


def test_wav_roundtrip(tmp_path: Path) -> None:
    audio = np.column_stack(
        (
            np.linspace(-0.8, 0.8, 100),
            np.linspace(0.8, -0.8, 100),
        )
    )
    path = tmp_path / "roundtrip.wav"

    write_wav(path, 8_000, audio)
    sample_rate, restored = read_wav(path)

    assert sample_rate == 8_000
    assert restored.shape == audio.shape
    assert np.max(np.abs(restored - audio)) < 1 / 32767
