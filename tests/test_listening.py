import json
from pathlib import Path

import numpy as np

from open_music.listening import RATING_DIMENSIONS, blind_assignment, write_listening_pack
from open_music.wav import read_wav


def test_blind_assignment_is_deterministic_and_balanced_by_identity() -> None:
    first = blind_assignment("recording-a")

    assert first == blind_assignment("recording-a")
    assert set(first) == {"A.wav", "B.wav"}
    assert set(first.values()) == {"source", "reconstruction"}


def test_listening_pack_exports_playable_audio_and_rating_page(tmp_path: Path) -> None:
    source = np.linspace(-0.5, 0.5, 128)[:, None]
    reconstruction = np.repeat(source * 0.9, 2, axis=1)
    residual = np.repeat(source * 0.1, 2, axis=1)

    answer = write_listening_pack(
        tmp_path,
        "mono-test",
        8_000,
        source,
        reconstruction,
        residual,
        {"explained_energy": 0.99},
    )

    expected = {
        "source.wav",
        "reconstruction.wav",
        "residual-true.wav",
        "residual-amplified.wav",
        "A.wav",
        "B.wav",
        "answer-key.json",
        "index.html",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert answer == json.loads((tmp_path / "answer-key.json").read_text(encoding="utf-8"))
    assert answer["rating_dimensions"] == list(RATING_DIMENSIONS)

    for name in ("source.wav", "reconstruction.wav", "A.wav", "B.wav"):
        _, audio = read_wav(tmp_path / name)
        assert audio.shape == (128, 1)
    _, amplified = read_wav(tmp_path / "residual-amplified.wav")
    assert abs(float(np.max(np.abs(amplified))) - 0.95) < 1e-4

    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Play A" in page
    assert "Play B" in page
    assert "Peak-normalized residual" in page
    assert all(dimension in page for dimension in RATING_DIMENSIONS)
