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
    assert "Final residual after all layers" in page
    assert all(dimension in page for dimension in RATING_DIMENSIONS)


def test_numerical_floor_is_not_amplified_and_structural_residual_is_separate(
    tmp_path: Path,
) -> None:
    source = np.linspace(-0.5, 0.5, 128)[:, None]
    numerical = source * 1e-16
    structural = source * 0.1

    answer = write_listening_pack(
        tmp_path,
        "residual-test",
        8_000,
        source,
        source - numerical,
        numerical,
        {},
        structural_residual=structural,
    )

    assert answer["final_residual"]["classification"] == "numerical_floor"
    assert answer["final_residual"]["amplification_gain"] == 1.0
    assert not answer["final_residual"]["amplification_applied"]
    assert answer["structural_residual"]["classification"] == "structural"
    assert answer["structural_residual"]["amplification_applied"]
    assert (tmp_path / "structural-residual-true.wav").exists()
    assert (tmp_path / "structural-residual-amplified.wav").exists()
