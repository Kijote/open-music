import json
from pathlib import Path

import numpy as np

from open_music.corpus import load_external_manifest
from open_music.discovery import repeat_similarity_matrix


def test_external_manifest_has_pinned_cc0_recording() -> None:
    corpus = load_external_manifest(Path("corpus/external-small.json"))
    recording = corpus.recordings[0]

    assert recording.id == "prehistoric-drum-loop"
    assert recording.size == 996_880
    assert recording.checksum_algorithm == "sha256"
    assert len(recording.checksum_value) == 64
    assert recording.license_spdx == "CC0-1.0"
    assert recording.source["author"] == "hornpipe2"


def test_repeat_similarity_is_symmetric_and_finds_repeats() -> None:
    attack = np.exp(-np.arange(256) / 30.0)[:, None]
    different = np.sin(np.arange(256) * 0.7)[:, None] * np.exp(-np.arange(256) / 30.0)[:, None]
    matrix = repeat_similarity_matrix((attack, different, attack.copy()), feature_frames=256)

    assert matrix[0][0] == 1.0
    assert matrix[0][2] == 1.0
    assert matrix[0][1] < matrix[0][2]
    assert np.allclose(matrix, np.transpose(matrix))


def test_external_manifest_is_a_recording_not_a_known_event_timeline() -> None:
    raw = json.loads(Path("corpus/external-small.json").read_text(encoding="utf-8"))

    assert "recordings" in raw
    assert "assets" not in raw
    assert "tracks" not in raw
    assert "events" not in raw["recordings"][0]
