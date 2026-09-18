import json
from pathlib import Path

import numpy as np

from open_music import Sample
from open_music.corpus import checksum, load_external_manifest
from open_music.discovery import (
    cluster_candidates,
    fit_candidate_gain,
    fit_module_event,
    fit_modules_iteratively,
    repeat_similarity_matrix,
    select_canonical_candidates,
)
from open_music.external_benchmark import run_external_benchmark
from open_music.wav import write_wav


def test_external_manifest_has_pinned_cc0_recording() -> None:
    corpus = load_external_manifest(Path("corpus/external-small.json"))
    recording = corpus.recordings[0]

    assert recording.id == "prehistoric-drum-loop"
    assert recording.size == 996_880
    assert recording.checksum_algorithm == "sha256"
    assert len(recording.checksum_value) == 64
    assert recording.license_spdx == "CC0-1.0"
    assert recording.source["author"] == "hornpipe2"
    assert {item.id for item in corpus.recordings} == {
        "prehistoric-drum-loop",
        "wip-loop-track-02",
        "slowdrum-track-02",
    }
    assert all(len(item.checksum_value) == 64 for item in corpus.recordings)


def test_repeat_similarity_is_symmetric_and_finds_repeats() -> None:
    attack = np.exp(-np.arange(256) / 30.0)[:, None]
    different = np.sin(np.arange(256) * 0.7)[:, None] * np.exp(-np.arange(256) / 30.0)[:, None]
    matrix = repeat_similarity_matrix(
        (attack, different, attack.copy()),
        spectral_windows=(256,),
        time_offsets=(0,),
    )

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


def test_clustering_is_transitive_and_canonical_selection_is_deterministic() -> None:
    matrix = (
        (1.0, 0.998, 0.2, 0.1),
        (0.998, 1.0, 0.997, 0.1),
        (0.2, 0.997, 1.0, 0.1),
        (0.1, 0.1, 0.1, 1.0),
    )

    clusters = cluster_candidates(matrix, minimum_similarity=0.995)

    assert clusters == ((0, 1, 2), (3,))
    assert select_canonical_candidates(matrix, clusters) == (1, 3)


def test_candidate_gain_recovers_amplitude_scale() -> None:
    canonical = np.linspace(-1.0, 1.0, 32)[:, None]

    assert abs(fit_candidate_gain(canonical * 0.4, canonical) - 0.4) < 1e-12


def test_module_fit_refines_coarse_onset_and_gain() -> None:
    canonical = np.zeros((64, 1))
    canonical[:8, 0] = np.linspace(1.0, 0.1, 8)
    audio = np.zeros((256, 2))
    audio[103:167] = canonical * 0.4
    sample = Sample("module", canonical, 8_000, "CC0-1.0", "generated")

    event, score = fit_module_event(audio, sample, 96, search_radius=16, probe_frames=64)

    assert event.start_frame == 103
    assert abs(event.gain - 0.4) < 1e-12
    assert abs(score - 1.0) < 1e-12


def test_iterative_module_fit_returns_the_render_residual() -> None:
    canonical = np.zeros((32, 1))
    canonical[:4, 0] = (1.0, 0.7, 0.3, 0.1)
    audio = np.zeros((160, 2))
    audio[20:52] += canonical * 0.5
    audio[100:132] += canonical * 0.8
    sample = Sample("module", canonical, 8_000, "CC0-1.0", "generated")

    events, residual, scores = fit_modules_iteratively(
        audio,
        (sample, sample),
        (16, 96),
        search_radius=8,
        probe_frames=32,
    )

    assert [event.start_frame for event in events] == [20, 100]
    assert np.allclose([event.gain for event in events], [0.5, 0.8])
    assert np.allclose(scores, [1.0, 1.0])
    assert np.allclose(residual, 0.0)


def test_external_benchmark_normalizes_mono_for_analysis(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    source_path = cache / "mono.wav"
    mono = np.zeros((2_000, 1))
    write_wav(source_path, 8_000, mono)
    data = source_path.read_bytes()
    manifest = {
        "schema_version": 1,
        "id": "mono-test",
        "recordings": [
            {
                "id": "mono",
                "url": source_path.as_uri(),
                "cache_path": "mono.wav",
                "size": len(data),
                "checksum": {"algorithm": "sha256", "value": checksum(data, "sha256")},
                "license": {
                    "spdx": "CC0-1.0",
                    "redistributable": True,
                    "derivatives": True,
                    "commercial_use": True,
                },
                "source": {"author": "generated"},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_external_benchmark(manifest_path, cache, tmp_path / "output")
    recording = report["recordings"][0]

    assert recording["channels"] == 1
    assert recording["analysis_channels"] == 2
    assert report["aggregate"]["recording_count"] == 1
