import numpy as np

from open_music import Event, Sample, render
from open_music.discovery import (
    detect_onsets,
    detection_metrics,
    discover_events,
    extract_candidates,
)


SAMPLE_RATE = 8_000


def pulse(frames: int = 320) -> np.ndarray:
    time = np.arange(frames)
    return np.exp(-time / 35.0) * np.sin(2 * np.pi * time / 17.0)


def test_detects_repeated_attacks_and_extracts_windows() -> None:
    sample = Sample("pulse", pulse(), SAMPLE_RATE, "CC0-1.0", "generated")
    library = {sample.id: sample}
    expected = [Event("pulse", frame, gain=0.8) for frame in (0, 2_000, 4_000, 6_000)]
    mixture = render(library, expected, 8_000, SAMPLE_RATE)

    onsets = detect_onsets(
        mixture,
        SAMPLE_RATE,
        frame_length=128,
        hop_length=32,
        minimum_spacing_seconds=0.1,
    )
    candidates = extract_candidates(mixture, onsets, maximum_frames=500)

    assert len(onsets) == 4
    assert len(candidates) == 4
    assert all(candidate.shape[0] <= 500 for candidate in candidates)


def test_discovers_library_events_without_expected_timeline() -> None:
    sample = Sample("pulse", pulse(), SAMPLE_RATE, "CC0-1.0", "generated")
    library = {sample.id: sample}
    expected = [Event("pulse", frame, gain=gain) for frame, gain in ((0, 0.5), (3_000, 0.8))]
    mixture = render(library, expected, 5_000, SAMPLE_RATE)

    result = discover_events(
        mixture,
        library,
        SAMPLE_RATE,
        search_radius=1024,
        probe_frames=256,
    )
    metrics = detection_metrics(expected, result.events, tolerance_frames=32)

    assert len(result.events) == 2
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert all(event.sample_id == "pulse" for event in result.events)
    assert abs(result.events[0].gain - 0.5) < 1e-9
    assert abs(result.events[1].gain - 0.8) < 1e-9


def test_silence_has_no_onsets_or_events() -> None:
    silence = np.zeros((2_000, 2))
    sample = Sample("pulse", pulse(), SAMPLE_RATE, "CC0-1.0", "generated")

    assert detect_onsets(silence, SAMPLE_RATE) == ()
    assert discover_events(silence, {"pulse": sample}, SAMPLE_RATE).events == ()
