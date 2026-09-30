import numpy as np

from open_music.validation import ValidationThresholds, validate_replacement
from open_music.discovery import cluster_candidates, repeat_similarity_matrix


def tone(frequency: float, frames: int = 8192, sample_rate: int = 8000) -> np.ndarray:
    time = np.arange(frames) / sample_rate
    return np.sin(2 * np.pi * frequency * time)


def test_accepts_identical_signal_and_explicit_gain() -> None:
    reference = tone(220)
    result = validate_replacement(reference, reference * 0.5, maximum_lag_frames=8)

    assert result.accepted
    assert result.score > 0.99
    assert abs(result.gain - 2.0) < 1e-6


def test_rejects_equal_average_spectrum_with_reversed_temporal_order() -> None:
    low = tone(180, frames=4096)
    high = tone(620, frames=4096)
    forward = np.concatenate((low, high))
    reversed_order = np.concatenate((high, low))

    result = validate_replacement(forward, reversed_order, maximum_lag_frames=0)

    assert not result.accepted
    assert any(reason.startswith("local_time_frequency") for reason in result.rejection_reasons)


def test_rejects_local_note_change_hidden_by_compatible_frames() -> None:
    reference = tone(220)
    candidate = reference.copy()
    candidate[3072:5120] = tone(440, frames=2048)

    result = validate_replacement(reference, candidate, maximum_lag_frames=0)

    assert not result.accepted
    assert any(item.maximum_incompatible_run > 3 for item in result.resolutions)


def test_rejects_envelope_change_despite_same_global_rms() -> None:
    reference = tone(220)
    envelope = np.concatenate((np.full(4096, 0.25), np.full(4096, 1.75)))
    candidate = reference * envelope
    candidate *= np.sqrt(np.mean(reference**2) / np.mean(candidate**2))

    result = validate_replacement(reference, candidate, maximum_lag_frames=0)

    assert not result.accepted
    assert max(item.p95_log_energy_error for item in result.resolutions) > 0.18


def test_does_not_allow_one_resolution_to_compensate_for_another() -> None:
    reference = tone(220)
    candidate = reference.copy()
    candidate[:256] *= np.hanning(512)[:256]
    strict = ValidationThresholds(maximum_incompatible_run=0)

    result = validate_replacement(
        reference,
        candidate,
        maximum_lag_frames=0,
        thresholds=strict,
    )

    assert not result.accepted
    assert any(not resolution.accepted for resolution in result.resolutions)


def test_complete_link_clustering_rejects_transitive_family() -> None:
    matrix = (
        (1.0, 0.99, 0.50),
        (0.99, 1.0, 0.99),
        (0.50, 0.99, 1.0),
    )

    assert cluster_candidates(matrix, minimum_similarity=0.98) == ((0, 1), (2,))


def test_repeat_similarity_preserves_temporal_order() -> None:
    low = tone(180, frames=4096)
    high = tone(620, frames=4096)
    forward = np.concatenate((low, high))
    reversed_order = np.concatenate((high, low))

    matrix = repeat_similarity_matrix(
        (forward, forward.copy(), reversed_order),
        time_offsets=(0,),
    )

    assert matrix[0][1] > 0.99
    assert matrix[0][2] < 0.9
