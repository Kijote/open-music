from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .core import Audio, Event, Sample
from .validation import validate_replacement


@dataclass(frozen=True)
class Discovery:
    events: tuple[Event, ...]
    onset_frames: tuple[int, ...]
    match_scores: tuple[float, ...]


def _mono(audio: Audio) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        return values
    return np.max(np.abs(values), axis=1)


def detect_onsets(
    audio: Audio,
    sample_rate: int,
    *,
    frame_length: int = 1024,
    hop_length: int = 256,
    minimum_spacing_seconds: float = 0.2,
    relative_threshold: float = 0.08,
) -> tuple[int, ...]:
    mono = np.abs(_mono(audio))
    if mono.size == 0 or float(np.max(mono)) == 0.0:
        return ()

    previous = np.pad(mono[:-hop_length], (hop_length, 0))
    novelty_samples = np.maximum(mono - previous, 0.0)
    frame_count = max(1, (mono.size + hop_length - 1) // hop_length)
    novelty = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        start = index * hop_length
        end = min(start + frame_length, mono.size)
        novelty[index] = float(np.max(novelty_samples[start:end], initial=0.0))

    median = float(np.median(novelty))
    deviation = float(np.median(np.abs(novelty - median)))
    threshold = max(float(np.max(novelty)) * relative_threshold, median + 6.0 * deviation)
    candidates = [
        index
        for index, value in enumerate(novelty)
        if value >= threshold
        and value >= (novelty[index - 1] if index else 0.0)
        and value >= (novelty[index + 1] if index + 1 < novelty.size else 0.0)
    ]

    minimum_spacing = max(1, round(minimum_spacing_seconds * sample_rate / hop_length))
    selected: list[int] = []
    for candidate in sorted(candidates, key=lambda index: novelty[index], reverse=True):
        if all(abs(candidate - current) >= minimum_spacing for current in selected):
            selected.append(candidate)

    return tuple(sorted(index * hop_length for index in selected))


def extract_candidates(
    audio: Audio,
    onset_frames: Sequence[int],
    *,
    maximum_frames: int,
) -> tuple[Audio, ...]:
    candidates: list[Audio] = []
    for index, onset in enumerate(onset_frames):
        next_onset = onset_frames[index + 1] if index + 1 < len(onset_frames) else audio.shape[0]
        end = min(audio.shape[0], onset + maximum_frames, next_onset)
        candidates.append(audio[onset:end])
    return tuple(candidates)


def repeat_similarity_matrix(
    candidates: Sequence[Audio],
    *,
    spectral_windows: Sequence[int] = (256, 1024, 4096),
    time_offsets: Sequence[int] = (0, 512, 1024, 2048),
) -> tuple[tuple[float, ...], ...]:
    """Compare candidates using non-compensating local time-frequency validation.

    This score is intended for candidate retrieval. Replacement acceptance must
    still inspect ``validate_replacement`` and its per-resolution diagnostics.
    No spectrum is averaged over time.
    """
    if not spectral_windows or any(size <= 1 for size in spectral_windows):
        raise ValueError("spectral windows must contain positive sizes greater than one")
    if not time_offsets or any(offset < 0 for offset in time_offsets):
        raise ValueError("time offsets must contain non-negative positions")

    size = len(candidates)
    matrix = np.eye(size, dtype=np.float64)
    maximum_lag = min(max(time_offsets), min(spectral_windows))
    fft_sizes = tuple(int(value) for value in spectral_windows)
    for left in range(size):
        for right in range(left + 1, size):
            result = validate_replacement(
                candidates[left],
                candidates[right],
                fft_sizes=fft_sizes,
                maximum_lag_frames=maximum_lag,
            )
            score = 1.0 if np.isclose(result.score, 1.0, atol=1e-12) else result.score
            matrix[left, right] = score
            matrix[right, left] = score
    return tuple(tuple(float(value) for value in row) for row in matrix)


def cluster_candidates(
    similarity_matrix: Sequence[Sequence[float]],
    *,
    minimum_similarity: float = 0.989,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic complete-link groups of mutually similar candidates.

    A candidate may join a group only when it is compatible with every existing
    member. This deliberately rejects the invalid A≈B, B≈C ⇒ A≈C assumption.
    """
    size = len(similarity_matrix)
    if any(len(row) != size for row in similarity_matrix):
        raise ValueError("similarity matrix must be square")

    groups: list[list[int]] = []
    for candidate in range(size):
        destination = next(
            (
                group
                for group in groups
                if all(
                    similarity_matrix[candidate][member] >= minimum_similarity
                    for member in group
                )
            ),
            None,
        )
        if destination is None:
            groups.append([candidate])
        else:
            destination.append(candidate)
    return tuple(tuple(group) for group in groups)


def select_canonical_candidates(
    similarity_matrix: Sequence[Sequence[float]],
    clusters: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Select each cluster's medoid; the earliest candidate breaks exact ties."""
    return tuple(
        max(
            cluster,
            key=lambda candidate: (
                sum(similarity_matrix[candidate][peer] for peer in cluster),
                -candidate,
            ),
        )
        for cluster in clusters
    )


def fit_candidate_gain(candidate: Audio, canonical: Audio) -> float:
    available = min(candidate.shape[0], canonical.shape[0])
    query = np.asarray(candidate[:available], dtype=np.float64).reshape(-1)
    source = np.asarray(canonical[:available], dtype=np.float64).reshape(-1)
    source_energy = float(np.dot(source, source))
    return max(0.0, float(np.dot(query, source) / source_energy)) if source_energy else 0.0


def _stereo(audio: Audio) -> Audio:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] == 1:
        return np.repeat(values, 2, axis=1)
    return values


def _fit_at(
    mixture: Audio,
    sample: Sample,
    start: int,
    probe_frames: int,
) -> tuple[float, float]:
    source = _stereo(sample.audio)[:probe_frames]
    available = min(source.shape[0], mixture.shape[0] - start)
    if available <= 0:
        return -1.0, 0.0
    source = source[:available].reshape(-1)
    query = mixture[start : start + available].reshape(-1)
    source_energy = float(np.dot(source, source))
    query_energy = float(np.dot(query, query))
    if source_energy == 0.0 or query_energy == 0.0:
        return -1.0, 0.0
    gain = max(0.0, float(np.dot(query, source) / source_energy))
    similarity = float(np.dot(query, source) / np.sqrt(query_energy * source_energy))
    return similarity, gain


def fit_module_event(
    audio: Audio,
    sample: Sample,
    onset_frame: int,
    *,
    search_radius: int = 512,
    probe_frames: int = 4096,
) -> tuple[Event, float]:
    """Refine a coarse onset and gain for one already-selected module."""
    mixture = _stereo(audio)
    first = max(0, onset_frame - search_radius)
    last = min(mixture.shape[0] - 1, onset_frame + search_radius)
    best = max(
        (
            (*_fit_at(mixture, sample, start, probe_frames), start)
            for start in range(first, last + 1)
        ),
        key=lambda result: (result[0], -abs(result[2] - onset_frame), -result[2]),
    )
    similarity, gain, start = best
    return Event(sample.id, start, gain=gain), similarity


def fit_modules_iteratively(
    audio: Audio,
    modules: Sequence[Sample],
    onset_frames: Sequence[int],
    *,
    search_radius: int = 512,
    probe_frames: int = 4096,
) -> tuple[tuple[Event, ...], Audio, tuple[float, ...]]:
    """Fit each selected module to the residual and subtract it before the next fit."""
    if len(modules) != len(onset_frames):
        raise ValueError("modules and onset frames must have equal length")

    residual = _stereo(audio).copy()
    events: list[Event] = []
    scores: list[float] = []
    for sample, onset in zip(modules, onset_frames, strict=True):
        event, score = fit_module_event(
            residual,
            sample,
            onset,
            search_radius=search_radius,
            probe_frames=probe_frames,
        )
        source = _stereo(sample.audio) * event.gain
        end = min(residual.shape[0], event.start_frame + source.shape[0])
        if end > event.start_frame:
            residual[event.start_frame : end] -= source[: end - event.start_frame]
        events.append(event)
        scores.append(score)
    return tuple(events), residual, tuple(scores)


def discover_events(
    audio: Audio,
    library: Mapping[str, Sample],
    sample_rate: int,
    *,
    search_radius: int = 1024,
    probe_frames: int = 4096,
) -> Discovery:
    mixture = _stereo(audio)
    onsets = detect_onsets(mixture, sample_rate)
    events: list[Event] = []
    scores: list[float] = []

    for onset in onsets:
        best: tuple[float, str, int, float] | None = None
        first = max(0, onset - search_radius)
        last = min(mixture.shape[0] - 1, onset + search_radius)
        for sample in library.values():
            for start in range(first, last + 1):
                similarity, gain = _fit_at(mixture, sample, start, probe_frames)
                candidate = (similarity, sample.id, start, gain)
                if best is None or candidate[0] > best[0]:
                    best = candidate

        if best is not None and best[0] > 0.1:
            similarity, sample_id, start, gain = best
            events.append(Event(sample_id, start, gain=gain))
            scores.append(similarity)

    events.sort(key=lambda event: event.start_frame)
    return Discovery(tuple(events), onsets, tuple(scores))


def detection_metrics(
    expected: Sequence[Event],
    discovered: Sequence[Event],
    *,
    tolerance_frames: int,
) -> dict[str, float | int]:
    unmatched = set(range(len(discovered)))
    matches = 0
    for target in expected:
        candidates = [
            index
            for index in unmatched
            if abs(discovered[index].start_frame - target.start_frame) <= tolerance_frames
        ]
        if candidates:
            chosen = min(
                candidates,
                key=lambda index: abs(discovered[index].start_frame - target.start_frame),
            )
            unmatched.remove(chosen)
            matches += 1

    precision = matches / len(discovered) if discovered else 0.0
    recall = matches / len(expected) if expected else 1.0
    return {
        "true_positives": matches,
        "false_positives": len(discovered) - matches,
        "false_negatives": len(expected) - matches,
        "precision": precision,
        "recall": recall,
    }
