from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .core import Audio, Event, Sample


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
