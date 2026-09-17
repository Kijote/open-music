from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Audio = NDArray[np.float64]


@dataclass(frozen=True)
class Sample:
    id: str
    audio: Audio
    sample_rate: int
    license: str
    source: str

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio, dtype=np.float64)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.ndim != 2 or audio.shape[1] not in (1, 2):
            raise ValueError("audio must be mono or stereo")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "audio", audio)


@dataclass(frozen=True)
class Event:
    sample_id: str
    start_frame: int
    gain: float = 1.0
    pan: float = 0.0

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("start_frame cannot be negative")
        if not -1.0 <= self.pan <= 1.0:
            raise ValueError("pan must be between -1 and 1")


@dataclass(frozen=True)
class Metrics:
    mean_absolute_error: float
    snr_db: float
    explained_energy: float


@dataclass(frozen=True)
class Match:
    sample_id: str
    similarity: float


def _as_stereo(audio: Audio, pan: float) -> Audio:
    if audio.shape[1] == 2:
        left_scale = 1.0 if pan <= 0 else 1.0 - pan
        right_scale = 1.0 if pan >= 0 else 1.0 + pan
        return audio * np.array([left_scale, right_scale])

    mono = audio[:, 0]
    left = mono * (1.0 if pan <= 0 else 1.0 - pan)
    right = mono * (1.0 if pan >= 0 else 1.0 + pan)
    return np.column_stack((left, right))


def render(
    library: Mapping[str, Sample],
    events: Sequence[Event],
    total_frames: int,
    sample_rate: int,
) -> Audio:
    if total_frames < 0:
        raise ValueError("total_frames cannot be negative")

    output = np.zeros((total_frames, 2), dtype=np.float64)
    for event in events:
        sample = library[event.sample_id]
        if sample.sample_rate != sample_rate:
            raise ValueError("sample-rate conversion is not implemented yet")

        rendered = _as_stereo(sample.audio, event.pan) * event.gain
        end = min(event.start_frame + rendered.shape[0], total_frames)
        if end > event.start_frame:
            output[event.start_frame:end] += rendered[: end - event.start_frame]

    return output


def measure(reference: Audio, approximation: Audio) -> Metrics:
    reference = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(approximation, dtype=np.float64)
    if reference.shape != approximation.shape:
        raise ValueError("signals must have the same shape")

    residual = reference - approximation
    reference_energy = float(np.sum(reference**2))
    residual_energy = float(np.sum(residual**2))
    mae = float(np.mean(np.abs(residual)))

    if residual_energy == 0.0:
        snr = float("inf")
    elif reference_energy == 0.0:
        snr = float("-inf")
    else:
        snr = float(10.0 * np.log10(reference_energy / residual_energy))

    explained = 1.0 if reference_energy == 0.0 and residual_energy == 0.0 else (
        1.0 - residual_energy / reference_energy if reference_energy else 0.0
    )
    return Metrics(mae, snr, explained)


def match_sample(query: Audio, library: Mapping[str, Sample]) -> Match:
    query_vector = np.asarray(query, dtype=np.float64).reshape(-1)
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        raise ValueError("cannot match a silent query")

    best: Match | None = None
    for sample in library.values():
        candidate = sample.audio.reshape(-1)
        size = max(query_vector.size, candidate.size)
        left = np.pad(query_vector, (0, size - query_vector.size))
        right = np.pad(candidate, (0, size - candidate.size))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        similarity = float(np.dot(left, right) / denominator) if denominator else -1.0
        if best is None or similarity > best.similarity:
            best = Match(sample.id, similarity)

    if best is None:
        raise ValueError("library cannot be empty")
    return best
