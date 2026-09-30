from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import Audio


@dataclass(frozen=True)
class RetrievalCandidate:
    query_index: int
    candidate_index: int
    score: float


def _mono(audio: Audio) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    return values if values.ndim == 1 else np.mean(values, axis=1)


def local_time_frequency_descriptor(
    audio: Audio,
    *,
    temporal_cells: int = 12,
    spectral_cells: int = 48,
) -> np.ndarray:
    """Return an ordered retrieval descriptor; no spectrum is averaged over time.

    This is deliberately only a cheap candidate index. A high descriptor score
    never authorizes replacement: callers must use the strict local validator.
    """
    if temporal_cells < 2 or spectral_cells < 2:
        raise ValueError("descriptor dimensions must be at least two")
    values = _mono(audio)
    if values.size == 0:
        raise ValueError("audio cannot be empty")
    boundaries = np.linspace(0, values.size, temporal_cells + 1, dtype=int)
    spectra: list[np.ndarray] = []
    envelope: list[float] = []
    for index in range(temporal_cells):
        cell = values[boundaries[index] : boundaries[index + 1]]
        if cell.size < 2:
            cell = np.pad(cell, (0, 2 - cell.size))
        windowed = cell * np.hanning(cell.size)
        spectrum = np.log1p(np.abs(np.fft.rfft(windowed)))
        source_positions = np.linspace(0.0, 1.0, spectrum.size)
        target_positions = np.linspace(0.0, 1.0, spectral_cells)
        reduced = np.interp(target_positions, source_positions, spectrum)
        norm = float(np.linalg.norm(reduced))
        spectra.append(reduced / norm if norm else reduced)
        envelope.append(float(np.sqrt(np.mean(cell**2))))
    envelope_values = np.asarray(envelope)
    envelope_peak = float(np.max(envelope_values))
    if envelope_peak:
        envelope_values /= envelope_peak
    descriptor = np.concatenate((*spectra, envelope_values))
    norm = float(np.linalg.norm(descriptor))
    return descriptor / norm if norm else descriptor


def retrieve_global_candidates(
    blocks: Sequence[Audio],
    *,
    neighbors_per_block: int = 8,
    minimum_duration_ratio: float = 0.8,
    minimum_score: float = 0.7,
) -> tuple[RetrievalCandidate, ...]:
    """Retrieve globally similar blocks while preserving temporal variability."""
    if neighbors_per_block <= 0:
        raise ValueError("neighbors_per_block must be positive")
    if not 0.0 <= minimum_duration_ratio <= 1.0:
        raise ValueError("minimum_duration_ratio must be between zero and one")
    if not -1.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between minus one and one")
    if not blocks:
        return ()
    descriptors = np.stack([local_time_frequency_descriptor(block) for block in blocks])
    similarities = descriptors @ descriptors.T
    lengths = np.asarray([np.asarray(block).shape[0] for block in blocks])
    results: list[RetrievalCandidate] = []
    for query in range(len(blocks)):
        duration_ratio = np.minimum(lengths[query], lengths) / np.maximum(lengths[query], lengths)
        eligible = np.flatnonzero(
            (np.arange(len(blocks)) != query)
            & (duration_ratio >= minimum_duration_ratio)
            & (similarities[query] >= minimum_score)
        )
        ordered = sorted(eligible, key=lambda index: (-similarities[query, index], index))
        results.extend(
            RetrievalCandidate(query, int(candidate), float(similarities[query, candidate]))
            for candidate in ordered[:neighbors_per_block]
        )
    return tuple(results)
