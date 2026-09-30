from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .harmonic_groups import HarmonicFamily
from .partials import PartialDecomposition


@dataclass(frozen=True)
class HarmonicFamilyMatch:
    left: int
    right: int
    similarity: float
    duration_ratio: float
    fundamental_ratio: float


def harmonic_family_descriptor(
    family: HarmonicFamily,
    decomposition: PartialDecomposition,
    *,
    maximum_harmonic: int = 12,
    temporal_cells: int = 16,
) -> np.ndarray:
    """Describe harmonic shape and ordered envelope without absolute pitch."""
    if maximum_harmonic < 2 or temporal_cells < 2:
        raise ValueError("descriptor dimensions must be at least two")
    profile = np.zeros(maximum_harmonic, dtype=np.float64)
    for member in family.members:
        if member.harmonic_number <= maximum_harmonic:
            energy = decomposition.tracks[member.component_index].spectral_energy
            profile[member.harmonic_number - 1] = np.sqrt(energy)
    profile_norm = float(np.linalg.norm(profile))
    if profile_norm:
        profile /= profile_norm

    values = np.asarray(family.audio, dtype=np.float64)
    mono = values if values.ndim == 1 else np.mean(values, axis=1)
    start = max(0, family.start_frame * decomposition.hop_size)
    end = min(mono.size, family.end_frame * decomposition.hop_size + decomposition.fft_size)
    active = mono[start:end]
    boundaries = np.linspace(0, active.size, temporal_cells + 1, dtype=int)
    envelope = np.asarray(
        [
            float(np.sqrt(np.mean(active[boundaries[index] : boundaries[index + 1]] ** 2)))
            if boundaries[index + 1] > boundaries[index]
            else 0.0
            for index in range(temporal_cells)
        ]
    )
    envelope_norm = float(np.linalg.norm(envelope))
    if envelope_norm:
        envelope /= envelope_norm
    descriptor = np.concatenate((profile, envelope))
    norm = float(np.linalg.norm(descriptor))
    return descriptor / norm if norm else descriptor


def harmonic_family_similarity_matrix(
    families: Sequence[HarmonicFamily],
    decompositions: Sequence[PartialDecomposition],
) -> tuple[tuple[float, ...], ...]:
    if len(families) != len(decompositions):
        raise ValueError("families and decompositions must have equal length")
    if not families:
        return ()
    descriptors = np.stack(
        [
            harmonic_family_descriptor(family, decomposition)
            for family, decomposition in zip(families, decompositions)
        ]
    )
    matrix = np.clip(descriptors @ descriptors.T, 0.0, 1.0)
    matrix[np.isclose(matrix, 1.0, atol=1e-12)] = 1.0
    return tuple(tuple(float(value) for value in row) for row in matrix)


def cluster_harmonic_families(
    families: Sequence[HarmonicFamily],
    similarity_matrix: Sequence[Sequence[float]],
    *,
    minimum_similarity: float = 0.94,
    minimum_duration_ratio: float = 0.65,
) -> tuple[tuple[int, ...], ...]:
    """Create deterministic complete-link groups of reusable harmonic families."""
    size = len(families)
    if len(similarity_matrix) != size or any(len(row) != size for row in similarity_matrix):
        raise ValueError("similarity matrix must be square and match families")
    groups: list[list[int]] = []
    durations = [family.end_frame - family.start_frame for family in families]
    for candidate in range(size):
        destination = next(
            (
                group
                for group in groups
                if all(
                    similarity_matrix[candidate][member] >= minimum_similarity
                    and min(durations[candidate], durations[member])
                    / max(durations[candidate], durations[member])
                    >= minimum_duration_ratio
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
