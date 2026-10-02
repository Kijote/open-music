from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Audio
from .partials import PartialDecomposition


@dataclass(frozen=True)
class HarmonicMember:
    component_index: int
    track_id: int
    harmonic_number: int
    frequency_hz: float
    cents_error: float


@dataclass(frozen=True)
class HarmonicFamily:
    id: int
    fundamental_hz: float
    start_frame: int
    end_frame: int
    members: tuple[HarmonicMember, ...]
    audio: Audio
    energy: float


@dataclass(frozen=True)
class HarmonicGrouping:
    families: tuple[HarmonicFamily, ...]
    standalone_component_indexes: tuple[int, ...]
    standalone: tuple[Audio, ...]
    residual: Audio
    reconstruction_max_error: float


def _overlap_ratio(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> float:
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    shortest = min(left_end - left_start, right_end - right_start)
    return overlap / shortest if shortest else 0.0


def group_harmonic_partials(
    decomposition: PartialDecomposition,
    *,
    maximum_harmonic: int = 12,
    maximum_cents_error: float = 55.0,
    minimum_temporal_overlap: float = 0.5,
    minimum_family_members: int = 2,
) -> HarmonicGrouping:
    """Group simultaneous partial trajectories into explicit harmonic series."""
    if maximum_harmonic < 2 or minimum_family_members < 2:
        raise ValueError("harmonic and family limits must be at least two")
    if maximum_cents_error < 0.0:
        raise ValueError("maximum_cents_error cannot be negative")
    if not 0.0 <= minimum_temporal_overlap <= 1.0:
        raise ValueError("minimum_temporal_overlap must be between zero and one")
    if len(decomposition.components) != len(decomposition.tracks):
        raise ValueError("partial components and tracks must have equal length")

    available = set(range(len(decomposition.tracks)))
    candidates = sorted(
        available,
        key=lambda index: (
            decomposition.tracks[index].mean_frequency_hz,
            -decomposition.tracks[index].spectral_energy,
            decomposition.tracks[index].id,
        ),
    )
    families: list[HarmonicFamily] = []
    for fundamental_index in candidates:
        if fundamental_index not in available:
            continue
        fundamental = decomposition.tracks[fundamental_index]
        members = [
            HarmonicMember(
                component_index=fundamental_index,
                track_id=fundamental.id,
                harmonic_number=1,
                frequency_hz=fundamental.mean_frequency_hz,
                cents_error=0.0,
            )
        ]
        for component_index in sorted(available - {fundamental_index}):
            track = decomposition.tracks[component_index]
            ratio = track.mean_frequency_hz / fundamental.mean_frequency_hz
            harmonic = round(ratio)
            if not 2 <= harmonic <= maximum_harmonic:
                continue
            cents_error = 1200.0 * np.log2(ratio / harmonic)
            overlap = _overlap_ratio(
                fundamental.start_frame,
                fundamental.end_frame,
                track.start_frame,
                track.end_frame,
            )
            if abs(cents_error) <= maximum_cents_error and overlap >= minimum_temporal_overlap:
                members.append(
                    HarmonicMember(
                        component_index=component_index,
                        track_id=track.id,
                        harmonic_number=harmonic,
                        frequency_hz=track.mean_frequency_hz,
                        cents_error=float(cents_error),
                    )
                )
        # Keep one strongest trajectory for each harmonic number so two nearby
        # peaks cannot both claim the same position in a series.
        unique: dict[int, HarmonicMember] = {}
        for member in members:
            current = unique.get(member.harmonic_number)
            if current is None or (
                decomposition.tracks[member.component_index].spectral_energy
                > decomposition.tracks[current.component_index].spectral_energy
            ):
                unique[member.harmonic_number] = member
        members = [unique[number] for number in sorted(unique)]
        if len(members) < minimum_family_members:
            continue
        indexes = [member.component_index for member in members]
        family_audio = sum(
            (decomposition.components[index] for index in indexes),
            np.zeros_like(decomposition.residual),
        )
        start_frame = min(decomposition.tracks[index].start_frame for index in indexes)
        end_frame = max(decomposition.tracks[index].end_frame for index in indexes)
        families.append(
            HarmonicFamily(
                id=len(families),
                fundamental_hz=fundamental.mean_frequency_hz,
                start_frame=start_frame,
                end_frame=end_frame,
                members=tuple(members),
                audio=family_audio,
                energy=float(np.sum(family_audio**2)),
            )
        )
        available.difference_update(indexes)

    standalone_indexes = tuple(sorted(available))
    standalone = tuple(decomposition.components[index] for index in standalone_indexes)
    reconstructed = sum(
        (family.audio for family in families), np.zeros_like(decomposition.residual)
    )
    reconstructed += sum(standalone, np.zeros_like(decomposition.residual))
    residual = decomposition.residual
    original = (
        sum(decomposition.components, np.zeros_like(decomposition.residual))
        + decomposition.residual
    )
    error = float(np.max(np.abs(original - reconstructed - residual)))
    return HarmonicGrouping(
        families=tuple(families),
        standalone_component_indexes=standalone_indexes,
        standalone=standalone,
        residual=residual,
        reconstruction_max_error=error,
    )
