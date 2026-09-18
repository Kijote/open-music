from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Audio, Event, Sample
from .discovery import (
    cluster_candidates,
    detect_onsets,
    extract_candidates,
    fit_modules_iteratively,
    repeat_similarity_matrix,
    select_canonical_candidates,
)


@dataclass(frozen=True)
class ResidualPass:
    index: int
    onset_frames: tuple[int, ...]
    candidates: tuple[Audio, ...]
    clusters: tuple[tuple[int, ...], ...]
    accepted_clusters: tuple[tuple[int, ...], ...]
    canonical_candidates: tuple[int, ...]
    modules: tuple[Sample, ...]
    events: tuple[Event, ...]
    match_scores: tuple[float, ...]
    energy_before: float
    energy_after: float
    accepted: bool
    stop_reason: str | None


@dataclass(frozen=True)
class ResidualDiscovery:
    passes: tuple[ResidualPass, ...]
    modules: tuple[Sample, ...]
    events: tuple[Event, ...]
    residual: Audio
    stop_reason: str


def _energy(audio: Audio) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sum(np.square(values)))


def discover_residual_layers(
    audio: Audio,
    sample_rate: int,
    *,
    license_spdx: str,
    source_id: str,
    maximum_candidate_seconds: float = 0.75,
    maximum_passes: int = 3,
    minimum_cluster_size: int = 2,
    minimum_improvement_ratio: float = 1e-6,
) -> ResidualDiscovery:
    """Discover only reusable residual modules until deterministic convergence."""
    residual = np.asarray(audio, dtype=np.float64).copy()
    initial_energy = _energy(residual)
    maximum_frames = round(maximum_candidate_seconds * sample_rate)
    passes: list[ResidualPass] = []
    discovered_modules: list[Sample] = []
    discovered_events: list[Event] = []
    stop_reason = "maximum_passes"

    for pass_index in range(1, maximum_passes + 1):
        energy_before = _energy(residual)
        if pass_index > 1 and energy_before <= initial_energy * minimum_improvement_ratio:
            stop_reason = "residual_floor"
            passes.append(
                ResidualPass(
                    pass_index,
                    (),
                    (),
                    (),
                    (),
                    (),
                    (),
                    (),
                    (),
                    energy_before,
                    energy_before,
                    False,
                    stop_reason,
                )
            )
            break
        onsets = detect_onsets(residual, sample_rate, frame_length=256)
        candidates = extract_candidates(residual, onsets, maximum_frames=maximum_frames)
        similarity = repeat_similarity_matrix(candidates)
        clusters = cluster_candidates(similarity)
        accepted_clusters = tuple(
            cluster for cluster in clusters if len(cluster) >= minimum_cluster_size
        )

        if not onsets:
            stop_reason = "no_onsets"
        elif not accepted_clusters:
            stop_reason = "no_reusable_clusters"
        else:
            canonicals = select_canonical_candidates(similarity, accepted_clusters)
            candidate_cluster = {
                candidate: cluster_index
                for cluster_index, cluster in enumerate(accepted_clusters)
                for candidate in cluster
            }
            modules = tuple(
                Sample(
                    id=f"layer-{pass_index:02d}-module-{cluster_index:03d}",
                    audio=candidates[canonical],
                    sample_rate=sample_rate,
                    license=license_spdx,
                    source=f"residual:{source_id}:pass-{pass_index}:candidate-{canonical:03d}",
                )
                for cluster_index, canonical in enumerate(canonicals)
            )
            accepted_candidates = sorted(candidate_cluster)
            selected_modules = tuple(
                modules[candidate_cluster[candidate]] for candidate in accepted_candidates
            )
            selected_onsets = tuple(onsets[candidate] for candidate in accepted_candidates)
            events, next_residual, match_scores = fit_modules_iteratively(
                residual,
                selected_modules,
                selected_onsets,
            )
            energy_after = _energy(next_residual)
            improvement = energy_before - energy_after
            required_improvement = initial_energy * minimum_improvement_ratio

            if improvement <= 0.0 or improvement < required_improvement:
                stop_reason = "insufficient_improvement"
            else:
                passes.append(
                    ResidualPass(
                        pass_index,
                        onsets,
                        candidates,
                        clusters,
                        accepted_clusters,
                        canonicals,
                        modules,
                        events,
                        match_scores,
                        energy_before,
                        energy_after,
                        True,
                        None,
                    )
                )
                discovered_modules.extend(modules)
                discovered_events.extend(events)
                residual = next_residual
                continue

        passes.append(
            ResidualPass(
                pass_index,
                onsets,
                candidates,
                clusters,
                accepted_clusters,
                (),
                (),
                (),
                (),
                energy_before,
                energy_before,
                False,
                stop_reason,
            )
        )
        break

    return ResidualDiscovery(
        tuple(passes),
        tuple(discovered_modules),
        tuple(discovered_events),
        residual,
        stop_reason,
    )
