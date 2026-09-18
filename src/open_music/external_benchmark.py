from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import Sample, measure, render
from .corpus import fetch_assets, load_external_manifest
from .discovery import (
    cluster_candidates,
    detect_onsets,
    extract_candidates,
    fit_modules_iteratively,
    repeat_similarity_matrix,
    select_canonical_candidates,
)
from .residual import discover_residual_layers
from .wav import read_wav, write_wav


def _coverage(onsets: tuple[int, ...], lengths: list[int], total_frames: int) -> int:
    intervals = sorted(zip(onsets, lengths, strict=True))
    covered = 0
    end = 0
    for start, length in intervals:
        next_end = min(total_frames, start + length)
        covered += max(0, next_end - max(start, end))
        end = max(end, next_end)
    return covered


def run_external_benchmark(
    manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    maximum_candidate_seconds: float = 0.75,
) -> dict:
    corpus = load_external_manifest(manifest_path)
    paths = fetch_assets(corpus.recordings, cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for recording in corpus.recordings:
        sample_rate, source_audio = read_wav(paths[recording.id])
        source_channels = int(source_audio.shape[1])
        audio = np.repeat(source_audio, 2, axis=1) if source_channels == 1 else source_audio
        onsets = detect_onsets(audio, sample_rate)
        maximum_frames = round(maximum_candidate_seconds * sample_rate)
        candidates = extract_candidates(audio, onsets, maximum_frames=maximum_frames)
        lengths = [int(candidate.shape[0]) for candidate in candidates]
        covered_frames = _coverage(onsets, lengths, audio.shape[0])
        similarity = repeat_similarity_matrix(candidates)
        clusters = cluster_candidates(similarity)
        canonicals = select_canonical_candidates(similarity, clusters)

        candidate_clusters = {
            candidate: cluster_index
            for cluster_index, cluster in enumerate(clusters)
            for candidate in cluster
        }
        library = {
            f"module-{cluster_index:03d}": Sample(
                id=f"module-{cluster_index:03d}",
                audio=candidates[canonical],
                sample_rate=sample_rate,
                license=recording.license_spdx,
                source=f"discovered:{recording.id}:candidate-{canonical:03d}",
            )
            for cluster_index, canonical in enumerate(canonicals)
        }
        selected_modules = [
            library[f"module-{candidate_clusters[index]:03d}"] for index in range(len(candidates))
        ]
        events, iterative_residual, match_scores = fit_modules_iteratively(
            audio,
            selected_modules,
            onsets,
        )
        initial_reconstruction = render(library, events, audio.shape[0], sample_rate)
        initial_residual = audio - initial_reconstruction
        if not np.allclose(initial_residual, iterative_residual):
            raise RuntimeError("iterative residual does not match rendered events")
        residual_discovery = discover_residual_layers(
            initial_residual,
            sample_rate,
            license_spdx=recording.license_spdx,
            source_id=recording.id,
        )
        library.update({sample.id: sample for sample in residual_discovery.modules})
        all_events = (*events, *residual_discovery.events)
        reconstruction = render(library, all_events, audio.shape[0], sample_rate)
        residual = audio - reconstruction
        if not np.allclose(residual, residual_discovery.residual):
            raise RuntimeError("residual discovery does not match rendered events")
        fidelity = measure(audio, reconstruction)

        recording_dir = output_dir / recording.id
        write_wav(recording_dir / "source.wav", sample_rate, source_audio)
        for index, candidate in enumerate(candidates):
            write_wav(
                recording_dir / "candidates" / f"candidate-{index:03d}.wav", sample_rate, candidate
            )
        for cluster_index, canonical in enumerate(canonicals):
            write_wav(
                recording_dir / "modules" / f"module-{cluster_index:03d}.wav",
                sample_rate,
                candidates[canonical],
            )
        for residual_pass in residual_discovery.passes:
            pass_dir = recording_dir / "residual-discovery" / f"pass-{residual_pass.index:02d}"
            for index, candidate in enumerate(residual_pass.candidates):
                write_wav(
                    pass_dir / "candidates" / f"candidate-{index:03d}.wav",
                    sample_rate,
                    candidate,
                )
            for module in residual_pass.modules:
                write_wav(pass_dir / "modules" / f"{module.id}.wav", sample_rate, module.audio)
        write_wav(recording_dir / "reconstruction.wav", sample_rate, reconstruction)
        write_wav(recording_dir / "residual.wav", sample_rate, residual)

        duration_seconds = audio.shape[0] / sample_rate
        results.append(
            {
                "recording_id": recording.id,
                "source": recording.source,
                "license": recording.license_spdx,
                "sample_rate": sample_rate,
                "channels": source_channels,
                "analysis_channels": int(audio.shape[1]),
                "duration_seconds": duration_seconds,
                "event_count": len(onsets),
                "event_density_hz": len(onsets) / duration_seconds if duration_seconds else 0.0,
                "onset_frames": list(onsets),
                "candidate_frames": lengths,
                "candidate_durations_seconds": [length / sample_rate for length in lengths],
                "coverage": {
                    "covered_frames": covered_frames,
                    "total_frames": int(audio.shape[0]),
                    "ratio": covered_frames / audio.shape[0] if audio.shape[0] else 0.0,
                },
                "repeat_similarity_matrix": [
                    [round(value, 12) for value in row] for row in similarity
                ],
                "similarity_parameters": {
                    "spectral_windows": [512, 1024, 2048, 4096],
                    "time_offsets": [0, 512, 1024, 2048],
                    "minimum_similarity": 0.989,
                },
                "modules": {
                    "cluster_assignments": [
                        candidate_clusters[index] for index in range(len(candidates))
                    ],
                    "clusters": [list(cluster) for cluster in clusters],
                    "canonical_candidates": list(canonicals),
                    "module_count": len(library),
                    "reuse_ratio": len(events) / len(library) if library else 0.0,
                    "event_gains": [round(event.gain, 12) for event in events],
                    "event_frames": [event.start_frame for event in events],
                    "event_match_scores": [round(score, 12) for score in match_scores],
                },
                "residual_discovery": {
                    "maximum_passes": 3,
                    "minimum_cluster_size": 2,
                    "minimum_improvement_ratio": 1e-6,
                    "stop_reason": residual_discovery.stop_reason,
                    "accepted_module_count": len(residual_discovery.modules),
                    "accepted_event_count": len(residual_discovery.events),
                    "passes": [
                        {
                            "index": residual_pass.index,
                            "onset_frames": list(residual_pass.onset_frames),
                            "candidate_count": len(residual_pass.candidates),
                            "clusters": [list(cluster) for cluster in residual_pass.clusters],
                            "accepted_clusters": [
                                list(cluster) for cluster in residual_pass.accepted_clusters
                            ],
                            "canonical_candidates": list(residual_pass.canonical_candidates),
                            "module_ids": [module.id for module in residual_pass.modules],
                            "event_frames": [event.start_frame for event in residual_pass.events],
                            "event_gains": [
                                round(event.gain, 12) for event in residual_pass.events
                            ],
                            "match_scores": [
                                round(score, 12) for score in residual_pass.match_scores
                            ],
                            "energy_before": residual_pass.energy_before,
                            "energy_after": residual_pass.energy_after,
                            "energy_improvement": (
                                residual_pass.energy_before - residual_pass.energy_after
                            ),
                            "accepted": residual_pass.accepted,
                            "stop_reason": residual_pass.stop_reason,
                        }
                        for residual_pass in residual_discovery.passes
                    ],
                },
                "reconstruction": {
                    "mean_absolute_error": fidelity.mean_absolute_error,
                    "snr_db": fidelity.snr_db,
                    "explained_energy": fidelity.explained_energy,
                    "residual_energy": float(np.sum(np.square(residual))),
                    "source_energy": float(np.sum(np.square(audio))),
                },
            }
        )

    source_energy = sum(item["reconstruction"]["source_energy"] for item in results)
    residual_energy = sum(item["reconstruction"]["residual_energy"] for item in results)
    total_modules = sum(
        item["modules"]["module_count"] + item["residual_discovery"]["accepted_module_count"]
        for item in results
    )
    total_events = sum(
        item["event_count"] + item["residual_discovery"]["accepted_event_count"] for item in results
    )
    report = {
        "corpus": corpus.id,
        "recordings": results,
        "aggregate": {
            "recording_count": len(results),
            "duration_seconds": sum(item["duration_seconds"] for item in results),
            "event_count": total_events,
            "module_count": total_modules,
            "reuse_ratio": total_events / total_modules if total_modules else 0.0,
            "source_energy": source_energy,
            "residual_energy": residual_energy,
            "explained_energy": (1.0 - residual_energy / source_energy if source_energy else 1.0),
            "snr_db": (
                10.0 * np.log10(source_energy / residual_energy) if residual_energy else "infinity"
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze external recordings without a known library"
    )
    parser.add_argument("--manifest", type=Path, default=Path("corpus/external-small.json"))
    parser.add_argument("--cache", type=Path, default=Path("corpus/cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/external-benchmark"))
    args = parser.parse_args()
    report = run_external_benchmark(args.manifest, args.cache, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
