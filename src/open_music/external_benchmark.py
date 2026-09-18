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
        sample_rate, audio = read_wav(paths[recording.id])
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
        reconstruction = render(library, events, audio.shape[0], sample_rate)
        residual = audio - reconstruction
        if not np.allclose(residual, iterative_residual):
            raise RuntimeError("iterative residual does not match rendered events")
        fidelity = measure(audio, reconstruction)

        recording_dir = output_dir / recording.id
        write_wav(recording_dir / "source.wav", sample_rate, audio)
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
        write_wav(recording_dir / "reconstruction.wav", sample_rate, reconstruction)
        write_wav(recording_dir / "residual.wav", sample_rate, residual)

        duration_seconds = audio.shape[0] / sample_rate
        results.append(
            {
                "recording_id": recording.id,
                "source": recording.source,
                "license": recording.license_spdx,
                "sample_rate": sample_rate,
                "channels": int(audio.shape[1]),
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
                "reconstruction": {
                    "mean_absolute_error": fidelity.mean_absolute_error,
                    "snr_db": fidelity.snr_db,
                    "explained_energy": fidelity.explained_energy,
                    "residual_energy": float(np.sum(np.square(residual))),
                },
            }
        )

    report = {"corpus": corpus.id, "recordings": results}
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
