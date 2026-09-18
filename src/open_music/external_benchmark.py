from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import fetch_assets, load_external_manifest
from .discovery import detect_onsets, extract_candidates, repeat_similarity_matrix
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

        recording_dir = output_dir / recording.id
        write_wav(recording_dir / "source.wav", sample_rate, audio)
        for index, candidate in enumerate(candidates):
            write_wav(
                recording_dir / "candidates" / f"candidate-{index:03d}.wav", sample_rate, candidate
            )

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
                    [round(value, 12) for value in row]
                    for row in repeat_similarity_matrix(candidates)
                ],
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
