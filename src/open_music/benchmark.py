from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import Event, Sample, match_sample, measure, render
from .corpus import fetch_corpus, load_manifest
from .wav import read_wav, resample_linear, write_wav


def run_benchmark(manifest_path: Path, cache_dir: Path, output_dir: Path) -> dict:
    corpus = load_manifest(manifest_path)
    paths = fetch_corpus(corpus, cache_dir)

    library: dict[str, Sample] = {}
    for asset in corpus.assets:
        source_rate, audio = read_wav(paths[asset.id])
        audio = resample_linear(audio, source_rate, corpus.target_sample_rate)
        library[asset.id] = Sample(
            id=asset.id,
            audio=audio,
            sample_rate=corpus.target_sample_rate,
            license=asset.license_spdx,
            source=f"{asset.source['repository']}@{asset.source['commit']}:{asset.source['path']}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for track in corpus.tracks:
        events = [
            Event(
                sample_id=item["sample_id"],
                start_frame=round(item["time_seconds"] * corpus.target_sample_rate),
                gain=float(item.get("gain", 1.0)),
                pan=float(item.get("pan", 0.0)),
            )
            for item in track["events"]
        ]
        total_frames = round(float(track["duration_seconds"]) * corpus.target_sample_rate)
        reference = render(library, events, total_frames, corpus.target_sample_rate)

        interpreted_events: list[Event] = []
        for event in events:
            query = library[event.sample_id].audio
            match = match_sample(query, library)
            interpreted_events.append(
                Event(match.sample_id, event.start_frame, event.gain, event.pan)
            )

        reconstruction = render(
            library, interpreted_events, total_frames, corpus.target_sample_rate
        )
        residual = reference - reconstruction
        metrics = measure(reference, reconstruction)
        used_samples = {event.sample_id for event in interpreted_events}
        stored_frames = sum(library[sample_id].audio.shape[0] for sample_id in used_samples)

        track_dir = output_dir / track["id"]
        write_wav(track_dir / "original.wav", corpus.target_sample_rate, reference)
        write_wav(track_dir / "reconstruction.wav", corpus.target_sample_rate, reconstruction)
        write_wav(track_dir / "residual.wav", corpus.target_sample_rate, residual)

        results.append(
            {
                "track_id": track["id"],
                "metrics": {
                    "mean_absolute_error": metrics.mean_absolute_error,
                    "snr_db": metrics.snr_db,
                    "explained_energy": metrics.explained_energy,
                },
                "modularity": {
                    "event_count": len(interpreted_events),
                    "unique_module_count": len(used_samples),
                    "stored_sample_seconds": stored_frames / corpus.target_sample_rate,
                    "track_seconds": total_frames / corpus.target_sample_rate,
                    "reuse_ratio": len(interpreted_events) / len(used_samples),
                },
            }
        )

    report = {"corpus": corpus.id, "tracks": results}
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False).replace("Infinity", '"infinity"'),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Open Music corpus benchmark")
    parser.add_argument("--manifest", type=Path, default=Path("corpus/open-small.json"))
    parser.add_argument("--cache", type=Path, default=Path("corpus/cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    args = parser.parse_args()
    report = run_benchmark(args.manifest, args.cache, args.output)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
