# Roadmap

## M0 — Executable reconstruction contract

- [x] Core sample and event representation.
- [x] Deterministic renderer.
- [x] Objective waveform metrics.
- [x] Basic library matching.
- [x] GitHub Actions on pushes and pull requests.

## M1 — Open corpus harness

- [x] Versioned corpus manifest with URL, checksum, license, and attribution.
- [x] VCSL importer, initially percussion only.
- [x] Small openly licensed song/loop evaluation set.
- [x] Local corpus cache excluded from Git.
- [x] CI smoke corpus.
- [x] Separate scheduled external-recording benchmark.
- [x] Publish benchmark metrics and audio as workflow artifacts.
- [x] Publish blind A/B listening packs and subjective rating pages.

## M2 — Module discovery

- [x] Onset detection.
- [x] Candidate-window extraction.
- [ ] Revalidate repeat similarity without temporal spectral averaging.
- [ ] Revalidate similarity clustering with complete-link compatibility.
- [ ] Revalidate canonical sample selection against every replacement.
- [x] Iterative reconstruction, subtraction, and residual analysis.
- [x] Convergent residual rediscovery with reusable-module filtering.
- [x] Lossless granular stream fallback for sustained/onset-free material.
- [ ] Cluster and transform continuous grains to improve stream modularity.

## M2.1 — Replacement validity correction

- [x] Frame-local multiresolution FFT validation (256/1024/4096).
- [x] Independent local limits for spectrum, energy, duration, and incompatible runs.
- [x] Explicit bounded lag and robust gain transforms.
- [x] Complete-link clustering; no assumed transitivity.
- [x] Suggested split boundaries from localized validation failures.
- [ ] Recursive adaptive subdivision at incompatible spans.
- [ ] Windowed overlap-add reconstruction and boundary continuity metrics.
- [ ] Re-run Block Rockin' Beats and calibrate thresholds with blind controls.
- [ ] Regression run on Stairway to Heaven without song-specific parameters.

## M3 — Musical transformations

- [x] Per-event gain estimation.
- [ ] Stereo placement estimation.
- [ ] Pitch estimation and transposition.
- [ ] Envelope fitting.
- [ ] Tempo, beat, and bar inference.
- [ ] Hierarchical modules: hit, note, chord, phrase, pattern.

## M4 — Open Music format

- [ ] Versioned schema.
- [ ] Content-addressed sample references.
- [ ] Embedded and external sample resolution.
- [ ] License/provenance graph.
- [ ] WAV/FLAC renderer and tracker exporters.
