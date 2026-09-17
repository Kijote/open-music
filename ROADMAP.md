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
- [ ] Separate scheduled extended benchmark.
- [x] Publish benchmark metrics and audio as workflow artifacts.

## M2 — Module discovery

- [x] Onset detection.
- [x] Candidate-window extraction.
- [ ] Similarity clustering without a known library.
- [ ] Canonical sample selection.
- [ ] Iterative subtraction and residual analysis.

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
