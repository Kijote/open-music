# Dataset policy

Open Music separates code licensing from audio licensing. A file being downloadable or royalty-free does not necessarily permit redistribution as part of a dataset.

Every imported asset must record:

- canonical source URL and provider identifier;
- author or uploader when applicable;
- exact license and license URL;
- retrieval date;
- cryptographic checksum;
- whether redistribution, derivatives, and commercial use are allowed.

## Initial providers

### VCSL

Versilian Community Sample Library is the preferred first library because its collection is published under CC0. Start with percussion so module discovery can be evaluated against transient, repeated events.

### NSynth

NSynth provides isolated notes with pitch, velocity, instrument family, and source labels. It is useful for matching and classification experiments. Keep its provenance and dataset terms attached to every imported item.

### University of Iowa

The University of Iowa Musical Instrument Samples are useful acoustic references. Preserve the institution's usage statement and original URL in the manifest.

### Freesound

Freesound is a provider, not one homogeneous license. The first importer must accept CC0 only. CC BY may be added after automatic attribution exists. CC BY-NC must not enter the generally redistributable corpus.

## Corpus tiers

- **smoke**: tiny deterministic/generated fixtures committed to the repository and run on every PR;
- **open-small**: pinned, checksum-verified open audio downloaded in CI;
- **extended**: larger datasets run on a schedule or manually;
- **local**: user-owned music that never enters CI or repository history.

The extended benchmark must never make pull-request CI depend on a large or unreliable external download.

## External recording benchmark

`corpus/external-small.json` is intentionally separate from the known-sample corpus. Its first
recording is the CC0 *Prehistoric Drum Loop* by hornpipe2 from OpenGameArt. The manifest pins the
downloaded byte size and SHA-256 digest and contains no source samples or event timeline.

The scheduled/manual `external-benchmark.yml` workflow receives only that mixed WAV. It detects
onsets, exports every candidate window, computes coverage and a repeat-similarity matrix, and
publishes the source, candidates, and deterministic JSON report as an artifact.

The discovery stage compares time-local spectra at multiple window sizes, clusters candidates by
similarity, selects a deterministic medoid for each cluster, estimates per-event gain, and renders
a first-pass modular reconstruction. Keeping the spectral scales explicit also provides a baseline
for later learned embeddings or fine-tuned similarity weights.
The workflow also publishes each canonical module, the reconstruction, and its residual.

Residual discovery then repeats the same process on the current residual. A residual cluster is
accepted only when at least two events reuse its canonical module and subtraction produces a
minimum energy improvement. Singleton clusters are exported for diagnosis but rejected as modules,
preventing a lower error from being achieved by memorizing each unexplained fragment. The report
records every pass and one deterministic stopping reason: residual floor, no onsets, no reusable
clusters, insufficient improvement, or the configured pass limit.

The external corpus also includes two 6.58-second, 44.1 kHz mono CC0 tracks from
*CC0 Scraps* by celestialghost8: `wip loop - Track 02` and `slowdrum - Track 02`. They deliberately
contrast with the stereo transient percussion loop. The current transient pipeline reconstructs
neither tonal track, making the aggregate score a regression baseline for sustained-note and
harmonic-change discovery rather than presenting the percussion result as general performance.

## Listening evaluation

Every external recording artifact contains a `listening/index.html` page and directly playable WAV
files. The pack includes deterministic blind A/B source and reconstruction files, the true-level
residual, and a separately labeled peak-normalized residual for hearing low-level artifacts. The
page records independent 1–5 ratings for timbre, transients, continuity, spatial image, and overall
similarity. Objective metrics remain available in a collapsed section and do not pre-populate the
subjective assessment.
