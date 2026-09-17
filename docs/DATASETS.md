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
