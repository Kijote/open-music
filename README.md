# Open Music

Open Music compiles linear audio into a modular, reproducible representation made of samples, events, transformations, patterns, and an optional residual.

The project optimizes for **modularity first**: explain as much of a recording as possible with the smallest useful set of reusable audio modules.

## First executable contract

The initial vertical slice establishes the invariant that every change must preserve:

1. build a deterministic reference mix from reusable samples;
2. represent it as a library plus timed events;
3. render that representation;
4. compare it with the reference;
5. report waveform error, SNR, and explained energy.

This is intentionally smaller than automatic source discovery. It gives future extraction and matching algorithms an objective regression harness.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Core model

```text
OpenMusicDocument
├── sample library
├── timeline events
│   ├── sample reference
│   ├── position
│   ├── gain
│   └── pan
├── rendered approximation
└── residual = reference - approximation
```

See [VISION.md](VISION.md), [ROADMAP.md](ROADMAP.md), and [docs/DATASETS.md](docs/DATASETS.md).

## License

Code is released under the MIT License. Audio assets retain their own licenses and must carry provenance metadata.
