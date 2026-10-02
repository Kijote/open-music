# Block Rockin' Beats — strict reuse experiment (2026-10-02)

This directory is the durable record for the attack-anchored reuse experiment previously explored interactively.

## Confirmed baseline

- 964 indexed attacks/events.
- 3,856 candidate pairs retrieved.
- 16 pairs accepted by strict ordered time-frequency validation.
- Listening review: all 16 accepted comparisons were judged excellent.
- 12 non-overlapping bundles in the strict reconstruction stage.
- 4.35% temporal bundle coverage.
- 2.50% replaced energy.
- Exact decomposition error: 0.

## Continuous pitch refinement

Pitch is refined continuously rather than by integer semitone steps.

- Coarse ratio step: 0.002.
- Fine ratio step: 0.0002.
- Activation threshold: 0.90.
- 19 directed validated activation candidates.
- After grouping reciprocal/duplicate representations, three currently examined modules expose 3, 3 and 7 unique occurrences.
- Useful pitch corrections are microtonal (cents), not integer semitone transpositions.

See `pitch-cross.json`, `pitch-cross-grouped.json`, and `validated-activation-candidates.json`.

## Source audio policy

The commercial source track is intentionally **not committed**. It is an external experiment input. Generated results must never depend on a chat attachment: scripts/configuration/result metadata belong in this repository, while source audio is supplied locally.

Audio audition exports should be generated deterministically from the local source and these manifests. This prevents experiment state from being trapped in chat history without redistributing the complete copyrighted track.

## Next

1. Export A/B auditions for every newly discovered occurrence after pitch correction.
2. Listening-review them against the 16/16 positive baseline.
3. Promote confirmed occurrences into the reusable-module benchmark.
4. Expand recall while preserving perceptual precision.
