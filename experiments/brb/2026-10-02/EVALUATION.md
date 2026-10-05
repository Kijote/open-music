# BRB iterative discovery evaluation

This evaluation exists to prevent the research pipeline from living only in
temporary notebooks or runtime scripts.

## Acceptance criteria

- Every accepted batch is persisted by iteration.
- Discovery retains ordered temporal spectral information for final validation.
- Reusable modules are preferred during discovery; unique fragments do not count
  as successful reusable discovery.
- Fine pitch/alignment metadata is retained in placements.
- The PCM invariant `source = sampled-track + residual` is independently checked.
- A run stops only by the configured convergence/stall rule or iteration limit.
- Runtime is recorded so retrieval optimizations can be compared without
  weakening the validation gate.

## Performance regression

The current exhaustive retrieval can approach quadratic pair comparison. The
next detector optimization must shortlist candidates using an indexed/vectorized
retrieval representation, then run the existing ordered-STFT multiresolution and
waveform validation only on that shortlist.

A performance change is accepted only when the evaluation shows materially lower
runtime while preserving the high-confidence module set (allowing documented
tie/order differences).
