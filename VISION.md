# Vision

Open Music is an open intermediate representation for music reconstructed from reusable audio modules.

Its primary goal is not bit-perfect compression. It is to maximize musical modularity while retaining an objectively measurable relationship with the source recording.

## Product invariant

A representation is useful when it can:

- render without access to the original recording;
- identify every module by content and provenance;
- replace a module with a compatible library candidate;
- expose what remains unexplained as a residual;
- measure regression after every algorithmic change.

## Non-goals

- Being limited to MOD, MIDI, or a specific DAW.
- Assuming every module is a single musical note.
- Hiding license or provenance information.
- Claiming perfect source separation from arbitrary mastered audio.
