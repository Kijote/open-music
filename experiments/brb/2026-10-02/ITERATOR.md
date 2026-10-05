# Iterative residual dismantling

`iterate_residual.py` owns resumable state and calls `render_sampled_track.py` after every accepted batch, preserving `source = sampled-track + residual`.

It starts in **discovery** mode: only high-confidence candidates with at least two occurrences are automatically promoted. After the configured number of stalled iterations it enters **exhaustion**, where the discovery backend may broaden toward persistent components, extensions/subdivisions and ultimately unique material.

Discovery is deliberately an external command boundary. It receives the current residual and writes `candidates.json` following `candidate.schema.json`. This lets ordered-STFT, fine alignment, micro-pitch and later tempo/component search evolve independently from iteration state.

Every successful iteration persists candidates, accepted candidates, cumulative placements, sampled-track, residual and metrics under `iterations/NNN/`. `state.json` permits resume. Temporally averaged spectra must not be used as the final acceptance representation.

Example:
```bash
python experiments/brb/2026-10-02/iterate_residual.py --source /local/brb.mp3 --samples-dir /local/brb-samples --work-dir /tmp/brb-run --discover-cmd 'python discover.py --input {residual} --output {candidates}'
```
