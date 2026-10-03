# Reproducible sample reconstruction

The experiment must produce the sample reconstruction **before** calculating a residual.

```
sampled-track = silence + transformed samples at their detected offsets
residual      = source - sampled-track
```

`render_sampled_track.py` implements that invariant. It never uses the original
recording as the background of the sampled track. Each manifest placement can
specify `sample`, `start_seconds`, `pitch_ratio`, and optional `gain`.

The renderer also verifies:

```
sampled-track + residual ~= source
```

Full commercial source audio and full-length derived audio are not committed to
this public repository. The script, placement/pitch metadata, validation data,
and experiment results are persisted so the run is reproducible with a locally
supplied source recording.

Example:

```bash
python experiments/brb/2026-10-02/render_sampled_track.py \
  --source /path/to/source.mp3 \
  --manifest /path/to/render-manifest.json \
  --samples-dir /path/to/samples \
  --out-dir ./out
```

The iterative research loop is:

```
source -> discover/validate reusable samples -> sampled-track -> residual
       -> discover/validate on residual -> extend sampled-track -> residual ...
```
