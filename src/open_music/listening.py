from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np

from .core import Audio
from .wav import write_wav

RATING_DIMENSIONS = ("timbre", "transients", "continuity", "spatial_image", "overall")


def blind_assignment(recording_id: str) -> dict[str, str]:
    source_first = hashlib.sha256(recording_id.encode()).digest()[0] % 2 == 0
    return (
        {"A.wav": "source", "B.wav": "reconstruction"}
        if source_first
        else {"A.wav": "reconstruction", "B.wav": "source"}
    )


def _match_channels(audio: Audio, channels: int) -> Audio:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] == channels:
        return values
    if channels == 1:
        return np.mean(values, axis=1, keepdims=True)
    if values.shape[1] == 1 and channels == 2:
        return np.repeat(values, 2, axis=1)
    raise ValueError("unsupported listening channel conversion")


def _listening_html(recording_id: str, metrics: dict, answer: dict[str, str]) -> str:
    dimensions = "\n".join(
        f"<label>{escape(name.replace('_', ' ').title())}"
        f'<input type="range" min="1" max="5" value="3" data-rating="{name}"></label>'
        for name in RATING_DIMENSIONS
    )
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Open Music listening test — {escape(recording_id)}</title>
<style>
body{{font:16px system-ui;max-width:820px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}}
button,input{{margin:.4rem}} button{{padding:.6rem 1rem}} label{{display:block}}
pre{{white-space:pre-wrap;background:#222;padding:1rem}} .hidden{{display:none}}
</style>
<h1>{escape(recording_id)}</h1>
<p>Use headphones. Match loudness before judging. A/B is blind until you reveal the answer.</p>
<audio id="player" controls preload="metadata" src="A.wav"></audio>
<p><button data-track="A.wav">Play A</button><button data-track="B.wav">Play B</button></p>
<h2>Ratings: 1 poor — 5 indistinguishable</h2>
{dimensions}
<p><button id="download">Download ratings</button><button id="reveal">Reveal A/B</button></p>
<pre id="answer" class="hidden"></pre>
<h2>Artifact inspection</h2>
<p>True residual:</p><audio controls preload="metadata" src="residual-true.wav"></audio>
<p>Peak-normalized residual (amplified for diagnosis):</p>
<audio controls preload="metadata" src="residual-amplified.wav"></audio>
<details><summary>Objective metrics</summary><pre>{escape(json.dumps(metrics, indent=2))}</pre></details>
<script>
const player=document.querySelector('#player');
document.querySelectorAll('[data-track]').forEach(b=>b.onclick=()=>{{
 const time=player.currentTime,playing=!player.paused;player.src=b.dataset.track;
 player.currentTime=time;if(playing)player.play();
}});
const answer={json.dumps(answer, sort_keys=True)};
document.querySelector('#reveal').onclick=()=>{{
 const out=document.querySelector('#answer');out.textContent=JSON.stringify(answer,null,2);
 out.classList.remove('hidden');
}};
document.querySelector('#download').onclick=()=>{{
 const ratings=Object.fromEntries([...document.querySelectorAll('[data-rating]')]
   .map(x=>[x.dataset.rating,Number(x.value)]));
 const data={{recording_id:{json.dumps(recording_id)},ratings}};
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(
   [JSON.stringify(data,null,2)],{{type:'application/json'}}));a.download='ratings.json';a.click();
}};
</script>
</html>
"""


def write_listening_pack(
    output_dir: Path,
    recording_id: str,
    sample_rate: int,
    source: Audio,
    reconstruction: Audio,
    residual: Audio,
    metrics: dict,
) -> dict:
    channels = 1 if np.asarray(source).ndim == 1 else int(np.asarray(source).shape[1])
    source = _match_channels(source, channels)
    reconstruction = _match_channels(reconstruction, channels)
    residual = _match_channels(residual, channels)
    peak = float(np.max(np.abs(residual), initial=0.0))
    amplified_gain = 0.95 / peak if peak else 1.0
    assignment = blind_assignment(recording_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_wav(output_dir / "source.wav", sample_rate, source)
    write_wav(output_dir / "reconstruction.wav", sample_rate, reconstruction)
    write_wav(output_dir / "residual-true.wav", sample_rate, residual)
    write_wav(output_dir / "residual-amplified.wav", sample_rate, residual * amplified_gain)
    tracks = {"source": source, "reconstruction": reconstruction}
    for filename, identity in assignment.items():
        write_wav(output_dir / filename, sample_rate, tracks[identity])

    answer = {
        "recording_id": recording_id,
        "assignment": assignment,
        "residual_amplification_gain": amplified_gain,
        "rating_dimensions": list(RATING_DIMENSIONS),
    }
    (output_dir / "answer-key.json").write_text(
        json.dumps(answer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(
        _listening_html(recording_id, metrics, answer), encoding="utf-8"
    )
    return answer
