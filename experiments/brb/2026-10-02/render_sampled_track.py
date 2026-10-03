#!/usr/bin/env python3
"""Render a sample-only reconstruction and its exact residual.

Pipeline:
  sampled_track = silence + transformed samples at manifest offsets
  residual = source - sampled_track
  invariant: sampled_track + residual == source (within float tolerance)

Requires: Python 3, numpy, scipy, ffmpeg with rubberband filter.
The source recording and generated full-length audio are external inputs/outputs
and are intentionally not stored in this public repository.
"""
from __future__ import annotations
import argparse, json, math, subprocess, tempfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile

def ffmpeg_decode(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(src),
                    "-ar","44100","-ac","2","-c:a","pcm_f32le",str(dst)],check=True)

def pitch(src: Path, ratio: float, dst: Path) -> None:
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(src),
                    "-af",f"rubberband=pitch={ratio:.12g}",
                    "-c:a","pcm_f32le",str(dst)],check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--samples-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--fade-ms",type=float,default=12.0)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.manifest.read_text())
    placements=cfg["placements"]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); source_wav=td/"source.wav"; ffmpeg_decode(a.source,source_wav)
        sr,source=wavfile.read(source_wav); source=source.astype(np.float32)
        modeled=np.zeros_like(source,dtype=np.float32)
        fade=round(sr*a.fade_ms/1000)
        rendered=[]
        for i,p in enumerate(placements):
            sample=a.samples_dir/p["sample"]
            ratio=float(p.get("pitch_ratio",1.0))
            shifted=td/f"{i:04d}.wav"; pitch(sample,ratio,shifted)
            sr2,x=wavfile.read(shifted)
            if sr2!=sr: raise ValueError(f"sample rate mismatch: {sample}")
            x=x.astype(np.float32)
            start=round(float(p["start_seconds"])*sr)
            n=min(len(x),len(modeled)-start)
            if n<=0: continue
            x=x[:n]; w=np.ones(n,dtype=np.float32); f=min(fade,n//2)
            if f:
                w[:f]=np.linspace(0,1,f,endpoint=False)
                w[-f:]=np.linspace(1,0,f,endpoint=False)
            gain=float(p.get("gain",1.0))
            modeled[start:start+n]+=gain*x*w[:,None]
            rendered.append({**p,"rendered_frames":n})
        residual=source-modeled
        err=(modeled+residual)-source
        maxerr=float(np.max(np.abs(err)))
        rmserr=float(np.sqrt(np.mean(err.astype(np.float64)**2)))
        wavfile.write(a.out_dir/"sampled-track.wav",sr,modeled)
        wavfile.write(a.out_dir/"residual.wav",sr,residual)
        result={"sample_rate":sr,"frames":len(source),"duration_seconds":len(source)/sr,
                "placements_rendered":len(rendered),"max_reconstruction_error":maxerr,
                "rms_reconstruction_error":rmserr,"placements":rendered}
        (a.out_dir/"render-result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))

if __name__=="__main__": main()
