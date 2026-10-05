#!/usr/bin/env python3
"""Evaluate autonomous residual discovery until convergence.

Runs the repository iterator/detector, records per-iteration discovery cost and
quality metrics, and verifies the exact reconstruction invariant. Commercial
source audio is a local input and is never committed.
"""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from scipy.io import wavfile
def load(p,d=None): return json.loads(p.read_text()) if p.exists() else d
def main():
 p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--samples-dir",type=Path,required=True); p.add_argument("--work-dir",type=Path,required=True); p.add_argument("--max-iterations",type=int,default=100); p.add_argument("--threshold",type=float,default=.90); a=p.parse_args()
 here=Path(__file__).parent; runner=here/"iterate_residual.py"; t=time.perf_counter()
 subprocess.run(["python3",str(runner),"--source",str(a.source),"--samples-dir",str(a.samples_dir),"--work-dir",str(a.work_dir),"--max-iterations",str(a.max_iterations),"--auto-threshold",str(a.threshold)],check=True)
 elapsed=time.perf_counter()-t; state=load(a.work_dir/"state.json",{})
 rows=[]
 for d in sorted((a.work_dir/"iterations").glob("[0-9][0-9][0-9]")):
  m=load(d/"metrics.json")
  if m: rows.append(m)
 # Independently verify final PCM-domain reconstruction when outputs exist.
 maxerr=rmserr=None
 if rows:
  d=a.work_dir/"iterations"/f"{rows[-1]['iteration']:03d}"; sw=d/"sampled-track.wav"; rw=d/"residual.wav"
  if sw.exists() and rw.exists():
   sr,s=wavfile.read(sw); _,r=wavfile.read(rw)
   import tempfile
   with tempfile.TemporaryDirectory() as td:
    src=Path(td)/"source.wav"; subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(a.source),"-ar",str(sr),"-ac","2","-c:a","pcm_f32le",str(src)],check=True); _,x=wavfile.read(src)
   n=min(len(x),len(s),len(r)); e=s[:n].astype(np.float64)+r[:n].astype(np.float64)-x[:n].astype(np.float64); maxerr=float(np.max(np.abs(e))); rmserr=float(np.sqrt(np.mean(e*e)))
 report={"elapsed_seconds":elapsed,"iterations_with_metrics":len(rows),"final_state":state,"max_reconstruction_error":maxerr,"rms_reconstruction_error":rmserr,"iterations":rows}
 (a.work_dir/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
