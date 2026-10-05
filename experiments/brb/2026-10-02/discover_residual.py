#!/usr/bin/env python3
"""Ordered time-frequency residual reuse detector.

Finds attack-anchored 600 ms segments, retrieves candidates with a compact
ordered-STFT descriptor, validates with the full ordered STFT sequence and
waveform correlation under bounded lag, then performs continuous micro-pitch
refinement. No temporally averaged spectrum is used for final acceptance.
"""
from __future__ import annotations
import argparse,json,math,subprocess,tempfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft,find_peaks,resample

def decode(src,dst):
 subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(src),"-ar","44100","-ac","1","-c:a","pcm_f32le",str(dst)],check=True)
def ordered(x,sr,n=1024):
 _,_,z=stft(x,fs=sr,nperseg=n,noverlap=3*n//4,boundary=None,padded=False); a=np.abs(z).T
 a=np.log1p(20*a); return a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
def sim(a,b):
 m=min(len(a),len(b)); return float(np.mean(np.sum(a[:m]*b[:m],axis=1))) if m else 0.
def corr_lag(a,b,maxlag):
 best=(-2.,0)
 for lag in range(-maxlag,maxlag+1):
  x=a[max(0,lag):min(len(a),len(b)+lag)]; y=b[max(0,-lag):min(len(b),len(a)-lag)]
  if len(x)<32: continue
  c=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
  if c>best[0]: best=(c,lag)
 return best
def pitch_audio(x,ratio):
 # Fast in-memory pitch proxy for fine search: frequency-axis change via
 # resampling followed by duration restoration. Final render uses rubberband.
 n=max(8,round(len(x)/ratio)); y=resample(x,n); return resample(y,len(x)).astype(np.float32)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--samples-dir",type=Path,required=True); p.add_argument("--duration",type=float,default=.6); p.add_argument("--threshold",type=float,default=.90); p.add_argument("--top-k",type=int,default=12); p.add_argument("--min-separation",type=float,default=.25); a=p.parse_args(); a.samples_dir.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory() as td:
  w=Path(td)/"x.wav"; decode(a.input,w); sr,x=wavfile.read(w); x=x.astype(np.float32)
 n=round(a.duration*sr); env=np.convolve(np.abs(x),np.ones(max(1,sr//200))/max(1,sr//200),mode="same")
 peaks,_=find_peaks(env,distance=round(a.min_separation*sr),prominence=max(1e-7,float(np.std(env))*.35))
 starts=[int(q) for q in peaks if q+n<=len(x)]
 seg=[x[q:q+n] for q in starts]; desc=[ordered(v,sr,256) for v in seg]
 # Retrieval uses ordered low-resolution descriptors; validate shortlisted pairs at 1024/4096.
 pairs=[]
 for i in range(len(seg)):
  scored=[]
  for j in range(i):
   if abs(starts[i]-starts[j])<n: continue
   s=sim(desc[i],desc[j])
   if s>.72: scored.append((s,j))
  for rs,j in sorted(scored,reverse=True)[:6]:
   multi=np.mean([sim(ordered(seg[i],sr,k),ordered(seg[j],sr,k)) for k in (1024,4096)])
   c,lag=corr_lag(seg[i],seg[j],round(.004*sr))
   score=.55*multi+.45*max(0,c)
   if score>=a.threshold*.94: pairs.append((score,i,j,multi,c,lag))
 pairs=sorted(pairs,reverse=True)[:a.top_k]
 rows=[]
 for rank,(score,i,j,multi,c,lag) in enumerate(pairs,1):
  # continuous micro-pitch around identity; the earlier BRB experiment showed useful corrections in cents.
  best=(score,1.0,c,lag)
  for ratio in np.arange(.988,1.0121,.002):
   y=pitch_audio(seg[j],float(ratio)); cc,ll=corr_lag(seg[i],y,round(.004*sr)); ss=.55*multi+.45*max(0,cc)
   if ss>best[0]: best=(ss,float(ratio),cc,ll)
  lo=max(.98,best[1]-.002); hi=min(1.02,best[1]+.002)
  for ratio in np.arange(lo,hi+.00001,.0002):
   y=pitch_audio(seg[j],float(ratio)); cc,ll=corr_lag(seg[i],y,round(.004*sr)); ss=.55*multi+.45*max(0,cc)
   if ss>best[0]: best=(ss,float(ratio),cc,ll)
  module=f"iter-module-{rank:03d}"; sample=f"{module}.wav"; wavfile.write(a.samples_dir/sample,sr,seg[j])
  cents=1200*math.log2(best[1])
  # A validated pair creates two reusable occurrences. The canonical exemplar is candidate j.
  for q,ratio in ((j,1.0),(i,1.0/best[1])):
   rows.append({"module_id":module,"sample":sample,"start_seconds":starts[q]/sr,"pitch_ratio":ratio,
    "gain":1.0,"score":best[0],"occurrence_count":2,"ordered_tf_similarity":multi,
    "waveform_correlation":best[2],"lag_samples":best[3],"pitch_cents":cents,
    "needs_review":bool(best[0]<a.threshold)})
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(rows,indent=2)+"\n")
if __name__=="__main__": main()
