#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
def load(p,d): return json.loads(p.read_text()) if p.exists() else d
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2)+"\n")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--samples-dir",type=Path,required=True); p.add_argument("--work-dir",type=Path,required=True); p.add_argument("--discover-cmd"); p.add_argument("--max-iterations",type=int,default=100); p.add_argument("--min-reuse",type=int,default=2); p.add_argument("--auto-threshold",type=float,default=.90); p.add_argument("--patience",type=int,default=3); p.add_argument("--renderer",type=Path,default=Path(__file__).with_name("render_sampled_track.py")); p.add_argument("--detector",type=Path,default=Path(__file__).with_name("discover_residual.py")); a=p.parse_args()
 sp=a.work_dir/"state.json"; s=load(sp,{"iteration":0,"placements":[],"stalled":0,"mode":"discovery"}); a.work_dir.mkdir(parents=True,exist_ok=True)
 for _ in range(a.max_iterations):
  n=s["iteration"]; d=a.work_dir/"iterations"/f"{n:03d}"; d.mkdir(parents=True,exist_ok=True); residual=a.source if n==0 else a.work_dir/"iterations"/f"{n-1:03d}"/"residual.wav"; cp=d/"candidates.json"
  if a.discover_cmd: subprocess.run(a.discover_cmd.format(residual=residual,iteration_dir=d,candidates=cp),shell=True,check=True)
  else: subprocess.run(["python3",str(a.detector),"--input",str(residual),"--output",str(cp),"--samples-dir",str(a.samples_dir),"--threshold",str(a.auto_threshold)],check=True)
  rows=load(cp,[]); acc=[c for c in rows if c.get("human_accepted") or (c.get("occurrence_count",0)>=a.min_reuse and c.get("score",0)>=a.auto_threshold and not c.get("needs_review",False))]; save(d/"accepted.json",acc)
  if not acc:
   s["stalled"]+=1
   if s["stalled"]>=a.patience:
    if s["mode"]=="discovery": s["mode"]="exhaustion"; s["stalled"]=0
    else: save(sp,s); break
   s["iteration"]+=1; save(sp,s); continue
  s["stalled"]=0; known={(round(x["start_seconds"],6),x.get("module_id")) for x in s["placements"]}
  for c in acc:
   k=(round(c["start_seconds"],6),c.get("module_id"))
   if k not in known: s["placements"].append({x:c[x] for x in ("module_id","sample","start_seconds","pitch_ratio","gain") if x in c}); known.add(k)
  mf=d/"placements.json"; save(mf,{"placements":s["placements"]}); subprocess.run(["python3",str(a.renderer),"--source",str(a.source),"--manifest",str(mf),"--samples-dir",str(a.samples_dir),"--out-dir",str(d)],check=True)
  r=load(d/"render-result.json",{}); m={"iteration":n,"mode":s["mode"],"candidate_count":len(rows),"accepted_count":len(acc),"total_placements":len(s["placements"]),**r}; save(d/"metrics.json",m); s["last_metrics"]=m; s["iteration"]+=1; save(sp,s)
if __name__=="__main__": main()
