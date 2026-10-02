from pathlib import Path
import json, wave
import numpy as np

ROOT = Path(__file__).parent / 'block-rockin-beats-analysis'

def read(path):
    with wave.open(str(path), 'rb') as w:
        x=np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').reshape(-1,w.getnchannels()).astype(float)/32768
        return x, w.getframerate()

def mono(x): return x.mean(axis=1) if x.ndim==2 else x

def main():
    residual, rate = read(ROOT/'bundle-analysis/bundle-residual.wav')
    source, _ = read(ROOT/'source.wav')
    index=json.loads((ROOT/'bundle-analysis/seeds/index.json').read_text())
    excluded=np.zeros(len(residual), dtype=bool)
    for s in index['seeds']:
        a=round(s['target_start_seconds']*rate); b=min(len(excluded),a+round(s['duration_seconds']*rate)); excluded[a:b]=True
    r=mono(residual); y=mono(source); hop=round(.12*rate); size=round(.6*rate)
    results=[]
    for s in index['seeds']:
        a=round(s['target_start_seconds']*rate); target=y[a:a+size]; target=target-target.mean(); tn=np.linalg.norm(target)
        best=[]
        for pos in range(0,len(r)-size,hop):
            if excluded[pos:pos+size].mean()>.05: continue
            cand=r[pos:pos+size]; cand=cand-cand.mean(); cn=np.linalg.norm(cand)
            if tn and cn: best.append((float(np.dot(target,cand)/(tn*cn)),pos))
        best.sort(reverse=True)
        for score,pos in best[:3]: results.append({'seed':s['seed'],'target_start_seconds':a/rate,'residual_candidate_start_seconds':pos/rate,'duration_seconds':size/rate,'score':score})
    out=ROOT/'bundle-analysis/residual-seed-search.json'; out.write_text(json.dumps({'seeds':len(index['seeds']),'candidates':results,'hop_seconds':hop/rate},indent=2)+'\n'); print(out, len(results))

if __name__=='__main__': main()
