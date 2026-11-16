
"""Fixed-model paired intervals. Repeat RNGs are averaged inside each map."""
import argparse,json
from pathlib import Path
import numpy as np
FIELDS=('U','validity','first','selected')
def paired(a,b,draws=20000,seed=2026111661):
    def table(rows):
        out={}
        for r in rows:
            if r['K']==32 and r['rule']=='path':out.setdefault(r['id'],[]).append([r[f] for f in FIELDS])
        return {k:np.mean(v,axis=0) for k,v in out.items()}
    aa,bb=table(a),table(b)
    if aa.keys()!=bb.keys():raise ValueError('map identity mismatch')
    X=np.stack([aa[k]-bb[k] for k in sorted(aa)]);p=X.mean(0)
    rng=np.random.default_rng(seed);boot=[]
    for off in range(0,draws,250):
        W=rng.multinomial(len(X),np.full(len(X),1/len(X)),size=min(250,draws-off))/len(X)
        boot.append(W@X)
    boot=np.concatenate(boot);se=boot.std(0,ddof=1)
    crit=float(np.quantile(np.abs((boot-p)/np.where(se>1e-15,se,1)).max(1),.95))
    metrics={f:dict(delta=float(v),lower=float(v-crit*s),upper=float(v+crit*s)) for f,v,s in zip(FIELDS,p,se)}
    gates={f:'protected' if v['lower']>=-.01 else 'damage_beyond_margin' if v['upper']<-.01 else 'inconclusive' for f,v in metrics.items() if f!='U'}
    return dict(metrics=metrics,gates=gates,family='one pair x four metrics',draws=draws,critical=crit,
                method='fixed-standard-deviation standardized maximum error; not bootstrap-t',
                interval_scope='fixed models and maps, not training or selection uncertainty')
def main():
    p=argparse.ArgumentParser();p.add_argument('--set',required=True);p.add_argument('--multi',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();v=paired(json.loads(Path(a.set).read_text()),json.loads(Path(a.multi).read_text()))
    with Path(a.out).open('x') as f:json.dump(v,f,indent=2)
    print(json.dumps(v,indent=2))
if __name__=='__main__':main()
