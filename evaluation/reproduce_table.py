
"""Recompute all reported six fixed-pair families from archived per-map metrics."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',required=True);p.add_argument('--expected',default='docs/results.json');p.add_argument('--out',required=True)
    a=p.parse_args();expected=json.loads(Path(a.expected).read_text());z=np.load(a.metrics,allow_pickle=False)
    fields=z['fields'].tolist();models=expected['models'];keys=[k for k in models if not k.startswith('old__')]
    # Only exact six fixed comparisons; never select an endpoint from these test arrays.
    ix=[fields.index(f) for f in ('U','validity','first','selected_trace')]
    flows={'1301':[(2026111404,'linear1404'),(2026111405,'linear1405'),(2026111511,'c1301_1511')],
           '1501':[(2026111502,'legacy1502'),(2026111503,'legacy1503'),(2026111512,'c1501_1512')]}
    assert z['ids'].shape==(1200,) and z['K'].tolist()==[1,4,8,16,32]
    for name in keys:
        v=z[name];assert v.shape==(3,1200,5,len(fields)) and np.isfinite(v).all()
        for f,m in models[name].items():assert abs(v[:,:,-1,fields.index(f)].mean()-m)<1e-12
        assert np.all(v[:,:,:,fields.index('V_target')]-v[:,:,:,fields.index('D_target')]==v[:,:,:,fields.index('U')])
    W=np.random.default_rng(2026111621).multinomial(1200,np.full(1200,1/1200),size=20000)/1200
    out=[]
    for source,pairs in flows.items():
        X=np.concatenate([(z[n+'_set']-z[n+'_multi']).mean(0)[:,-1,ix] for _,n in pairs],1)
        pt=X.mean(0);b=W@X;sd=b.std(0,ddof=1);critical=float(np.quantile(np.abs((b-pt)/np.where(sd>1e-15,sd,1)).max(1),.95))
        ref=next(f for f in expected['families'] if f['name']=='new_rng_source'+source)
        assert abs(critical-ref['critical'])<1e-10
        for j,(stream,n) in enumerate(pairs):
            for k,f in enumerate(('U','validity','first','selected_trace')):
                v=dict(delta=float(pt[j*4+k]),lower=float(pt[j*4+k]-critical*sd[j*4+k]),upper=float(pt[j*4+k]+critical*sd[j*4+k]))
                rr=ref['comparisons'][j]['metrics'][f]
                assert max(abs(v[t]-rr[t]) for t in v)<1e-10
            out.append(dict(source=source,stream=stream,U_multi=float(z[n+'_multi'][:,:,-1,0].mean()),U_set=float(z[n+'_set'][:,:,-1,0].mean())))
    result=dict(status='PASS',models=len(keys),maps=1200,repeats=3,comparisons=6,bootstrap_draws=20000,
                numerical_tolerance=1e-10,training_or_model_calls=0,scope='saved metric recomputation, not regeneration of 1200-map predictions',results=out)
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
