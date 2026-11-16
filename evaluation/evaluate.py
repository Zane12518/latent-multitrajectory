
"""Generate K32 prior candidates. No filtering/repair/search; metrics use a checker afterwards."""
import argparse,json,time
from pathlib import Path
import torch
from latent_multi.runtime import Session,save_new
from latent_multi.sampling import predict
from latent_multi.selection import scores
from latent_multi.maze import metrics,validate_input,KS

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--limit',type=int);p.add_argument('--batch-maps',type=int,default=2)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());full=json.loads(Path(a.data).read_text())
    rows=full[:a.limit] if a.limit else full
    if not 1<=a.repeats<=3:raise ValueError('registered repeats are 1..3')
    if len({r['id'] for r in full})!=len(full):raise ValueError('duplicate map IDs')
    for row in rows:validate_input(row)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);s=Session(cfg,rows,a.device)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if 'model_config' in c:
        for key,value in c['model_config'].items():
            if key!='forward_dtype' and cfg['model'].get(key)!=value:raise ValueError('checkpoint/config mismatch: '+key)
    s.m.load_state_dict(c['model']);s.m.eval()
    records=[];begin=time.monotonic()
    for repeat in range(a.repeats):
        raw=[]
        for off in range(0,len(rows),a.batch_maps):
            items=rows[off:off+a.batch_maps];ys,lp,z=predict(s,items,list(range(off,off+len(items))),repeat)
            raw.append(dict(ids=[r['id'] for r in items],actions=ys,chosen_logprobs=lp,logits=z))
            for row,y,l in zip(items,ys.tolist(),lp.tolist()):
                for rule in ('grid','path'):
                    mm=metrics(row,y,scores(row['x'],y,l,rule))
                    records.extend(dict(id=row['id'],repeat=repeat,K=k,rule=rule,**v) for k,v in mm.items())
        save_new(out/f'predictions_{repeat}.pt',dict(ids=[r['id'] for r in rows],repeat=repeat,raw=raw))
    (out/'per_map.json').write_text(json.dumps(records))
    fields=('U','U_all','validity','first','H','H_all','selected','V','D','V_target','D_target')
    means={rule:{str(k):{f:sum(r[f] for r in records if r['K']==k and r['rule']==rule)/(len(rows)*a.repeats) for f in fields} for k in KS} for rule in ('grid','path')}
    report=dict(status='complete',count=len(rows),population_count=len(full),complete_population=len(rows)==len(full),
                means=means,repeats=a.repeats,K=32,outer_steps=16,random_nodes=32,
                protocol='project K32 / logical-map-node-v2',device=a.device,dtype=cfg['model']['forward_dtype'],
                seconds=time.monotonic()-begin,actual_bill=None)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
