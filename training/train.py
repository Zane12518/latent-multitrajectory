
"""Bounded training. Output paths must be new; full resume does not reseed."""
import argparse,json,time
from pathlib import Path
import torch
from latent_multi.runtime import Session,save_new
from latent_multi.maze import validate_input

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--data',required=True)
    p.add_argument('--arm',choices=['mass','multi','set'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--resume');p.add_argument('--stream',type=int);p.add_argument('--updates',type=int)
    p.add_argument('--save-every',type=int,default=250)
    a=p.parse_args();config=json.loads(Path(a.config).read_text());rows=json.loads(Path(a.data).read_text())
    if len(rows)<2 or len({r['id'] for r in rows})!=len(rows):raise ValueError('training IDs')
    for r in rows:validate_input(r,references=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    s=Session(config,rows,a.device)
    if a.resume:s.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    if a.stream is not None:s.new_stream(a.stream)
    save_new(out/'start.pt',s.state());start=s.step;updates=a.updates if a.updates is not None else config['training']['updates']
    if updates<1:raise ValueError('updates must be positive')
    begin=time.monotonic()
    with (out/'training.jsonl').open('x') as f:
        for j in range(updates):
            row=s.update(a.arm);row['elapsed_seconds']=time.monotonic()-begin
            f.write(json.dumps(row)+'\n');f.flush()
            if (j+1)%a.save_every==0 or j+1==updates:save_new(out/f'step{s.step:05d}.pt',s.state())
            print(json.dumps({k:row[k] for k in ('step','lr','grad_preclip','update_norm')}),flush=True)
    summary=dict(status='complete',arm=a.arm,start=start,end=s.step,updates=updates,
                 offered_maps=2*config['training']['microbatches']*updates,
                 native_ACT_exposure='see training.jsonl',prior_K=32,outer_steps=16,
                 device=str(s.device),hardware=torch.cuda.get_device_name() if s.device.type=='cuda' else 'CPU',
                 seconds=time.monotonic()-begin,actual_bill=None)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
