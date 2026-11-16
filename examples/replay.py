
"""Interactive, offline replay of saved candidates; does not run a model."""
import argparse,json
from pathlib import Path
from latent_multi.maze import metrics,check,validate_input
from latent_multi.objective import nextcell
from latent_multi.selection import scores

def trace(row,actions):
    """Display a candidate up to its first error, without changing its selection score."""
    cells=row['x'];loc=cells.index(3);goal=cells.index(4);seen=set();route=[loc]
    while loc!=goal:
        if loc in seen:return route,'loop'
        seen.add(loc);action=int(actions[loc])
        if action not in range(4):return route,'stop'
        dest=nextcell(loc,action)
        if dest<0:return route,'boundary'
        route.append(dest)
        if cells[dest]==2:return route,'wall'
        loc=dest
    return route,'shortest' if len(route)-1==row['optimal'] else 'longer'

def build_payload(data):
    maps=data['maps'];ids=[row['id'] for row in maps];models={}
    for row in maps:validate_input(row)
    for name,value in data['models'].items():
        if value['map_ids']!=ids or len(value['candidates'])!=len(maps):
            raise ValueError('map/candidate identity mismatch')
        panels=[]
        for row,pool in zip(maps,value['candidates']):
            actions=pool['actions'];lp=pool['chosen_logprobs']
            if len(actions)!=32 or len(lp)!=32 or any(len(y)!=100 for y in actions+lp):
                raise ValueError('expected K32, 100-cell outputs and scores')
            confidence=scores(row['x'],actions,lp)
            mm=metrics(row,actions,confidence,ks=range(1,33))
            curves=[{key:value for key,value in mm[k].items() if not key.endswith('_set')} for k in range(1,33)]
            traces=[]
            for y,score in zip(actions,confidence):
                route,status=trace(row,y);legal,short,identity,length=check(row,y)
                assert legal==(status in ('shortest','longer')) and short==(status=='shortest')
                if legal:assert tuple(route)==identity
                traces.append(dict(route=route,status=status,length=length,confidence=score))
            panels.append(dict(traces=traces,metrics=curves))
        models[name]=panels
    pairs=[]
    for name in models:
        if name.endswith('_multi'):
            prefix=name[:-6];other=prefix+'_set'
            if other not in models:raise ValueError('missing matched Set endpoint')
            pairs.append(dict(id=prefix,multi=name,set=other))
    if len(pairs)*2!=len(models):raise ValueError('unpaired endpoints')
    return dict(mode=data['mode'],selection=data['selection'],maps=maps,models=models,pairs=pairs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/demo_replay.json');p.add_argument('--out',default='outputs/replay.html');p.add_argument('--map-index',type=int,default=0)
    a=p.parse_args();payload=build_payload(json.loads(Path(a.data).read_text()))
    if a.map_index not in range(len(payload['maps'])):p.error('map-index is outside the saved map range')
    template=Path(__file__).with_name('replay.html').read_text()
    body=template.replace('__REPLAY_DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')).replace('__INITIAL_MAP__',str(a.map_index))
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:f.write(body)
    output=[dict(model=name,map_id=payload['maps'][a.map_index]['id'],**panels[a.map_index]['metrics'][-1]) for name,panels in payload['models'].items()]
    print(json.dumps(dict(mode='interactive-saved-experiment-replay',maps=len(payload['maps']),K=32,models=output),indent=2))
if __name__=='__main__':main()
