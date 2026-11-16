
"""Offline experimental-result replay; does not run a model."""
import argparse,json,html
from pathlib import Path
from latent_multi.maze import metrics,check
from latent_multi.selection import scores

def draw(row,candidates):
    palette=['#1677ff','#d64f84','#04a777','#eb9e36','#815cc7','#565c70']
    parts=['<svg viewBox="0 0 200 200" width="300" role="img" aria-label="saved candidate paths">']
    for i,v in enumerate(row['x']):
        r,c=divmod(i,10);fill={1:'#ffffff',2:'#253746',3:'#52b788',4:'#e9c46a'}[v]
        parts.append(f'<rect x="{c*20}" y="{r*20}" width="20" height="20" fill="{fill}" stroke="#ddd"/>')
    seen=set()
    for y in candidates:
        legal,short,route,L=check(row,y)
        if not legal or route in seen:continue
        seen.add(route);points=' '.join(f'{(v%10)*20+10},{(v//10)*20+10}' for v in route)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{palette[(len(seen)-1)%len(palette)]}" stroke-width="2.5" opacity=".6"/>')
    return ''.join(parts)+'</svg>'

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/demo_replay.json');p.add_argument('--out',default='outputs/replay.html');p.add_argument('--map-index',type=int,default=0)
    a=p.parse_args();data=json.loads(Path(a.data).read_text());row=data['maps'][a.map_index];output=[];cards=[]
    for name,v in data['models'].items():
        pool=v['candidates'][a.map_index];y=pool['actions'];lp=pool['chosen_logprobs'];mm=metrics(row,y,scores(row['x'],y,lp))[32]
        output.append(dict(model=name,map_id=row['id'],**{k:mm[k] for k in ('U','U_all','V_target','D_target','validity','first','H','selected')}))
        cards.append('<section><h2>'+html.escape(name)+'</h2>'+draw(row,y)+'<pre>'+html.escape(json.dumps(output[-1],indent=2))+'</pre></section>')
    title='实验结果回放 / Saved experiment replay'
    body='<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px system-ui;margin:32px;color:#203446}main{display:flex;flex-wrap:wrap;gap:18px}section{border:1px solid #ccc;padding:16px;max-width:340px}h2{font-size:18px}pre{font-size:13px}</style><h1>'+title+'</h1><p>非实时模型推理。同一地图、K32、第一评价随机流，展示全部12个固定端点。当前为固定前4张地图之一，不代表完整1200图测试。路径与指标由离线checker核验，不参与模型选答。</p><p>绿色：起点；黄色：终点；深色：墙。彩色线仅显示不同合法路径，未画出非法候选；数值仍统计全部32条候选。U为不同最短路径数，validity为全部候选合法率，selected为固定无工具路径读出是否选中最短路径。</p><main>'+''.join(cards)+'</main>'
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:f.write(body)
    print(json.dumps(dict(mode='saved-experiment-replay',map_id=row['id'],K=32,models=output),indent=2))
if __name__=='__main__':main()
