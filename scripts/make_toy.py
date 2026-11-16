
"""Small synthetic fixtures, not the historical benchmark population."""
import argparse,json
from pathlib import Path
def make():
    rows=[]
    for q,(top,left) in enumerate(((1,1),(3,4),(5,2),(6,6))):
        x=[2]*100;corners=[(top+r)*10+left+c for r in range(3) for c in range(3)]
        for i in corners:x[i]=1
        start=top*10+left;goal=(top+2)*10+left+2;x[start]=3;x[goal]=4
        paths=[]
        def walk(v,ix,ac):
            if v==goal:paths.append((ix,ac));return
            r,c=divmod(v,10)
            if r<top+2:walk(v+10,ix+[v],ac+[1])
            if c<left+2:walk(v+1,ix+[v],ac+[3])
        walk(start,[],[])
        rows.append(dict(id=q,x=x,optimal=4,path_indices=[p[0] for p in paths],path_actions=[p[1] for p in paths]))
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    with Path(a.out).open('x') as f:json.dump(make(),f,indent=2)
    print('4 synthetic maps; 6 shortest paths per map; no benchmark claim')
