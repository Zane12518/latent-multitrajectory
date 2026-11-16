
"""Maze10 task semantics. Validators are not used to choose a candidate."""
import math
from collections import deque
from .objective import nextcell, shortest_dag, refs
KS=(1,4,8,16,32)

def shortest_length(cells):
    return len(shortest_dag(tuple(cells))[0])

def validate_input(row, references=False):
    x=row['x']
    if len(x)!=100 or set(x)-{1,2,3,4} or x.count(3)!=1 or x.count(4)!=1:
        raise ValueError('expected 100 cells, floor=1, wall=2, start=3, goal=4')
    L=shortest_length(x)
    if L<1 or row['optimal']!=L: raise ValueError('unreachable or inconsistent shortest length')
    if references:
        for ix,ac in refs(row):
            y=[-1]*100
            for i,a in zip(ix,ac): y[i]=a
            if not check(row,y)[1]: raise ValueError('invalid complete shortest reference')

def policy(row,visit):
    x=row['x'];goal=x.index(4);dist={goal:0};queue=deque([goal])
    while queue:
        j=queue.popleft()
        for a in range(4):
            n=nextcell(j,a)
            if n>=0 and x[n]!=2 and n not in dist:dist[n]=dist[j]+1;queue.append(n)
    y=[0]*100
    for j in dist:
        if j==goal:continue
        for a in range(4):
            if dist.get(nextcell(j,a))==dist[j]-1:y[j]=a+1;break
    bank=refs(row)
    for j,a in zip(*bank[visit%len(bank)]):y[j]=a+1
    assert check(row,[v-1 for v in y])[1]
    return y

def check(row,y):
    if len(y)!=100: raise ValueError('expected one action per cell')
    x=row['x'];loc=x.index(3);goal=x.index(4);visited=set();route=[loc]
    while loc!=goal:
        if loc in visited or x[loc]==2 or y[loc] not in range(4):return False,False,None,None
        visited.add(loc);n=nextcell(loc,int(y[loc]))
        if n<0 or x[n]==2:return False,False,None,None
        loc=n;route.append(loc)
    length=len(route)-1
    return True,length==row['optimal'],tuple(route),length

def metrics(row,actions,scores,ks=KS):
    # Selection is determined before validation. Tie: smallest candidate index.
    selected={k:max(range(k),key=lambda i:(scores[i],-i)) for k in ks}
    flags=[check(row,y) for y in actions];out={}
    for k in ks:
        fs=flags[:k];legal={f[2] for f in fs if f[0]};target={f[2] for f in fs if f[1]}
        v=sum(f[0] for f in fs);vt=sum(f[1] for f in fs)
        out[k]=dict(U=len(target),U_all=len(legal),V=v,D=v-len(legal),V_target=vt,
                    D_target=vt-len(target),validity=v/k,first=int(fs[0][1]),
                    H=int(vt>0),H_all=int(v>0),selected=int(flags[selected[k]][1]),
                    selected_index=selected[k],target_set=sorted(target),legal_set=sorted(legal))
    return out
