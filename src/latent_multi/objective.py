"""Frozen train-only complete-reference assignment; no inference communication."""
import math
import random
from collections import deque
from functools import lru_cache
K=32
DIRS=((-1,0),(1,0),(0,-1),(0,1))
def subset(bank,question,visit,cap=32,seed=2026111211,rotate=True):
    # Sorting and independent local RNG make the choice independent of model, slot and scores.
    if cap<1 or cap>K or visit<0:raise ValueError('bad cap or visit')
    bank=tuple(sorted(set(bank)))
    if len(bank)<=cap:return bank
    order=list(bank);random.Random(seed*1000003+int(question)).shuffle(order)
    off=visit*cap if rotate else 0
    return tuple(sorted(order[(off+i)%len(order)] for i in range(cap)))

def auxiliary(scores,denom):
    """Both auxiliaries use identical complete-reference scores, no train gradient through the plan."""
    import torch as t
    if scores.ndim!=2 or scores.shape[0]!=K or denom<1:raise ValueError('bad score shape')
    sc=scores.double();n=sc.shape[1]
    if not n:
        zero=sc.sum()*0
        return zero,zero,sc.detach(),{'row_err':0.,'col_err':0.}
    if not t.isfinite(sc).all():raise FloatingPointError('nonfinite log score')
    multi=-sc.mean()/denom
    if n==1:
        return multi,multi,t.full_like(sc.detach(),1/K),{'row_err':0.,'col_err':0.}
    lw=sc.detach()/.5
    for _ in range(12):
        lw=lw-lw.logsumexp(1,keepdim=True)-math.log(K)
        lw=lw-lw.logsumexp(0,keepdim=True)-math.log(n)
    w=lw.exp()
    if w.requires_grad or not t.isfinite(w).all():raise FloatingPointError('allocation')
    ce=(w.sum(0)-1/n).abs().max();re=(w.sum(1)-1/K).abs().max()
    if float(ce)>1e-10 or abs(float(w.sum())-1)>1e-10:raise ArithmeticError('column/total mass')
    return multi,-(w*sc).sum()/denom,w,{'row_err':float(re),'col_err':float(ce)}

def nextcell(j,a):
    x,y=divmod(j,10);dx,dy=DIRS[a];x+=dx;y+=dy
    return 10*x+y if 0<=x<10 and 0<=y<10 else -1

@lru_cache(maxsize=8192)
def shortest_dag(cells):
    """Goal distances, then only edges on a complete shortest start-goal path."""
    start=cells.index(3);end=cells.index(4);dist={end:0};pending=deque([end])
    while pending:
        j=pending.popleft()
        for a in range(4):
            h=nextcell(j,a)
            if h>=0 and cells[h]!=2 and h not in dist:dist[h]=dist[j]+1;pending.append(h)
    if start not in dist:return (),0
    levels=[];here=(start,)
    for dep in range(dist[start],0,-1):
        edges=[[(a,nextcell(j,a)) for a in range(4) if dist.get(nextcell(j,a))==dep-1] for j in here]
        dest=tuple(sorted({h for row in edges for _,h in row}));ids={h:i for i,h in enumerate(dest)}
        mapping=[[(ids[dict(row)[a]] if a in dict(row) else -1) for a in range(4)] for row in edges]
        levels.append((here,tuple(map(tuple,mapping))));here=dest
    assert here==(end,)
    # Independent integer path count used to tell complete stored banks from truncated banks.
    counts=[1]
    for _,ix in reversed(levels):counts=[sum(counts[v] for v in ar if v>=0) for ar in ix]
    return tuple(levels),counts[0]

def maze_exact_logmass(lp,row):
    import torch as t
    lv,count=shortest_dag(tuple(row['x']))
    if not count or len(lv)!=row['optimal']:raise ValueError('invalid shortest-path target')
    prev=lp.new_zeros((lp.shape[0],1))
    for cells,mapping in reversed(lv):
        ix=t.as_tensor(mapping,device=lp.device);xx=t.as_tensor(cells,device=lp.device)
        branches=lp[:,xx,:]+prev[:,ix.clamp_min(0)]
        prev=branches.masked_fill(ix[None]<0,-float('inf')).logsumexp(2)
    return prev[:,0]

def refs(row):
    if len(row['path_indices']) != len(row['path_actions']):
        raise ValueError('unpaired paths')
    bank=tuple(sorted(set((tuple(i),tuple(a)) for i,a in zip(row['path_indices'],row['path_actions']))))
    if any(len(i)!=len(a) or len(i)!=row['optimal'] for i,a in bank):
        raise ValueError('not complete shortest paths')
    if not bank: raise ValueError('training requires at least one complete shortest reference')
    return bank

def score_refs(lp, bank):
    import torch as t
    if not bank: return lp[:,:0,0].double()
    ix=t.as_tensor([v[0] for v in bank],device=lp.device)
    ac=t.as_tensor([v[1] for v in bank],device=lp.device)
    return lp[:,ix,ac].double().sum(-1)

def terms(logits, rows, visits):
    import torch as t
    lp=logits.float().log_softmax(-1).reshape(len(rows),32,100,5)
    columns=[[],[],[]]; diagnostics=[]
    for row,a,visit in zip(rows,lp,visits,strict=True):
        aa=a[:,:,1:5]  # Null remains in the original five-class normalizer.
        bank=refs(row); sub=subset(bank,row['id'],visit)
        scores=score_refs(aa,sub)
        multi,setloss,plan,err=auxiliary(scores,row['optimal'])
        mass=-maze_exact_logmass(aa.double(),row).mean()/row['optimal']
        for col,value in zip(columns,(mass,multi,setloss)): col.append(value)
        diagnostics.append(dict(id=row['id'],visit=visit,references=len(bank),exposed=len(sub),
                                plan_mass=float(plan.sum()),**err))
    return [t.stack(col).mean() for col in columns],diagnostics
