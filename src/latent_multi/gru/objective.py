"""Train-only allocation. The native prior, rollout, model and inference remain outside this module."""
import itertools
import math
import random
from collections import deque
from functools import lru_cache

K=32
DIRS=((-1,0),(1,0),(0,-1),(0,1))

def canonical_coloring(values):
    s={};out=[]
    for c in values:
        if c not in (3,4,5):raise ValueError('invalid active graph label')
        if c not in s:s[c]=len(s)+3
        out.append(s[c])
    return tuple(out)

def refs(row,task):
    if task=='sat':data=[int(v) for v in row['solution_bits']]
    elif task=='maze':
        if len(row['path_indices'])!=len(row['path_actions']):raise ValueError('unpaired paths')
        data=[(tuple(ix),tuple(ac)) for ix,ac in zip(row['path_indices'],row['path_actions'])]
        if any(len(ix)!=len(ac) or len(ix)!=row['optimal'] for ix,ac in data):raise ValueError('not complete shortest paths')
    elif task=='graph':data=[canonical_coloring(y[:10]) for y in row['ys']]
    else:raise ValueError(task)
    return tuple(sorted(set(data)))

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

def score_refs(lp,bank,task):
    import torch as t
    if not bank:return lp[:,:0,0].double()
    if task=='sat':
        codes=t.as_tensor(bank,device=lp.device)
        ys=(codes[:,None]>>t.arange(lp.shape[1],device=lp.device))&1
    elif task=='maze':
        ix=t.as_tensor([v[0] for v in bank],device=lp.device);ac=t.as_tensor([v[1] for v in bank],device=lp.device)
        return lp[:,ix,ac].double().sum(-1)
    elif task=='graph':
        raw=[];owner=[]
        for j,x in enumerate(bank):
            orb=sorted({tuple(p[q-3] for q in x) for p in itertools.permutations((3,4,5))})
            raw+=orb;owner += [j]*len(orb)
        ys=t.as_tensor(raw,device=lp.device)
    else:raise ValueError(task)
    scores=lp[:,None].expand(-1,len(ys),-1,-1).gather(-1,ys[None,:,:,None].expand(lp.shape[0],-1,-1,1)).squeeze(-1).sum(-1)
    if task=='graph':
        owner=t.as_tensor(owner,device=lp.device)
        return t.stack([scores[:,owner==j].logsumexp(1) for j in range(len(bank))],1)
    return scores

def terms(logits,rows,task,visits,phase='main',cap=32):
    """Returns bank mass, complete mass or scoped finite mass, multi, set. Caller adds native task loss."""
    import torch as t
    n={'sat':20,'maze':100,'graph':10}[task];z=logits.reshape(len(rows),K,-1,logits.shape[-1])[:,:,:n,:]
    lp=(z.double() if task=='graph' else z.float()).log_softmax(-1);columns=[[] for _ in range(4)];diagnostics=[]
    for row,a,visit in zip(rows,lp,visits,strict=True):
        if phase=='limited' and task!='sat':raise ValueError('finite pilot is SAT only')
        bank=refs(row,task)
        if phase=='limited' and len(bank)>cap:raise ValueError('finite worker may only receive the pre-bound restricted pool')
        sub=subset(bank,row['id'],visit,cap,rotate=phase=='main');L=row['optimal'] if task=='maze' else n
        sc=score_refs(a,sub,task);multi,setloss,w,err=auxiliary(sc,L);zero=a.sum()*0
        bankmass=-sc.logsumexp(1).mean()/L if len(sub) else zero
        if phase=='limited':mass=bankmass   # never ask for full labels or a solver here
        elif task=='maze':mass=-maze_exact_logmass(a.double(),row).mean()/L
        else:
            if task=='sat' and len(bank)!=int(row['legal_N']):raise ValueError('whole-space mass given incomplete train bank')
            if task=='graph' and len(bank)!=row['complete_classes']:raise ValueError('unverified coloring orbit count')
            mass=bankmass if len(sub)==len(bank) else -score_refs(a,bank,task).logsumexp(1).mean()/L
        for dest,value in zip(columns,(bankmass,mass,multi,setloss)):dest.append(value)
        diagnostics.append(dict(input_id=int(row['id']),visit=int(visit),M=len(bank),exposed=len(sub),**err))
    return [t.stack(col).mean() for col in columns],diagnostics

def combine(parts,arm,task,native=None,aux_coef=.1,original=None):
    bank,mass,independent,assignment=parts
    if arm=='native' and task in ('sat','maze'):
        if original is None:raise ValueError('missing exact original strong native loss')
        return original
    if task=='sat':common=mass
    else:
        if native is None:raise ValueError('missing real native quality objective')
        own=native;common=native+(mass*.1 if task=='graph' else mass)
    if arm=='native':return native
    if arm=='mass':return common
    if arm=='multi':return common+aux_coef*independent
    if arm=='set':return common+aux_coef*assignment
    raise ValueError(arm)
