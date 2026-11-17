"""Evaluate already-decoded candidates. No model, test labels, execution, reranking or repair in selection."""
import math
from .objective import K,canonical_coloring,nextcell

def identities(task,row,pred):
    if task=='sat':
        if len(pred)!=20 or any(v not in (0,1) for v in pred):return None,None
        if not all(any(bool(pred[abs(v)-1])==(v>0) for v in c) for c in row['clauses']):return None,None
        h=sum(int(v)<<j for j,v in enumerate(pred));return h,h
    if task=='graph':
        if len(pred)!=55:raise ValueError('unbound graph padding layout')
        y=pred[:10]
        if any(v not in (3,4,5) for v in y):return None,None
        if len(row['x'])!=55:raise ValueError('unbound graph input')
        pos=0
        for i in range(10):
            for j in range(i,10):
                if i!=j and row['x'][pos]==2 and y[i]==y[j]:return None,None
                pos+=1
        ident=canonical_coloring(y);return ident,ident
    if task!='maze':raise ValueError(task)
    cells=row['x']
    if len(cells)!=100 or cells.count(3)!=1 or cells.count(4)!=1:raise ValueError('unbound maze input')
    if len(pred)!=100 or any(v not in range(4) for v in pred):return None,None
    p=cells.index(3);g=cells.index(4);seen={p};path=[p]
    while p!=g:
        p=nextcell(p,int(pred[p]))
        if p<0 or p in seen or cells[p]==2:return None,None
        seen.add(p);path.append(p)
    h=tuple(path);return h,h if len(h)-1==row['optimal'] else None

def curve(task,row,ys,selector_scores):
    # selector_scores: frozen mean-token confidence / masked Maze confidence / official Graph q-halt only.
    if len(ys)!=K or len(selector_scores)!=K or not all(math.isfinite(float(s)) for s in selector_scores):
        raise ValueError('need 32 ordered candidates and true saved selector scores; no fallback')
    ids=[identities(task,row,p) for p in ys];by_k={}
    for k in (1,2,4,8,16,32):
        # Pick solely by score/index. The checker is applied for measurement, never to filter this selection.
        best=max(range(k),key=lambda i:(float(selector_scores[i]),-i));legal=[i[0] for i in ids[:k] if i[0] is not None];target=[i[1] for i in ids[:k] if i[1] is not None]
        v=len(legal);u=len(set(legal));vt=len(target);ut=len(set(target))
        by_k[k]=dict(K=k,U_all=u,U=ut,V=v,D=v-u,V_target=vt,D_target=vt-ut,validity=v/k,first_valid=int(ids[0][0] is not None),first=int(ids[0][1] is not None),atleastone=int(v>0),atleastone_target=int(vt>0),selected=int(ids[best][1] is not None),selected_valid=int(ids[best][0] is not None),selected_slot=best,D_per_K=(v-u)/k,D_per_V=(v-u)/v if v else None,legal_set=frozenset(legal),target_set=frozenset(target))
    return by_k

def churn(cur,ref):
    a,b=set(cur),set(ref);added=a-b;lost=b-a;retained=a&b
    assert len(added)-len(lost)==len(a)-len(b)
    return dict(added=len(added),lost=len(lost),retained=len(retained))

def check_metric_identities():
    c={'clauses':[[1]]};a=[1]+[0]*19;b=[1,1]+[0]*18;bad=[0]*20
    t=curve('sat',c,[bad,a,b,a]*8,[3,4,1,4]*8);h=t[32]
    assert (h['U'],h['V'],h['D'],h['first'],h['selected'],h['selected_slot'])==(2,24,22,0,1,1) and t[1]['D_per_V'] is None
    shifted=[bad]*32;shifted[31]=b
    poor=curve('sat',c,shifted,[1]+[0]*31)[32];assert poor['atleastone'] and not poor['selected'] and churn(h['target_set'],poor['target_set'])==dict(added=1,lost=0,retained=1)
    grid=[2]*100;grid[0]=3;grid[1]=grid[10]=grid[11]=grid[12]=1;grid[2]=4;row=dict(x=grid,optimal=2)
    direct=[3]*100;long=direct.copy();long[0]=1;long[12]=0;cycle=long.copy();cycle[10]=0
    z=curve('maze',row,[direct,long,cycle,cycle]*8,[1,4,1,2]*8)[32]
    assert (z['U_all'],z['U'],z['V'],z['D'],z['V_target'],z['selected_valid'],z['selected'])==(2,1,16,14,8,1,0)
    edges=[2 if (i,j)==(0,1) else 0 if i==j else 1 for i in range(10) for j in range(i,10)];y=(3,4,5,3,4,5,3,4,5,3)
    gr=dict(x=edges);p=identities('graph',gr,y+(9,)*45);q=identities('graph',gr,tuple({3:4,4:5,5:3}[i] for i in y)+(-100,)*45)
    assert p==q and p[0] is not None and identities('graph',gr,(3,)*55)==(None,None)
    return dict(status='PASS',scope='synthetic saved-candidate metric identities, not model/GPU scores',new_model_calls=0,tests=['SAT zero-valued bit mask and V=U+D','actual selection vs pool success','Maze shortest vs longer legal vs cycle','Graph color orbit/padding','matched K-prefix and churn'])
