
"""Fixed no-fit readouts. No validator, optimal length or reference access."""
import math
from .objective import nextcell

def trace_score(cells,actions,chosen_logp):
    loc=cells.index(3);goal=cells.index(4);seen=set();values=[]
    for _ in range(100):
        if loc==goal or loc in seen:break
        seen.add(loc);values.append(float(chosen_logp[loc]));a=int(actions[loc])
        if a not in range(4):break
        loc=nextcell(loc,a)
        if loc<0:break
    if not values or not all(math.isfinite(v) for v in values):raise ValueError('empty or nonfinite path confidence')
    return sum(values)/len(values),len(values)

def scores(cells,actions,chosen_logp,rule='path'):
    if rule=='path':return [trace_score(cells,y,p)[0] for y,p in zip(actions,chosen_logp)]
    if rule=='grid':
        mask=[i for i,v in enumerate(cells) if v not in (2,4)]
        return [sum(float(p[i]) for i in mask)/len(mask) for p in chosen_logp]
    raise ValueError(rule)
