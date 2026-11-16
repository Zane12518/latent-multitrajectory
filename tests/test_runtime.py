
import importlib.util,json
from pathlib import Path
import torch
from latent_multi.runtime import Session,save_new

def eq(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b

def test_cpu_update_save_restore(tmp_path):
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('toy',root/'scripts/make_toy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg=json.loads((root/'configs/cpu_smoke.json').read_text());s=Session(cfg,m.make())
    assert sum(p.numel() for p in s.m.parameters())==546882
    r=s.update('multi');assert r['update_norm']>0
    mid=s.state();p=tmp_path/'resume.pt';save_new(p,mid)
    expected=s.update('set');end=s.state()
    s.restore(torch.load(p,weights_only=True));actual=s.update('set')
    assert eq(expected,actual) and eq(end,s.state())
