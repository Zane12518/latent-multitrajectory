
import importlib.util
from pathlib import Path
import torch
import pytest
from latent_multi import objective as o
from latent_multi.maze import check,policy,metrics,validate_input
from latent_multi.selection import trace_score
from latent_multi.sampling import seed_key

def toy():
    sp=importlib.util.spec_from_file_location('toy',Path(__file__).parents[1]/'scripts/make_toy.py')
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m.make()[0]

def test_validator_and_breaks():
    r=toy();validate_input(r,True);y=[v-1 for v in policy(r,0)]
    assert check(r,y)[:2]==(True,True)
    y[r['x'].index(3)]=-1;assert not check(r,y)[0]
    with pytest.raises(ValueError):validate_input(dict(r,optimal=99))

@pytest.mark.parametrize('M',[0,1,3,31,32,37])
def test_sinkhorn(M):
    x=torch.randn(32,M,dtype=torch.float64,requires_grad=True)
    u,s,w,err=o.auxiliary(x,4)
    assert not w.requires_grad and torch.isfinite(s)
    gu=torch.autograd.grad(u,x,retain_graph=True)[0];gs=torch.autograd.grad(s,x)[0]
    if M:assert abs(float(w.sum())-1)<1e-10 and err['col_err']<1e-10
    if M<=1:assert torch.equal(gu,gs) and torch.equal(u,s)

def test_assignment_permutations():
    torch.manual_seed(3);x=torch.randn(32,7,dtype=torch.float64);p=torch.randperm(32);q=torch.randperm(7)
    a=o.auxiliary(x,4);b=o.auxiliary(x[p][:,q],4)
    assert torch.allclose(a[1],b[1],atol=1e-12)
    assert torch.allclose(a[2][p][:,q],b[2],atol=1e-12)

def test_nonfinite_and_bad_shape():
    with pytest.raises(FloatingPointError):o.auxiliary(torch.full((32,2),float('nan')),4)
    with pytest.raises(ValueError):o.auxiliary(torch.zeros(8,2),4)

def test_dag_equals_complete_path_sum():
    r=toy();lp=torch.randn(32,100,5,dtype=torch.float64).log_softmax(-1)[:,:,1:]
    a=o.maze_exact_logmass(lp,r);b=o.score_refs(lp,o.refs(r)).logsumexp(1)
    assert torch.allclose(a,b,atol=1e-12)
    assert o.shortest_dag(tuple(r['x']))[1]==6

def test_identity_and_counting():
    r=toy();a=[v-1 for v in policy(r,0)];b=[v-1 for v in policy(r,1)]
    z=metrics(r,[a]*16+[b]*16,[0]*32)[32]
    assert z['U']==2 and z['V_target']-z['D_target']==z['U'] and z['selected_index']==0

def test_selector_does_not_filter_walls_or_null():
    r=toy();a=[-1]*100;lp=[-2.]*100
    assert trace_score(r['x'],a,lp)==(-2.,1)
    a[r['x'].index(3)]=0
    assert trace_score(r['x'],a,lp)[1]==2  # It visits a wall; no checker/filter call.

def test_unique_logical_keys():
    keys={seed_key(r,i,n) for r in range(3) for i in range(1200) for n in range(32)}
    assert len(keys)==115200

def test_subset_is_model_independent():
    bank=tuple(range(71));a=o.subset(bank,9,0);b=o.subset(tuple(reversed(bank)),9,0)
    assert a==b and len(a)==32 and o.subset(bank,9,1)!=a
