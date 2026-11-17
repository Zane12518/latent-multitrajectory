"""Small checks only for the new continuation adapter; no archived experiment reruns."""
import copy
import math
import random
import numpy as np
import pytest
import torch
from latent_multi.gru.train import content_identity, maze_quality, protocol, rng_state, set_rng


def test_content_identity_includes_input_and_references():
    rows=[dict(id=1,x=[1,2],solution_bits=[0,3])]
    for key,value in [('x',[2,1]),('solution_bits',[0,2])]:
        changed=copy.deepcopy(rows);changed[0][key]=value
        assert content_identity(rows)!=content_identity(changed)
    assert content_identity(rows)==content_identity(copy.deepcopy(rows))


def test_optimal_action_mass_and_mask_gradient():
    z=torch.zeros(32,100,4,requires_grad=True)
    targets=[[False]*4 for _ in range(100)]
    targets[0]=[True,True,False,False];targets[1]=[False,False,True,False]
    loss=maze_quality(z,[dict(optimal_actions=targets)])
    assert float(loss)==pytest.approx((math.log(2)+math.log(4))/2)
    loss.backward()
    assert torch.count_nonzero(z.grad[:,2:])==0
    assert z.grad[:,0,0].mean()<0 and z.grad[:,0,2].mean()>0


def test_protocol_rejects_unsupported_target():
    with pytest.raises(ValueError):protocol('maze','native',1e-4)
    assert protocol('sat','set',1e-4)['accumulation']==4


def test_rng_roundtrip():
    state=rng_state()
    expected=(random.random(),np.random.random(),torch.rand(4))
    set_rng(state)
    actual=(random.random(),np.random.random(),torch.rand(4))
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])
