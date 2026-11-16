
import copy,importlib.util,json
from pathlib import Path
import pytest
import torch
from latent_multi.runtime import Session,save_new

def eq(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b

@pytest.fixture(scope='module')
def trained():
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('toy',root/'scripts/make_toy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg=json.loads((root/'configs/cpu_smoke.json').read_text());s=Session(cfg,m.make())
    assert sum(p.numel() for p in s.m.parameters())==546882
    r=s.update('multi');assert r['update_norm']>0
    return cfg,m.make(),s.state()

def test_cpu_update_save_restore(tmp_path,trained):
    cfg,rows,mid=trained;s=Session(cfg,rows);s.restore(mid)
    p=tmp_path/'resume.pt';save_new(p,mid)
    expected=s.update('multi');end=s.state()
    s.restore(torch.load(p,weights_only=True));actual=s.update('multi')
    assert eq(expected,actual) and eq(end,s.state())

@pytest.mark.parametrize('field',['x','refs'])
def test_same_ids_changed_content_rejected(trained,field):
    cfg,rows,mid=trained;rows=copy.deepcopy(rows)
    if field=='x':rows[-1]['x'][0]=(rows[-1]['x'][0]+1)%5
    else:
        rows[-1]['path_indices']=rows[-1]['path_indices'][:1]
        rows[-1]['path_actions']=rows[-1]['path_actions'][:1]
    with pytest.raises(ValueError,match='content identity'):Session(cfg,rows).restore(mid)

@pytest.mark.parametrize('field,value',[('aux_coef',.2),('microbatches',2),('schedule','post-linear'),('updates',3)])
def test_configuration_change_rejected(trained,field,value):
    cfg,rows,mid=trained;cfg=copy.deepcopy(cfg);cfg['training'][field]=value
    with pytest.raises(ValueError,match='configuration mismatch'):Session(cfg,rows).restore(mid)

@pytest.mark.parametrize('field',['optimizer','rng','prior_rng','carry','release_config','train_ids','identity','order_rng','stream'])
def test_missing_state_rejected(trained,field):
    cfg,rows,mid=trained;c=copy.deepcopy(mid);del c[field]
    with pytest.raises(ValueError,match='missing'):Session(cfg,rows).restore(c)

def test_missing_adam_slot_rejected(trained):
    cfg,rows,mid=trained;c=copy.deepcopy(mid);del c['optimizer']['state'][next(iter(c['optimizer']['state']))]
    with pytest.raises(ValueError,match='optimizer slot'):Session(cfg,rows).restore(c)

def test_explicit_arm_and_rng_fork(trained):
    cfg,rows,mid=trained;s=Session(cfg,rows)
    with pytest.raises(ValueError,match='fork'):s.restore(mid,arm='set')
    s.restore(mid,mode='fork',arm='set');s.new_stream(91,fork=True)
    assert s.lineage[-2]['from_arm']=='multi' and s.lineage[-2]['to_arm']=='set'
    assert s.update('set')['update_norm']>0
    with pytest.raises(ValueError,match='fork'):s.update('multi')
    with pytest.raises(ValueError,match='fork'):s.new_stream(92)

def test_explicit_new_experiment(trained):
    cfg,rows,mid=trained;cfg=copy.deepcopy(cfg);cfg['training']['aux_coef']=.05
    s=Session(cfg,rows)
    with pytest.raises(ValueError,match='exactly'):s.restore(mid,mode='new-experiment')
    s.restore(mid,mode='new-experiment',allow_changes=['training.aux_coef'])
    assert s.lineage[-1]['config_changes']['training.aux_coef']==dict(before=.1,after=.05)

def test_legacy_is_explicit_and_stays_marked(trained):
    cfg,rows,mid=trained;c=copy.deepcopy(mid);c['format']='latent-multitrajectory-resume-v1'
    for key in ('identity','lineage','arm'):del c[key]
    s=Session(cfg,rows)
    with pytest.raises(ValueError,match='explicit legacy'):s.restore(c)
    with pytest.warns(RuntimeWarning,match='not verified exact'):s.restore(c,mode='legacy',arm='multi')
    saved=s.state();s.restore(saved)
    assert s.lineage[-1]['legacy_identity_unverified'] is True
