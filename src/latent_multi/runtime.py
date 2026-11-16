
"""GRAM Maze task adapter: native posterior ACT + target-free prior episodes."""
import copy
import dataclasses
import random
from pathlib import Path
import numpy as np
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch as t
from .vendor.gram.recursive_reasoning.gram import GenerativeRecursiveReasoningModel
from .vendor.gram.losses import ACTLossHead
from . import maze, objective

def configure():
    t.set_num_threads(4);t.use_deterministic_algorithms(True)
    t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
    t.backends.cuda.enable_flash_sdp(False);t.backends.cuda.enable_mem_efficient_sdp(False)
    t.backends.cuda.enable_math_sdp(True)

def seed(n):
    random.seed(n);np.random.seed(n%2**32);t.manual_seed(n)
    if t.cuda.is_available():t.cuda.manual_seed_all(n)

def plain(x):
    if t.is_tensor(x):return x.detach().cpu().clone()
    if dataclasses.is_dataclass(x):return {f.name:plain(getattr(x,f.name)) for f in dataclasses.fields(x)}
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(plain(v) for v in x)
    return copy.deepcopy(x)

def move(x,device):
    if t.is_tensor(x):return x.to(device)
    if dataclasses.is_dataclass(x):return type(x)(**{f.name:move(getattr(x,f.name),device) for f in dataclasses.fields(x)})
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    return x

def carry_restore(template,saved,device):
    if dataclasses.is_dataclass(template):
        return type(template)(**{f.name:carry_restore(getattr(template,f.name),saved[f.name],device) for f in dataclasses.fields(template)})
    return move(saved,device)

def save_new(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('xb') as f:t.save(value,f);f.flush();__import__('os').fsync(f.fileno())
    # Exclusive link publishes a complete artifact without replacing an old file.
    __import__('os').link(tmp,path);tmp.unlink()

class Session:
    def __init__(self,config,train,device='cpu'):
        configure();self.config=copy.deepcopy(config);self.device=t.device(device)
        self.train=train;self.seed=config['training']['initial_seed'];seed(self.seed)
        with t.device(self.device):self.m=GenerativeRecursiveReasoningModel(config['model'])
        self.opt=t.optim.AdamW(self.m.parameters(),lr=.0003,weight_decay=.01)
        self.head=ACTLossHead(self.m,'stablemax_cross_entropy',kl_coef=.1,kl_balance_alpha=.8,free_bits=None,value_coef=1)
        self.step=0;self.rr=random.Random(self.seed+100);self.order=[];self.cur=0;self.visits={}
        self.carry=None;self.actual_ids=[];self.calls=dict(native_outer_lanes=0,prior_outer_lanes=0)
        self.prior_rng=t.Generator(device=self.device);self.prior_rng.manual_seed(self.seed+100000)
        self.stream=None;self.m.train()

    def batch(self,items,visits=None,K=1,labels=False):
        b=dict(inputs=t.tensor([r['x'] for r in items],device=self.device,dtype=t.int32).repeat_interleave(K,0),
               puzzle_identifiers=t.zeros(len(items)*K,device=self.device,dtype=t.int32),
               ids=t.tensor([r['id'] for r in items],device=self.device).repeat_interleave(K))
        if labels:b['labels']=t.tensor([maze.policy(r,v) for r,v in zip(items,visits)],device=self.device).repeat_interleave(K,0)
        return b

    def episode(self,batch,training=False):
        if 'labels' in batch:raise ValueError('prior generation cannot receive targets')
        was=self.m.training;self.m.eval();B=len(batch['inputs']);inner=self.m.inner
        c=move(inner.empty_carry(B),self.device);c=inner.reset_carry(t.ones(B,device=self.device,dtype=t.bool),c)
        for j in range(16):
            with t.set_grad_enabled(training and j==15):c,z,_,_,_=inner(c,batch)
        self.m.train(was);self.calls['prior_outer_lanes']+=B*16
        return z

    def isolated_episode(self,batch,training):
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with t.random.fork_rng(devices=devices):
            if devices:t.cuda.set_rng_state(self.prior_rng.get_state(),devices[0])
            else:t.set_rng_state(self.prior_rng.get_state())
            z=self.episode(batch,training)
            self.prior_rng.set_state(t.cuda.get_rng_state(devices[0]) if devices else t.get_rng_state())
        return z

    def state(self):
        ns=np.random.get_state()
        return dict(format='latent-multitrajectory-resume-v1',numpy_state_as_list=True,
            model=plain(self.m.state_dict()),optimizer=plain(self.opt.state_dict()),
            rng=dict(py=random.getstate(),np=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),cpu=t.get_rng_state(),
                     cuda=t.cuda.get_rng_state_all() if self.device.type=='cuda' else []),
            prior_rng=self.prior_rng.get_state(),step=self.step,order_rng=self.rr.getstate(),order=self.order[:],
            cursor=self.cur,reference_visits=self.visits.copy(),carry=plain(self.carry),actual_ids=self.actual_ids[:],
            calls=self.calls.copy(),host='gram',deterministic=False,model_training=self.m.training,
            scheduler=None,scaler=None,initial_seed=self.seed,quality_mode='canonical',replay_index=0,stream=self.stream,
            release_config=self.config,train_ids=[r['id'] for r in self.train],device_type=self.device.type)

    def restore(self,c):
        if c.get('format')!='latent-multitrajectory-resume-v1':raise ValueError('not a full training checkpoint')
        if c.get('device_type', 'cuda')!=self.device.type:raise ValueError('exact resume requires matching CPU/CUDA RNG backend; use weights-only for inference')
        if c.get('train_ids',[r['id'] for r in self.train])!=[r['id'] for r in self.train]:raise ValueError('training population/order mismatch')
        self.m.load_state_dict(c['model']);self.opt.load_state_dict(c['optimizer'])
        r=c['rng'];random.setstate(r['py']);ns=r['np'];np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]));t.set_rng_state(r['cpu'])
        if self.device.type=='cuda':t.cuda.set_rng_state_all(r['cuda'])
        self.prior_rng.set_state(c['prior_rng']);self.step=c['step'];self.rr.setstate(c['order_rng'])
        self.order=c['order'][:];self.cur=c['cursor'];self.visits=c['reference_visits'].copy();self.actual_ids=c['actual_ids'][:]
        self.calls=c['calls'].copy();self.m.train(c['model_training']);self.seed=c['initial_seed'];self.stream=c.get('stream')
        self.carry=carry_restore(self.m.initial_carry(self.batch(self.train[:2],[0,0],labels=True)),c['carry'],self.device) if c['carry'] is not None else None

    def new_stream(self,n):
        seed(n);self.prior_rng.manual_seed(n+100000);self.rr.seed(n+100);self.stream=n

    def update(self,arm):
        if arm not in ('mass','multi','set'):raise ValueError(arm)
        self.opt.zero_grad(set_to_none=True);start=self.step+1;train=self.config['training']
        if train['schedule']=='prefix':lr=.0003*min(1,start/200)*(1-(2/3)*max(0,min(1,(start-200)/2800)))
        else:lr=.0001*(1-.75*min(1,max(0,(start-9001)/999)))
        for group in self.opt.param_groups:group['lr']=lr
        ids=[];records=[];parts_log=[];act=[];admitted=0
        for micro in range(train['microbatches']):
            items=[];vis=[]
            for _ in range(2):
                if self.cur>=len(self.order):self.order=list(range(len(self.train)));self.rr.shuffle(self.order);self.cur=0
                r=self.train[self.order[self.cur]];self.cur+=1;v=self.visits.get(r['id'],0)
                self.visits[r['id']]=v+1;items.append(r);vis.append(v);ids.append(r['id'])
            b=self.batch(items,vis,labels=True)
            if self.carry is None:self.carry=move(self.m.initial_carry(b),self.device)
            before=plain(self.carry.halted);admitted+=int(self.carry.halted.sum())
            self.carry,native,_,_,_=self.head(return_keys=[],carry=self.carry,batch=b);native=native/2
            actual=self.carry.current_data['ids'].tolist();self.actual_ids+=actual;self.calls['native_outer_lanes']+=2
            act.append(dict(before_halted=before.tolist(),after_halted=self.carry.halted.tolist(),steps=self.carry.steps.tolist(),native_ids=actual))
            z=self.isolated_episode(self.batch(items,K=32),True);parts,info=objective.terms(z,items,vis)
            common=native+parts[0];loss=common+train['aux_coef']*(parts[1] if arm=='multi' else parts[2] if arm=='set' else 0)
            if not t.isfinite(loss):raise FloatingPointError('loss')
            (loss/train['microbatches']).backward();records+=info
            parts_log.append(dict(total=float(loss.detach()),native=float(native.detach()),mass=float(parts[0].detach()),multi=float(parts[1].detach()),set=float(parts[2].detach())))
        norm=float(t.nn.utils.clip_grad_norm_(self.m.parameters(),1,error_if_nonfinite=True))
        if not 0<norm<100000:raise FloatingPointError('gradient gate')
        before=[p.detach().clone() for p in self.m.parameters()];self.opt.step();self.step+=1
        delta=sum(float((p.detach()-v).double().square().sum()) for p,v in zip(self.m.parameters(),before))**.5
        if not all(t.isfinite(p).all() for p in self.m.parameters()):raise FloatingPointError('weights')
        return dict(step=self.step,offered_ids=ids,native_admissions=admitted,ACT=act,lr=lr,grad_preclip=norm,
                    update_norm=delta,parts=parts_log,references=records,calls=self.calls.copy())
