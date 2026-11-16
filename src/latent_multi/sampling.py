
"""Auditable map/node keys. Key uniqueness is not a claim of statistical independence."""
import types
import torch as t

def seed_key(repeat,ordinal,node):
    if not (0<=repeat<4 and 0<=ordinal<65536 and 0<=node<64):raise ValueError('key fields out of range')
    return (((1*16+1)*4+repeat)*65536+ordinal)*64+node

def predict(session,items,ordinals,repeat=0):
    if len(items)!=len(ordinals) or len(set(ordinals))!=len(ordinals):raise ValueError('map binding')
    layer=session.m.inner.variational;old=layer.reparameterize;node=[0]
    def sample(self,mu,logvar):
        std=t.exp(.5*logvar);chunks=[]
        if std.shape[0]!=len(items)*32*100 or node[0]>=32:raise ValueError('random-node shape')
        for ordinal in ordinals:
            g=t.Generator(device=std.device);g.manual_seed(seed_key(repeat,ordinal,node[0]))
            chunks.append(t.randn((32*100,std.shape[-1]),dtype=std.dtype,device=std.device,generator=g))
        node[0]+=1
        return mu+t.cat(chunks)*std
    layer.reparameterize=types.MethodType(sample,layer)
    try:
        with t.no_grad():z=session.episode(session.batch(items,K=32),False)
    finally:layer.reparameterize=old
    if node[0]!=32:raise AssertionError('expected 2 random nodes per outer step')
    z=z.float().reshape(len(items),32,100,5)
    if not t.isfinite(z).all():raise FloatingPointError('logits')
    actions=z.argmax(-1)-1
    logp=z.log_softmax(-1).gather(-1,(actions+1).unsqueeze(-1)).squeeze(-1)
    return actions.cpu(),logp.cpu(),z.cpu()
