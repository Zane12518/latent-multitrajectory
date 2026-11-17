"""GRU task hosts extracted from archived implementation; no platform dependencies."""
import copy
import torch

def build_sat(n=20,clause_max=91,device="cpu"):
    t=torch;L=3*clause_max+n
    class Solver(t.nn.Module):
        def __init__(self):
            super().__init__();self.emb=t.nn.Embedding(2*n+4,16);self.condition=t.nn.Sequential(t.nn.Linear(L*16,512),t.nn.GELU(),t.nn.Linear(512,n*128));self.pos=t.nn.Parameter(t.randn(1,n,128)*.02);self.attn=t.nn.MultiheadAttention(128,4,batch_first=True,dropout=0);self.cell=t.nn.GRUCell(388,128);self.norm=t.nn.LayerNorm(128);self.head=t.nn.Linear(128,2)
        def forward(self,x,depth,training=False):
            ex=self.condition(self.emb(x).flatten(1)).reshape(-1,n,128)+self.pos;eps=t.randn_like(ex);state=.25*eps;ex=ex+.25*eps
            lit=x[:,:3*clause_max].reshape(-1,clause_max,3);sg=t.stack([(lit==i+1).float().sum(-1)-(lit==i+n+1).float().sum(-1) for i in range(n)],-1)
            for _ in range(depth):
                with t.set_grad_enabled(training):
                    bit=self.head(state).float().softmax(-1)[:,:,1];positive=sg>0;negative=sg<0;literal=t.where(positive,bit[:,None,:],1-bit[:,None,:]);fail=t.where(sg!=0,1-literal,t.ones_like(literal)).clamp_min(1e-5);others=(fail.log().sum(-1,keepdim=True)-fail.log()).exp();force=(sg.sign()*others).sum(1)/clause_max;feat=t.stack([force,positive.float().mean(1),negative.float().mean(1),bit],-1);message=self.attn(state,state,state,need_weights=False)[0];state=self.norm(self.cell(t.cat([ex,message,state,feat],-1).reshape(-1,388),state.reshape(-1,128)).reshape_as(state))+.05*t.randn_like(state)
            for _ in range(16-depth):t.randn_like(state)
            return self.head(state)
    return Solver().to(device)

def build_maze(device="cpu"):
    v={"task":"maze","arch":"shared"}
    t=torch
    if v['task']=='nq10':
        dim=256
        class Queens(t.nn.Module):
            def __init__(self):
                super().__init__();self.emb=t.nn.Embedding(3,32);self.condition=t.nn.Sequential(t.nn.Linear(3200,1024),t.nn.GELU(),t.nn.Linear(1024,10*dim));self.pos=t.nn.Parameter(t.randn(1,10,dim)*.02);self.attn=t.nn.MultiheadAttention(dim,8,batch_first=True,dropout=0);self.cell=t.nn.GRUCell(dim*3+4,dim);self.norm=t.nn.LayerNorm(dim);self.head=t.nn.Linear(dim,10)
            def forward(self,x,depth,training=False):
                ex=self.condition(self.emb(x).flatten(1)).reshape(-1,10,dim)+self.pos;eps=t.randn_like(ex);state=.25*eps;ex=ex+.25*eps;vis=(x.reshape(-1,10,10)==2).float()
                for _ in range(depth):
                    with t.set_grad_enabled(training):
                        pr=self.head(state).float().softmax(-1);column=pr.sum(1,keepdim=True)-pr;diag=t.zeros_like(pr)
                        for delta in range(1,10):
                            diag[:,delta:,delta:]+=pr[:,:-delta,:-delta];diag[:,:-delta,:-delta]+=pr[:,delta:,delta:];diag[:,delta:,:-delta]+=pr[:,:-delta,delta:];diag[:,:-delta,delta:]+=pr[:,delta:,:-delta]
                        clue=vis.sum(-1);features=t.stack([(pr*(column+diag)).sum(-1)/10,clue,clue-(pr*vis).sum(-1),pr.max(-1).values],-1);message=self.attn(state,state,state,need_weights=False)[0];state=self.norm(self.cell(t.cat([ex,message,state,features],-1).reshape(-1,dim*3+4),state.reshape(-1,dim)).reshape_as(state))+.05*t.randn_like(state)
                for _ in range(16-depth):t.randn_like(state)
                return self.head(state)
        return Queens().to(device)
    class Block(t.nn.Module):
        def __init__(self):
            super().__init__();self.message=t.nn.Conv2d(64,64,3,padding=1);self.cell=t.nn.GRUCell(128,64);self.norm=t.nn.LayerNorm(64)
        def forward(self,state,ex):
            message=self.message(state).tanh();x=t.cat([ex,message],1).permute(0,2,3,1).reshape(-1,128);s=state.permute(0,2,3,1).reshape(-1,64);out=self.norm(self.cell(x,s));return out.reshape(-1,10,10,64).permute(0,3,1,2)
    class Planner(t.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=t.nn.Sequential(t.nn.Conv2d(6,64,3,padding=1),t.nn.GELU(),t.nn.Conv2d(64,64,3,padding=1));self.pos=t.nn.Parameter(t.randn(1,64,10,10)*.02);self.blocks=t.nn.ModuleList([Block()]);self.head=t.nn.Conv2d(64,4,1)
            if v['arch']=='untied':self.blocks.extend(copy.deepcopy(self.blocks[0]) for _ in range(15))
        def forward(self,x,depth,training=False):
            grid=x.reshape(-1,10,10);coords=t.linspace(-1,1,10,device=x.device);channels=t.stack([(grid==i).float() for i in [1,2,3,4]]+[coords[None,:,None].expand(len(x),10,10),coords[None,None,:].expand(len(x),10,10)],1);ex=self.encoder(channels)+self.pos;eps=t.randn_like(ex);state=.15*eps;ex=ex+.15*eps
            for j in range(depth):
                with t.set_grad_enabled(training):state=self.blocks[j if v['arch']=='untied' else 0](state,ex)+.03*t.randn_like(state)
            for _ in range(16-depth):t.randn_like(state)
            return self.head(state).flatten(2).transpose(1,2)
    return Planner().to(device)

def encode(row,n,clause_max):
    literals=[v if v>0 else n-v for c in row['clauses'] for v in c]
    return literals+[0]*(3*clause_max-len(literals))+[2*n+1]*n

def path(row,actions):
    if len(actions)!=100 or any(int(x) not in range(4) for x in actions):return False,None,None,row['optimal']
    start=row['x'].index(3);goal=row['x'].index(4);at=start;visited={at};result=[at]
    for _ in range(100):
        if at==goal:return True,tuple(result),len(result)-1,row['optimal']
        rr,cc=divmod(at,10);dr,dc=[(-1,0),(1,0),(0,-1),(0,1)][int(actions[at])];rr+=dr;cc+=dc
        if not(0<=rr<10 and 0<=cc<10):return False,None,None,row['optimal']
        at=rr*10+cc
        if row['x'][at]==2 or at in visited:return False,None,None,row['optimal']
        visited.add(at);result.append(at)
    return False,None,None,row['optimal']
