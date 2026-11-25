"""Separate projector and local-atlas output types; no global frame supervision."""
import torch
from torch import nn
from ..factory import make_embedding,make_attention


def torch_projector(z):
    z=z/torch.linalg.vector_norm(z,dim=-1,keepdim=True).clamp_min(1e-8)
    return z[:,:,:,None]*z[:,:,None,:].conj()


class QWZNetwork(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c;self.atlas=c['wavefunction']['type']=='local_atlas'
        m=dict(c['model']);self.charged=c['attention']['type']=='gauge_covariant'
        self.complex_stream=c['attention']['type'] in ['complex','gauge_covariant'];m['complex_stream']=self.complex_stream
        self.embedding=make_embedding(c['embedding'],'qwz_topology',c['physics'],m)
        factory=make_attention(c['attention'],'qwz_topology')
        self.attentions=nn.ModuleList([factory(m['width'],m['heads']) for _ in range(m['depth'])])
        dtype=torch.complex64 if self.complex_stream else torch.float32
        self.ff=nn.ModuleList([nn.Linear(m['width'],m['width'],bias=not self.charged,dtype=dtype) for _ in range(m['depth'])])
        # Charged outputs are sections relative to the INPUT bundle; topology is supplied in this design.
        self.readout=nn.Linear(m['width'],2 if self.complex_stream else (4 if self.atlas else 3),bias=not self.charged,dtype=dtype)

    def forward(self,batch,intervention=None,mask=None):
        spec=intervention or {};z=batch['z']
        if spec.get('type')=='zero_phase_channel' and self.complex_stream:z=z.real.to(z.dtype)
        x=self.embedding(z,batch['coords'],batch['mass'],batch['valid'],spec)
        if mask is not None:x=x*mask[...,None]
        records=[]
        for li,(attention,ff) in enumerate(zip(self.attentions,self.ff)):
            current=spec if spec.get('layer',0)==li else {}
            update,aux=attention(x,z,batch['valid'],batch['distance'],batch['chart'],current)
            x=x+update
            y=ff(x)
            x=x+(y*torch.tanh(abs(y))/abs(y).clamp_min(1e-7) if self.complex_stream else torch.tanh(y))
            aux['features']=x;records.append(aux)
        output=self.readout(x)
        if self.complex_stream:frame=output
        elif self.atlas:frame=output[...,:2]+1j*output[...,2:]
        else:
            vector=output/torch.linalg.vector_norm(output,dim=-1,keepdim=True).clamp_min(1e-8)
            p=torch.empty((*vector.shape[:-1],2,2),dtype=torch.complex64)
            p[...,0,0]=(1+vector[...,2])/2;p[...,1,1]=(1-vector[...,2])/2
            p[...,0,1]=(vector[...,0]-1j*vector[...,1])/2;p[...,1,0]=p[...,0,1].conj()
            return dict(projector=p,raw_norm=torch.linalg.vector_norm(output,dim=-1),records=records,features=x,raw_output=output)
        norm=torch.linalg.vector_norm(frame,dim=-1)
        frame=frame/norm.clamp_min(1e-8)[...,None]
        p=torch_projector(frame)
        if self.atlas:
            b,nodes=p.shape[:2];n=nodes//2
            p=(p.reshape(b,2,n,2,2)*batch['cover'].transpose(1,2)[...,None,None]).sum(1)
            # A convex atlas mixture is Hermitian; select its top eigenline to obtain rank1 idempotent output.
            value,vec=torch.linalg.eigh(p);p=torch_projector(vec[...,-1]);mix_gap=value[...,-1]-value[...,-2]
            active=batch['cover'].transpose(1,2)>1e-6
            active_norm=torch.where(active,norm.reshape(b,2,n),torch.full_like(norm.reshape(b,2,n),1e9)).min(1).values
            norm=torch.minimum(active_norm,mix_gap)
        return dict(projector=p,frame=frame,raw_norm=norm,records=records,features=x,raw_output=output)

    def loss(self,result,batch):
        if not self.atlas:
            loss=(abs(result['projector']-batch['target_projector'])**2).mean()
        else:
            valid=batch['valid'];error=(abs(result['frame']-batch['z'])**2).sum(-1)
            loss=(error*valid).sum()/valid.sum().clamp_min(1)
            b,nodes=valid.shape;n=nodes//2;frames=result['frame'].reshape(b,2,n,2)
            predicted=(frames[:,0].conj()*frames[:,1]).sum(-1)
            target=(batch['z'].reshape(b,2,n,2)[:,0].conj()*batch['z'].reshape(b,2,n,2)[:,1]).sum(-1)
            overlap=valid.reshape(b,2,n).all(1)
            loss=loss+.3*((abs(predicted-target)**2)*overlap).sum()/overlap.sum().clamp_min(1)
        regularization=self.config['attention']['window']['regularization']
        if regularization:loss=loss+regularization*torch.stack([a.widths().mean()/self.config['attention']['window']['upper'] for a in self.attentions]).mean()
        return loss
