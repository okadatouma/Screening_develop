"""Complex channels retain phase; charged embeddings use no additive bias."""
import torch
from torch import nn


def links(z,valid=None,tolerance=1e-7):
    """T_ij=<u_j|u_i>/|.| transforms exp(i chi_i) T_ij exp(-i chi_j)."""
    overlap=torch.einsum('bic,bjc->bij',z,z.conj())
    good=abs(overlap)>tolerance
    if valid is not None:good=good & valid[:,:,None] & valid[:,None,:]
    transport=torch.where(good,overlap/abs(overlap).clamp_min(tolerance),torch.zeros_like(overlap))
    return transport,good


class PhaseEmbedding(nn.Module):
    def __init__(self,config,model):
        super().__init__();self.kind=config['type'];self.width=model['width']
        self.charged=self.kind=='gauge_covariant';self.complex=model.get('complex_stream',False)
        if self.complex:
            self.projection=nn.Linear(2,self.width,bias=not self.charged,dtype=torch.complex64)
            self.neutral_gate=nn.Linear(5,self.width,bias=False)
        else:
            self.projection=nn.Linear((8 if self.kind=='projector' else 4)+5,self.width)

    def forward(self,z,coords,mass,valid,intervention=None):
        spec=intervention or {}
        neutral=torch.cat([coords,torch.broadcast_to(mass[:,None,None]/4,(*z.shape[:2],1))],-1)
        if self.complex:
            value=self.projection(z)*(1+.1*torch.tanh(self.neutral_gate(neutral)))
            value=value*torch.tanh(abs(value))/abs(value).clamp_min(1e-7)
        else:
            if self.kind=='projector':
                p=z[:,:,:,None]*z[:,:,None,:].conj();flat=p.flatten(-2)
                features=torch.cat([flat.real,flat.imag],-1)
            else:features=torch.cat([z.real,z.imag],-1)
            if spec.get('type')=='zero_phase_channel':features=features.clone();features[...,features.shape[-1]//2:]=0
            value=torch.tanh(self.projection(torch.cat([features,neutral],-1)))
        return value*valid[...,None]
