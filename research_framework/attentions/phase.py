"""Real, native-complex and U(1)-covariant Attention including output projection."""
import math
import torch
from torch import nn
from ..embeddings.phase import links


class PhaseAttention(nn.Module):
    def __init__(self,config,width,heads):
        super().__init__();self.config=config;self.width=width;self.heads=heads
        if width%heads:raise ValueError('QWZ width must be divisible by heads')
        self.covariant=config['type']=='gauge_covariant';self.is_complex=config['type'] in ['complex','gauge_covariant']
        dtype=torch.complex64 if self.is_complex else torch.float32
        self.qkv=nn.Linear(width,3*width,bias=not self.covariant,dtype=dtype)
        self.output=nn.Linear(width,width,bias=not self.covariant,dtype=dtype)
        c=config['window']
        if c['mode'].startswith('trainable'):
            frac=(c['width']-c['lower'])/(c['upper']-c['lower'])
            self.window_logit=nn.Parameter(torch.full((heads if c['mode']=='trainable_per_head' else 1,),math.log(frac/(1-frac))))
        else:self.register_buffer('window_logit',torch.zeros(1))

    def widths(self):
        c=self.config['window']
        if c['mode'].startswith('trainable'):return (c['lower']+(c['upper']-c['lower'])*torch.sigmoid(self.window_logit)).expand(self.heads)
        return torch.full((self.heads,),c['upper'] if c['width']=='full' else float(c['width']))

    def window(self,distance,spec):
        c=self.config['window'];w=self.widths()
        if spec.get('type')=='window_width':w=torch.full_like(w,float(spec['width']))
        if (c['mode']=='full' or c['width']=='full') and spec.get('type')!='window_width':window=torch.ones((self.heads,)+distance.shape,device=distance.device)
        elif c['shape']=='hard':window=(distance[None]<=w[:,None,None]).float()
        elif c['shape']=='sigmoid':window=torch.sigmoid((w[:,None,None]-distance)/c['temperature'])
        elif c['shape']=='gaussian':window=torch.exp(-(distance[None]/w[:,None,None])**2)
        else:
            x=(distance[None]/w[:,None,None]).clamp(0,1)
            if c['shape']=='smooth7':window=1-x**4*(35-84*x+70*x*x-20*x**3)
            else:window=.5*(1+torch.cos(torch.pi*x))
        if spec.get('type')=='distance_band':window=window*(distance<=spec.get('radius',1.))[None]
        return window

    def forward(self,x,z,valid,distance,chart,intervention=None):
        spec=intervention or {};b,n,width=x.shape;h=self.heads;dh=width//h
        q,k,v=self.qkv(x).reshape(b,n,3,h,dh).permute(2,0,3,1,4).unbind(0)
        transport,good=links(z,valid)
        if self.is_complex:
            raw=torch.einsum('bhid,bhjd->bhij',q.conj(),k)
            scores=(raw*transport[:,None] if self.covariant else raw).real/math.sqrt(dh)
        else:scores=q@k.transpose(-1,-2)/math.sqrt(dh)
        window=self.window(distance,spec)
        allowed=valid[:,:,None]&valid[:,None,:]
        if self.covariant:allowed=allowed&good
        if spec.get('type')=='patch_connections':allowed=allowed&(chart[None,:,None]==chart[None,None,:])
        if spec.get('type')=='remove_connections':
            allowed=allowed.clone()
            for i,j in spec.get('edges',[(0,1),(1,0)]):allowed[:,i,j]=False
        allowed_h=allowed[:,None]&(window[None]>0)
        logits=scores+torch.log(window[None].clamp_min(1e-30))
        weights=torch.softmax(logits.masked_fill(~allowed_h,-1e9),-1)*allowed_h
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        use_transport=self.covariant and spec.get('type')!='disable_gauge_transport'
        effective=weights.to(v.dtype)*(transport[:,None] if use_transport else 1)
        head=effective@v
        if spec.get('type')=='head_ablation':
            mask=torch.ones(h,device=x.device);mask[spec.get('head',0)]=0;head=head*mask[None,:,None,None]
        blocks=self.output.weight.reshape(width,h,dh).permute(1,2,0)
        contribution=torch.einsum('bhnd,hdw->bhnw',head,blocks)
        update=contribution.sum(1)
        if self.output.bias is not None:update=update+self.output.bias
        alpha=torch.softmax(scores.masked_fill(~allowed[:,None],-1e9),-1)*allowed[:,None]
        alpha=alpha/alpha.sum(-1,keepdim=True).clamp_min(1e-12)
        record=dict(Q=q,K=k,V=v,QK=scores,Alpha=alpha,
                    Window=window,weights=weights,transport=transport,valid_links=good,head_contribution=contribution,
                    widths=self.widths(),output=update)
        return update*valid[...,None],record
