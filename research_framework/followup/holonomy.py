"""Masked QWZ Wilson-loop composition, reusing the existing atlas and blocks.

The supplied observations are local links, NOT raw globally visible projectors.
This tests composition of supplied geometric information, not discovery of Chern.
"""
import numpy as np
import torch
from torch import nn
from ..paths import wannier_source
from ..embeddings.phase import PhaseEmbedding
from ..attentions.phase import PhaseAttention

wannier_source()
from wannier_ph.geometry import qwz
from wannier_ph.wannier import canonical_atlas

INPUT_KEYS=('z','carrier','coords','valid','distance','chart')


def geometry(length=12):
    if length%4 or length<12:raise ValueError('length must be a multiple of 4 >=12')
    w=length//4
    path=np.array([(i,0) for i in range(w)]+[(w,j) for j in range(w)]+
                  [(w-i,w) for i in range(w)]+[(0,w-j) for j in range(w)])
    n=1+2*length
    # Graph geodesic distance: two edge-token rings joined to a query at edge 0.
    distance=np.full((n,n),1e3);np.fill_diagonal(distance,0)
    for loop in range(2):
        ix=1+loop*length
        distance[0,ix]=distance[ix,0]=1
        for j in range(length):
            a=ix+j;b=ix+(j+1)%length
            distance[a,b]=distance[b,a]=1
    for k in range(n):distance=np.minimum(distance,distance[:,k,None]+distance[None,k,:])
    return path,distance.astype('float32')


def _frames(u):
    f,valid,_=canonical_atlas(u)
    # Use the better conditioned chart at every point; its identity is hidden.
    chart=np.argmax(abs(u)**2,axis=-1)
    return np.take_along_axis(f,chart[...,None,None],axis=-1)[...,0]


def _rotate(u,k,center,angle,axis,radius):
    delta=(k-center+np.pi)%(2*np.pi)-np.pi
    bump=np.maximum(1-np.sum(delta**2,axis=-1)/radius**2,0)**4
    a=angle*bump/2
    # Smooth periodic SU(2) conjugation H'=UHU† preserves the exact gap and Chern.
    pauli=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
    sigma=np.einsum('a,aij->ij',axis,pauli)
    return np.cos(a)[...,None]*u-1j*np.sin(a)[...,None]*np.einsum('ij,...j->...i',sigma,u)


def dataset(groups=128,seed=0,length=12):
    rng=np.random.default_rng(seed);path,distance=geometry(length);n=len(distance)
    zs=[];carriers=[];coords=[];targets=[];ids=[];minimum=[];meta=[]
    grid=4*(length//4)
    for group in range(groups):
        mass=float(rng.choice([-.4,-.8,-1.3,.4,.8,1.3,2.7,-2.7]))
        k,u,gap=qwz(grid,mass,shift=rng.uniform(0,1,2))
        starts=rng.integers(0,grid,(2,2));selected=int(rng.integers(2))
        # Geometric exclusion, independent of the target: the smooth deformation
        # must be invisible to BOTH loops' query-local two-hop neighbourhoods.
        remote=(path[length//2]+starts[selected])%grid
        for attempt in range(1000):
            other=(path+starts[1-selected])%grid
            local_vertices=other[[0,1,2,-1]]
            delta=(local_vertices-remote+grid/2)%grid-grid/2
            if np.min(np.linalg.norm(delta,axis=-1))>1.55:break
            starts[1-selected]=rng.integers(0,grid,2)
        else:raise RuntimeError('could not place distractor outside local exclusion')
        indices=[(path+s)%grid for s in starts];ix=indices[selected][length//2]
        center=k[ix[0],ix[1]];axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
        angle=rng.uniform(1.,2.9)
        carrier=np.exp(1j*rng.uniform(-np.pi,np.pi,n))
        c=np.zeros((n,4),np.float32);c[0,0]=1
        for loop in range(2):
            sl=slice(1+loop*length,1+(loop+1)*length)
            t=2*np.pi*np.arange(length)/length
            c[sl]=np.stack([np.zeros(length),np.full(length,1 if loop==selected else -1),np.cos(t),np.sin(t)],-1)
        for twin,sign in enumerate([-1,1]):
            changed=_rotate(u,k,center,sign*angle,axis,1.55*2*np.pi/grid)
            frames=_frames(changed)
            z=np.zeros((n,2),complex);z[0,1]=carrier[0]
            loops=[]
            for loop in range(2):
                p=indices[loop];v=frames[p[:,0],p[:,1]]
                # T_ij=<u_j|u_i>, conjugate of the standard positive-direction link.
                link=np.sum(v*np.roll(v,-1,axis=0).conj(),-1)
                minimum.append(float(abs(link).min()))
                if abs(link).min()<1e-6:raise ValueError('non-identifiable zero-overlap edge; no data silently discarded')
                link/=abs(link);loops.append(np.prod(link))
                sl=slice(1+loop*length,1+(loop+1)*length)
                z[sl,0]=carrier[sl]*link/np.sqrt(2);z[sl,1]=carrier[sl]/np.sqrt(2)
            zs.append(z);carriers.append(carrier);coords.append(c);targets.append(loops[selected]);ids.append(group)
            meta.append((mass,gap,selected,twin))
    z=torch.tensor(np.array(zs),dtype=torch.complex64)
    carrier=torch.tensor(np.array(carriers),dtype=torch.complex64)
    return dict(z=z,carrier=carrier,coords=torch.tensor(np.array(coords)),valid=torch.ones(z.shape[:2],dtype=torch.bool),
                distance=torch.tensor(distance),chart=torch.tensor(np.r_[0,np.ones(length),2*np.ones(length)],dtype=torch.long),
                target=torch.view_as_real(torch.tensor(targets,dtype=torch.complex64)),group=np.array(ids),
                metadata=np.array(meta),minimum_overlap=min(minimum),length=length)


def inputs(batch):return {k:batch[k] for k in INPUT_KEYS}


def subset(batch,index):
    return {k:(v[index] if k in ['z','carrier','coords','valid','target'] else v) for k,v in batch.items()}


def gauge(batch,chi):
    out=dict(batch);g=torch.exp(1j*chi)
    out['z']=batch['z']*g[...,None];out['carrier']=batch['carrier']*g
    return out


def oracle(batch):
    length=batch['length'];ratio=batch['z'][:,1:,0]/batch['z'][:,1:,1]
    product=ratio.reshape(-1,2,length).prod(-1)
    selected=(batch['coords'][:,1,1]<0).long()
    return torch.view_as_real(product[torch.arange(len(product)),selected])


def contract(batch,depth=2):
    visible=batch['distance'][0]<=depth
    twin_error=float(abs(batch['z'][::2,visible]-batch['z'][1::2,visible]).max())
    lower=float(((batch['target'][::2]-batch['target'][1::2])**2).mean()/4)
    oracle_error=float(abs(oracle(batch)-batch['target']).max())
    transformed=gauge(batch,torch.randn(batch['z'].shape[:2]))
    gauge_error=float(abs(oracle(transformed)-batch['target']).max())
    assert twin_error<2e-6,'local twins differ: local information may leak'
    assert lower>1e-4,'remote changes produce insufficient target diversity'
    assert oracle_error<3e-5 and gauge_error<3e-5,'target is not recoverable from permitted input'
    assert set(inputs(batch))==set(INPUT_KEYS)
    return dict(passed=True,oracle_max_error=oracle_error,gauge_oracle_max_error=gauge_error,
                identical_local_twin_max_error=twin_error,local_twins_mse_lower_bound=lower,
                local_visible_tokens=int(visible.sum()),total_tokens=len(visible),
                minimum_overlap=batch['minimum_overlap'],groups=len(np.unique(batch['group'])),
                input_keys=INPUT_KEYS,hidden=['mass','physical k','raw occupied states','projectors','target holonomy','atlas transitions'],
                supplied='individual oriented links and graph/query metadata; flat carrier transport is a coordinate convention',
                learned='selection/composition of supplied links; not inference of Berry links or Chern',
                split='independent physical groups across train/validation/test; both twins stay in the same split')


class HolonomyModel(nn.Module):
    """New query readout around existing phase embedding and attention blocks."""
    def __init__(self,kind,width=24,heads=3,depth=2):
        super().__init__();self.kind=kind;self.depth=depth
        cov=kind in ['gauge_covariant','trainable_window']
        self.cov=cov
        self.embedding=PhaseEmbedding(dict(type='gauge_covariant' if cov else 'complex'),dict(width=width,complex_stream=cov))
        mode='trainable_per_head' if kind=='trainable_window' else ('fixed' if kind=='local_attention' else 'full')
        window=dict(mode=mode,shape='gaussian' if mode.startswith('trainable') else 'hard',
                    width=2. if mode.startswith('trainable') else (1. if mode=='fixed' else 'full'),
                    lower=.25,upper=12.,temperature=.25)
        self.blocks=nn.ModuleList([PhaseAttention(dict(type='gauge_covariant' if cov else 'real',window=window),width,heads)
                                  for _ in range(depth)]) if 'mlp' not in kind else nn.ModuleList()
        self.pointwise=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh()) if 'mlp' in kind else nn.Identity()
        self.readout=nn.Sequential(nn.Linear(width*(2 if cov else 1),width),nn.Tanh(),nn.Linear(width,2))

    def forward(self,batch,intervention=None,record=False):
        # Access only these six permitted input fields, even during training.
        b=inputs(batch);z=b['z'];carrier=b['carrier'];valid=b['valid']
        # Give every real-valued control exact pointwise carrier removal. This is
        # a stronger baseline than asking it to learn division by a random gauge.
        embedding_z=z if self.cov else z*carrier.conj()[...,None]
        x=self.embedding(embedding_z,b['coords'],torch.zeros(len(z),device=z.device),valid)
        records=[]
        flat=torch.stack([carrier,torch.zeros_like(carrier)],-1)
        for layer,block in enumerate(self.blocks):
            spec=intervention if intervention and intervention.get('layer',0)==layer else None
            update,trace=block(x,flat,valid,b['distance'],b['chart'],spec)
            x=x+update
            x=x*torch.tanh(abs(x))/abs(x).clamp_min(1e-7) if self.cov else torch.tanh(x)
            if record:records.append(trace)
        if self.kind=='local_mlp':
            local=b['distance'][0]<=self.depth
            h=self.pointwise(x[:,local]).mean(1)
        elif self.kind=='pooled_mlp':h=self.pointwise(x).mean(1)
        else:h=x[:,0]
        if self.cov:
            h=h*carrier[:,0,None].conj()
            h=torch.cat([h.real,h.imag],-1)
        prediction=self.readout(h)
        return (prediction,dict(representation=h,trace=records)) if record else prediction
