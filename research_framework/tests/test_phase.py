from copy import deepcopy
import numpy as np
import torch
import pytest
from research_framework.experiments.plans import qwz_base
from research_framework.factory import make_wavefunction
from research_framework.embeddings.phase import links
from research_framework.task_adapters.qwz import dataset,transform_gauge,QWZTask
from research_framework.paths import wannier_source
wannier_source()
from wannier_ph.geometry import qwz,berry_data,phase_lift,euclidean
from wannier_ph.persistence import compute,plateau


@pytest.mark.parametrize('atlas',[False,True])
def test_transport_embedding_attention_and_projection_covariance(atlas):
    torch.set_num_threads(1);torch.manual_seed(12)
    c=qwz_base();c['embedding']['type']='gauge_covariant';c['attention'].update(type='gauge_covariant',phase_transport=True)
    if atlas:c['wavefunction']['type']='local_atlas'
    c['physics']['nk']=5
    model=make_wavefunction(c['wavefunction'],'qwz_topology',full_config=c)
    batch=dataset(c,[1.,3.]);chi=2*torch.pi*torch.rand(batch['valid'].shape)
    other=transform_gauge(batch,chi)
    left=model(batch);right=model(other)
    phase=torch.exp(1j*chi)
    a,valid=links(batch['z'],batch['valid']);b,_=links(other['z'],other['valid'])
    torch.testing.assert_close(b,phase[:,:,None]*a*phase[:,None,:].conj(),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(right['features'],phase[...,None]*left['features'],atol=4e-6,rtol=4e-6)
    torch.testing.assert_close(right['projector'],left['projector'],atol=5e-6,rtol=5e-6)
    for x,y in zip(left['records'],right['records']):
        torch.testing.assert_close(x['QK'],y['QK'],atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(y['output'],phase[...,None]*x['output'],atol=4e-6,rtol=4e-6)
    loss=model.loss(left,batch);assert torch.isfinite(loss);loss.backward()
    assert torch.isfinite(model.readout.weight.grad).all()
    p=left['projector'];torch.testing.assert_close(p@p,p,atol=5e-6,rtol=5e-6)
    torch.testing.assert_close(p,p.transpose(-1,-2).conj())
    if atlas:
        torch.testing.assert_close(right['frame'],phase[...,None]*left['frame'],atol=4e-6,rtol=4e-6)
    broken=model(batch,{'type':'disable_gauge_transport'})
    assert not torch.allclose(broken['features'],left['features'])


def test_zero_overlap_links_are_masked_not_fake_phases():
    z=torch.tensor([[[1,0],[0,1],[0,0]]],dtype=torch.complex64)
    t,good=links(z)
    assert t[0,0,1]==0 and not good[0,0,1]
    assert not good[0,2].any()
    assert torch.isfinite(t).all()


@pytest.mark.parametrize('kind,attention',[('baseline','real'),('projector','real'),('complex','real'),('complex','complex')])
def test_non_covariant_models_have_valid_losses_and_head_intervention(kind,attention):
    torch.manual_seed(4)
    c=qwz_base();c['embedding']['type']=kind;c['attention']['type']=attention;c['physics']['nk']=4
    model=make_wavefunction(c['wavefunction'],'qwz_topology',full_config=c);data=dataset(c,[1.])
    a=model(data);b=model(data,{'type':'head_ablation','head':0})
    assert not torch.allclose(a['features'],b['features'])
    loss=model.loss(a,data);assert torch.isfinite(loss);loss.backward()
    assert model.readout.weight.grad is not None


def test_qwz_physical_reference_and_existing_plateau():
    from ripser import ripser
    for mass,expected in [(3,0),(1,1),(-1,-1)]:
        k,u,_=qwz(8,mass);chi=np.random.default_rng(0).uniform(-3,3,u.shape[:2])
        assert abs(berry_data(u*np.exp(1j*chi)[...,None])['chern']-expected)<1e-12
        assert abs(berry_data(u.conj())['chern']+expected)<1e-12
    k,u,_=qwz(8,1);d=euclidean(phase_lift(k,u,8));r=ripser(d,distance_matrix=True)
    assert plateau(r['dgms'],2,d.max())['plateau_screen']
    u[:]=[1,0];d=euclidean(phase_lift(k,u,8));r=ripser(d,distance_matrix=True)
    assert plateau(r['dgms'],3,d.max())['plateau_screen']


def test_relative_scale_failure_is_not_hidden_by_normalization():
    from ripser import ripser
    k,u,_=qwz(8,3);widths=[]
    for scale in [1.,2.]:
        d=euclidean(phase_lift(k,u,12,base_scale=scale,seed=0,offset=True))
        diagrams=ripser(d,distance_matrix=True)['dgms']
        widths.append(plateau(diagrams,3,d.max())['plateau_width'])
    assert widths[0]>.08 and widths[1]<.08


def test_torch_head_width_gradient_tail_and_restore():
    from research_framework.attentions.phase import PhaseAttention
    c=qwz_base()['attention'];c['window'].update(mode='trainable_per_head',width=2.,lower=.1,upper=5.,shape='gaussian')
    module=PhaseAttention(c,12,2)
    with torch.no_grad():module.window_logit.copy_(torch.tensor([-.4,.8]))
    distance=torch.tensor([[0.,3.],[3.,0.]])
    total=module.window(distance,{})[0].sum();total.backward();analytic=float(module.window_logit.grad[0])
    eps=1e-3
    with torch.no_grad():
        module.window_logit[0]+=eps;a=float(module.window(distance,{})[0].sum());module.window_logit[0]-=2*eps;b=float(module.window(distance,{})[0].sum());module.window_logit[0]+=eps
    assert abs(analytic-(a-b)/(2*eps))<1e-3
    assert module.window(distance,{})[0,0,1]>0
    other=PhaseAttention(c,12,2);other.load_state_dict(module.state_dict());torch.testing.assert_close(other.widths(),module.widths())


def test_width_intervention_changes_a_full_attention_window():
    from research_framework.attentions.phase import PhaseAttention
    c=qwz_base()['attention'];module=PhaseAttention(c,12,2)
    d=torch.tensor([[0.,3.],[3.,0.]])
    assert torch.all(module.window(d,{})==1)
    assert module.window(d,{'type':'window_width','width':1.})[0,0,1]<.1


def test_large_graph_preserves_tiny_positive_edges_and_absent_zeros(tmp_path):
    from research_framework.diagnostics.topology import graph_ph
    w=np.zeros((65,65))
    for i,j in [(0,1),(1,2),(2,3),(3,0)]:w[i,j]=w[j,i]=1e-10
    graph_ph(tmp_path,'tiny',{'Screening':w})
    d=np.load(tmp_path/'tiny_ph.npz')['Screening_H1']
    assert len(d)==1 and np.isinf(d[0,1])
