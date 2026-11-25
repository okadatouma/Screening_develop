import importlib.util
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from research_framework.experiments.plans import continuous_base
from research_framework.factory import make_wavefunction
from research_framework.embeddings.periodic import PeriodicEmbedding
from research_framework.attentions.screening import ScreeningFactory
from periodicwave import CustomScreenformer as cs,network_blocks


def reference_module():
    path=Path(cs.__file__).parents[1]/'research_framework/results/implementation_audit/CustomScreenformer_before.py.txt'
    spec=importlib.util.spec_from_loader('periodicwave._research_before',loader=None)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    return module


def test_legacy_factory_exact_initialization_wavefunction_and_derivatives():
    from periodicwave import envelopes
    from periodicwave.pbc import feature_layer
    before=reference_module();lat=jnp.asarray([[4.,-.3],[0.,3.8]])
    kwargs=dict(ndim=2,determinants=1,envelope=envelopes.make_null_envelope(),
        feature_layer=feature_layer.make_pbc_feature_layer(1,(2,0),ndim=2,lattice=lat,include_r_ae=False),
        jastrow='NONE',complex_output=True,pbc_lattice=lat,num_layers=1,num_heads=2,attn_dim=2,value_dim=2,
        mlp_dim=8,num_perceptrons_per_layer=1,use_layer_norm=False,mlp_activation_fct='TANH')
    old=before.make_fermi_net((2,0),jnp.zeros(1),**kwargs);new=cs.make_fermi_net((2,0),jnp.zeros(1),**kwargs)
    po=old.init(jax.random.PRNGKey(7));pn=new.init(jax.random.PRNGKey(7))
    for a,b in zip(jax.tree_util.tree_leaves(po),jax.tree_util.tree_leaves(pn)):np.testing.assert_array_equal(a,b)
    x=jnp.asarray([.23,.41,1.17,.89]);s=jnp.array([1.,-1.]);at=jnp.zeros((1,2));q=jnp.zeros(1)
    left=lambda p,z:old.apply(p,z,s,at,q);right=lambda p,z:new.apply(p,z,s,at,q)
    np.testing.assert_array_equal(left(po,x),right(pn,x))
    np.testing.assert_array_equal(jax.grad(lambda z:left(po,z)[1])(x),jax.grad(lambda z:right(pn,z)[1])(x))
    np.testing.assert_array_equal(jax.hessian(lambda z:left(po,z)[1])(x),jax.hessian(lambda z:right(pn,z)[1])(x))


@pytest.mark.parametrize('kind,dim',[('baseline',5),('six_theta',11),('spin_onehot',6),('six_theta_onehot',12),('learned_fourier',11)])
def test_embedding_pbc_crystal_permutation_and_derivatives(kind,dim):
    c=continuous_base();c['physics']['lattice']=[[4.,-.5],[0.,3.]];c['physics']['crystal_supercell']=[[6,0],[0,6]];c['embedding']['type']=kind
    emb=PeriodicEmbedding(c['embedding'],c['physics'],c['model'])
    weight,extra=emb.init(jax.random.PRNGKey(3),4,8);lat=jnp.asarray(c['physics']['lattice'])
    x=jnp.asarray([[.2,.4],[1.2,.7]]);s=jnp.asarray([1.,-1.])
    def features(xx):
        theta=2*jnp.pi*(xx@jnp.linalg.inv(lat).T);base=jnp.concatenate([jnp.sin(theta),jnp.cos(theta)],-1)
        return emb.features(base,s,xx[:,None,:],extra)
    f=features(x);assert f.shape==(2,dim)
    np.testing.assert_allclose(features(x+lat[:,0]),f,atol=2e-13)
    original=np.concatenate([np.sin(2*np.pi*np.asarray(x)@np.linalg.inv(lat).T),np.cos(2*np.pi*np.asarray(x)@np.linalg.inv(lat).T),np.asarray(s)[:,None]],-1)
    if kind=='baseline':np.testing.assert_allclose(f,original,atol=1e-14)
    if emb.onehot:np.testing.assert_array_equal(f[:,-2:],[[1,0],[0,1]])
    if emb.high:
        angles=2*np.pi*(x@jnp.linalg.inv(lat).T)@jnp.asarray([[6,0],[0,6],[6,-6]]).T
        np.testing.assert_allclose(f[:,4:10],np.stack([np.cos(angles),np.sin(angles)],-1).reshape(2,6),atol=1e-13)
        crystal=lat@jnp.linalg.inv(jnp.asarray(c['physics']['crystal_supercell']))
        np.testing.assert_allclose(features(x+crystal[:,0])[:,4:10],f[:,4:10],atol=1e-13)
    base=f[:,:4]
    np.testing.assert_allclose(emb.apply(weight,extra,base[::-1],s[::-1],x[::-1,None,:]),emb.apply(weight,extra,base,s,x[:,None,:])[::-1],atol=1e-13)
    analytic=jax.jacfwd(features)(x);eps=1e-6;direction=jnp.asarray([[.7,-.2],[.3,.5]])
    np.testing.assert_allclose(jnp.einsum('ijab,ab->ij',analytic,direction),(features(x+eps*direction)-features(x-eps*direction))/(2*eps),rtol=2e-7,atol=2e-7)


def test_window_matches_existing_and_finite_difference():
    c=continuous_base();cfg=c['attention'];cfg['window'].update(mode='trainable_per_head',lower=.1,upper=3.,width=1.5)
    factory=ScreeningFactory(cfg);init,apply=factory(2,8,2,2,2,2.,use_softmask=True,use_fixed_value=False,use_tanh_norm=True,use_gate=True,use_output_scale=True)
    p=init(jax.random.PRNGKey(2));d=jnp.array([[0.,.4],[.4,0.]])
    w=factory.widths(p,2,2.)
    from periodicwave.attention import screening_window
    np.testing.assert_allclose(factory.window(p,d,2,2.),screening_window(dict(mask_omega=jnp.log(jnp.expm1(w/2.))),d,2.,win_shape='cos',freeze_w=False))
    def total(raw):return factory.window(dict(p,window_logit=raw),d,2,2.).sum()
    g=jax.grad(total)(p['window_logit']);eps=1e-6
    finite=np.array([(total(p['window_logit'].at[i].add(eps))-total(p['window_logit'].at[i].add(-eps)))/(2*eps) for i in range(2)])
    np.testing.assert_allclose(g,finite,rtol=1e-6,atol=1e-8)


def test_fermionic_exchange_twist_and_local_energy(tmp_path):
    from research_framework.task_adapters.continuous import synthetic_hf,ContinuousTask
    from periodicwave import networks
    c=continuous_base();c['wavefunction']['type']='twist_dependent';c['physics']['twist']=[.2,-.3]
    reference=synthetic_hf(tmp_path/'plane.txt',2,c['physics']['lattice'])
    wave=make_wavefunction(c['wavefunction'],'continuous_electrons',full_config=c,reference_path=reference)
    p=wave.init(jax.random.PRNGKey(3));x=jnp.array([.2,.4,1.2,.9]);s=jnp.array([1.,1.])
    f=lambda xx,ss:wave.apply(p,xx,ss)
    a,l=f(x,s);aa,ll=f(x.reshape(2,2)[::-1].ravel(),s[::-1]);np.testing.assert_allclose(l,ll,atol=1e-11)
    np.testing.assert_allclose(np.exp(1j*(aa-a)),-1,atol=1e-11)
    shifted=x.at[:2].add(wave.lattice[:,0]);b,lb=f(shifted,s)
    np.testing.assert_allclose(lb,l,atol=1e-10);np.testing.assert_allclose(np.exp(1j*(b-a)),np.exp(.2j),atol=1e-10)
    # No NN correction, no Jastrow: exact two plane-wave kinetic energy under twist.
    c['wavefunction']['net_scale']=0.;task=ContinuousTask(c);task.build(tmp_path);p=task.wave.init(jax.random.PRNGKey(3))
    el=task.energy_batch(p,x[None],s[None])[0]
    modes=np.loadtxt(reference)[:,:2];kappa=np.linalg.solve(np.asarray(c['physics']['lattice']).T,c['physics']['twist'])
    np.testing.assert_allclose(el,.5*np.sum((modes+kappa)**2),rtol=1e-10,atol=1e-10)
    from research_framework.wavefunctions.twist import twist_overlaps
    result,states=twist_overlaps(task.wave,p,[[0.,0.],[.2,0.],[.2,.2],[0.,.2]],mesh=3)
    np.testing.assert_allclose(np.sum(abs(states)**2,axis=1),1,atol=1e-12)
    np.testing.assert_allclose(result['overlap'],result['overlap'].conj().T,atol=1e-12)
