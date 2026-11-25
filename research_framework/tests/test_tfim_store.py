import numpy as np
import torch
import jax
import jax.numpy as jnp
import pytest
from research_framework.config import validate_config
from research_framework.experiments.store import RunStore
from research_framework.task_adapters.tfim import build_model,intervene,TFIM,configurations,cmi_matrix
from research_framework.diagnostics.information import born_mi,quantum_mi,reduced_density


def test_tfim_existing_hamiltonian_local_energy_and_parity():
    c=validate_config({'physics':{'n':4}});m=build_model(c);p=m.init(jax.random.PRNGKey(1));spins=jnp.asarray(configurations(4))
    logs=np.asarray(m.batch(p,spins));psi=np.exp(logs-logs.max());psi/=np.linalg.norm(psi)
    h=TFIM(4);local=np.asarray(h.local_energy(m.batch,p,spins))
    np.testing.assert_allclose(np.dot(psi**2,local),np.vdot(psi,h.operator()@psi),atol=1e-10)
    _,ed,res=h.ground_state();assert res<1e-10
    np.testing.assert_allclose(np.vdot(ed,ed[::-1]),1,atol=1e-10)
    target,changed=intervene(m,p,dict(type='head_ablation',head=0))
    assert not np.allclose(m.batch(p,spins),target.batch(changed,spins))


def test_classical_cmi_and_quantum_mi_are_distinct():
    bell=np.array([1.,0.,0.,1.])/np.sqrt(2)
    assert abs(born_mi(bell,2)[0,1]-np.log(2))<1e-12
    assert abs(quantum_mi(bell,2)[0,1]-2*np.log(2))<1e-12
    np.testing.assert_allclose(cmi_matrix(bell,2)[0,1],np.log(2),atol=1e-12)
    product=np.ones(16)/4
    np.testing.assert_allclose(cmi_matrix(product,4),0,atol=1e-12)
    np.testing.assert_allclose(quantum_mi(product,4),0,atol=1e-12)


@pytest.mark.parametrize('mode',['fixed','trainable_per_head'])
def test_adapter_soft_width_is_not_passed_as_hard_integer_radius(mode):
    c=validate_config({'physics':{'n':8},'attention':{'window':{'mode':mode,'shape':'sigmoid','width':2.,'lower':0.,'upper':4.}}})
    model=build_model(c);params=model.init(jax.random.PRNGKey(1))
    np.testing.assert_allclose(model.widths(params['layers'][0]),[2.,2.],atol=1e-12)
    assert model.config.radius=='full'
    if mode=='trainable_per_head':
        grad=jax.grad(lambda p:model.layer_window(p['layers'][0]).sum())(params)
        assert np.linalg.norm(grad['layers'][0]['window_logit'])>0


def test_durable_resume_preserves_rng_optimizer_and_rejects_changes(tmp_path):
    c=validate_config({});store=RunStore(tmp_path/'run',c)
    state=dict(params=dict(x=jnp.array([1.,2.])),torch=torch.tensor([3.,4.]),rng=torch.get_rng_state())
    store.save_checkpoint(2,state);step,new=store.restore();assert step==2
    np.testing.assert_array_equal(new['params']['x'],state['params']['x'])
    torch.testing.assert_close(new['torch'],state['torch']);torch.testing.assert_close(new['rng'],state['rng'])
    with pytest.raises(FileExistsError):store.save_checkpoint(2,state)
    resumed=RunStore(tmp_path/'run',c,resume=True);assert resumed.restore()[0]==2
    changed=validate_config({'experiment':{'seed':202}})
    with pytest.raises(ValueError):RunStore(tmp_path/'run',changed,resume=True)
