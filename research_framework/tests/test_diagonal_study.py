"""Scientific controls for the new six-condition bounded study."""
import copy
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research_framework.config import DEFAULTS, validate_config
from research_framework.embeddings.periodic import PeriodicEmbedding
from research_framework.followup.continuous import MODES
from research_framework.followup.diagonal_study import (
    DiagonalEmbedding, DiagonalTask, StudyStore, VARIANTS, measurement_statistics, history_rows)

jax.config.update("jax_enable_x64",True)


def config(variant):
    c=copy.deepcopy(DEFAULTS)
    c['experiment']['task']='continuous_electrons'
    c['physics'].update(n=2,lattice=[[3.,0.],[0.,3.]],crystal_supercell=[[1,0],[0,1]],rs=4.,lam=.72,delta=4.,hf_path=None)
    c['model'].update(capacity_mode='padded',embedding_budget=20,width=4,heads=1,head_dim=2,depth=1,perceptrons=1)
    c['embedding'].update(type='baseline' if variant==VARIANTS[0] else 'learned_fourier',modes=MODES,spin_encoding='scalar',sparse_penalty=0. if variant==VARIANTS[0] else .01)
    c['wavefunction'].update(type='hf_nn_jastrow',jastrow=False,complex_phase=True,net_scale=.05)
    c['training'].update(method='vmc',steps=10000,walkers=4,burn_in=2,sweeps=2,checkpoint_every=500)
    c['attention']['window'].update(mode='fixed',shape='cos',width=1.5)
    c['evaluation'].update(ph=False,interventions=False,probe_samples=2)
    c['analysis'].update(phase_sensitivity=False,persistent_homology=False,entanglement=False,berry=False)
    return validate_config(c)


def test_plus_diagonal_preserves_baseline_rows_and_periodicity():
    c=config(VARIANTS[0]);p=PeriodicEmbedding(c['embedding'],c['physics'],c['model'])
    d=DiagonalEmbedding(c['embedding'],c['physics'],c['model'],variant=VARIANTS[0])
    w,e=d.init(jax.random.PRNGKey(1),4,4);w0,e0=p.init(jax.random.PRNGKey(1),4,4)
    np.testing.assert_array_equal(w,w0)
    assert sum(v.size for v in e.values())==sum(v.size for v in e0.values())==6
    base=jnp.arange(8.).reshape(2,4);spin=jnp.array([1.,-1.]);ae=jnp.array([[[.3,.7]],[[1.2,.4]]])
    f=d.features(base,spin,ae,e);old=p.features(base,spin,ae,e)
    np.testing.assert_allclose(f[:,:5],old)
    theta=2*np.pi*np.asarray(ae[:,0,:])/3
    np.testing.assert_allclose(f[:,5],np.cos(theta[:,0]-theta[:,1]))
    np.testing.assert_allclose(f[:,6],np.sin(theta[:,0]-theta[:,1]))
    np.testing.assert_allclose(f,d.features(base,spin,ae+jnp.array([[[3.,0.]]]),e),atol=1e-12)
    masked=d.features(base,spin,ae,dict(e,_feature_mask=jnp.ones(7).at[5:7].set(0)))
    np.testing.assert_array_equal(masked[:,5:],0)
    np.testing.assert_array_equal(d.feature_mask('remove_spin'),[1,1,1,1,0,1,1])


def test_disabled_candidate_cannot_reappear_through_data_or_penalty_gradients():
    c=config(VARIANTS[1]);d=DiagonalEmbedding(c['embedding'],c['physics'],c['model'],variant=VARIANTS[1])
    w,e=d.init(jax.random.PRNGKey(2),4,4)
    assert float(e['fourier_amplitude'][2])==0
    base=jnp.ones((2,4));spin=jnp.array([1.,-1.]);ae=jnp.array([[[.3,.7]],[[1.2,.4]]])
    def loss(amp):
        extra=dict(e,fourier_amplitude=amp)
        return jnp.sum(d.apply(w,extra,base,spin,ae))+d.penalty(extra)
    perturbed=e['fourier_amplitude'].at[2].set(999.)
    np.testing.assert_allclose(loss(perturbed),loss(e['fourier_amplitude']))
    assert float(jax.grad(loss)(perturbed)[2])==0
    np.testing.assert_array_equal(d.features(base,spin,ae,dict(e,fourier_amplitude=perturbed))[:,8:10],0)
    assert np.any(np.asarray(jax.grad(loss)(perturbed))[[0,1,3,4,5]]!=0)


def test_two_start_diagnostics_reject_persistent_start_dependence():
    rng=np.random.default_rng(123)
    p=dict(initial_times=128,energy_rhat_max=1.05,energy_ess_min=400,
           energy_sem_Ha_per_electron_max=1e-4,structure_rhat_max=1.1,
           start_equivalence_Ha_per_electron=5e-4)
    good=[rng.normal(0,1e-6,(128,16)).astype(complex) for _ in range(2)]
    sf=[rng.normal(1,1e-3,(128,16,6)) for _ in range(2)]
    result=measurement_statistics(good,sf,p,1.)
    assert result['sampling_pass'] and result['total_chains']==32
    bad=[good[0],good[1]+1.]
    result=measurement_statistics(bad,sf,p,1.)
    assert not result['sampling_pass'] and result['energy_Ha_per_electron']['rhat']>1.05
    good[0][0,0]=np.nan
    with pytest.raises(ValueError):measurement_statistics(good,sf,p,1.)


def test_deadline_collection_retains_only_committed_history_records(tmp_path):
    path=tmp_path/'history.jsonl'
    path.write_bytes(b'{"step":1,"energy":-2.0}\n{"step":2,"energy":')
    assert history_rows(path)==[dict(step=1,energy=-2.)]


@pytest.mark.parametrize('variant',VARIANTS)
def test_existing_vmc_runs_two_finite_updates_with_new_embedding(tmp_path,variant):
    c=config(variant);c['experiment']['name']=variant
    # Both spinor components need support: the generic toy table includes a
    # k=0 orbital with an identically zero minus component, unlike production HF.
    hf=tmp_path/'two_supported_spinors.txt'
    np.savetxt(hf,[[2*np.pi/3,0,0,0,1,0],[0,2*np.pi/3,0,0,1,0]])
    c['physics']['hf_path']=str(hf)
    task=DiagonalTask(c,variant)
    store=StudyStore(tmp_path/variant,c,task,time.time()+120)
    task.run(store,until=2)
    import json
    rows=[json.loads(x) for x in (store.path/'history.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(np.isfinite(r['energy']) for r in rows)
    assert (store.path/'diagnostics/step_0000000_internal.npz').exists()
    _,state=store.restore()
    if variant==VARIANTS[1]:
        assert float(state['params']['layers']['embedding_extra']['fourier_amplitude'][2])==0
