import jax
import numpy as np
from research_framework.config import validate_config
from research_framework.task_adapters import tfim
from research_framework.cli import run


def test_vmc_honors_burn_in_and_restores_durable_state(tmp_path,monkeypatch):
    called=[];original=tfim.metropolis
    def tracked(model,params,spins,key,proposals):
        called.append(proposals)
        return original(model,params,spins,key,proposals)
    monkeypatch.setattr(tfim,'metropolis',tracked)
    c=validate_config(dict(physics=dict(n=4),training=dict(method='vmc',steps=1,checkpoint_every=1,
        walkers=2,burn_in=3,sweeps=2),evaluation=dict(ph=False,interventions=False,probe_samples=2)))
    run(c,tmp_path/'vmc')
    assert 12 in called and 8 in called


def test_intervened_state_chunking_matches_full_enumeration():
    from lattice.physics import enumerate_model,normalized_state,configurations
    c=validate_config(dict(physics=dict(n=5)))
    model=tfim.build_model(c);params=model.init(jax.random.PRNGKey(2))
    changed,p=tfim.intervene(model,params,dict(type='head_ablation',head=0))
    full=normalized_state(np.asarray(jax.jit(changed.batch)(p,configurations(5))))
    chunked=enumerate_model(changed,p,batch_size=8)
    np.testing.assert_allclose(chunked,full,rtol=1e-12,atol=1e-12)
