"""Tests only for the new follow-up task/protocol, not existing diagnostics."""
import copy
import numpy as np
import pytest
import torch
from research_framework.followup.holonomy import dataset,contract,HolonomyModel,inputs,gauge,oracle
from research_framework.followup.continuous import zero_correction,remove_candidate,cases,MODES
from research_framework.embeddings.periodic import PeriodicEmbedding


@pytest.fixture(scope='module')
def batch():
    torch.set_num_threads(2)
    return dataset(16,410)


def test_identifiability_and_local_ambiguity(batch):
    c=contract(batch)
    assert c['passed'] and c['local_twins_mse_lower_bound']>1e-4
    assert 'target' not in inputs(batch) and 'metadata' not in inputs(batch)


@pytest.mark.parametrize('kind',['local_mlp','pooled_mlp','local_attention','full_attention','gauge_covariant','trainable_window'])
def test_new_readout_and_gauge_invariance(batch,kind):
    torch.manual_seed(1);m=HolonomyModel(kind)
    b=inputs(batch);y=m(b)
    transformed=gauge(b,torch.randn(b['z'].shape[:2])*2)
    assert y.shape==(32,2) and torch.isfinite(y).all()
    torch.testing.assert_close(y,m(transformed),atol=3e-6,rtol=3e-5)
    if kind in ['local_mlp','local_attention']:
        torch.testing.assert_close(y[::2],y[1::2],atol=2e-6,rtol=2e-5)
    if len(m.blocks):
        changed=m(b,dict(type='head_ablation',layer=0,head=0))
        assert torch.max(abs(changed-y))>1e-8
    ((y-batch['target'])**2).mean().backward()
    if kind=='trainable_window':assert m.blocks[0].window_logit.grad is not None


def test_hidden_target_cannot_change_prediction(batch):
    m=HolonomyModel('full_attention');bad=dict(batch,target=torch.randn_like(batch['target']),metadata=None)
    torch.testing.assert_close(m(batch),m(bad))


def test_splits_and_longer_path(batch):
    other=dataset(16,411)
    assert not np.array_equal(batch['metadata'],other['metadata'])
    contract(dataset(12,412,length=16))


def test_initialization_and_candidate_mask_do_not_mutate():
    import jax.numpy as jnp
    from research_framework.config import DEFAULTS
    p={'orbital':[{'w':jnp.ones((2,3))}], 'layers':{'embedding_extra':{'fourier_amplitude':jnp.ones(6)}}}
    zero=zero_correction(p)
    assert np.asarray(zero['orbital'][0]['w']).sum()==0
    assert np.asarray(p['orbital'][0]['w']).sum()==6
    c=copy.deepcopy(DEFAULTS);c['embedding'].update(type='learned_fourier',modes=MODES)
    e=PeriodicEmbedding(c['embedding'],c['physics'],c['model'])
    changed=remove_candidate(p,e,2)
    mask=np.asarray(changed['layers']['embedding_extra']['_feature_mask'])
    assert np.array_equal(np.flatnonzero(mask==0),[8,9])
    assert '_feature_mask' not in p['layers']['embedding_extra']
    with pytest.raises(IndexError):remove_candidate(p,e,6)


def test_n36_plan_and_gate_rejects_mismatched_walkers(tmp_path):
    import json
    from research_framework.paths import ROOT
    from research_framework.followup.cluster import gate
    base=json.loads((ROOT/'research_framework/results/2026-11-06_cluster/plan.json').read_text())[-1]['config']
    plan=cases(base)
    assert len(plan)==18
    assert all(x['config']['physics']==base['physics'] for x in plan)
    for x in plan:
        folder=tmp_path/x['name'];folder.mkdir()
        evidence=dict(phase=[0.],logabs=[1.],position_sha256='same',burned_walker_sha256='same',parameter_count=11439)
        (folder/'initial_state.json').write_text(json.dumps(evidence))
        (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(energy=1.,gradient_norm=1.)) for _ in range(2)))
    bad=tmp_path/plan[1]['name']/'initial_state.json'
    evidence=json.loads(bad.read_text());evidence['burned_walker_sha256']='mismatch';bad.write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match='unmatched burned_walker'):gate(tmp_path,plan)
    assert not (tmp_path/'PILOT_GATE_PASSED.json').exists()
    evidence['burned_walker_sha256']='same';bad.write_text(json.dumps(evidence));gate(tmp_path,plan)
    assert (tmp_path/'PILOT_GATE_PASSED.json').exists()


def test_global_mlp_control_same_information_and_capacity(batch):
    from research_framework.followup.controls import GlobalMLP
    m=GlobalMLP();b=inputs(batch)
    assert sum(p.numel() for p in m.parameters())==5474
    torch.testing.assert_close(m(b),m(gauge(b,torch.randn(b['z'].shape[:2]))),atol=2e-6,rtol=2e-5)
    ((m(b)-batch['target'])**2).mean().backward()
