import pickle
import numpy as np
import pytest
from research_framework.cli import run
from research_framework.config import validate_config
from research_framework.experiments.plans import qwz_base


def last(path):
    files=sorted((path/'checkpoints').glob('*.pkl'))
    with files[-1].open('rb') as f:return pickle.load(f)['state']


@pytest.mark.parametrize('task',['lattice_tfim','qwz_topology'])
def test_interrupted_training_restores_exact_optimizer_rng_and_outputs(tmp_path,task):
    if task=='lattice_tfim':c=validate_config({'physics':{'n':4},'training':{'steps':4,'checkpoint_every':2}})
    else:
        c=qwz_base();c['physics']['nk']=4;c['training'].update(steps=4,checkpoint_every=2)
    c['evaluation'].update(ph=False,interventions=False,probe_samples=2)
    c['analysis']['phase_sensitivity']=False
    uninterrupted=tmp_path/'uninterrupted';interrupted=tmp_path/'interrupted'
    run(c,uninterrupted)
    run(c,interrupted,until=2)
    assert not (interrupted/'COMPLETE.json').exists()
    run(c,interrupted,resume=True)
    a,b=last(uninterrupted),last(interrupted)
    if task=='lattice_tfim':
        import jax
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(np.load(uninterrupted/'states.npz')['variational'],np.load(interrupted/'states.npz')['variational'])
    else:
        import torch
        for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
        torch.testing.assert_close(a['rng'],b['rng'],rtol=0,atol=0)
