"""Controlled initial HF state and individual Fourier-removal measurements.

No new Hamiltonian, VMC update, sampling, PH or attention implementation.
"""
import copy
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from ..config import validate_config
from ..experiments.store import digest,json_save,csv_save
from ..task_adapters.continuous import ContinuousTask

MODES=[[1,0],[0,1],[1,-1],[6,0],[0,6],[6,-6]]
KINDS=['baseline','six_theta','spin_onehot','six_theta_onehot','learned_fourier']


def zero_correction(params):
    p=copy.deepcopy(params)
    p['orbital'][0]['w']=jnp.zeros_like(p['orbital'][0]['w'])
    return p


class MatchedHFTask(ContinuousTask):
    def build(self,folder):
        super().build(folder)
        old_init=self.wave.init
        self.wave.init=lambda key: zero_correction(old_init(key))

    def initial_walkers(self,key,count):
        data=super().initial_walkers(key,count)
        # Record the actual physical initial state, not just the nominal seed.
        return data


def cases(base,steps=10000):
    result=[]
    for seed in [101,202,303]:
        for kind,penalty in [(k,0.) for k in KINDS]+[('learned_fourier',.01)]:
            c=copy.deepcopy(base)
            name=f'n36_{kind}_l1{penalty:g}_s{seed}'
            c['experiment'].update(name=name,seed=seed,stage='followup')
            c['model'].update(capacity_mode='padded',embedding_budget=20)
            c['embedding'].update(type=kind,spin_encoding='onehot' if 'onehot' in kind else 'scalar',
                                  modes=MODES,sparse_penalty=penalty)
            c['training'].update(steps=steps,checkpoint_every=min(500,steps))
            c['evaluation'].update(ph=False,interventions=False,samples=64,chains=8,burn_in=256,thinning=4)
            c['analysis'].update(persistent_homology=False)
            result.append(dict(name=name,config=validate_config(c)))
    return result


def initial_evidence(task,seed):
    _,key,walk=jax.random.split(jax.random.PRNGKey(seed),3)
    p=task.wave.init(key)
    data=task.initial_walkers(walk,4)
    phase,logabs=jax.vmap(task.wave.apply,in_axes=(None,0,0))(p,data.positions,data.spins)
    if not np.isfinite(np.asarray(logabs)).all():raise FloatingPointError('Initial HF support not finite')
    return dict(phase=np.asarray(phase).tolist(),logabs=np.asarray(logabs).tolist(),
                position_sha256=digest(np.asarray(data.positions).tobytes()),
                parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(p)),
                nn_readout_zero=bool(np.all(np.asarray(p['orbital'][0]['w'])==0)),
                state='identical supplied HF; NN orbital readout zero; Jastrow disabled')


def remove_candidate(params,embedding,index):
    if not embedding.learned:raise ValueError('candidate removal requires learned_fourier')
    if not 0<=index<len(embedding.modes):raise IndexError(index)
    p=copy.deepcopy(params)
    mask=np.ones(embedding.dimensions())
    mask[4+2*index:6+2*index]=0
    p['layers']['embedding_extra']['_feature_mask']=jnp.asarray(mask)
    return p


def sq(task,x):
    q=np.asarray(MODES)
    frac=x.reshape(-1,task.c['physics']['n'],2)@np.linalg.inv(task.c['physics']['lattice']).T
    rho=np.exp(2j*np.pi*np.einsum('bnd,qd->bnq',frac,q)).sum(1)
    return (abs(rho)**2/task.c['physics']['n']).mean(0)


def candidate_response(task,store,params):
    """Independent fresh Born resampling for each removed pair of input channels."""
    # RunStore intentionally restores host arrays. The existing normalized
    # Fourier projection uses JAX .at updates outside jit during phase probes.
    params=jax.tree_util.tree_map(jnp.asarray,params)
    rows=[]; seed=task.c['experiment']['seed']+40000
    ref,x,s,e=task.measured_arrays(params,seed)
    base_sq=sq(task,x)
    # Matched probes distinguish phase response from a change in sample distribution.
    phase0=np.asarray(jax.vmap(task.wave.apply,in_axes=(None,0,0))(params,jnp.asarray(x[:8]),jnp.asarray(s[:8]))[0])
    for index,q in enumerate(MODES):
        p=remove_candidate(params,task.wave.embedding,index)
        value,xx,ss,ee=task.measured_arrays(p,seed+index+1)
        phase=np.asarray(jax.vmap(task.wave.apply,in_axes=(None,0,0))(p,jnp.asarray(x[:8]),jnp.asarray(s[:8]))[0])
        rows.append(dict(candidate=index,qx=q[0],qy=q[1],amplitude=float(params['layers']['embedding_extra']['fourier_amplitude'][index]),
                         **value,delta_energy=value['energy']-ref['energy'],baseline_sem=ref['block_sem'],
                         delta_sq=(sq(task,xx)-base_sq).tolist(),phase_circular_rms=float(np.sqrt(np.mean(np.angle(np.exp(1j*(phase-phase0)))**2))),
                         resampled=True,importance_certified=False))
        np.savez_compressed(store.path/'diagnostics'/f'candidate_{index}_born.npz',positions=xx,spins=ss,local_energy=ee)
        task.diagnostics(store,f'candidate_{index}',p,x[:8],s[:8])
    json_save(store.path/'candidate_response.json',dict(reference=ref,modes=MODES,reference_sq=base_sq,rows=rows,
        limitation='Amplitudes are coordinates, not unique physical importance. Finite-chain SEM does not certify convergence.'))
    csv_save(store.path/'candidate_response.csv',rows)
