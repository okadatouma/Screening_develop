"""Independent C3 operator / numerical Ewald symmetry checks on fresh MC states.

No Chern number or phase label is assigned from a zero-twist rotation expectation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from periodicwave.pbc import hamiltonian
from screening_analysis.checkpoint_model import load_model
from screening_analysis.physical_loops import make_network_evaluator
from screening_analysis.rotation_probe import c3_local_ratios,c3_expectation
from screening_analysis.topology_run import json_safe


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    records=json.loads((args.analysis/'records.json').read_text());cases=json.loads((args.benchmark/'summary.json').read_text())
    folders={c['id']:Path(c['checkpoint']).parent for c in cases};results=[];cache={}
    theta=2*np.pi/3;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    for r in records:
        if r['view']!='A':continue
        folder=folders[r['id'].split('-step')[0]];path=folder/f'qmcjax_ckpt_{r["step"]:06d}.npz'
        with load_model(folder,path) as (net,params,data,info):
            with np.load(args.analysis/r['id']/'fresh_measurement.npz',allow_pickle=False) as z:
                x=z['positions'];s=z['spins']
            N=s.shape[-1];x=x.reshape(-1,N,2);C=4
            evaluate=make_network_evaluator(net,params,jnp.asarray(data['atoms'][0]),jnp.asarray(data['charges'][0]),'angle')
            ratios=c3_local_ratios(evaluate,x,s,info['lattice']);value=c3_expectation(ratios)
            chain_mean=ratios.reshape(-1,C).mean(axis=0)
            sem_real=float(chain_mean.real.std(ddof=1)/np.sqrt(C));sem_imag=float(chain_mean.imag.std(ddof=1)/np.sqrt(C))
            cfg=info['config'];kw=cfg['system']['make_local_energy_kwargs'];L=info['lattice']
            radius=kw.get('convergence_radius',5) # declared lambda_jellium.local_energy default
            key=json.dumps([L.tolist(),kw['potential_kwargs']['interaction_energy_scale'],radius])
            if key not in cache:
                potential=hamiltonian.make_2DCoulomb_potential(jnp.asarray(L),jnp.zeros((1,2)),jnp.zeros(1),
                    radius,kw['potential_kwargs']['interaction_energy_scale'])
                cache[key]=jax.jit(jax.vmap(lambda xx:potential(jnp.zeros((N,1,2)),xx[:,None,:]-xx[None,:,:])))
            v1=np.asarray(cache[key](jnp.asarray(x[:8])));v2=np.asarray(cache[key](jnp.asarray(x[:8]@rotation)))
            ewald_error=float(np.max(abs(v1-v2)))
            result=dict(id=r['id'],expectation_real=float(value['expectation'].real),expectation_imag=float(value['expectation'].imag),
                expectation_magnitude=float(abs(value['expectation'])),chain_sem_real=sem_real,chain_sem_imag=sem_imag,
                nearest_root_real=float(value['nearest_root_candidate'].real),nearest_root_imag=float(value['nearest_root_candidate'].imag),
                eigenstate_candidate_residual=value['candidate_eigenstate_residual'],ewald_C3_max_difference=ewald_error,ewald_radius=radius,
                interpretation='raw C3 expectation/residual at zero twist, short MC; neither a certified eigenvalue nor Chern/phase label')
            results.append(result);np.savez_compressed(out/(r['id']+'.npz'),c3_local_ratios=ratios)
            print(r['id'],result['expectation_magnitude'],result['eigenstate_candidate_residual'],flush=True)
    (out/'summary.json').write_text(json.dumps(json_safe(dict(results=results,phase_status='unverified',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())),indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('benchmark',type=Path);p.add_argument('analysis',type=Path)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
