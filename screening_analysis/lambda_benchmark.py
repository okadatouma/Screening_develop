"""Fresh-output CPU benchmark through the EXISTING run_lambda_jellium entry point.

Candidate lambdas are NOT verified phase labels. Core: 3 seeds x 2 lambdas x
baseline/PH/spectral/weight smoothing. Stress jobs vary actual model/initial state.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def plan(stress=True):
    cases=[]
    for lam in (.33,.72):
        for mode in ('off','persistence','laplacian','weights'):
            for seed in (11,23,37):
                cases.append(dict(id=f'N9-lam{str(lam).replace(".","")}-{mode}-seed{seed}',n=9,rs=15.,lam=lam,seed=seed,
                    mode=mode,gamma=0. if mode=='off' else .1,iterations=24,kind='core',initial_wavefunction='random_screenformer',overrides={}))
    if stress:
        changes=dict(width_short={'PW_W_LO':'.6','PW_W_HI':'1.2'},width_wide={'PW_W_LO':'2.2','PW_W_HI':'3.8'},
            smooth7={'PW_WIN_SHAPE':'smooth7'},gauss_ring={'PW_ENV':'1','PW_ENV_KERNEL':'gauss'},
            trim4={'PW_TRIM_POW':'4'},heads4={'PW_HEADS':'4'},layers2={'PW_LAYERS':'2'},N4={})
        for name,overrides in changes.items():cases.append(dict(id='stress-'+name,n=4 if name=='N4' else 9,rs=15.,lam=.33,seed=11,
                mode='off',gamma=0.,iterations=12,kind='stress',factor=name,initial_wavefunction='random_screenformer',overrides=overrides))
        for lam in (.33,.72):cases.append(dict(id=f'plane-reference-lam{lam}',n=9,rs=15.,lam=lam,seed=11,
            mode='off',gamma=0.,iterations=12,kind='initial_wavefunction',initial_wavefunction='lower_band_plane_wave_reference_NOT_self_consistent_HF',overrides={}))
        cases.append(dict(id='liquid-candidate-lam0',n=9,rs=15.,lam=0.,seed=11,mode='off',gamma=0.,iterations=12,
            kind='out_of_domain',initial_wavefunction='random_screenformer',overrides={}))
    for case in cases:
        case.update(phase='unknown',phase_verified=False,seed_group=f'seed{case["seed"]}',candidate_parameter_label=f'lambda={case["lam"]:g}')
    return cases


def reference_table(out):
    """Known plane-wave Slater in the existing 6-column format; NOT an HF solution."""
    from periodicwave.pbc import lattices
    L,_=lattices._triangular_lattice_vecs_periodic_potential(np.sqrt(2*np.pi/np.sqrt(3)),9)
    modes=np.array([(i,j) for i in (-1,0,1) for j in (-1,0,1)])
    momenta=2*np.pi*modes@np.linalg.inv(np.asarray(L))
    table=np.column_stack((momenta,np.zeros((9,2)),np.ones(9),np.zeros(9)))
    path=out/'lower_band_plane_wave_reference.txt'
    with path.open('x') as f:np.savetxt(f,table,fmt='%.17g')
    (out/'reference_origin.json').write_text(json.dumps(dict(kind='analytic lower-band plane-wave Slater reference',
        self_consistent_HF=False,physical_phase='unverified',integer_momenta=modes.tolist(),lattice=np.asarray(L).tolist(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
    return path


def summarize(case,folder):
    file=folder/'train_stats.csv'
    if not file.exists():return dict(status='missing_stats')
    with file.open() as f:rows=list(csv.DictReader(f))
    energy=np.array([float(r['energy']) for r in rows]);finite=np.isfinite(energy)
    cfg=json.loads((folder/'runtime_config.json').read_text())
    expected=folder/f'qmcjax_ckpt_{case["iterations"]-1:06d}.npz'
    ok=len(rows)==case['iterations'] and finite.all() and expected.exists()
    return dict(status='complete_short_unconverged' if ok else 'incomplete_or_nonfinite',steps=len(rows),
        nonfinite_batch_means=int((~finite).sum()),training_final_energy=float(energy[-1]) if len(rows) and finite[-1] else None,
        mean_final4_batches=float(energy[-4:].mean()) if len(rows)>=4 and finite[-4:].all() else None,
        mean_final4_local_variance=float(np.mean([float(r['locstd'])**2 for r in rows[-4:]])) if len(rows)>=4 else None,
        checkpoint=str(expected),config=cfg,independent_phase_status='UNVERIFIED',
        note='training batches, not a post-equilibration phase/energy convergence certificate')


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=args.reuse_complete)
    cases=plan(not args.skip_stress)
    with (out/'plan.json').open('x') as f:json.dump(dict(cases=cases,
        unavailable=[dict(condition='same-Hamiltonian validated WC-HF vs AHC-HF',status='missing verified HF data'),
                     dict(condition='Halo Wigner Crystal',status='supported arbitrary label/schema; no validated reference supplied')],
        physical_labels='none assigned; lambda alone is insufficient'),f,indent=2)
    (out/'logs').mkdir(exist_ok=False)
    reference=reference_table(out) if not args.skip_stress else None
    results=[]
    for case in cases:
        folder=out/case['id'];start=time.perf_counter();reused=False
        env={k:v for k,v in os.environ.items() if not k.startswith('PW_')}
        env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='1',PW_OPTIMIZER='adam',PW_CHECK_NAN='1',
            PW_BATCH='16',PW_WIDTH='16',PW_HEADS='2',PW_ATTN_DIM='4',PW_VALUE_DIM='4',PW_DET='1',PW_LR='.002',
            PW_MCMC_STEPS='2',PW_FLIPS='1',PW_FLIP_BATCH='3',PW_EL_CHUNK='4',PW_BURN_IN='16',PW_SAVE_MIN='0',
            PW_LAPL='default',PW_SEED=str(case['seed']),PW_TOPO_MODE=case['mode'],PW_TOPO_GAMMA=str(case['gamma']),
            PW_TOPO_SAMPLES='4',PW_OUTPUT_DIR=str(folder))
        env.update(case['overrides'])
        if case['initial_wavefunction'].startswith('lower_band'):
            env.update(PW_LJ_HF=str(reference),PW_LJ_JASTROW='0',PW_LJ_NETSCALE='.05')
        if folder.exists():
            if not args.reuse_complete:raise FileExistsError(folder)
            existing=summarize(case,folder)
            cfg=existing.get('config',{});kw=cfg.get('system',{}).get('make_local_energy_kwargs',{})
            if existing['status']!='complete_short_unconverged' or cfg.get('seed')!=case['seed'] or kw.get('lam')!=case['lam']:
                raise ValueError('existing directory is not the declared completed case: '+str(folder))
            reused=True;returncode=0
        else:
            command=[sys.executable,'-m','periodicwave.configs.run_lambda_jellium',str(case['n']),str(case['rs']),str(case['lam']),str(case['iterations']),case['id']]
            with (out/'logs'/(case['id']+'.txt')).open('x') as f:
                try:returncode=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800,check=False).returncode
                except subprocess.TimeoutExpired:returncode=-999
        summary=summarize(case,folder);result=dict(**case,**summary,returncode=returncode,reused_complete=reused,wall_seconds=time.perf_counter()-start)
        with (folder/'benchmark_case.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        results.append(result);print(case['id'],returncode,summary['status'],flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    root=Path(__file__).resolve().parents[1]
    (out/'source.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
        ('periodicwave/configs/run_lambda_jellium.py','periodicwave/train_screen.py','periodicwave/topology_regularization.py',
         'periodicwave/base_config.py','screening_analysis/lambda_benchmark.py')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-complete',action='store_true');p.add_argument('--skip-stress',action='store_true');run(p.parse_args())
