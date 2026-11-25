"""Immutable 18-condition N36 pilot -> matched-state gate -> production array."""
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import shlex
import shutil
import subprocess
import sys
import numpy as np
from ..paths import ROOT,lattice_source,wannier_source
from ..cluster import _copy_code
from ..experiments.store import json_save,digest,RunStore
from .continuous import cases,MatchedHFTask,initial_evidence,candidate_response


def prepare(root,steps=10000):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    for d in ['inputs','logs','jobs']:(root/d).mkdir()
    source=root/'source';project=source/ROOT.name
    _copy_code(ROOT,project)
    _copy_code(lattice_source()/'lattice',source/'Lattice_screening/lattice')
    _copy_code(wannier_source()/'wannier_ph',source/'wannier_ph/wannier_ph')
    _copy_code(wannier_source()/'vendor',source/'wannier_ph/vendor')
    prior=ROOT/'research_framework/results/2026-11-06_cluster'
    base=json.loads((prior/'plan.json').read_text())[-1]['config']
    hf=root/'inputs/N36_HF_orbitals.txt';shutil.copy2(base['physics']['hf_path'],hf)
    shutil.copy2(prior/'inputs/N36_HF_metadata.json',root/'inputs/N36_HF_metadata.json')
    base['physics']['hf_path']=str(hf)
    plan=cases(base,steps)
    json_save(root/'plan.json',plan)
    hashes={str(p.relative_to(root)):digest(p.read_bytes()) for p in source.rglob('*') if p.is_file()}
    hashes['inputs/N36_HF_orbitals.txt']=digest(hf.read_bytes())
    hashes['plan.json']=digest((root/'plan.json').read_bytes())
    json_save(root/'frozen_sha256.json',hashes)
    json_save(root/'protocol.json',dict(steps=steps,seeds=[101,202,303],n=36,conditions=18,
        supplied='same converged external HF orbitals, Hamiltonian, cell; initial NN orbital readout zero; no Jastrow',
        learned='NN correction, embedding projections/gains, attention; crystal modes explicitly supplied to six_theta',
        pilot_updates=2,gate='all finite; same physical initial amplitudes, phase, post-burn walkers and first-step energies per seed',
        evaluation='fresh Born chains; existing QK/spin/phase diagnostics; individual Fourier candidate removal and resampling',
        nominal_capacity='padded 20; same nominal parameter count, distinct active function classes',
        sparse_penalty=.01,penalty_selection='predeclared, no validation-based tuning',
        resources='CPU4,96GB, array concurrency2; existing jobs untouched'))
    q=shlex.quote;python=ROOT/'research_framework/.venv/bin/python'
    common=f'''set -euo pipefail
cd {q(str(project))}
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RF_LATTICE_SOURCE={q(str(source/'Lattice_screening'))}
export RF_WANNIER_SOURCE={q(str(source/'wannier_ph'))}
export PWT_SOURCE={q(str(project))}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
'''
    for action,hours,cpus,mem in [('pilot',2,4,96),('gate',1,1,4),('train',168,4,96),('collect',2,2,8)]:
        array='#SBATCH --array=0-17%2\n' if action in ['pilot','train'] else ''
        script=f'''#!/bin/bash
#SBATCH --job-name=rf-follow-{action}
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --time={hours}:00:00
#SBATCH --output={root}/logs/{action}-%A_%a.out
#SBATCH --error={root}/logs/{action}-%A_%a.err
{array}{common}
{q(str(python))} -m research_framework.followup.cluster worker --root {q(str(root))} --action {action}
'''
        (root/'jobs'/f'{action}.sbatch').write_text(script)
    return root


def verify(root):
    for p,h in json.loads((root/'frozen_sha256.json').read_text()).items():
        if digest((root/p).read_bytes())!=h:raise ValueError(f'frozen source/input changed: {p}')


def submit(root):
    root=Path(root).resolve();verify(root)
    with (root/'submission.lock').open('x') as f:f.write('exclusive submission\n')
    jobs={}
    for action,dep in [('pilot',None),('gate','pilot'),('train','gate'),('collect','train')]:
        args=['sbatch','--parsable']
        if dep:args.append(f'--dependency={"afterany" if action=="collect" else "afterok"}:{jobs[dep]}')
        args.append(str(root/'jobs'/f'{action}.sbatch'))
        job=subprocess.run(args,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
        jobs[action]=job
        json_save(root/f'submitted_{action}.json',dict(job=job,command=args))
    json_save(root/'submission.json',jobs);return jobs


def gate(root,plan):
    evidence=[]
    for case in plan:
        folder=root/case['name'];v=json.loads((folder/'initial_state.json').read_text())
        rows=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
        if len(rows)!=2 or not all(np.isfinite(r['energy']) and np.isfinite(r['gradient_norm']) for r in rows):raise ValueError('pilot must have two finite updates')
        v.update(seed=case['config']['experiment']['seed'],first_energy=rows[0]['energy'],name=case['name'])
        evidence.append(v)
    for seed in [101,202,303]:
        group=[x for x in evidence if x['seed']==seed];ref=group[0]
        for v in group:
            for k in ['phase','logabs','first_energy']:
                if not np.allclose(v[k],ref[k],rtol=1e-10,atol=1e-9):raise ValueError(f'initial physical state mismatch {seed} {k}')
            for k in ['position_sha256','burned_walker_sha256','parameter_count']:
                if v[k]!=ref[k]:raise ValueError(f'unmatched {k}')
    json_save(root/'PILOT_GATE_PASSED.json',dict(passed=True,conditions=evidence))


def worker(root,action):
    root=Path(root).resolve();verify(root);plan=json.loads((root/'plan.json').read_text())
    if action=='gate':gate(root,plan);return
    if action=='collect':
        from .report import report
        report(root.parent,root);return
    index=int(os.environ['SLURM_ARRAY_TASK_ID']);case=plan[index];c=case['config']
    if action=='train' and not (root/'PILOT_GATE_PASSED.json').exists():raise RuntimeError('matched-state pilot gate missing')
    store=RunStore(root/case['name'],c,resume=action=='train');task=MatchedHFTask(c)
    try:
        if action=='pilot':
            task.build(store.path);v=initial_evidence(task,c['experiment']['seed'])
            task.run(store,until=2)
            with (store.path/'checkpoints/step_0000000.pkl').open('rb') as f:state=pickle.load(f)['state']
            v['burned_walker_sha256']=digest(np.asarray(state['walkers'].positions).tobytes()+np.asarray(state['walkers'].spins).tobytes())
            json_save(store.path/'initial_state.json',v)
        else:
            task.run(store,resume=True)
            if task.wave.embedding.learned:
                _,state=store.restore();candidate_response(task,store,state['params'])
            json_save(store.path/'FOLLOWUP_COMPLETE.json',dict(candidate_response=task.wave.embedding.learned))
    except Exception as error:store.fail(error);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','submit','worker']);p.add_argument('--root',required=True)
    p.add_argument('--action',choices=['pilot','gate','train','collect']);p.add_argument('--steps',type=int,default=10000)
    a=p.parse_args()
    if a.command=='prepare':print(prepare(a.root,a.steps))
    elif a.command=='submit':print(submit(a.root))
    else:worker(a.root,a.action)
