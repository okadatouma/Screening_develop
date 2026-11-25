"""Freeze, submit and inspect the seven remaining production conditions.

CPU Slurm jobs use the complete tested framework environment. N36 production
resumes the SAME config/checkpoint only after a two-update feasibility gate.
Existing research results, production plans and Slurm jobs are never modified.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .paths import ROOT,lattice_source,wannier_source
from .config import validate_config
from .experiments.store import digest,json_save,environment


def _copy_code(source,destination):
    """Source only; never recursively copy results, environments or checkpoints."""
    destination.mkdir(parents=True,exist_ok=False)
    for parent,dirs,files in os.walk(source):
        dirs[:]=[d for d in dirs if d not in ['.venv','results','__pycache__','.pytest_cache','.git','figures'] and 'results' not in d]
        for name in files:
            if not name.endswith(('.py','.md','.yaml','.json','.txt','.toml')):continue
            path=Path(parent)/name;target=destination/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)


def prepare(original_plan,output,python=None):
    original_plan=Path(original_plan).resolve();root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();(root/'jobs').mkdir();(root/'inputs').mkdir()
    original=json.loads(original_plan.read_text());cases=deepcopy(original)
    if len(cases)!=7 or [c['config']['experiment']['task'] for c in cases]!=['lattice_tfim']*6+['continuous_electrons']:
        raise ValueError('expected six TFIM conditions followed by the guarded N36 condition')
    python=Path(python or ROOT/'research_framework/.venv/bin/python').absolute()
    if not python.is_file():raise FileNotFoundError(python)
    source=root/'source';project=source/'Screening_topology_gpu'
    project.mkdir(parents=True)
    for folder in ['research_framework','periodicwave','screening_analysis']:
        _copy_code(ROOT/folder,project/folder)
    _copy_code(lattice_source()/'lattice',source/'Lattice_screening/lattice')
    for folder in ['wannier_ph','vendor']:_copy_code(wannier_source()/folder,source/'wannier_ph'/folder)
    # Retain exact-before factory needed by the regression test, without copying old outputs.
    archive=ROOT/'research_framework/results/implementation_audit/CustomScreenformer_before.py.txt'
    target=project/'research_framework/results/implementation_audit'/archive.name
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(archive,target)
    hf=Path(cases[6]['config']['physics']['hf_path'])
    frozen_hf=root/'inputs'/'N36_HF_orbitals.txt';shutil.copy2(hf,frozen_hf)
    if (hf.parent/'metadata.json').exists():shutil.copy2(hf.parent/'metadata.json',root/'inputs/N36_HF_metadata.json')
    cases[6]['config']['physics']['hf_path']=str(frozen_hf)
    cases[6]['executable']=False
    cases[6]['reason']='requires successful two-update N36 feasibility gate; cluster worker enforces dependency and gate hash'
    for case in cases:validate_config(case['config'])
    json_save(root/'original_plan.json',original)
    json_save(root/'plan.json',cases)
    hashes={str(p.relative_to(root)):digest(p.read_bytes()) for p in sorted(source.rglob('*')) if p.is_file()}
    hashes[str(frozen_hf.relative_to(root))]=digest(frozen_hf.read_bytes())
    json_save(root/'frozen_sha256.json',hashes)
    json_save(root/'cluster_config.json',dict(created=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        original_plan=str(original_plan),original_plan_sha256=digest(original_plan.read_bytes()),
        python=str(python),source=str(project),environment=environment(),
        resources=dict(tfim=dict(cpus=2,memory_gb=16,hours=24,concurrency=2),
                       n36_pilot=dict(cpus=4,memory_gb=96,hours=2,updates=2),
                       n36_production=dict(cpus=4,memory_gb=96,hours=168),
                       collect=dict(cpus=2,memory_gb=8,hours=2)),
        existing_jobs='untouched; no GPU requested to avoid competing with existing GPU queue',
        feasibility_criteria=dict(finite_updates=2,estimated_training_seconds_max=6*24*3600,peak_rss_gb_max=80),
        semantic_changes=['HF table path replaced by immutable byte-identical copy',
                          'TFIM intervention enumeration is batch-streamed and configured VMC burn-in is honored'],
        hamiltonian_changed=False))
    q=shlex.quote
    common=f'''set -euo pipefail
cd {q(str(project))}
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-2}}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RF_LATTICE_SOURCE={q(str(source/'Lattice_screening'))}
export RF_WANNIER_SOURCE={q(str(source/'wannier_ph'))}
export PWT_SOURCE={q(str(project))}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
'''
    def job(name,cpus,mem,hours,action,extra=''):
        text=f'''#!/bin/bash
#SBATCH --job-name=rf-{name}
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --time={hours}:00:00
#SBATCH --output={root}/logs/{name}-%A_%a.out
#SBATCH --error={root}/logs/{name}-%A_%a.err
{extra}{common}
exec {q(str(python))} -u -m research_framework.cluster worker {q(str(root))} {action}
'''
        path=root/'jobs'/f'{name}.sbatch';path.write_text(text);return path
    job('tfim',2,16,24,'tfim --index "$SLURM_ARRAY_TASK_ID"','#SBATCH --array=0-5%2\n')
    job('n36-pilot',4,96,2,'n36-pilot')
    job('n36',4,96,168,'n36')
    job('collect',2,8,2,'collect')
    (root/'README_JA.md').write_text('''# 残り7条件のクラスター実行

元のproduction planは変更せず、sourceとHF入力を凍結した。
TFIMはN=12/16×3seed、各10000更新。arrayは最大2条件を同時実行。
N36は同一設定で2更新の計算負荷・有限性テストを行い、成功後のみ同じcheckpointから10000更新へ継続する。
N36の判定に失敗した場合は本計算を開始せず、失敗ログを保持する。
全計算の終了後に集計jobがcomparison.csvとCLUSTER_STATUS_JA.mdを生成する。
job IDはsubmission.json、標準出力/エラーはlogs/、学習結果は各条件名のディレクトリ。
既存のGPU/Slurmジョブと以前の結果は変更しない。
''')
    return root


def verify(root):
    for name,wanted in json.loads((root/'frozen_sha256.json').read_text()).items():
        if digest((root/name).read_bytes())!=wanted:raise ValueError('frozen source/input changed: '+name)
    expected=json.loads((root/'cluster_config.json').read_text())['environment']['packages']
    current=environment()['packages']
    if expected!=current:raise ValueError('execution environment differs from the recorded package versions')


def worker(root,action,index=None):
    root=Path(root).resolve();verify(root)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('production workers must run inside a Slurm allocation')
    import jax
    jax.config.update('jax_enable_x64',True)
    from .cli import run,collect
    plan=json.loads((root/'plan.json').read_text());case_index=index if action=='tfim' else 6
    if action=='collect':
        results=collect(root)
        lines=['# クラスター計算の状態','',f'完了 {sum(r["status"]=="complete" for r in results)}/{len(results)} 条件。','',
               '| 条件 | 状態 |','|---|---|']+[f'| {r["run"]} | {r["status"]} |' for r in results]
        pilot=root/'n36_feasibility.json'
        if pilot.exists():lines+=['','N36検証: '+json.loads(pilot.read_text())['status']]
        (root/'CLUSTER_STATUS_JA.md').write_text('\n'.join(lines)+'\n');return
    if case_index is None or not 0<=case_index<len(plan):raise ValueError('case index out of range')
    case=plan[case_index];folder=root/case['name'];config=case['config']
    if action=='tfim' and config['experiment']['task']!='lattice_tfim':raise ValueError('TFIM array index mismatch')
    if action=='n36':
        gate=json.loads((root/'n36_feasibility.json').read_text())
        if gate['status']!='passed' or gate['config_sha256']!=digest(json.dumps(config,sort_keys=True).encode()):raise ValueError('N36 feasibility gate not passed for this config')
        run(config,folder,resume=True);return
    if action=='n36-pilot':
        started=time.monotonic()
        try:
            run(config,folder,resume=(folder/'latest.json').exists(),until=2)
            history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
            actual=[r for r in history if r['step']==2]
            if len(actual)!=1:raise ArithmeticError('pilot did not complete the requested two finite updates')
            seconds=float(actual[0]['update_seconds'])
            expected=seconds*config['training']['steps']
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
            criteria=json.loads((root/'cluster_config.json').read_text())['feasibility_criteria']
            if expected>criteria['estimated_training_seconds_max'] or peak>criteria['peak_rss_gb_max']:
                raise ArithmeticError(f'N36 exceeds planned resources: estimated_seconds={expected}, peak_rss_gb={peak}')
            json_save(root/'n36_feasibility.json',dict(status='passed',updates=2,steady_update_seconds=seconds,
                estimated_training_seconds=expected,peak_rss_gb=peak,total_pilot_seconds=time.monotonic()-started,
                config_sha256=digest(json.dumps(config,sort_keys=True).encode()),slurm_job_id=os.environ['SLURM_JOB_ID']))
        except Exception as error:
            if not (root/'n36_feasibility.json').exists():json_save(root/'n36_feasibility.json',dict(status='failed',error=repr(error),total_pilot_seconds=time.monotonic()-started))
            raise
        return
    run(config,folder,resume=(folder/'latest.json').exists())


def submit(root):
    root=Path(root).resolve();verify(root)
    lock=root/'submission.lock'
    # Exclusive marker prevents accidentally submitting a second copy of the same study.
    with lock.open('x') as f:f.write(datetime.now(ZoneInfo('Asia/Tokyo')).isoformat())
    submitted={};commands=[]
    def add(key,script,arguments=()):
        command=['sbatch','--parsable',*arguments,str(root/'jobs'/script)]
        result=subprocess.run(command,capture_output=True,text=True,check=True)
        job_id=result.stdout.strip().split(';')[0]
        if not job_id.isdigit():raise ValueError('unexpected sbatch output: '+result.stdout)
        submitted[key]=job_id;commands.append(command)
        json_save(root/'submission_progress.json',dict(jobs=submitted,commands=commands),replace=(root/'submission_progress.json').exists())
        return job_id
    try:
        tfim=add('tfim_array','tfim.sbatch')
        pilot=add('n36_pilot','n36-pilot.sbatch')
        full=add('n36_production','n36.sbatch',[f'--dependency=afterok:{pilot}','--kill-on-invalid-dep=yes'])
        add('collect','collect.sbatch',[f'--dependency=afterany:{tfim}:{full}','--kill-on-invalid-dep=yes'])
        json_save(root/'submission.json',dict(submitted=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),jobs=submitted,commands=commands))
    except Exception as error:
        json_save(root/'submission_error.json',dict(error=repr(error),already_submitted=submitted))
        raise
    return submitted


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('plan',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('submit');p.add_argument('root',type=Path)
    p=sub.add_parser('worker');p.add_argument('root',type=Path);p.add_argument('action',choices=['tfim','n36-pilot','n36','collect']);p.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.command=='prepare':print(prepare(args.plan,args.output))
    elif args.command=='submit':print(json.dumps(submit(args.root),indent=2))
    else:worker(args.root,args.action,args.index)


if __name__=='__main__':main()
