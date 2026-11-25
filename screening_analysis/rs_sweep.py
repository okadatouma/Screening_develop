"""Prepare and execute rs sweeps. Never submits scheduler jobs.

prepare --config CONFIG --output NEW_ROOT
run PLAN --index INTEGER [--dry-run]
analyze PLAN --index INTEGER [--output NEW_DIR]
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OWNED={'PW_OUTPUT_DIR','PW_LJ_HF','PW_SEED','PW_SEED_FROM_TAG','PW_TOPO_MODE','PW_TOPO_GAMMA',
       'PW_INIT_LOCALIZE','PW_DELTA_MULT','PW_COULOMB'}


def number(value):
    return format(float(value),'.12g').replace('-','m').replace('.','p')


def write_conditions(plan,output):
    """Human-readable conditions alongside results; does not create run folders."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=plan['config']
    lines=['# 計算条件', '', '相の名前は初期化・想定する候補です。学習後の物理相は未確認です。', '',
           '| Array index | ケース | λ | rs | 電子数 | seed | 初期化 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    names={'plane':'液体候補（平面波）','ahc':'AHC候補（HF）','wc':'WC候補（HF）'}
    for case in plan['cases']:
        index=next(i for i,c in enumerate(plan.get('all_cases',plan['cases'])) if c['id']==case['id'])
        lines.append(f"| {index} | {case['id']} | {case['lam']:g} | {case['rs']:g} | {case['n']} | {case['seed']} | {names[case['initialization']]} |")
    lines+=['',f"学習ステップ数：{config['iterations']}。",'',
            '解析：PH、Windowの幅、Alphaの距離依存性、Attention固有モード、接続エントロピー、接続に対する波動関数の振幅・位相感度。',
            '', '保存walkerと、モデル間で共通の固定入力配置を分けて比較します。',
            '', 'PHは学習後の解析です。PHによる学習正則化は使用しません。',
            '', '## 学習設定', '', '```json',json.dumps(config['training_environment'],indent=2,ensure_ascii=False),'```',
            '', '## HF設定', '', '```json',json.dumps(config['hf'],indent=2,ensure_ascii=False),'```',
            '', '## 解析設定', '', '```json',json.dumps(config['analysis'],indent=2,ensure_ascii=False),'```',
            '',f"結果の保存先：`{plan['root']}`",'']
    (output/'CONDITIONS.md').write_text('\n'.join(lines),encoding='utf-8')


def cases_from_config(config):
    n=config['n']
    if type(n) is not int or n<4 or n>64 or math.isqrt(n)**2!=n:
        raise ValueError('n must be a square integer, 4<=n<=64 (reference PH backend limit)')
    hf=config['hf']
    if not np.isfinite(hf['gcut']) or hf['gcut']<1 or not 0<hf['mix']<=1 or not np.isfinite(hf['tolerance']) or hf['tolerance']<=0:
        raise ValueError('HF requires gcut>=1, 0<mix<=1 and finite tolerance>0')
    if type(hf['max_iterations']) is not int or hf['max_iterations']<1:raise ValueError('positive HF max_iterations')
    if not config['rs'] or not config['seeds'] or not config['branches']:
        raise ValueError('nonempty rs, seeds and branches required')
    if len(set(config['rs']))!=len(config['rs']) or len(set(config['seeds']))!=len(config['seeds']):
        raise ValueError('duplicate rs/seeds')
    if any(not np.isfinite(r) or r<=0 for r in config['rs']):raise ValueError('positive finite rs required')
    if any(type(s) is not int or not 0<=s<2**31 for s in config['seeds']):raise ValueError('integer seed in [0,2^31) required')
    if type(config['iterations']) is not int or config['iterations']<1:raise ValueError('positive iterations required')
    env=config.get('training_environment',{})
    if OWNED.intersection(env):raise ValueError('sweep-owned environment keys: '+str(sorted(OWNED.intersection(env))))
    if any(not k.startswith('PW_') or not isinstance(v,str) for k,v in env.items()):raise ValueError('training_environment must map PW_* names to strings')
    if env.get('PW_ENV','0') not in ('0','false','') or env.get('PW_FIXV','0') not in ('0','false',''):
        raise ValueError('this sweep uses learned Value and monotone Window; PW_ENV/PW_FIXV unsupported')
    if int(env.get('PW_HEADS','2'))<1 or int(env.get('PW_LAYERS','1'))<1:raise ValueError('positive heads/layers')
    from periodicwave.attention import WINDOW_SHAPES
    shape=env.get('PW_WIN_SHAPE','cos')
    if shape not in WINDOW_SHAPES:raise ValueError('unsupported PW_WIN_SHAPE')
    power=int(env.get('PW_WIN_POWER','4'));eta=float(env.get('PW_WIN_ETA','.1'))
    fixed=env.get('PW_FREEZE_ETA','0').lower() not in ('0','','false')
    if power<2 or power%2:raise ValueError('PW_WIN_POWER must be an even integer >= 2')
    if not np.isfinite(eta) or not 0<=eta<=1:raise ValueError('PW_WIN_ETA must be in [0,1]')
    if shape=='hybrid' and not fixed and not 0<eta<1:raise ValueError('learned Hybrid eta must start inside (0,1)')
    analysis=config['analysis']
    for key in ('native_samples','common_samples','ph_samples','phase_samples','null_repeats'):
        if type(analysis[key]) is not int or analysis[key]<1:raise ValueError('positive integer analysis '+key)
    research=analysis.get('research',{})
    if research.get('enabled',False):
        for key in ('samples','repeats','bins'):
            if type(research.get(key)) is not int or research[key]<1:raise ValueError('positive research '+key+' required')
        if not research.get('ks') or any(type(k) is not int or k<1 for k in research['ks']):raise ValueError('positive research ks required')
        if type(research.get('energy_samples',0)) is not int or research.get('energy_samples',0)<0:raise ValueError('nonnegative research energy_samples')
    labels=set();cases=[]
    for branch in config['branches']:
        label=branch['label']
        if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in label) or label in labels:
            raise ValueError('unique simple branch labels required')
        labels.add(label)
        if not np.isfinite(branch['lam']) or branch['lam']<0:raise ValueError('finite nonnegative lambda')
        if branch['initialization'] not in ('plane','wc','ahc'):raise ValueError('initialization must be plane/wc/ahc')
        if branch['initialization']=='ahc' and not branch.get('hf_reference'):raise ValueError('AHC branch requires an explicit HF reference seed table')
        branch_rs=branch.get('rs',config['rs'])
        if not branch_rs or len(set(branch_rs))!=len(branch_rs) or any(r not in config['rs'] for r in branch_rs):
            raise ValueError('branch rs must be a nonempty unique subset of config rs')
        for rs in branch_rs:
            for seed in config['seeds']:
                cases.append(dict(id=f'{label}-lam{number(branch["lam"])}-rs{number(rs)}-seed{seed}',
                    n=n,rs=float(rs),lam=float(branch['lam']),seed=seed,branch=label,
                    initialization=branch['initialization'],hf_reference=branch.get('hf_reference'),
                    phase='unknown',phase_verified=False))
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('case id collision')
    return cases


def shared_bank(n,count,seed=41031):
    from periodicwave.pbc.lj_hf import geometry
    side=math.isqrt(n);lattice=geometry(side)[1]*side
    rng=np.random.default_rng(seed)
    positions=(rng.uniform(size=(count,n,2))@lattice.T).reshape(count,-1)
    spins=rng.choice([-1.,1.],size=(count,n))
    spins[::2]=1 # Include fully polarized inputs as well as mixed pseudospins.
    return dict(positions=positions,spins=spins,atoms=np.zeros((count,1,2)),charges=np.zeros((count,1)),lattice=lattice)


def prepare(config_path,output,reference_inputs=None):
    config=json.loads(config_path.read_text());cases=cases_from_config(config)
    for case in cases:
        if case['hf_reference']:
            p=Path(case['hf_reference']);p=p if p.is_absolute() else ROOT/p
            if not p.is_file():raise FileNotFoundError(p)
            from periodicwave.pbc.lj_hf import geometry,gvectors,kgrid
            _,_,B=geometry(math.isqrt(config['n']))
            expected_k=kgrid(B,math.isqrt(config['n']))[2]
            expected_g=gvectors(B,config['hf']['gcut'])@B.T
            table=np.loadtxt(p)
            if table.ndim!=2 or table.shape[1]!=6 or not np.isfinite(table).all():raise ValueError('finite six-column HF seed table required')
            observed_k=np.unique(table[:,:2],axis=0);observed_g=np.unique(table[:,2:4],axis=0)
            ordered=lambda a:a[np.lexsort((a[:,1],a[:,0]))]
            if observed_k.shape!=expected_k.shape or observed_g.shape!=expected_g.shape or not np.allclose(ordered(observed_k),ordered(expected_k),atol=1e-10,rtol=0) or not np.allclose(ordered(observed_g),ordered(expected_g),atol=1e-10,rtol=0):
                raise ValueError('HF seed table does not match requested N/gcut; provide a compatible reference')
            case['hf_reference']=str(p.resolve())
            case['hf_reference_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    bank=shared_bank(config['n'],config['analysis']['common_samples'])
    if reference_inputs is not None:
        with np.load(reference_inputs,allow_pickle=False) as z:
            for key in bank:
                if key not in z:raise ValueError('reference input missing '+key)
                bank[key]=z[key]
        n=config['n'];count=len(bank['positions'])
        if bank['positions'].shape!=(count,2*n) or bank['spins'].shape!=(count,n) or count<1:
            raise ValueError('reference coordinate/spin shapes differ')
        if not np.allclose(bank['lattice'],shared_bank(n,1)['lattice'],rtol=0,atol=1e-12):raise ValueError('reference lattice differs')
    count=len(bank['positions']);n=config['n']
    if any(not np.isfinite(v).all() for v in bank.values()) or not np.isin(bank['spins'],[-1,1]).all():
        raise ValueError('finite common inputs and pseudospins +/-1 required')
    if bank['atoms'].shape!=(count,1,2) or bank['charges'].shape!=(count,1):
        raise ValueError('jellium shared atoms/charges require shapes [B,1,2] and [B,1]')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'common-inputs.npz',**bank)
    plan=dict(schema='rs-sweep-v1',config=config,cases=cases,root=str(output),
        common_inputs=str(output/'common-inputs.npz'),common_inputs_sha256=hashlib.sha256((output/'common-inputs.npz').read_bytes()).hexdigest(),
        common_input_origin=str(reference_inputs.resolve()) if reference_inputs else 'fixed synthetic uniform positions; alternating all-up and random pseudospins; not equilibrium samples',
        repository=str(ROOT),phase_policy='lambda and initialization are candidates only; no trained phase labels inferred')
    (output/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False,allow_nan=False))
    write_conditions(plan,output)
    write_conditions(plan,ROOT/'results'/'manual'/output.name)
    command=['sbatch',f'--array=0-{len(cases)-1}%2',f'--export=ALL,SWEEP_PLAN={output / "plan.json"}',str(ROOT/'rs_sweep.job')]
    print(f'{len(cases)} cases; no jobs submitted\n'+shlex.join(command))
    return plan


def prepare_reference(case,settings,output):
    """SCF at the requested rs, or analytic liquid seed; never relabel trained phases."""
    from periodicwave.pbc.lj_hf import LJHF
    from periodicwave.lj_orbitals import load_hf_table
    output.mkdir(parents=True,exist_ok=False)
    side=math.isqrt(case['n']);hf=LJHF(case['lam'],case['rs'],L=side,gcut=settings['gcut'])
    if case['initialization']=='plane':
        # N lowest supercell momenta, no forced one-electron-per-crystal-cell occupancy.
        from itertools import product
        modes=np.array(list(product(range(-side,side+1),repeat=2)))
        momenta=2*np.pi*modes@np.linalg.inv(hf.A*side)
        order=np.lexsort((modes[:,1],modes[:,0],np.sum(momenta**2,axis=1)))[:case['n']]
        table=np.column_stack((momenta[order],np.zeros((case['n'],2)),np.ones(case['n']),np.zeros(case['n'])))
        info=dict(kind='analytic lower-band plane-wave seed, NOT SCF or verified liquid',converged=None,
                  open_shell_ties_possible=True,phase='unknown')
    else:
        if case['initialization']=='wc':
            initial=hf.seed_wc()
        else:
            path=Path(case['hf_reference'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=case['hf_reference_sha256']:raise ValueError('HF reference changed since prepare')
            k,g,c=load_hf_table(str(path))
            if k.shape!=hf.k.shape or g.shape!=hf.G.shape or not np.allclose(k,hf.k,atol=1e-10) or not np.allclose(g,hf.G,atol=1e-10):
                raise ValueError('AHC seed table k/G basis differs from configured N/gcut')
            initial=np.einsum('ka,kb->kab',c,c.conj())
        solved=hf.solve(initial,maxit=settings['max_iterations'],mix=settings['mix'],tol=settings['tolerance'])
        hH,hF=hf.hartree(solved['D']),hf.fock(solved['D'])
        H=hH+hF;H[:,np.arange(hf.NG),np.arange(hf.NG)]+=hf.eps
        _,v=np.linalg.eigh((H+H.conj().transpose(0,2,1))/2)
        fresh=np.einsum('ka,kb->kab',v[:,:,0],v[:,:,0].conj())
        residual=float(np.max(abs(fresh-solved['D'])))
        info=dict(kind='SCF at requested rs; branch is seed provenance',converged=residual<settings['tolerance'],
            residual=residual,iterations=len(solved['hist']),energy=solved['E'],
            hf_chern_diagnostic=float(hf.chern(solved['c'])),density_fourier=float(hf.density_fourier(solved['c'])),
            phase='unknown',note='HF diagnostics do not certify the trained many-body phase')
        if not info['converged']:
            (output/'failed.json').write_text(json.dumps(info,indent=2))
            raise RuntimeError('HF did not converge at requested rs; see '+str(output/'failed.json'))
        table=hf.orbitals_text(solved['c'])
    path=output/'orbitals.txt';np.savetxt(path,table,fmt='%.17g')
    info.update(n=case['n'],lam=case['lam'],rs=case['rs'],gcut=settings['gcut'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output/'metadata.json').write_text(json.dumps(info,indent=2,allow_nan=False))
    return path


def read_plan(path,index):
    plan=json.loads(path.read_text())
    if not 0<=index<len(plan['cases']):raise ValueError('case index out of range')
    common=Path(plan['common_inputs'])
    if hashlib.sha256(common.read_bytes()).hexdigest()!=plan['common_inputs_sha256']:raise ValueError('common inputs changed')
    return plan,plan['cases'][index]


def run_case(plan,case,dry_run=False):
    config=plan['config'];folder=Path(plan['root'])/case['id']
    env={k:v for k,v in os.environ.items() if not k.startswith('PW_')}
    env.update(config['training_environment'])
    env.update(PW_OUTPUT_DIR=str(folder/'training'),PW_LJ_HF=str(folder/'reference/orbitals.txt'),
        PW_INIT_LOCALIZE='0' if case['initialization']=='plane' else '1',PW_SEED=str(case['seed']),
        PW_SEED_FROM_TAG='0',PW_TOPO_MODE='off',PW_TOPO_GAMMA='0',PW_COULOMB='1',PW_DELTA_MULT='1')
    command=[sys.executable,'-u','-m','periodicwave.configs.run_lambda_jellium',str(case['n']),str(case['rs']),str(case['lam']),str(config['iterations']),case['id']]
    if dry_run:
        print(json.dumps(dict(case=case,environment={k:v for k,v in env.items() if k.startswith('PW_')},command=command),indent=2));return
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'case.json').write_text(json.dumps(case,indent=2))
    case_plan=dict(plan,cases=[case],all_cases=plan['cases'])
    write_conditions(case_plan,folder)
    write_conditions(case_plan,ROOT/'results'/'manual'/Path(plan['root']).name/case['id'])
    prepare_reference(case,config['hf'],folder/'reference')
    with (folder/'training.log').open('x') as f:
        subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    analyze_case(plan,case,folder/'analysis')


def analyze_case(plan,case,output):
    from screening_analysis.rs_observables import analyze_checkpoint
    folder=Path(plan['root'])/case['id']
    analyze_checkpoint(folder/'training',output,Path(plan['common_inputs']),plan['config']['analysis'],case)
    write_conditions(dict(plan,cases=[case],all_cases=plan['cases']),output)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prepare_cmd=sub.add_parser('prepare');prepare_cmd.add_argument('--config',type=Path,required=True)
    prepare_cmd.add_argument('--output',type=Path,required=True);prepare_cmd.add_argument('--reference-inputs',type=Path)
    for name in ('run','analyze'):
        cmd=sub.add_parser(name);cmd.add_argument('plan',type=Path);cmd.add_argument('--index',type=int,required=True)
        if name=='run':cmd.add_argument('--dry-run',action='store_true')
        else:cmd.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.config,args.output,args.reference_inputs)
    else:
        plan,case=read_plan(args.plan,args.index)
        if args.command=='run':run_case(plan,case,args.dry_run)
        else:analyze_case(plan,case,args.output or Path(plan['root'])/case['id']/'analysis-recomputed')


if __name__=='__main__':main()
