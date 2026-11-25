"""python -m research_framework.cli {validate,plan,run,case,suite,collect,report}."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from .config import load_config,validate_config
from .factory import make_task
from .experiments.store import RunStore,json_save,csv_save,environment
from .experiments.retries import resolve,reserve


def run(config,output,resume=False,until=None):
    c=validate_config(config)
    store=RunStore(output,c,resume)
    if (store.path/'COMPLETE.json').exists():return json.loads((store.path/'summary.json').read_text())
    try:return make_task(c).run(store,resume,until)
    except Exception as error:store.fail(error);raise


def collect(root):
    root=Path(root);cases=json.loads((root/'plan.json').read_text());rows=[]
    for case in cases:
        path=resolve(root,case['name']);summary=path/'summary.json'
        row=dict(run=case['name'],group=case['group'],task=case['config']['experiment']['task'],seed=case['config']['experiment']['seed'],
                 folder=path.name,original_attempt_retained=path.name!=case['name'],
                 status='complete' if (path/'COMPLETE.json').exists() else ('failed_or_partial' if path.exists() else 'not_run'))
        if summary.exists():
            s=json.loads(summary.read_text());row.update({k:v for k,v in s.items() if isinstance(v,(str,float,int,bool)) or v is None})
        rows.append(row)
    csv_save(root/'comparison.csv',rows,replace=True)
    json_save(root/'collection.json',dict(runs=len(rows),complete=sum(r['status']=='complete' for r in rows),rows=rows),replace=True)
    return rows


def main():
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    import jax
    jax.config.update('jax_enable_x64',True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('validate');p.add_argument('config',type=Path)
    p=sub.add_parser('plan');p.add_argument('output',type=Path);p.add_argument('--tier',choices=['pilot','production'],default='pilot')
    p=sub.add_parser('run');p.add_argument('config',type=Path);p.add_argument('output',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--until',type=int)
    p=sub.add_parser('case');p.add_argument('plan',type=Path);p.add_argument('index',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--retry',action='store_true');p.add_argument('--until',type=int)
    p=sub.add_parser('suite');p.add_argument('plan',type=Path);p.add_argument('--task',choices=['continuous_electrons','lattice_tfim','qwz_topology']);p.add_argument('--timeout',type=int,default=600);p.add_argument('--resume',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p=sub.add_parser('collect');p.add_argument('root',type=Path)
    p=sub.add_parser('report');p.add_argument('root',type=Path);p.add_argument('--supplement',type=Path)
    args=parser.parse_args()
    if args.command=='validate':print(json.dumps(load_config(args.config),indent=2));return
    if args.command=='plan':
        from .experiments.plans import make_plan
        print(len(make_plan(args.output,args.tier)));return
    if args.command=='run':run(load_config(args.config),args.output,args.resume,args.until);return
    if args.command in ['case','suite']:
        cases=json.loads(args.plan.read_text());root=args.plan.parent
        if args.command=='case':
            if not 0<=args.index<len(cases):raise ValueError('case index out of range')
            case=cases[args.index]
            if case.get('executable') is False:raise ValueError(case['reason'])
            if args.resume and args.retry:raise ValueError('choose --resume (same code/state) or --retry (new source attempt)')
            target=reserve(root,case) if args.retry else resolve(root,case['name'])
            run(case['config'],target,args.resume,args.until);return
        (root/'logs').mkdir(exist_ok=True)
        for index,case in enumerate(cases):
            if args.task and case['config']['experiment']['task']!=args.task:continue
            output=resolve(root,case['name'])
            if (output/'COMPLETE.json').exists():print('already complete',case['name'],flush=True);continue
            if case.get('executable') is False:print('non-executable template',case['name'],flush=True);continue
            retry=args.retry_failed and output.exists()
            command=[sys.executable,'-m','research_framework.cli','case',str(args.plan),str(index)]+(['--retry'] if retry else (['--resume'] if args.resume else []))
            attempt=len(list((root/'logs').glob(case['name']+'*.log')))
            logfile=root/'logs'/f'{case["name"]}{"__retry_"+str(attempt) if retry else ""}.log'
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',JAX_PLATFORMS='cpu')
            try:
                with logfile.open('a' if args.resume else 'x') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout,env=env)
                print(case['name'],'exit',result.returncode,flush=True)
            except subprocess.TimeoutExpired:
                print(case['name'],'TIMEOUT; checkpoint retained',flush=True)
                with logfile.open('a') as log:log.write('\nTIMEOUT: retained checkpoints; resume is explicit.\n')
        result=collect(root);print(dict(total=len(result),complete=sum(r['status']=='complete' for r in result),csv=str(root/'comparison.csv')),flush=True);return
    if args.command=='collect':
        result=collect(args.root);print(dict(total=len(result),complete=sum(r['status']=='complete' for r in result),csv=str(args.root/'comparison.csv')));return
    from .experiments.report import report
    report(args.root,args.supplement)


if __name__=='__main__':main()
