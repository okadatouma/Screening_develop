"""Compare saved checkpoints on each run's fixed final probe configurations.

This isolates parameter changes; it is NOT sampling from historical |psi|^2.
No theta_0 is reconstructed and no energy is estimated on these fixed inputs.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from screening_analysis.attn_probe import probe, save_probe
from screening_analysis.density_ph import analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Reuse completed outputs, never overwrite probes')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    rows=[]
    for run in sorted(args.root.glob('*/training')):
        if not (run.parent/'analysis/topology/final/probe.npz').exists():
            continue
        with np.load(run.parent/'analysis/topology/final/probe.npz',allow_pickle=False) as z:
            fixed={k:z[k] for k in ('positions','spins','atoms','charges')}
            lattice=z['lattice']
        checkpoints=sorted(run.glob('qmcjax_ckpt_*.npz'))
        targets=[0,20000,40000,60000,79999]
        selected=sorted(set(min(checkpoints,key=lambda p:abs(int(p.stem.rsplit('_',1)[1])-target)) for target in targets))
        inventory=dict(available_steps=[int(p.stem.rsplit('_',1)[1]) for p in checkpoints],
                       selected=[p.name for p in selected],
                       limitation='No step 0 checkpoint; earliest saved step is not initialization. Fixed inputs from final checkpoint, not historical equilibrium samples.')
        runout=args.output/run.parent.name
        runout.mkdir(exist_ok=args.resume)
        (runout/'inventory.json').write_text(json.dumps(inventory,indent=2))
        for checkpoint in selected:
            step=int(checkpoint.stem.rsplit('_',1)[1])
            dest=runout/f'step-{step:06d}'
            dest.mkdir(exist_ok=args.resume)
            if not (dest/'probe.npz').exists():
                result=probe(run,checkpoint=checkpoint,config_path=run/'runtime_config.json',
                         lattice=lattice,evaluation_data=fixed,nconf=len(fixed['positions']))
                save_probe(result,dest/'probe.npz')
            if (dest/'density-ph/summary.json').exists():
                summary=json.loads((dest/'density-ph/summary.json').read_text())
            else:
                summary=analyze(dest/'probe.npz',dest/'density-ph',repeats=1)
            with (run/'train_stats.csv').open() as f:
                stats=[r for r in csv.DictReader(f) if max(0,step-1999)<=int(r['step'])<=step]
            energy=float(np.mean([float(r['energy']) for r in stats]))
            row=dict(run=run.parent.name,step=step,energy_previous_2000_updates=energy,
                     common_max_k=summary['common_max_k'],
                     **summary['mean_beta1_per_graph_shared_untied_k'])
            rows.append(row)
            print(json.dumps(row),flush=True)
        # Use exactly the same k interval across all times and configurations.
        runrows=[r for r in rows if r['run']==run.parent.name]
        fixed_max=min(r['common_max_k'] for r in runrows)
        raw_by_step={}
        common_keys=None
        for row in runrows:
            path=runout/f"step-{row['step']:06d}"/'density-ph/curves.csv'
            with path.open() as f:
                selected=[r for r in csv.DictReader(f) if r['control']=='raw' and
                          r['common_untied']=='True' and int(r['k'])<=fixed_max]
            raw_by_step[row['step']]=selected
            keys={(r['sample'],r['layer'],r['head'],r['k']) for r in selected}
            common_keys=keys if common_keys is None else common_keys & keys
        for row in runrows:
            row['history_fixed_max_k']=fixed_max
            for source in ('qk','trim','window','screening','distance'):
                values=[int(r['beta1']) for r in raw_by_step[row['step']] if r['source']==source and
                        (r['sample'],r['layer'],r['head'],r['k']) in common_keys]
                row['fixed_k_'+source]=float(np.mean(values))
    with (args.output/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
