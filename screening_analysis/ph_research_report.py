"""Report conditional PH controls and run-level physical correlations.

audit RESULTS_ROOT --output NEW_DIR
collect PLAN --output NEW_DIR [--analysis-dir analysis]
Requires matplotlib; no network evaluation or scheduler submission.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def mean(rows,key):
    a=[float(r[key]) for r in rows if r.get(key) not in (None,'','nan')]
    return float(np.mean(a)) if a else None


def null_summary(folder):
    result=[];strict=read(folder/'constrained-null.csv');hierarchy=read(folder/'constraint-hierarchy.csv')
    for conditional in ('spin_degree_distance','spin_degree_only','spin_distance_only'):
        rows=strict if conditional=='spin_degree_distance' else [r for r in hierarchy if r['conditional']==conditional]
        keys=sorted({(r['source'],r['layer'],r['head'],r['k']) for r in rows})
        for source,layer,head,k in keys:
            selected=[r for r in rows if (r['source'],r['layer'],r['head'],r['k'])==(source,layer,head,k)]
            original=np.array([float(r['observed_beta1_at_full_support']) for r in selected])
            control=np.array([float(r['null_beta1_at_full_support']) for r in selected])
            result.append(dict(source=source,layer=int(layer),head=int(head),k=int(k),conditional=conditional,
                conditional_draws=len(selected),configurations=len({r['sample'] for r in selected}),
                observed_beta1_endpoint=float(original.mean()),null_beta1_endpoint=float(control.mean()),
                endpoint_agreement_fraction=float(np.mean(original==control)),
                observed_beta1_mean=mean(selected,'observed_beta1_mean'),null_beta1_mean=mean(selected,'null_beta1_mean'),
                degree_rmse=mean(selected,'degree_rmse'),
                degree_only_zero_move_fraction=float(np.mean([int(r['available_two_switches'])==0 for r in selected])) if conditional=='spin_degree_only' else None,
                strict_target_reached_fraction=float(np.mean([r['target_reached']=='True' for r in selected])) if conditional=='spin_degree_distance' else None,
                caution='Conditional diagnostic, not a uniform-null p-value; inspect mobility and relaxed degree changes.'))
    return result


def audit(root,output,audit_source=None):
    output.mkdir(parents=True,exist_ok=False);summaries=[]
    folders=sorted(p for p in root.iterdir() if (p/'ph-features.csv').exists())
    fig,axes=plt.subplots(1,len(folders),figsize=(6*len(folders),4),squeeze=False)
    for ax,folder in zip(axes[0],folders):
        summary=null_summary(folder);write(output/(folder.name+'-null-summary.csv'),summary)
        for r in summary:r['model']=folder.name
        summaries.extend(summary)
        selected=[r for r in summary if r['source']=='qk' and r['head']==1]
        labels=[];orig=[];null=[]
        for r in selected:
            labels.append(r['conditional'].replace('spin_','').replace('_','\n'))
            orig.append(r['observed_beta1_mean']);null.append(r['null_beta1_mean'])
        x=np.arange(len(labels));ax.bar(x-.18,orig,.36,label='observed');ax.bar(x+.18,null,.36,label='conditional control')
        ax.set(xticks=x,xticklabels=labels,ylabel='Mean beta1 over weight thresholds',title=folder.name+' QK head 1')
        ax.legend(fontsize=8)
    fig.suptitle('Head 1: distinguish degree constraints from spin/distance constraints; finite swap chains')
    fig.tight_layout();fig.savefig(output/'qk-head1-controls.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(folders),figsize=(6*len(folders),4),squeeze=False)
    for ax,folder in zip(axes[0],folders):
        rows=read(folder/'ph-features.csv');sources=sorted({r['source'] for r in rows})
        for head in (0,1):
            a=[mean([r for r in rows if r['source']==s and r['control']=='positive_rank' and r['head']==str(head)],'beta1_mean') for s in sources]
            ax.plot(np.arange(len(sources)),a,'o-',label=f'head {head}')
        ax.set(xticks=np.arange(len(sources)),xticklabels=[s.replace('importance_','sensitivity\n') for s in sources],
               ylabel='Rank-filtration mean beta1',title=folder.name);ax.tick_params(axis='x',labelrotation=20);ax.legend()
    fig.suptitle('Separate sensitivity-based filtrations: not physical Chern numbers')
    fig.tight_layout();fig.savefig(output/'sensitivity-ph.png',dpi=160);plt.close(fig)
    if audit_source is not None:
        from screening_analysis.ph_research import load_probe,importance_graph,top_graph,graph_features
        from screening_analysis.topology import symmetrize
        sources=('qk','screening','importance_phase','importance_complex')
        fig,axes=plt.subplots(len(folders),len(sources),figsize=(13,4*len(folders)),squeeze=False)
        for row,folder in enumerate(folders):
            summary=json.loads((folder/'summary.json').read_text());sample=summary['sample_indices'][0];k=summary['k'][0]
            p=load_probe(audit_source/folder.name/'shared-probe.npz')
            with np.load(audit_source/folder.name/'connection-sensitivity.npz') as g:gradient=g['derivatives'][sample,:,1]
            N=int(p['NE'])
            matrices={'qk':symmetrize((p['S'][sample,0,1]+1)/2),'screening':symmetrize(np.clip(p['W_window'][sample,0,1],0,1))}
            for channel in ('phase','complex'):matrices['importance_'+channel]=importance_graph(gradient,N,channel)[0]
            with np.load(folder/'diagrams.npz') as diagrams:
                for col,source in enumerate(sources):
                    ax=axes[row,col];initial,status=top_graph(matrices[source],k)
                    key=f'b{sample}_l0_h1_{source}_k{k}_null0_H1'
                    if initial is not None and key in diagrams:
                        _,original=graph_features(initial)
                        for label,d,color,marker in [('observed',original['diagrams'][1],'#2675b6','o'),
                                                      ('strict null 0',diagrams[key],'#d47c11','x')]:
                            finite=np.isfinite(d[:,1]);death=np.where(finite,d[:,1],1.04)
                            ax.scatter(d[:,0],death,s=20,alpha=.55,color=color,marker=marker,label=label)
                    ax.plot([0,1],[0,1],color='#aaa',linewidth=.6)
                    ax.set(xlim=(-.02,1.06),ylim=(-.02,1.08),title=folder.name+'\n'+source,xlabel='Birth',ylabel='Death');ax.legend(fontsize=7)
        fig.suptitle('Head 1, first shared input, same top-k support size; y=1.04: infinite/censored death; strict null may be frozen')
        fig.tight_layout();fig.savefig(output/'matched-null-diagrams.png',dpi=160);plt.close(fig)
    (output/'summary.json').write_text(json.dumps(dict(null_comparisons=summaries,
        physical_phase='unknown',inference='No significance or phase claims. Configuration correlations are within a single saved/fixed-input model.'),indent=2))
    return summaries


def collect(plan_path,output,analysis_dir='analysis'):
    plan=json.loads(plan_path.read_text());output.mkdir(parents=True,exist_ok=False)
    rows=[];missing=[]
    for case in plan['cases']:
        for subset in ('native','common'):
            folder=Path(plan['root'])/case['id']/analysis_dir/subset/'research'
            features=read(folder/'ph-features.csv');physical=read(folder/'configuration-physics.csv')
            if not features or not physical:missing.append([case['id'],subset]);continue
            keys=sorted({(r['source'],r['control'],r['layer'],r['head']) for r in features})
            for source,control,layer,head in keys:
                selected=[r for r in features if (r['source'],r['control'],r['layer'],r['head'])==(source,control,layer,head)]
                rows.append(dict(run=case['id'],seed=case['seed'],lam=case['lam'],rs=case['rs'],branch=case['branch'],subset=subset,
                    source=source,control=control,layer=int(layer),head=int(head),configurations=len(selected),
                    beta1_mean=mean(selected,'beta1_mean'),h1_finite_persistence=mean(selected,'h1_finite_persistence'),
                    structure_factor_shell=mean(physical,'structure_factor_shell'),local_energy_real=mean(physical,'local_energy_real'),
                    unit='one independent training run; configurations/heads not independent runs'))
            for source,control,layer in sorted({(r['source'],r['control'],r['layer']) for r in features}):
                selected=[r for r in features if (r['source'],r['control'],r['layer'])==(source,control,layer)]
                rows.append(dict(run=case['id'],seed=case['seed'],lam=case['lam'],rs=case['rs'],branch=case['branch'],subset=subset,
                    source=source,control=control,layer=int(layer),head=-1,configurations=len({r['sample'] for r in selected}),
                    beta1_mean=mean(selected,'beta1_mean'),h1_finite_persistence=mean(selected,'h1_finite_persistence'),
                    structure_factor_shell=mean(physical,'structure_factor_shell'),local_energy_real=mean(physical,'local_energy_real'),
                    unit='one run, mean of per-head PH (not PH of mean weights); head-permutation invariant'))
    write(output/'run-level-features.csv',rows)
    correlations=[]
    keys=sorted({(r['subset'],r['source'],r['control'],r['layer'],r['head']) for r in rows})
    for subset,source,control,layer,head in keys:
        base=[r for r in rows if (r['subset'],r['source'],r['control'],r['layer'],r['head'])==(subset,source,control,layer,head)]
        strata=[('all_parameters_mixed',base)]
        strata.extend((f'lambda={lam},rs={rs}',[r for r in base if r['lam']==lam and r['rs']==rs]) for lam,rs in sorted({(r['lam'],r['rs']) for r in base}))
        strata.extend((f'lambda={lam},rs={rs},initialization={branch}',[r for r in base if (r['lam'],r['rs'],r['branch'])==(lam,rs,branch)])
                      for lam,rs,branch in sorted({(r['lam'],r['rs'],r['branch']) for r in base}))
        for label,selected in strata:
            for observable in ('structure_factor_shell','local_energy_real'):
                a=np.array([(r['beta1_mean'],r[observable]) for r in selected if r[observable] is not None],float).reshape(-1,2)
                ok=len(a)>=5 and min(a.std(axis=0))>1e-12
                correlations.append(dict(subset=subset,source=source,control=control,layer=layer,head=head,stratum=label,
                    observable=observable,training_runs=len(a),unique_seeds=len({r['seed'] for r in selected if r[observable] is not None}),
                    pearson=float(np.corrcoef(a.T)[0,1]) if ok else None,
                    caution='Mixed parameters/initializations are confounded; repeated seeds can correlate runs. No p-values. Head=-1 is permutation invariant.'))
    write(output/'run-level-correlations.csv',correlations)
    (output/'status.json').write_text(json.dumps(dict(rows=len(rows),missing=missing,phase_verified=False),indent=2))
    print(len(rows),'run-level feature rows;',len(missing),'missing subsets')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('audit','collect'):
        parser=sub.add_parser(name);parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
        if name=='collect':parser.add_argument('--analysis-dir',default='analysis')
        else:parser.add_argument('--audit-source',type=Path)
    args=p.parse_args()
    if args.command=='audit':audit(args.input,args.output,args.audit_source)
    else:collect(args.input,args.output,args.analysis_dir)


if __name__=='__main__':main()
