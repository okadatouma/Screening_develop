"""Group-held-out comparison of all available internal features; no phase inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from screening_analysis import phase_metrics
from .workflow import provenance,write


def run(directory,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    ab=json.loads((directory/'summary.json').read_text());cde=json.loads((directory/'cde-summary.json').read_text());lookup={r['run']['id']:r for r in cde}
    rows=[]
    for view in ('A','B'):
        records=[r for r in ab if r['run']['view']==view];units={}
        for r in records:
            feats={}
            for a in r['aggregates']:
                name=a['source']+'_'+a['control']
                feats[name]=[a[k] for k in ('H0_total_persistence','H1_total_persistence','beta0_auc','beta1_auc','rank1_probability','rank2_probability')]
                if name=='screening_raw':
                    feats['PH_only']=[a[k] for k in ('H0_total_persistence','H1_total_persistence','H0_max_persistence','H1_max_persistence')]
                    feats['winding_only']=[a[f'rank{i}_probability'] for i in (0,1,2)]
            c=lookup[r['run']['id']]
            feats.update({k:v for k,v in c['features'].items() if k in ('C_qk','C_distance_only','C_qk_residual','C_window','D_qk','D_screening') or k.startswith('E_')})
            for name,value in list(feats.items()):
                if name.startswith('E_PH_'):
                    feats[name.replace('E_PH_','E_PH_only_',1)]=value[:8]
                    feats[name.replace('E_PH_','E_distance_only_',1)]=value[8:]
            units[r['run']['id']]=feats
        for key in sorted(set.intersection(*(set(v) for v in units.values()))):
            values=[units[r['run']['id']][key] for r in records]
            if any(v is None for v in values) or not np.isfinite(np.asarray(values)).all():
                rows.append(dict(view=view,method=key,status='undefined features retained as missing; no imputation'));continue
            result=phase_metrics.evaluate(values,[r['run']['condition'] for r in records],[r['run']['group'] for r in records],bootstrap_repetitions=1000)
            rows.append(dict(view=view,method=key,label_kind='lambda condition ONLY; WC/AHC remains unverified',**result))
    tested=[r for r in rows if 'balanced_accuracy' in r];p=np.array([r['permutation']['pvalue'] for r in tested]);order=np.argsort(p)
    corrected=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    for i,v in zip(order,corrected):tested[i]['holm_pvalue']=float(v)
    write(out/'metrics.json',dict(results=rows,physical_phase_status='未検証',samples_per_model=8,independent_seed_groups=3,N=9,rs=15,lambda_conditions=[.33,.72],
        provenance=provenance(20261026),selection='all available families/controls, no hyperparameter selection; no repeated-checkpoint data inflation'))
    from screening_analysis.report_topology_benchmark import csv_write,line_svg
    csv_write(out/'metrics.csv',[dict(view=r['view'],method=r['method'],BA=r['balanced_accuracy'],AUROC=r['auroc'],F1=r['f1_macro'],MCC=r['matthews_correlation'],p=r['permutation']['pvalue'],holm_p=r['holm_pvalue']) for r in tested])
    selected=['qk_raw','window_raw','trim_raw','screening_raw','PH_only','winding_only','C_qk','D_qk','D_screening','E_PH_only_mean_std_L0'];panels=[]
    for view in ('A','B'):
        values=[next(r['balanced_accuracy'] for r in tested if r['view']==view and r['method']==method) for method in selected]
        panels.append((view+'; method index defined in methods.json',[('8-configuration lambda-condition BA',np.arange(len(selected)),values)]))
    write(out/'methods.json',dict(enumerate(selected)))
    line_svg(panels,out/'methods.svg','Actual internal-feature ablation: NOT WC/AHC; 3 seed groups; no significant separation',xlabel='method index',ylabel='balanced accuracy')
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.output)
