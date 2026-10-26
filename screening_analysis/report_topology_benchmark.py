"""Export reviewable numeric tables and standalone SVG plots from G/benchmark data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from screening_analysis.topology_run import json_safe


COLORS=('#44566d','#4d8e65','#ca7b25','#9463a8','#369bb1','#d86668','#788b32')


def line_svg(panels,path,title,xlabel='update',ylabel='total code energy'):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="440">','<rect width="1100" height="440" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#17232d">',f'<text x="30" y="25" font-size="17">{escape(title)}</text>']
    for p,(name,curves) in enumerate(panels):
        x=60+p*540;y=65;w=440;h=260
        panel_y=ylabel[p] if isinstance(ylabel,(tuple,list)) else ylabel
        ys=np.concatenate([np.asarray(c[2]) for c in curves]);lo=ys.min();hi=ys.max();span=max(hi-lo,1e-10)
        low=lo-.05*span;high=hi+.05*span;xx=np.concatenate([np.asarray(c[1]) for c in curves]);xmin=xx.min();xmax=max(xx.max(),xmin+1e-12)
        svg.extend([f'<text x="{x}" y="{y-15}">{escape(name)}</text>',f'<path d="M{x},{y} V{y+h} H{x+w}" fill="none" stroke="#89939c"/>',
            f'<text x="{x-40}" y="{y+8}">{high:.3g}</text>',f'<text x="{x-40}" y="{y+h}">{low:.3g}</text>',
            f'<text x="{x}" y="{y+h+17}">{xmin:.3g}</text>',f'<text x="{x+w-15}" y="{y+h+17}">{xmax:.3g}</text>',
            f'<text x="{x+150}" y="{y+h+37}">{escape(xlabel)}</text>',
            f'<text x="{x-50}" y="{y+150}" text-anchor="middle" transform="rotate(-90 {x-50} {y+150})">{escape(panel_y)}</text>'])
        for i,(label,xx,yy) in enumerate(curves):
            color=COLORS[i%len(COLORS)];points=' '.join(f'{x+w*(a-xmin)/(xmax-xmin):.2f},{y+h*(1-(b-low)/(high-low)):.2f}' for a,b in zip(xx,yy))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<text x="{x+(i%3)*150}" y="{y+h+62+17*(i//3)}" fill="{color}">{escape(label)}</text>')
    svg.append('</g></svg>');path.write_text('\n'.join(svg))


def csv_write(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def regularizer_table(directory):
    rows=json.loads((directory/'summary.json').read_text());base={r['seed']:r for r in rows if r['regularizer']['mode']=='off'}
    groups={}
    for r in rows:groups.setdefault((r['regularizer']['mode'],r['regularizer']['gamma']),[]).append(r)
    table=[]
    for (mode,gamma),records in groups.items():
        measured=[r for r in records if r['measurement'] is not None]
        energies=np.array([r['measurement']['energy'] for r in measured]);variances=np.array([r['measurement']['variance_abs_complex'] for r in measured])
        table.append(dict(mode=mode,gamma=gamma,seeds=len(measured),energy_mean=float(energies.mean()),energy_seed_std=float(energies.std(ddof=1)),
            paired_energy_delta_mean=float(np.mean([r['measurement']['energy']-base[r['seed']]['measurement']['energy'] for r in measured])),
            variance_mean=float(variances.mean()),variance_seed_std=float(variances.std(ddof=1)),nan_steps=sum(r['nan_steps'] for r in records),
            median_update_seconds=float(np.median([r['median_update_seconds_excluding_first3'] for r in measured])),
            PH_consistency_mean=float(np.mean([r['common_configuration_stability']['persistence'] for r in measured])),
            spectral_consistency_mean=float(np.mean([r['common_configuration_stability']['laplacian'] for r in measured])),
            weight_consistency_mean=float(np.mean([r['common_configuration_stability']['weights'] for r in measured]))))
    return table,groups


def accuracy_svg(results,path):
    families=('density','gr','structure_factor','distances','widths','weight_distribution','topology_completed')
    names=('density','g(r)','S(k)','distances','widths','weights','PH completed')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="530">','<rect width="1200" height="530" fill="white"/>',
         '<g font-family="sans-serif" font-size="13" fill="#17232d">',
         '<text x="25" y="28" font-size="18">Lambda-condition balanced accuracy (NOT verified WC/AHC phases)</text>',
         '<text x="25" y="52">A: fresh fixed-psi MCMC; B: uniform shared inputs independent of every model. 16 configurations.</text>']
    for j,name in enumerate(names):svg.append(f'<text x="{180+j*140}" y="87">{name}</text>')
    for i,(view,mode) in enumerate((v,m) for v in ('A','B') for m in ('off','persistence','laplacian','weights')):
        y=105+43*i;svg.append(f'<text x="20" y="{y+22}">{view} / {mode}</text>')
        for j,family in enumerate(families):
            row=next(r for r in results if r['view']==view and r['regularizer']==mode and r['samples']==16 and r['feature']==family)
            a=row['balanced_accuracy'];color=f'rgb({int(245-195*a)},{int(249-117*a)},{int(252-79*a)})'
            svg.extend([f'<rect x="{170+j*140}" y="{y}" width="130" height="34" rx="3" fill="{color}"/>',
                        f'<text x="{213+j*140}" y="{y+23}">{a:.3f}</text>'])
    svg.extend(['<text x="25" y="485">Only 3 paired seed groups. Exact permutation p cannot be below 0.25 in this design.</text>',
                '<text x="25" y="508">No physical phase label or convergence certificate is inferred from these scores.</text>','</g></svg>'])
    path.write_text('\n'.join(svg))


def distance_svg(rows,path):
    panels=[]
    for view in ('A','B'):
        curves=[]
        for mode in ('off','persistence','laplacian','weights'):
            r=next(r for r in rows if r['view']==view and r['regularizer']==mode and r['graph']=='completed' and r['metric']=='bottleneck')
            curves.append((mode,[0,1],[r['within_parameter']['finite_mean'],r['between_parameter']['finite_mean']]))
        panels.append((view+' (same lambda x=0; different lambda x=1)',curves))
    line_svg(panels,path,'PH separation: lambda conditions, not verified phases; pair means share runs',xlabel='same / different parameter',ylabel='mean H0+H1 bottleneck')


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    table,groups=regularizer_table(args.regularization);csv_write(out/'regularizer_comparison.csv',table)
    (out/'regularizer_comparison.json').write_text(json.dumps(table,indent=2))
    curves=[];vcurves=[]
    for (mode,gamma),runs in groups.items():
        history=[];variance=[]
        for r in runs:
            with (args.regularization/r['id']/'stats.csv').open() as f:series=list(csv.DictReader(f))
            history.append([float(s['energy']) for s in series]);variance.append([float(s['variance_abs_complex']) for s in series])
        curves.append((f'{mode} g={gamma:g}',np.arange(len(history[0])),np.median(history,axis=0)))
        vcurves.append((f'{mode} g={gamma:g}',np.arange(len(variance[0])),np.median(variance,axis=0)))
    line_svg([('Energy: median of 3 seed batch means',curves),('Local variance: median of 3 seeds',vcurves)],out/'regularizer_training.svg',
             'Free N4 VMC: three DISTINCT regularizers, 64 equal updates; no WC/AHC labels',ylabel=('energy [code]','Var(E_L) [code squared]'))
    classification=json.loads((args.analysis/'classification.json').read_text())['candidate_condition_results']
    flat=[]
    for r in classification:
        ci=r['bootstrap']['interval_95']['balanced_accuracy']
        flat.append(dict(view=r['view'],regularizer=r['regularizer'],samples=r['samples'],feature=r['feature'],
            balanced_accuracy=r['balanced_accuracy'],auroc=r['auroc'],f1_macro=r['f1_macro'],mcc=r['matthews_correlation'],
            BA_CI_low=ci[0],BA_CI_high=ci[1],permutation_p=r['permutation']['pvalue'],holm_p=r['holm_pvalue'],independent_groups=r['independent_groups'],
            label_kind='lambda condition, not physical phase'))
    csv_write(out/'candidate_condition_metrics.csv',flat);accuracy_svg(classification,out/'candidate_condition_accuracy.svg')
    measurements=json.loads((args.analysis/'records.json').read_text());measurements=[r for r in measurements if r['view']=='A']
    csv_write(out/'natural_measurements.csv',[dict(id=r['id'],kind=r['kind'],regularizer=r['mode'],seed=r['seed'],condition=r['candidate_parameter_label'],
        updates=r['completed_updates'],energy=r['measurement']['energy'],energy_per_particle_Ha=r['measurement']['energy_per_particle_Ha'],
        variance=r['measurement']['variance_abs_complex'],imaginary_energy=r['measurement']['imaginary_energy'],chain_sem=r['measurement']['chain_sem'],
        phase_status='unverified') for r in measurements])
    ds=json.loads((args.analysis/'distances.json').read_text());distance_svg(ds,out/'within_between.svg')
    perturb=json.loads((args.analysis/'coordinate_perturbations.json').read_text());panels=[]
    for lam in ('033','072'):
        curves=[]
        for seed in (11,23,37):
            subset=[r for r in perturb if r['id']==f'N9-lam{lam}-off-seed{seed}']
            curves.append((f'seed{seed}',[r['sigma'] for r in subset],[r['completed_PH_bottleneck']['mean'] for r in subset]))
        panels.append((f'lambda={"0."+lam[1:]}',curves))
    line_svg(panels,out/'coordinate_sensitivity.svg','Small coordinate perturbations at shared inputs; no sampling-distribution claim',
             xlabel='Cartesian displacement sigma (code length)',ylabel='paired completed-PH bottleneck')
    if args.symmetry:
        rows=json.loads((args.symmetry/'summary.json').read_text())['results']
        csv_write(out/'rotation_diagnostics.csv',rows)
    (out/'source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('regularization',type=Path);p.add_argument('analysis',type=Path)
    p.add_argument('--symmetry',type=Path)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
