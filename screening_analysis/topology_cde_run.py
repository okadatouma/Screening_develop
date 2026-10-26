"""C/D/E analyses of immutable diagnostic probes, kept separate from graph PH A/B.

python -m screening_analysis.topology_cde_run --probe PROBE.npz --output NEW_DIR
python -m screening_analysis.topology_cde_run --previous-results PREVIOUS_PILOT --output NEW_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from screening_analysis import multiscale_topology as multi,directed_loops as directed,hidden_topology as hidden
from screening_analysis.topology_run import json_safe


def finite_mean(a,axis=None):
    a=np.asarray(a,dtype=float);good=np.isfinite(a);count=good.sum(axis=axis)
    return np.divide(np.where(good,a,0).sum(axis=axis),count,
                     out=np.full(np.asarray(count).shape,np.nan),where=count>0)


def heatmap_svg(panels,path,title,xlabel='distance cutoff b',ylabel='QK cutoff a',xvalues=None,yvalues=None):
    """Small standalone scientific heatmaps with explicit scales and missing cells."""
    rows=(len(panels)+2)//3;W=1200;H=80+rows*260
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
         f'<rect width="{W}" height="{H}" fill="white"/>','<g font-family="sans-serif" fill="#17232d" font-size="13">',
         f'<text x="25" y="28" font-size="17">{escape(title)}</text>']
    for p,(name,a) in enumerate(panels):
        a=np.asarray(a);ny,nx=a.shape;x=55+(p%3)*395;y=75+(p//3)*260;w=310;h=170
        values=a[np.isfinite(a)];lo=float(values.min()) if len(values) else 0.;hi=float(values.max()) if len(values) else 1.
        if hi<=lo:hi=lo+max(1e-12,.05*abs(lo))
        svg.append(f'<text x="{x}" y="{y-15}">{escape(name)} [range {lo:.3g}, {hi:.3g}]</text>')
        for iy in range(ny):
            for ix in range(nx):
                v=a[iy,ix];z=np.clip((v-lo)/(hi-lo),0,1)
                color='#bcbcbc' if not np.isfinite(v) else f'rgb({int(244-195*z)},{int(249-117*z)},{int(252-79*z)})'
                svg.append(f'<rect x="{x+ix*w/nx:.2f}" y="{y+(ny-1-iy)*h/ny:.2f}" width="{w/nx+.2:.2f}" height="{h/ny+.2:.2f}" fill="{color}"/>')
        svg.extend([f'<text x="{x+90}" y="{y+h+34}">{escape(xlabel)}</text>',
                    f'<text x="{x-30}" y="{y+85}" transform="rotate(-90 {x-30} {y+85})">{escape(ylabel)}</text>'])
        if xvalues is not None:
            svg.extend([f'<text x="{x}" y="{y+h+17}">{xvalues[0]:.3g}</text>',f'<text x="{x+w-20}" y="{y+h+17}">{xvalues[-1]:.3g}</text>'])
        if yvalues is not None:
            svg.extend([f'<text x="{x-26}" y="{y+h}">{yvalues[0]:.2g}</text>',f'<text x="{x-26}" y="{y+10}">{yvalues[-1]:.2g}</text>'])
    svg.extend(['</g></svg>']);Path(path).write_text('\n'.join(svg))


def load_probe(path):
    with np.load(path,allow_pickle=False) as z:
        result={k:z[k] for k in z.files if k!='metadata_json'}
        result['metadata']=json.loads(str(z['metadata_json']))
    return result


def analyze(probe,output,run=None,a_grid=None,b_grid=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    s=probe['S'];B,L,H,N,_=s.shape
    a=np.array([-1.,-.5,0,.25,.5,.75,1.]) if a_grid is None else np.asarray(a_grid)
    b=np.linspace(0,1.1*float(probe['dref']),9) if b_grid is None else np.asarray(b_grid)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.all(np.diff(a)>0) or not np.all(np.diff(b)>0):
        raise ValueError('finite ascending a and b grids required')
    window=probe.get('Window',np.ones_like(s));ms=[];loop_arrays={};loop_stats=[];excluded=[]
    perimeter=np.linspace(0,3*max(np.linalg.norm(probe['lattice'],axis=0)),9)
    for i in range(B):
        vectors=probe['head_vectors'][i] if 'head_vectors' in probe else None
        ms.append(multi.analyze_multiscreen(s[i],window[i],probe['positions'][i],probe['lattice'],a,b,
                                          head_vectors=vectors))
        loops=directed.analyze_loops(s[i],probe['W_window'][i],probe['positions'][i],probe['lattice'],perimeter_edges=perimeter)
        for key in ('triangles','perimeter','longest_edge','area','qk','screening','reverse_qk','reverse_screening',
                    'epsilon_values','epsilon_circulations','regularized_log_circulation'):
            loop_arrays[f'b{i}_{key}']=loops[key]
        loop_stats.append(loops['statistics']);excluded.append(loops['excluded'])
    multiscale={key:np.stack([record[key] for record in ms]) for key in ms[0]}
    rank=multiscale['head_rank'];valid=np.isfinite(rank).sum(axis=0)
    multiscale['rank_probability']=np.stack([np.divide((rank==r).sum(axis=0),valid,
                        out=np.full(valid.shape,np.nan),where=valid>0) for r in range(3)],axis=2)
    # Shape [L,H,rank,a,b]; heads are retained in data, pooled in invariant features.
    with (out/'multiscale.npz').open('xb') as f:np.savez_compressed(f,**multiscale)
    with (out/'directed-loops.npz').open('xb') as f:np.savez_compressed(f,**loop_arrays)
    stats={k:np.stack([r[k] for r in loop_stats]) for k in loop_stats[0]}
    with (out/'directed-statistics.npz').open('xb') as f:np.savez_compressed(f,**stats,perimeter_edges=perimeter)
    features={}
    pooled=multiscale['head_mean_betti'].mean(axis=0)
    prob=multiscale['rank_probability'].mean(axis=1)
    features['C_qk']=np.concatenate((pooled.ravel(),prob.ravel())).tolist()
    distance_betti=multiscale['distance_only_betti'].mean(axis=0)
    distance_rank=multiscale['distance_only_rank'];good=np.isfinite(distance_rank).sum(axis=0)
    distance_probability=np.stack([np.divide((distance_rank==r).sum(axis=0),good,
                 out=np.full(good.shape,np.nan),where=good>0) for r in range(3)])
    features['C_distance_only']=np.concatenate((distance_betti.ravel(),distance_probability.ravel())).tolist()
    features['C_qk_residual']=np.concatenate(((pooled-distance_betti[None,None,:,:]).ravel(),
                                             (prob-distance_probability[None,:,None,:]).ravel())).tolist()
    features['C_window']=np.concatenate((multiscale['integrated_betti'].mean(axis=0).ravel(),
                                         multiscale['union_betti'].mean(axis=0).ravel())).tolist()
    for source in ('qk','screening'):
        # Pool heads with mean/std (NOT indexed head identities).
        averages=finite_mean(stats[source],axis=0) # [L,H,stat]
        features['D_'+source]=np.concatenate((finite_mean(averages,axis=1).ravel(),
                                  np.nanstd(averages,axis=1).ravel())).tolist() if N>=3 else None
    diagrams={};hidden_summary={}
    if 'hidden_output' in probe:
        representations=hidden.analyze_hidden(probe['hidden_output'])
        for method,data in representations.items():
            layer_stats=[]
            for layer,record in enumerate(data['layers']):
                for dim,d in enumerate(record['diagrams']):diagrams[f'{method}_l{layer}_H{dim}']=d
                z=data['clouds'][:,layer];features[f'E_pool_{method}_L{layer}']=z.mean(axis=0).tolist()
                desc=[]
                for stat in record['summaries']:
                    desc.extend(stat[k] for k in ('total_persistence','max_persistence','essential_intervals','finite_intervals'))
                desc.extend(np.quantile(record['distance'],[.25,.5,.75,1]).tolist())
                features[f'E_PH_{method}_L{layer}']=desc
                layer_stats.append(record['summaries'])
            hidden_summary[method]=layer_stats
        with (out/'hidden-clouds.npz').open('xb') as f:np.savez_compressed(f,**{k:v['clouds'] for k,v in representations.items()},**diagrams)
    panels=[]
    for layer in range(L):
        for dim in (0,1):panels.append((f'L{layer} head-mean beta{dim}',pooled[layer,...,dim]))
        panels.append((f'L{layer} P(wind rank=2)',prob[layer,2]))
    heatmap_svg(panels,out/'multiscale.svg','Two-parameter graph surfaces (not 1D persistence diagrams)',xvalues=b,yvalues=a)
    heatmap_svg([(f'L{l} integrated Window beta1',multiscale['integrated_betti'].mean(axis=0)[l,...,1]) for l in range(L)]
                +[(f'L{l} head Jaccard',finite_mean(multiscale['head_jaccard'],axis=0)[l]) for l in range(L)],
                out/'windows-redundancy.svg','Window integration / redundancy; empty overlap is undefined',xvalues=b,yvalues=a)
    loop_panels=[]
    for source in ('qk','screening'):
        a_stats=finite_mean(stats[source+'_by_perimeter'],axis=(0,2)) # [L,bins,4]
        for l in range(L):loop_panels.append((f'{source} L{l}: CCW / abs / RMS / std',a_stats[l].T))
    heatmap_svg(loop_panels,out/'directed-loops.svg','Geometrically CCW network loops; nonquantized',
                xlabel='triangle perimeter bin',ylabel='statistic',xvalues=perimeter,yvalues=np.arange(4))
    summary=dict(run=run or {},shape=[B,L,H,N],a_grid=a,b_grid=b,features=features,hidden_PH=hidden_summary,
        loop_excluded=excluded,triangles_per_configuration=[len(loop_arrays[f'b{i}_triangles']) for i in range(B)],
        epsilon_sensitivity_max=[float(np.max(np.abs(loop_arrays[f'b{i}_epsilon_circulations']-loop_arrays[f'b{i}_screening'][None]),initial=0.)) for i in range(B)],
        probe_metadata=probe['metadata'],distance_metric='minimum-image Euclidean; Window uses the original network periodic norm',
        head_invariance='mean/std/max, window-weighted union/consensus and pair-averaged Jaccard; never a head-id feature',
        band_energy_proxy='window mass fractions times full nonlinear head norm squared, NOT an additive distance-resolved output',
        two_parameter='Betti/winding surfaces only; no 1D diagram or multiparameter rank invariant asserted',
        chern_relation='unverified unless supplied independently by a physical twist/C3 computation',
        hidden_topology='ordinary PH per layer; no zigzag persistence without inclusion maps',
        source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__,multi.__file__,directed.__file__,hidden.__file__,Path(multi.__file__).with_name('graph_wind.py'),__import__('screening_analysis.topology.persistence',fromlist=['']).__file__)})
    (out/'summary.json').write_text(json.dumps(json_safe(summary),indent=2,allow_nan=False))
    return summary


def evaluate_synthetic(summaries):
    rows=[]
    eligible=[r for r in summaries if r['run'].get('synthetic_geometry')]
    for mode in sorted({r['run']['window'] for r in eligible}):
        records=[r for r in eligible if r['run']['window']==mode]
        for feature in sorted(set.intersection(*(set(r['features']) for r in records))):
            values=[r['features'][feature] for r in records]
            if any(v is None for v in values) or not np.isfinite(np.asarray(values)).all():continue
            result=hidden.grouped_centroid(values,[r['run']['synthetic_geometry'] for r in records],
                                           [r['run']['seed'] for r in records])
            rows.append(dict(window=mode,feature=feature,**result))
    return dict(label='synthetic ordered/uniform geometry; not WC/AHC',results=rows,
        statistic='6 run-level decisions, 3 seed groups; shared geometry samples, no significance claim')


def evaluate_phases(summaries):
    """Physical labels must come from independent evidence, never folder names."""
    eligible=[r for r in summaries if r['run'].get('phase') in ('WC','AHC') and r['run'].get('phase_evidence') and r['run'].get('group')]
    if not eligible:return dict(status='unavailable',reason='independently labeled WC/AHC run metadata required')
    results=[]
    for condition in sorted({r['run'].get('condition','unspecified') for r in eligible}):
        records=[r for r in eligible if r['run'].get('condition','unspecified')==condition]
        if any(len({r['run']['group'] for r in records if r['run']['phase']==phase})<3 for phase in ('WC','AHC')):
            results.append(dict(condition=condition,status='need >=3 independent groups per phase'));continue
        for feature in sorted(set.intersection(*(set(r['features']) for r in records))):
            units={}
            for r in records:
                v=r['features'][feature]
                if v is None or not np.isfinite(v).all():continue
                units.setdefault((r['run']['group'],r['run']['phase']),[]).append(v)
            if len(units)<6 or len({len(v[0]) for v in units.values()})!=1:continue
            keys=list(units);features=[np.mean(units[k],axis=0) for k in keys]
            result=hidden.grouped_centroid(features,[k[1] for k in keys],[k[0] for k in keys])
            results.append(dict(condition=condition,feature=feature,**result))
    return dict(status='exploratory',results=results,caution='no automatic Chern verification; explicit condition strata and training-group holdout')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sources=parser.add_mutually_exclusive_group(required=True);sources.add_argument('--probe',type=Path);sources.add_argument('--previous-results',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.probe:
        analyze(load_probe(args.probe),args.output/'run',run=dict(id=args.probe.stem,phase='unknown'))
    else:
        records=[]
        for path in sorted(args.previous_results.glob('*/probe.npz')):
            original=json.loads((path.parent/'summary.json').read_text())
            record=analyze(load_probe(path),args.output/path.parent.name,run=original['run'])
            record['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();records.append(record)
            print(path.parent.name,flush=True)
        (args.output/'summary.json').write_text(json.dumps(json_safe(records),indent=2,allow_nan=False))
        (args.output/'geometry-classification.json').write_text(json.dumps(json_safe(evaluate_synthetic(records)),indent=2,allow_nan=False))
        (args.output/'phase-classification.json').write_text(json.dumps(json_safe(evaluate_phases(records)),indent=2,allow_nan=False))


if __name__=='__main__':main()
