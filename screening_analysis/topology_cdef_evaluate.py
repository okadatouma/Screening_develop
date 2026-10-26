"""Postprocess immutable CDE/F data: epsilon robustness, layer comparison, Window scales.

python -m screening_analysis.topology_cdef_evaluate CDE_DIR --physical F_DIR --output NEW_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from screening_analysis.hidden_topology import grouped_centroid
from screening_analysis.topology_cde_run import heatmap_svg,finite_mean
from screening_analysis.topology_run import json_safe


def epsilon_study(directory):
    records=[]
    for file in sorted(directory.glob('synthetic-*/summary.json')):
        summary=json.loads(file.read_text());B=summary['shape'][0];statistics=[];worst=0.
        with np.load(file.parent/'directed-loops.npz',allow_pickle=False) as z:
            eps=z['b0_epsilon_values']
            for b in range(B):
                v=z[f'b{b}_epsilon_circulations'] # [eps,L,H,triangles]
                reference=z[f'b{b}_screening']
                worst=max(worst,float(np.max(abs(v-reference[None]),initial=0.)))
                statistics.append(np.stack((v.mean(axis=-1),np.mean(abs(v),axis=-1),
                                            np.sqrt(np.mean(v*v,axis=-1)),v.std(axis=-1)),axis=-1))
        a=np.mean(statistics,axis=0) # [eps,L,H,4]
        features=np.concatenate((a.mean(axis=2).reshape(len(eps),-1),a.std(axis=2).reshape(len(eps),-1)),axis=-1)
        records.append(dict(run=summary['run'],epsilon=eps,features=features,max_circulation_change=worst,
            input_sha256=hashlib.sha256((file.parent/'directed-loops.npz').read_bytes()).hexdigest()))
    results=[]
    for mode in ('cos','wide','smooth7','band','sin','gauss','fixed'):
        subset=[r for r in records if r['run']['window']==mode]
        if not subset:continue
        for i,epsilon in enumerate(subset[0]['epsilon']):
            result=grouped_centroid([r['features'][i] for r in subset],[r['run']['synthetic_geometry'] for r in subset],
                                    [r['run']['seed'] for r in subset])
            results.append(dict(window=mode,epsilon=epsilon,max_circulation_change=max(r['max_circulation_change'] for r in subset),**result))
    return dict(label='synthetic geometry only; not WC/AHC',definition='(Wij-Wji)/(Wij+Wji+epsilon) then CCW triangle sum',
        results=results,source_inputs=[dict(id=r['run']['id'],sha256=r['input_sha256']) for r in records])


def window_study(physical_directory):
    summary=json.loads((physical_directory/'summary.json').read_text());rows=[]
    with np.load(physical_directory/'exact-density.npz',allow_pickle=False) as z:
        loop=z['imaginary_loop'];perimeter=z['perimeter'];edges=np.linspace(0,perimeter.max()*(1+1e-10),7)
        for mode in ('cos','wide','smooth7','band','sin','gauss','fixed'):
            mass=z[mode+'_loop_mass'] # [L,H,triangle]
            integrated=mass.sum(axis=(0,1))
            total=integrated.sum();value=float(np.dot(integrated,loop)/total) if total>0 else None
            bins=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                select=(perimeter>=lo)&(perimeter<hi);weight=integrated[select];den=weight.sum()
                bins.append(dict(perimeter_low=lo,perimeter_high=hi,count=int(select.sum()),window_mass=den,
                                 weighted_loop=float(np.dot(weight,loop[select])/den) if den>0 else None,
                                 unweighted_loop=float(loop[select].mean()) if select.any() else None))
            rows.append(dict(window=mode,integrated_window_loop_mean=value,integrated_mass=total,perimeter_bands=bins))
    return dict(case='same exactly validated continuous-space two-fermion G; frozen untrained-network Windows, not WC/AHC',
        integration='sum_{layer,head,triangle} m_ab*m_bc*m_ca*Im(G_abG_bcG_ca) / sum mass; invariant under Head permutations',
        empty_mass='undefined (null), no epsilon normalization',basis=summary['exact_density']['basis'],results=rows)


def head_study(directory):
    records=[]
    for path in sorted(directory.glob('synthetic-*/multiscale.npz')):
        with np.load(path,allow_pickle=False) as z:
            gram=z['head_gram'].mean(axis=0);trace=np.trace(gram,axis1=-2,axis2=-1)
            ratio=np.divide(gram.sum(axis=(-1,-2)),trace,out=np.full(trace.shape,np.nan),where=trace>0)
            negative=(gram<0)&(~np.eye(gram.shape[-1],dtype=bool))
            records.append(dict(id=path.parent.name,pooled_cancellation_ratio=ratio,
                                negative_head_pair_fraction=negative.sum(axis=(-1,-2))/(gram.shape[-1]*(gram.shape[-1]-1)),
                                mean_jaccard_over_grid=finite_mean(z['head_jaccard'],axis=(0,2,3))))
    return dict(definition='rho=<||sum c_h||²>/<sum ||c_h||²>, ratio of B/N moments, bias excluded',
                head_identity='no matching of head numbers across runs',records=records)


def hidden_feature_study(directory):
    records=[json.loads(p.read_text()) for p in directory.glob('synthetic-*/summary.json')];results=[]
    parts={'PH_only':slice(0,8),'H0_only':slice(0,4),'H1_only':slice(4,8),'distance_only':slice(8,None)}
    for mode in ('cos','wide','smooth7','band','sin','gauss','fixed'):
        subset=[r for r in records if r['run']['window']==mode]
        for feature in sorted(k for k in subset[0]['features'] if k.startswith('E_PH_')):
            for part,selection in parts.items():
                values=[r['features'][feature][selection] for r in subset]
                result=grouped_centroid(values,[r['run']['synthetic_geometry'] for r in subset],[r['run']['seed'] for r in subset])
                results.append(dict(window=mode,feature=feature,part=part,**result))
    return dict(label='synthetic ordered vs uniform; no WC/AHC labels',results=results,
        slices='PH summaries H0[0:4], H1[4:8]; ordinary distance quantiles [8:12] are a SEPARATE baseline')


def winding_figure(directory,path):
    source=directory/'synthetic-cos-seed11-ordered/multiscale.npz'
    with np.load(source,allow_pickle=False) as z:
        probability=finite_mean(z['rank_probability'],axis=1) # [L,rank,a,b]
        a=z['a_grid'][0];b=z['b_grid'][0];index=int(np.argmin(abs(a)))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="330">','<rect width="900" height="330" fill="white"/>',
         '<g font-family="sans-serif" font-size="13" fill="#17232d">',
         '<text x="35" y="25">Synthetic ordered geometry: winding probability vs physical distance cutoff, QK a=0</text>']
    colors=('#687886','#c27b1b','#7455a6')
    for layer in range(len(probability)):
        x=55+430*layer;y=65;w=340;h=180
        svg.extend([f'<text x="{x}" y="{y-13}">Layer {layer} (heads averaged)</text>',
            f'<path d="M{x},{y} V{y+h} H{x+w}" fill="none" stroke="#687886"/>',
            f'<text x="{x-20}" y="{y+10}">1</text>',f'<text x="{x-20}" y="{y+h}">0</text>',
            f'<text x="{x}" y="{y+h+18}">{b[0]:.2g}</text>',f'<text x="{x+w-15}" y="{y+h+18}">{b[-1]:.2g}</text>',
            f'<text x="{x+110}" y="{y+h+38}">distance cutoff b</text>'])
        for rank in range(3):
            v=probability[layer,rank,index];pts=' '.join(f'{x+w*(bb-b[0])/(b[-1]-b[0]):.2f},{y+h*(1-vv):.2f}' for bb,vv in zip(b,v))
            svg.append(f'<polyline points="{pts}" stroke="{colors[rank]}" stroke-width="2" fill="none"/>')
    for rank in range(3):svg.append(f'<text x="{150+240*rank}" y="320" fill="{colors[rank]}">winding rank {rank}</text>')
    svg.append('</g></svg>');path.write_text('\n'.join(svg))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('cde',type=Path);parser.add_argument('--physical',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    study=epsilon_study(args.cde)
    (args.output/'epsilon-study.json').write_text(json.dumps(json_safe(study),indent=2,allow_nan=False))
    modes=('cos','wide','smooth7','band','sin','gauss','fixed');eps=(1e-4,1e-8,1e-12)
    accuracy=np.array([[next(r['accuracy'] for r in study['results'] if r['window']==m and r['epsilon']==e) for m in modes] for e in eps])
    changes=np.array([[next(r['max_circulation_change'] for r in study['results'] if r['window']==m and r['epsilon']==e) for m in modes] for e in eps])
    heatmap_svg([('screening-loop geometry accuracy',accuracy),('max change across epsilon sweep',changes)],args.output/'epsilon-study.svg',
                'Epsilon sensitivity: columns cos/wide/smooth7/band/sin/gauss/fixed; rows 1e-4/1e-8/1e-12',xlabel='Window condition',ylabel='epsilon index')
    classification=json.loads((args.cde/'geometry-classification.json').read_text())['results'];panels=[]
    hidden_results=hidden_feature_study(args.cde)
    (args.output/'hidden-feature-controls.json').write_text(json.dumps(json_safe(hidden_results),indent=2,allow_nan=False))
    methods=('mean','mean_std','moments','gram_spectrum')
    for family in ('pool','PH_only','distance_only'):
        for layer in (0,1):
            if family=='pool':
                a=np.array([[next(r['accuracy'] for r in classification if r['window']==m and r['feature']==f'E_pool_{method}_L{layer}') for m in modes] for method in methods])
            else:
                a=np.array([[next(r['accuracy'] for r in hidden_results['results'] if r['window']==m and r['feature']==f'E_PH_{method}_L{layer}' and r['part']==family) for m in modes] for method in methods])
            panels.append((f'Hidden {family} L{layer} geometry accuracy',a))
    heatmap_svg(panels,args.output/'hidden-comparison.svg',
        'Hidden analyses: columns cos/wide/smooth7/band/sin/gauss/fixed; rows mean/mean_std/moments/Gram',xlabel='Window condition',ylabel='pooling method')
    (args.output/'head-redundancy-cancellation.json').write_text(json.dumps(json_safe(head_study(args.cde)),indent=2,allow_nan=False))
    winding_figure(args.cde,args.output/'distance-winding.svg')
    band_panels=[]
    for mode in ('cos','sin','gauss'):
        with np.load(args.cde/f'synthetic-{mode}-seed11-ordered/multiscale.npz',allow_pickle=False) as z:
            bands=finite_mean(z['window_band_fraction'],axis=0)
            for layer in range(len(bands)):band_panels.append((f'{mode} L{layer} off-diagonal Window mass',bands[layer]))
    heatmap_svg(band_panels,args.output/'head-distance-bands.svg','Head indices are local to each run; not classification features',
                xlabel='physical distance band',ylabel='Head (within run)')
    if args.physical:
        study=window_study(args.physical)
        (args.output/'physical-window-scales.json').write_text(json.dumps(json_safe(study),indent=2,allow_nan=False))
    (args.output/'source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
