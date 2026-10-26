"""Reproducible checkpoint/probe topology analysis; output directories must be new.

python -m screening_analysis.topology_run manifest.json --output NEW_DIRECTORY
Manifest: {"runs":[{"id":"...", "group":"independent training seed/run",
"phase":"WC|AHC|unknown", "phase_evidence":"...", "probe":"probe.npz"}]}
Or replace probe with folder, optional checkpoint/config_path/settings/lattice/nconf.
Paths are relative to the manifest. No input checkpoint/cache is written.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from screening_analysis import topology as ph
from screening_analysis.graph_wind import weighted_winding


def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [json_safe(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.bool_): return bool(value)
    return value


def graph_sources(probe):
    # A3 is alpha*m BEFORE optional clipping/diag interventions; aggregation is
    # saved as an extra check rather than silently replacing the requested A3.
    return {'window':probe.get('Window',np.ones_like(probe['Alpha'])),
            'trim':probe['Alpha'], 'screening':probe['W_window'],
            'qk':np.clip((probe['S']+1)/2,0,1)}


def analyze_probe(probe, output, run, rng_seed=20261026, null_repeats=4):
    """Full distributions [configuration,layer,head] and PH diagrams per graph."""
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    sources=graph_sources(probe); B,NL,NH,N,_=probe['S'].shape
    if min(B,NL,NH,N)<1 or type(null_repeats) is not int or null_repeats<1:
        raise ValueError('nonempty probe and positive integer null_repeats required')
    positions=np.asarray(probe['positions']).reshape(B,N,2); L=probe['lattice']
    ts=np.linspace(0,1,51); diagrams={}; rows=[]; curves={}; rng=np.random.default_rng(rng_seed)
    for name,weights in sources.items():
        for control in ('raw','rank','shuffle'):
            repeats=null_repeats if control=='shuffle' else 1
            bc=[]; densities=[]; ranks=[]
            for sample in range(B):
                for layer in range(NL):
                    for head in range(NH):
                        base=ph.symmetrize(weights[sample,layer,head])
                        for repeat in range(repeats):
                            w=(ph.rank_weights(base) if control=='rank' else
                               ph.shuffled_weights(base,rng) if control=='shuffle' else base)
                            a=ph.analyze_graph(w,ts)
                            try:
                                wind=weighted_winding(positions[sample],L,w,ts)
                                rank=wind['rank']; onset=(wind['critical_rank1'],wind['critical_rank2']); ambiguous=False
                            except ValueError as exc:
                                if 'degenerate shortest' not in str(exc): raise
                                rank=np.full(len(ts),np.nan); onset=(np.nan,np.nan); ambiguous=True
                            key=f'{name}_{control}_b{sample}_l{layer}_h{head}_r{repeat}'
                            for dim,d in enumerate(a['diagrams']):diagrams[key+f'_H{dim}']=d
                            diagrams[key+'_landscape']=a['landscape']
                            edges=w[np.triu_indices(N,1)]
                            row=dict(source=name,control=control,sample=sample,layer=layer,head=head,repeat=repeat,
                                critical_rank1=onset[0],critical_rank2=onset[1],ambiguous_embedding=ambiguous,
                                support_density=float(np.count_nonzero(base)/max(1,N*(N-1))),
                                weight_mean=float(edges.mean()) if len(edges) else 0.,
                                weight_std=float(edges.std()) if len(edges) else 0.,
                                beta0_auc=float(np.mean(a['betti'][0])),beta1_auc=float(np.mean(a['betti'][1])))
                            for dim,s in enumerate(a['summaries']):row.update({f'H{dim}_{k}':v for k,v in s.items()})
                            row.update({f'rank{r}_probability':float(np.mean(rank==r)) if not ambiguous else np.nan for r in range(3)})
                            rows.append(row); bc.append(a['betti']); densities.append(a['density']); ranks.append(rank)
            prefix=name+'_'+control
            curves[prefix+'_betti']=np.asarray(bc).reshape(B,NL,NH,repeats,2,len(ts))
            curves[prefix+'_density']=np.asarray(densities).reshape(B,NL,NH,repeats,len(ts))
            curves[prefix+'_wind']=np.asarray(ranks).reshape(B,NL,NH,repeats,len(ts))
            winding=curves[prefix+'_wind']; valid=np.isfinite(winding).sum(axis=(0,3))
            curves[prefix+'_wind_probability']=np.stack([
                np.divide((winding==r).sum(axis=(0,3)),valid,
                          out=np.full(valid.shape,np.nan),where=valid>0) for r in range(3)],axis=2)
    # Original periodic-norm distance baseline retained independently of learned weights.
    geometry=[]; geometry_rows=[]
    for sample in range(B):
        d=probe['d'][sample]; scale=float(probe['dref'])
        if not np.isfinite(scale) or scale<=0:raise ValueError('positive dref required')
        w=np.exp(-d/scale)
        geometry.append(ph.analyze_graph(w,ts)['betti'])
        try:
            wind=weighted_winding(positions[sample],L,w,ts)
            curves[f'geometry_b{sample}_wind']=wind['rank']
            geometry_rows.append(dict(sample=sample,**{f'critical_cutoff{r}':-scale*np.log(wind[f'critical_rank{r}'])
                                    for r in (1,2)},ambiguous_embedding=False))
        except ValueError as exc:
            if 'degenerate shortest' not in str(exc):raise
            curves[f'geometry_b{sample}_wind']=np.full(len(ts),np.nan)
            geometry_rows.append(dict(sample=sample,critical_cutoff1=np.nan,critical_cutoff2=np.nan,ambiguous_embedding=True))
    curves['geometry_betti']=np.asarray(geometry); curves['thresholds']=ts
    with (output/'diagrams.npz').open('xb') as f:np.savez_compressed(f,**diagrams)
    with (output/'curves.npz').open('xb') as f:np.savez_compressed(f,**curves)
    with (output/'metrics.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'geometry.json').write_text(json.dumps(json_safe(geometry_rows),indent=2,allow_nan=False))
    aggregate=[]
    for name in sources:
        for control in ('raw','rank','shuffle'):
            selected=[r for r in rows if r['source']==name and r['control']==control]
            keys=[k for k in rows[0] if k.startswith(('H0_','H1_','beta','rank','support_','weight_'))]
            aggregate.append(dict(source=name,control=control,**{k:float(np.mean([r[k] for r in selected])) for k in keys}))
    metadata=dict(run=run,probe_metadata=probe.get('metadata',{}),shape=[B,NL,NH,N],
        random_seed=rng_seed,null_repeats=null_repeats,aggregates=aggregate,
        statistical_unit='independent training group, NOT walkers/heads/checkpoints',
        absent_window='identity window (ones)',
        geometry='exp(-periodic_norm/dref); cutoff=-dref*log(t), strict d<cutoff; shortest-image edges',
        essential='infinite deaths retained in diagrams; excluded from finite persistence/landscape, counted separately',
        source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (__file__,ph.__file__,__import__('screening_analysis.topology.persistence',fromlist=['']).__file__,__import__('screening_analysis.graph_wind',fromlist=['']).__file__)})
    (output/'summary.json').write_text(json.dumps(json_safe(metadata),indent=2,ensure_ascii=False,allow_nan=False))
    plot_curves(curves,output/'curves.svg')
    return metadata


def plot_curves(curves,path):
    """Standalone SVG: standard line charts, no plotting dependency required."""
    colors={'window':'#3978a8','trim':'#cf7c18','screening':'#22865a','qk':'#a45da2'}
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="710" viewBox="0 0 1000 710">',
         '<rect width="1000" height="710" fill="white"/>',
         '<g font-family="sans-serif" font-size="13" fill="#222">',
         '<text x="45" y="24">Screening flag PH and embedded torus winding (not Chern number)</text>']
    panels=[('raw','betti',0,'Mean beta0'),('raw','betti',1,'Mean beta1'),
            ('rank','betti',1,'Mean beta1: positive-edge ranks'),('raw','density',None,'Edge density'),
            ('raw','wind',1,'P(winding rank &gt;=1)'),('raw','wind',2,'P(winding rank =2)')]
    for panel,(control,kind,index,title) in enumerate(panels):
        x=50+(panel%2)*490; y=68+(panel//2)*210; width=400; height=145
        lines={}
        for name in colors:
            a=curves[f'{name}_{control}_{kind}']
            if kind=='betti':a=a[...,index,:]
            if kind=='wind':
                valid=np.isfinite(a); a=np.where(valid,(a>=index).astype(float),np.nan)
            flat=a.reshape(-1,a.shape[-1]); count=np.isfinite(flat).sum(axis=0)
            lines[name]=np.divide(np.nansum(flat,axis=0),count,out=np.full(flat.shape[-1],np.nan),where=count>0)
        maximum=max(1.,max((np.nanmax(v) for v in lines.values() if np.isfinite(v).any()),default=1.))
        svg += [f'<text x="{x}" y="{y-12}">{title}</text>',f'<path d="M{x},{y} V{y+height} H{x+width}" fill="none" stroke="#999"/>',
                f'<text x="{x-32}" y="{y+8}">{maximum:.2g}</text>',f'<text x="{x-15}" y="{y+height}">0</text>',
                f'<text x="{x}" y="{y+height+18}">0</text>',f'<text x="{x+width-5}" y="{y+height+18}">1</text>',
                f'<text x="{x+140}" y="{y+height+18}">strict threshold t</text>']
        for name,values in lines.items():
            if not np.isfinite(values).all():continue
            points=' '.join(f'{x+j*width/(len(values)-1):.2f},{y+height-v/maximum*height:.2f}' for j,v in enumerate(values))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
    for i,(name,color) in enumerate(colors.items()):
        svg.append(f'<text x="{70+i*220}" y="700" fill="{color}">{name}</text>')
    svg.append('</g></svg>');Path(path).write_text('\n'.join(svg))


def compare_diagrams(first,second,output):
    """Matched graph keys; pairing is diagnostic, not independent phase evidence."""
    rows=[]
    with np.load(first,allow_pickle=False) as a,np.load(second,allow_pickle=False) as b:
        for key in sorted(a.files):
            if key not in b or not key.endswith(('_H0','_H1')):continue
            rows.append(dict(key=key,bottleneck=ph.diagram_distance(a[key],b[key],bottleneck=True),
                             wasserstein1=ph.diagram_distance(a[key],b[key])))
    Path(output).write_text(json.dumps(json_safe(rows),indent=2,allow_nan=False))
    return rows


def phase_classification(summaries):
    """Exploratory leave-training-group-out evaluation, never split walkers.

    Repeated checkpoints in a group/phase/window are averaged before fitting.
    No tuning, independent Chern validation, or significance claim is implied.
    """
    eligible=[r for r in summaries if r['run'].get('phase') in ('WC','AHC')
              and r['run'].get('phase_evidence') and r['run'].get('group')]
    if not eligible:return dict(status='unavailable',reason='WC/AHC labels with independent phase_evidence and training group are required')
    for phase in ('WC','AHC'):
        if len({r['run']['group'] for r in eligible if r['run']['phase']==phase})<3:
            return dict(status='insufficient_groups',reason='at least 3 independent training groups per phase required')
    keys=('H0_total_persistence','H1_total_persistence','H0_essential_intervals',
          'H1_essential_intervals','beta0_auc','beta1_auc','rank1_probability','rank2_probability')
    results=[]
    for source in ('window','trim','screening','qk'):
        for control in ('raw','rank','shuffle'):
            units={}
            for r in eligible:
                unit=(r['run']['group'],r['run']['phase'])
                a=next(a for a in r['aggregates'] if a['source']==source and a['control']==control)
                units.setdefault(unit,[]).append([a[k] for k in keys])
            X=np.array([np.mean(v,axis=0) for v in units.values()]);groups=np.array([k[0] for k in units])
            y=np.array([k[1]=='AHC' for k in units],int);predictions=[];truth=[]
            if not np.isfinite(X).all():
                results.append(dict(source=source,control=control,status='undefined_features'));continue
            for group in np.unique(groups):
                train=groups!=group;test=~train
                if len(np.unique(y[train]))<2:continue
                mean=X[train].mean(axis=0);scale=X[train].std(axis=0);scale=np.where(scale>1e-10,scale,1.)
                z=(X-mean)/scale;centroids=np.stack([z[train&(y==c)].mean(axis=0) for c in (0,1)])
                pred=np.argmin(np.sum((z[test,None,:]-centroids[None,:,:])**2,axis=-1),axis=-1)
                predictions.extend(pred.tolist());truth.extend(y[test].tolist())
            confusion=np.zeros((2,2),int)
            for actual,predicted in zip(truth,predictions):confusion[actual,predicted]+=1
            balanced=np.mean(np.diag(confusion)/confusion.sum(axis=1))
            results.append(dict(source=source,control=control,balanced_accuracy=float(balanced),
                                confusion_wc_ahc=confusion.tolist(),groups=len(set(groups))))
    return dict(status='exploratory',features=keys,results=results,
        caution='checkpoint means pooled per training group and phase; no independent Chern verification; no hyperparameter selection')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); manifest=json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True,exist_ok=False); summaries=[]
    identifiers=[r['id'] for r in manifest['runs']]
    if len(set(identifiers))!=len(identifiers) or any(Path(i).name!=i or i in ('.','..') for i in identifiers):
        raise ValueError('unique simple run ids required')
    base=args.manifest.resolve().parent
    for run in manifest['runs']:
        if 'probe' in run:
            path=base/run['probe']
            with np.load(path,allow_pickle=False) as z:
                probe={k:z[k] for k in z.files if k!='metadata_json'}
                probe['metadata']=json.loads(str(z['metadata_json']))
        else:
            from screening_analysis.attn_probe import probe as evaluate,save_probe
            kwargs={k:run[k] for k in ('lattice','nconf') if k in run}
            for k in ('checkpoint','config_path','settings'):
                if k in run:kwargs[k]=str(base/run[k]) if isinstance(run[k],str) else run[k]
            probe=evaluate(base/run['folder'],collect_internal=True,**kwargs)
        summary=analyze_probe(probe,args.output/run['id'],run,null_repeats=manifest.get('null_repeats',4))
        summaries.append(summary)
        if 'probe' not in run:save_probe(probe,args.output/run['id']/'probe.npz')
        print(run['id'],summary['shape'],flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (args.output/'summary.json').write_text(json.dumps(json_safe(summaries),indent=2,ensure_ascii=False,allow_nan=False))
    (args.output/'phase-classification.json').write_text(json.dumps(json_safe(phase_classification(summaries)),indent=2,allow_nan=False))
    if len(summaries)>1:
        first=args.output/identifiers[0]/'diagrams.npz'
        for identifier in identifiers[1:]:compare_diagrams(first,args.output/identifier/'diagrams.npz',args.output/f'distances-{identifier}.json')


if __name__=='__main__':main()
