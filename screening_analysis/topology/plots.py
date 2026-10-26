"""Scientific SVG panels from saved measurements, with explicit data lineage."""
import csv
import json
from pathlib import Path
import shutil
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
import numpy as np
from .workflow import ROOT, provenance, sha, write
from screening_analysis import topology as ph
from screening_analysis.report_topology_benchmark import line_svg


def canvas(title,width=1100,height=470):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/><g font-family="sans-serif" font-size="13" fill="#17232d">',
        f'<text x="25" y="26" font-size="17">{escape(title)}</text>']


def finish(svg,path):
    path.write_text('\n'.join(svg+['</g></svg>']));ET.parse(path)


def graphs(paths,path):
    from screening_analysis.graph_wind import minimum_image_edges
    svg=canvas('Actual trained candidates: identical input B0, L0 mean-head W; NOT verified WC/AHC',height=500)
    for panel,(label,probe_path) in enumerate(paths):
        with np.load(probe_path,allow_pickle=False) as z:
            L=z['lattice'];r=z['positions'][0].reshape(-1,2);w=ph.symmetrize(z['W_window'][0,0].mean(axis=0));f=(r@np.linalg.inv(L).T)%1
        displacement,length,ties=minimum_image_edges(r,L);ij=np.triu_indices(len(r),1)
        x=75+panel*540;y=85;size=320
        svg += [f'<text x="{x}" y="{y-22}">{escape(label)}; strict B &gt; 0.1</text>',
            f'<defs><clipPath id="c{panel}"><rect x="{x}" y="{y}" width="{size}" height="{size}"/></clipPath></defs>',
            f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="#fafafa" stroke="#999"/>',f'<g clip-path="url(#c{panel})">']
        for e,(i,j) in enumerate(zip(*ij)):
            if w[i,j]<=.1:continue
            for u in (-1,0,1):
                for v in (-1,0,1):
                    a=f[i]+[u,v];b=a+displacement[e]
                    svg.append(f'<line x1="{x+size*a[0]:.3f}" y1="{y+size*(1-a[1]):.3f}" x2="{x+size*b[0]:.3f}" y2="{y+size*(1-b[1]):.3f}" stroke="#31836a" stroke-opacity="{.25+.75*w[i,j]:.4f}" stroke-width="{.5+3*w[i,j]:.3f}"/>')
        for i,a in enumerate(f):svg += [f'<circle cx="{x+size*a[0]:.3f}" cy="{y+size*(1-a[1]):.3f}" r="5" fill="#255a91"/>']
        svg+=['</g>',f'<text x="{x}" y="{y+size+25}">fractional torus coordinates; edge width = B</text>']
    finish(svg,path)


def diagrams(paths,path):
    svg=canvas('Actual candidates B0, layer=0, head=0: finite H0/H1 and essential bars (not physical Chern)',width=1400,height=650)
    colors=['#2773a5','#c36b20']
    for row,(label,source) in enumerate(paths):
        with np.load(source,allow_pickle=False) as z:
            weights=[z['Window'][0,0,0],z['Alpha'][0,0,0],z['W_window'][0,0,0],np.clip((z['S'][0,0,0]+1)/2,0,1)]
        for col,(name,w) in enumerate(zip(['Window','Trim','Screening','QK mapped'],weights)):
            x=55+col*345;y=65+row*295;s=215
            svg += [f'<text x="{x}" y="{y-15}">{escape(label)} / {name}</text>',f'<path d="M{x},{y} V{y+s} H{x+s} L{x},{y}" fill="none" stroke="#aaa"/>',
                    f'<text x="{x+75}" y="{y+s+22}">birth u=1-B</text>',f'<text x="{x-32}" y="{y+110}" transform="rotate(-90 {x-32} {y+110})">death u</text>']
            for dim,d in enumerate(ph.persistence(w)):
                for birth,death in d:
                    yy=y+s*(1-death) if np.isfinite(death) else y-4
                    svg.append(f'<circle cx="{x+s*birth:.3f}" cy="{yy:.3f}" r="3" fill="{colors[dim]}" fill-opacity=".7"/>')
                count=int(np.isinf(d[:,1]).sum())
                svg.append(f'<text x="{x}" y="{y+s+40+16*dim}" fill="{colors[dim]}">H{dim}: {len(d)} bars, {count} essential (top)</text>')
    finish(svg,path)


def hidden_layers(directory,path):
    panels=[]
    for label,name in [('actual candidate','N9-lam033-off-seed11-B'),('synthetic untrained','synthetic-cos-seed11-ordered')]:
        folder=directory.parent/'layer-stress/stress-layers2-B/CDE' if label=='actual candidate' else ROOT/'screening_analysis/topology_cdef_results_2026-10-26_final'/name
        summary=json.loads((folder/'summary.json').read_text());curves=[]
        for pool,records in summary['hidden_PH'].items():
            values=[r[1]['total_persistence'] for r in records]
            curves.append((pool,np.arange(len(values)),values))
        panels.append((label+(' 2-layer stress' if label=='actual candidate' else '')+' (point-cloud PH)',curves))
    line_svg(panels,path,'Hidden H1 finite total persistence: actual vs synthetic kept separate; no zigzag claim',xlabel='layer index',ylabel='H1 total persistence [representation distance]')


def checkpoints_and_seeds(analysis,out):
    records=json.loads((analysis/'records.json').read_text());panels=[]
    for lam in ('lambda=0.33','lambda=0.72'):
        selected=[r for r in records if r['view']=='B' and r['mode']=='off' and r['seed']==11 and r['candidate_parameter_label']==lam and r['kind'] in ('core','checkpoint_drift')]
        selected.sort(key=lambda r:r['completed_updates'])
        y=[]
        for r in selected:
            a=np.asarray(r['features_by_count']['16']['topology_completed']);y.append(float(np.linalg.norm(a)))
        panels.append((lam+'; feature norm mixes saved PH statistics', [('seed11', [r['completed_updates'] for r in selected],y)]))
    line_svg(panels,out/'07-checkpoints.svg','Actual checkpoint drift on common inputs; not convergence certification',xlabel='completed updates',ylabel='completed-PH feature L2 norm')
    panels=[]
    for lam in ('lambda=0.33','lambda=0.72'):
        selected=sorted([r for r in records if r['view']=='A' and r['mode']=='off' and r['kind']=='core' and r['candidate_parameter_label']==lam],key=lambda r:r['seed'])
        panels.append((lam,[('fresh fixed-psi MC',[r['seed'] for r in selected],[r['measurement']['energy'] for r in selected])]))
    line_svg(panels,out/'08-seeds.svg','Actual off-model energies across independent seeds: short unconverged states',xlabel='training seed',ylabel='energy [total code units]')


def physical_panels(path):
    data=json.loads((ROOT/'screening_analysis/topology_f_results_2026-10-26_final/summary.json').read_text())
    records=sorted([r for r in data['independent_chern_benchmark']['records'] if r['size']==3 and r['grid']==10],key=lambda r:r['chern'])
    curves=[('independent QWZ',[r['chern'] for r in records],[r['max_abs_loop'] for r in records])]
    # Each point is a different known Hamiltonian mass; no Screenformer relation inferred.
    second=[('area-weighted loop',[r['chern'] for r in records],[r['area_weighted_loop_mean'] for r in records])]
    line_svg([('3x3 sites, exact occupied Slater; m=-1,+1,+3',curves),('Same independent model, 10x10 twist grid',second)],path,
        'Independent QWZ validation ONLY: fixed-basis loops vs many-body Chern; not WC/AHC',xlabel='independently computed Chern',ylabel=('max |Im(Gab Gbc Gca)|','mean area * Im(loop)'))


def report(input_dir,output):
    root=Path(input_dir).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    sa=ROOT/'screening_analysis';bench=root/'benchmark';analysis=sa/'topology_benchmark_analysis_2026-10-26_final'
    paths=[('lambda=.33 candidate',analysis/'N9-lam033-off-seed11/probe_B.npz'),('lambda=.72 candidate',analysis/'N9-lam072-off-seed11/probe_B.npz')]
    graphs(paths,out/'01-candidate-graphs.svg');diagrams(paths,out/'02-components-PD.svg')
    hidden_layers(bench,out/'06-hidden-layers.svg');checkpoints_and_seeds(analysis,out);physical_panels(out/'10-independent-physical-loops.svg')
    loop_panels=[]
    for field in ('qk','screening'):
        curves=[]
        for lam in ('033','072'):
            with np.load(bench/f'N9-lam{lam}-off-seed11-B/CDE/directed-statistics.npz',allow_pickle=False) as z:
                rms=z[field+'_by_perimeter'][...,2];valid=np.isfinite(rms).sum(axis=(0,1,2))
                means=np.divide(np.nansum(rms,axis=(0,1,2)),valid,out=np.full(rms.shape[-1],np.nan),where=valid>0)
                edges=z['perimeter_edges'];centres=(edges[:-1]+edges[1:])/2;keep=np.isfinite(means)
                curves.append(('lambda=0.'+lam[1:],centres[keep],means[keep]))
        loop_panels.append((field+' RMS, pooled 8 configurations and heads',curves))
    line_svg(loop_panels,out/'05-loop-comparison.svg','Actual candidate loop comparison B; empty perimeter bins omitted; not quantized',xlabel='triangle perimeter [code length]',ylabel='CCW circulation RMS')
    copies={
        '03-winding.svg':bench/'N9-lam033-off-seed11-B/AB/curves.svg',
        '04-multiscale.svg':bench/'N9-lam033-off-seed11-B/CDE/multiscale.svg',
        '05-directed-loops.svg':bench/'N9-lam033-off-seed11-B/CDE/directed-loops.svg',
        '09-condition-classification.svg':sa/'topology_g_benchmark_report_2026-10-26_final/candidate_condition_accuracy.svg',
        '11-regularized-training.svg':sa/'topology_g_benchmark_report_2026-10-26_final/regularizer_training.svg',
        'actual-projected-trace.svg':root/'physical-candidates/actual-projected-trace.svg',
        'synthetic-hidden-controls.svg':sa/'topology_cdef_comparison_2026-10-26_final/hidden-comparison.svg',
        'independent-QWZ-curvature.svg':sa/'topology_f_results_2026-10-26_final/chern-benchmark.svg'}
    for name,source in copies.items():shutil.copyfile(source,out/name);ET.parse(out/name)
    artifacts={};archive=out/'measurements';archive.mkdir()
    for key,folder,names in [
        ('AB',sa/'topology_results_2026-10-26_final',['pilot.json']),
        ('CDEF',sa/'topology_cdef_results_2026-10-26_final',['phase-classification.json','geometry-classification.json']),
        ('F',sa/'topology_f_results_2026-10-26_final',['summary.json']),
        ('G',sa/'topology_g_results_2026-10-26',['summary.json','source.json']),
        ('lambda',sa/'topology_benchmark_2026-10-26',['summary.json','plan.json','source.json']),
        ('analysis',analysis,['classification.json','records.json','distances.json','coordinate_perturbations.json','stress_comparisons.json']),
        ('symmetry',sa/'topology_benchmark_symmetry_2026-10-26_final',['summary.json'])]:
        dest=archive/key;dest.mkdir();artifacts[key]=[]
        for name in names:
            source=folder/name;shutil.copyfile(source,dest/name);artifacts[key].append(dict(source=str(source),sha256=sha(source),archived=str((dest/name).relative_to(root))))
    shutil.copyfile(root/'physical-candidates/summary.json',archive/'actual-candidate-1RDM.json')
    write(out/'provenance.json',provenance());write(out/'sources.json',dict(artifacts=artifacts,figures={name:dict(source=str(p),sha256=sha(p)) for name,p in copies.items()},
        retrospective_environment='Library versions captured at report collection, not reconstructed historical execution environments'))
    index=['# 実測図一覧','', 'WC/AHCの独立物理ラベルは未検証。01–09は実際の短時間学習候補（09はlambda条件分類）。',
        '06の右パネルとsyntheticは人工配置・未学習モデル。10とQWZは独立物理模型。11は自由N4の実VMC。',
        '存在しないWC/AHC精度・Screenformer Chern相関を描画していない。','']
    for p in sorted(out.glob('*.svg')):index += [f'## {p.stem}','',f'![{p.stem}]({p.name})','']
    (out/'FIGURES.md').write_text('\n'.join(index));print(out)
