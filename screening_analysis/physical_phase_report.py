"""Regenerate figures and Japanese reports using only saved CSV/NPZ/JSON.

This module deliberately has no JAX or model/checkpoint dependency.
python -m screening_analysis.physical_phase_report OUTPUT
"""
import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

COLORS = {4: '#2474b5', 25: '#d66b22'}
STYLES = {2: ('-', 'o'), 3: ('--', 's'), 4: ('-.', '^'), 10: (':', 'D')}
SOURCE = {'qk_affine': 'QK affine', 'alpha': 'Alpha', 'window': 'Window', 'screening': 'Screening'}
SC = dict(zip(SOURCE, ['#8856a7', '#2c7fb8', '#31a354', '#d95f0e']))
METRICS = {
    'structure_factor': ('S(G)', '通常の密度構造因子で、基準結晶波数における粒子密度の秩序を測る。'),
    'r90': ('R90 (minimum image)', '自己辺を除く総接続重みの90%を含む最小像距離。'),
    'crystal_increment': ('Fourier excess', '距離×擬スピン平均対照からの接続重み付き波数分布の増分。'),
    'residual_variance': ('Residual variance', '別配置で推定した距離×擬スピン条件平均からの標準化残差の分散。'),
    'residual_peak': ('Residual at G', '結晶参照波数に射影した符号付き標準化残差の平均。'),
    'edge_effective_heads': ('Effective heads: edges', '接続重みGram行列の参加率で、似た接続を持つheadの冗長性を測る。'),
    'output_effective_heads': ('Effective heads: outputs', 'Value・Gate・出力射影後のhead寄与Gram行列の参加率。'),
    'cancellation_ratio': ('Output sum/self power', 'head和の二乗ノルムを各head二乗ノルム和で割り、相殺・強め合いを測る。'),
    'phase_pearson': ('Phase-response Pearson', '実接続重みと内部辺介入の位相応答絶対値のPearson相関。'),
    'spin_rayleigh': ('Spin Rayleigh (norm.)', '次数規格化対称接続行列の中心化擬スピンベクトルへの射影。'),
    'crystal_rayleigh': ('Crystal Rayleigh (norm.)', '同じ次数規格化行列の結晶波数ベクトルへの射影。'),
    'circulation_rms': ('QK cycle RMS', '可縮な反時計回り三角形での順方向積−逆方向積のRMS。')}


def num(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


def mean(x):
    x=np.asarray(x,float);v=x[np.isfinite(x)]
    return float(v.mean()) if len(v) else np.nan


def avg(a, axis=0):
    a=np.asarray(a,float);valid=np.isfinite(a);count=valid.sum(axis=axis)
    return np.divide(np.where(valid,a,0).sum(axis=axis),count,
        out=np.full(np.shape(count),np.nan),where=count>0)


def fmt(x, digits=4):
    x=num(x)
    return f'{x:.{digits}g}' if np.isfinite(x) else 'NA'


def read_rows(root,name):
    path=root/'data'/(name+'.csv')
    if not path.exists(): return []
    with path.open(encoding='utf-8') as handle:return list(csv.DictReader(handle))


def take(rows, **conditions):
    return [r for r in rows if all(str(r.get(k))==str(v) or
        (isinstance(v,(int,float)) and num(r.get(k))==v) for k,v in conditions.items())]


def markdown_table(headers, rows):
    def cell(x):return str(x).replace('|',r'\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(cell(x) for x in headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])


def pearson(x,y):
    x,y=np.asarray(x,float).ravel(),np.asarray(y,float).ravel()
    finite=np.isfinite(x)&np.isfinite(y);x,y=x[finite],y[finite]
    return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and min(x.std(),y.std())>1e-14 else np.nan


def scale_adjusted_correlations(weight,response,screening,epsilon=1e-10):
    """Log-edge derivative already contains its edge-weight factor.

    At fixed directed forward/reverse ratio, dphi/dw_sym =
    (dphi/dlog_scale)/w_sym. Restrict to positive Screening support to avoid
    amplifying closed edges. This is still an internal, fixed-input diagnostic.
    """
    x,y,w=np.asarray(weight),np.asarray(response),np.asarray(screening)
    valid=(w>epsilon)&np.isfinite(x)&np.isfinite(y)&np.isfinite(w)
    return dict(active_fraction=float(valid.mean()),active_edges=int(valid.sum()),
        pearson_positive_support=pearson(x[valid],y[valid]),
        pearson_scale_adjusted=pearson(x[valid],y[valid]/w[valid]))


def eigenspace_groups(eigenvalues,crystal_overlap,spin_overlap,tolerance=1e-8):
    """Sum squared projections within degenerate eigenspaces, not eigenvectors."""
    ev=np.asarray(eigenvalues);groups=[];start=0
    while start<len(ev):
        stop=start+1
        while stop<len(ev) and abs(ev[stop]-ev[start])<=tolerance:stop+=1
        groups.append(dict(rank_start=start,dimension=stop-start,eigenvalue=mean(ev[start:stop]),
            crystal_projector_overlap=mean(np.sum(crystal_overlap[start:stop],axis=0)),
            spin_projector_overlap=mean(spin_overlap[start:stop])*(stop-start)))
        start=stop
    return groups


class Report:
    def __init__(self,root):
        self.root=Path(root);self.root.joinpath('figures').mkdir(exist_ok=True)
        self.rows={k:read_rows(self.root,k) for k in ('metrics','head_metrics','distance','distance_summary',
            'residual','head_pairs','phase','conditional','eigen','wavefunction','circulation','configuration_metrics')}
        self.config=json.loads((self.root/'diagnostics/analysis_config.json').read_text())
        self.validation=json.loads((self.root/'diagnostics/validation.json').read_text())
        self.manifest=json.loads((self.root/'diagnostics/input_manifest.json').read_text())
        self.banks={};self.captions={}
        for row in self.rows['metrics']:
            key=(int(num(row['rs'])),int(row['heads']),row['subset'])
            path=self.root/'cache'/row['model_id']/row['subset']/'analysis-arrays.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as z:self.banks[key]={k:z[k] for k in z.files}
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.grid':True,
            'grid.alpha':.2,'savefig.dpi':160,'figure.constrained_layout.use':True})
        self.hrep=4 if (4,4,'common') in self.banks else next(iter(self.banks),(4,2,'common'))[1]

    def save(self,fig,name,caption,normalization):
        fig.supxlabel(normalization,fontsize=8)
        fig.savefig(self.root/'figures'/(name+'.png'),bbox_inches='tight');plt.close(fig)
        self.captions[name]=caption

    def title(self,rs,h,subset):
        a=self.banks.get((rs,h,subset));b=len(a['positions']) if a else 0
        return f'rs={rs}, H={h}, {subset}, B={b}'

    def derived_tables(self):
        self.adjusted=[];self.subspaces=[]
        for key,a in self.banks.items():
            rs,H,subset=key;model=take(self.rows['metrics'],rs=rs,heads=H,subset=subset)[0]['model_id']
            for source in SOURCE:
                for h in range(H):
                    for channel in ('amplitude','phase'):
                        name=f'sensitivity_{source}_{channel}_l0_h{h}'
                        screen=f'sensitivity_screening_{channel}_l0_h{h}'
                        if name not in a:continue
                        self.adjusted.append(dict(model_id=model,rs=rs,heads=H,subset=subset,source=source,head=h,
                            channel=channel,**scale_adjusted_correlations(a[name][0],a[name][1],a[screen][0])))
            for b in range(len(a['positions'])):
                for h in range(H):
                    stem=f'b{b}_l0_h{h}'
                    groups=eigenspace_groups(a['eigen_eigenvalues_'+stem],a['eigen_crystal_mode_overlap_'+stem],a['eigen_spin_overlap_'+stem])
                    self.subspaces.extend(dict(model_id=model,rs=rs,heads=H,subset=subset,sample=b,head=h,**g) for g in groups)
        for name,rows in [('phase_adjusted',self.adjusted),('eigenspace_metrics',self.subspaces)]:
            with (self.root/'data'/(name+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['status']);writer.writeheader()
                writer.writerows([{k:('' if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()} for row in rows])

    def spectrum(self,a,source):
        keys=sorted(k for k in a if k.startswith('fourier_'+source+'_l0_h'))
        return avg(np.stack([a[k] for k in keys]),axis=(0,1))

    def fourier(self):
        h=self.hrep;fig,axes=plt.subplots(2,5,figsize=(19,7))
        for ri,rs in enumerate((4,25)):
            a=self.banks.get((rs,h,'native'))
            if a is None:continue
            q=a['q_vectors'];n=int(np.sqrt(len(q)));mode=a['integer_modes']
            for ci,source in enumerate(('structure','qk_affine','window','alpha','screening')):
                if source=='structure':value=avg(np.stack([a[k] for k in a if k.startswith('structure_b')]),axis=0)
                else:value=self.spectrum(a,source)
                value=value.copy();value[(mode==0).all(1)]=np.nan
                ax=axes[ri,ci]
                im=ax.pcolormesh(q[:,0].reshape(n,n),q[:,1].reshape(n,n),value.reshape(n,n),shading='nearest',cmap='viridis' if source=='structure' else 'coolwarm')
                ax.set(xlabel='qx [(rs aB)^-1]',ylabel='qy [(rs aB)^-1]',title=f'{source}\n'+self.title(rs,h,'native'))
                reference=a['shell']@(2*np.pi*np.linalg.inv(a['lattice']))
                ax.scatter(reference[:,0],reference[:,1],marker='o',s=22,facecolors='none',edgecolors='k',linewidths=.7)
                fig.colorbar(im,ax=ax,shrink=.8,label='S(q)' if source=='structure' else 'weighted pair cosine')
        self.save(fig,'attention_fourier',f'Native配置の波数分布。黒丸は基準結晶波数G。H={h}の全head平均、通常S(q)のみ自己項を含む。接続波数分布は自己辺を除き各配置の重み和で規格化。q=0は表示から除外。カラーバーは各panelで異なる。','S(q)=|rho|^2/N; S_A=sum A*cos/sum A, head mean; q=0 omitted; circles=reference G')
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ri,rs in enumerate((4,25)):
            for ci,subset in enumerate(('native','common')):
                a=self.banks.get((rs,h,subset));ax=axes[ri,ci]
                if a is None:continue
                shell=a['shell'];base=shell[0]//np.gcd.reduce(abs(shell[0]))
                mode=a['integer_modes'];cross=mode[:,0]*base[1]-mode[:,1]*base[0]
                ids=np.where(cross==0)[0];t=mode[ids]@base/(base@base);order=np.argsort(t);ids=ids[order];t=t[order]
                for j,source in enumerate(SOURCE):ax.plot(t,self.spectrum(a,source)[ids],list(STYLES.values())[j][0],color=SC[source],label=SOURCE[source])
                for control,ls in [('equal',':'),('distance','--'),('spin','-.')]:ax.plot(t,avg(a['control_'+control],axis=0)[ids],ls,color='0.5',label=control)
                qk=avg(a['control_qk_signed'],axis=0);ax.plot(t,qk[ids],':',color='#6a3d9a',label='signed QK / L1')
                ax.set(title=self.title(rs,h,subset),xlabel='integer multiple of primitive reciprocal direction',ylabel='weighted pair cosine [1]');ax.legend(ncol=2,fontsize=7)
        self.save(fig,'fourier_profiles',f'逆格子方向の断面。H={h}で全head平均。等重み・距離のみ・spinのみ・符号付きQK/L1を比較し、入力配置が作るピークを確認する。','Each configuration normalized before averaging; signed QK uses sum |S|; no uncertainty bars')

    def distances(self):
        h=self.hrep;fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ri,rs in enumerate((4,25)):
            for ci,subset in enumerate(('native','common')):
                a=self.banks.get((rs,h,subset));ax=axes[ri,ci]
                if a is None:continue
                distances=a['network_distances'].ravel();edges=np.linspace(0,float(distances.max())*1.00001,17);x=(edges[:-1]+edges[1:])/2
                for j,source in enumerate(SOURCE):
                    w=avg(np.stack([a[k] for k in a if k.startswith('weights_'+source+'_l0_h')]),axis=0).ravel()
                    density=np.histogram(distances,bins=edges,weights=w)[0];density=density/density.sum() if density.sum()>0 else density*np.nan
                    ax.plot(x,np.cumsum(density),list(STYLES.values())[j][0],color=SC[source],label=SOURCE[source])
                ax.axhline(.9,color='0.5',ls=':');ax.set(xlabel='network periodic distance [rs aB]',ylabel='cumulative pooled edge mass [1]',ylim=(0,1.04),title=self.title(rs,h,subset));ax.legend(fontsize=8)
        self.save(fig,'distance_profiles',f'距離ごとの累積接続重み。H={h}のhead平均をプールした図。個別headの正確なR50/R90/R95はdistance_summary.csvに保存。','Off-diagonal symmetric weights; pooled edge mass CDF (not average edge value)')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ci,subset in enumerate(('native','common')):
            ax=axes[ci]
            for rs in (4,25):
                for H,(ls,mk) in STYLES.items():
                    rows=take(self.rows['head_metrics'],rs=rs,heads=H,subset=subset,source='screening')
                    if not rows:continue
                    ax.scatter([num(r['width']) for r in rows],[num(r['r90_network']) for r in rows],marker=mk,color=COLORS[rs],label=f'rs={rs}, H={H}')
            lo,hi=ax.get_xlim();ax.plot([0,hi],[0,hi],':',color='0.5',label='R90 = width')
            ax.set(title=f'{subset}; B={self.config["samples"]}/model; each point=head',xlabel='Window width (network metric) [rs aB]',ylabel='Screening R90 (network metric) [rs aB]');ax.legend(fontsize=7,ncol=2)
        self.save(fig,'width_r90','Window幅と実ScreeningのR90。同じネットワーク周期ノルムで比較する。物理的な最小像距離R90とは区別する。','Pooled weighted quantile per head; no head matching between different H')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ri,rs in enumerate((4,25)):
            ax=axes[ri];rows=take(self.rows['distance'],rs=rs,heads=h,subset='native',distance_kind='minimum_image',source='screening')
            for spin,ls in [('same','-'),('opposite','--')]:
                rr=take(rows,spin=spin);bins=sorted({int(r['bin']) for r in rr})
                x=[];y=[]
                total=sum(num(r['total_per_configuration']) for r in rr)
                for b in bins:
                    selected=take(rr,bin=b);x.append(mean([num(r['distance_mid']) for r in selected]));y.append(sum(num(r['total_per_configuration']) for r in selected)/total if total>0 else np.nan)
                ax.plot(x,y,ls,color=COLORS[rs],marker='o' if spin=='same' else 's',label=spin)
            ax.set(title=self.title(rs,h,'native'),xlabel='minimum-image distance [rs aB]',ylabel='normalized mass per bin [1]');ax.legend()
        self.save(fig,'spin_distance','同・異擬スピンのScreening接続距離分布。各spin群の総重みで別々に規格化し、距離分布の形を比べる。','Symmetric off-diagonal Screening; each spin group has unit mass; all heads pooled')

    def residuals(self):
        h=self.hrep;fig,axes=plt.subplots(2,3,figsize=(16,8))
        for ri,rs in enumerate((4,25)):
            a=self.banks.get((rs,h,'common'))
            if a is None:continue
            for ci,source in enumerate(('qk_affine','screening')):
                x=a[f'residual_prediction_{source}_l0_h0'].ravel();y=a[f'weights_{source}_l0_h0'].ravel();valid=np.isfinite(x)&np.isfinite(y)
                ax=axes[ri,ci];ax.scatter(x[valid][::5],y[valid][::5],s=3,alpha=.15,color=COLORS[rs]);ax.set(xlabel='held-out distance x spin prediction [1]',ylabel='actual symmetric edge [1]',title=SOURCE[source]+'; head 0\n'+self.title(rs,h,'common'))
            ax=axes[ri,2]
            for source,ls in [('qk_affine','-'),('screening','--')]:
                vals=avg(np.stack([a[k] for k in a if k.startswith('residual_spectrum_'+source+'_l0_h')]),axis=(0,1));mask=np.any(np.all(a['integer_modes'][:,None]==a['shell'][None],axis=-1),axis=1)
                ax.plot(np.arange(mask.sum()),vals[mask],ls,marker='o' if source=='screening' else 's',label=SOURCE[source])
            ax.set(xlabel='reference G integer mode',xticks=np.arange(mask.sum()),xticklabels=[f'{i},{j}' for i,j in a['integer_modes'][mask]],ylabel='<Z cos(G dr)> [1]',title=self.title(rs,h,'common'));ax.legend()
        self.save(fig,'geometric_residuals',f'距離×spinだけから予測する条件平均と実重み、および標準化残差の結晶波数射影。共通配置H={h}。QK affineのZは線形変換が打ち切られない範囲で符号付きQKのZと同じ。head 0は可視化例でありモデル間で対応づけない。','2-fold whole-configuration crossfit; training bin count >= '+str(self.config['min_bin_count'])+'; zero SD / unseen bins = NA')
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ri,rs in enumerate((4,25)):
            for ci,subset in enumerate(('native','common')):
                ax=axes[ri,ci]
                for source,ls in [('qk_affine','-'),('screening','--')]:
                    rows=take(self.rows['residual'],rs=rs,heads=h,subset=subset,source=source)
                    ids=sorted({int(r['distance_bin']) for r in rows});x=[];y=[]
                    for b in ids:
                        rr=take(rows,distance_bin=b);x.append(mean([num(r['distance']) for r in rr]));y.append(mean([num(r['residual_mean']) for r in rr]))
                    ax.plot(x,y,ls,marker='o' if source=='screening' else 's',label=SOURCE[source])
                ax.axhline(0,color='0.5',ls=':');ax.set(title=self.title(rs,h,subset),xlabel='network periodic distance [rs aB]',ylabel='held-out mean Z [1]');ax.legend()
        self.save(fig,'residual_profiles','幾何残差の距離依存性。評価側bin平均が0になるよう強制していないため、推定の分布ずれも確認できる。','Configuration crossfit; bin counts and valid coverage recorded in residual.csv / head_metrics.csv')
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for ri,rs in enumerate((4,25)):
            a=self.banks.get((rs,h,'common'))
            if a is None:continue
            q=a['q_vectors'];n=int(np.sqrt(len(q)))
            for ci,source in enumerate(('qk_affine','screening')):
                v=avg(np.stack([a[k] for k in a if k.startswith('residual_spectrum_'+source+'_l0_h')]),axis=(0,1))
                ax=axes[ri,ci];bound=max(float(np.nanmax(abs(v))),1e-10)
                im=ax.pcolormesh(q[:,0].reshape(n,n),q[:,1].reshape(n,n),v.reshape(n,n),shading='nearest',cmap='coolwarm',vmin=-bound,vmax=bound)
                ax.set(title=SOURCE[source]+'\n'+self.title(rs,h,'common'),xlabel='qx [(rs aB)^-1]',ylabel='qy [(rs aB)^-1]');fig.colorbar(im,ax=ax,label='<Z cos(q dr)> [1]')
        self.save(fig,'residual_fourier','共通配置でのQK/Screening幾何残差の2次元波数スペクトル。正負を保存した有効辺平均であり、確率分布や構造因子ではない。','Held-out standardized residual; valid-edge mean per configuration, then head/configuration mean')

    def heads(self):
        h=self.hrep;fig,axes=plt.subplots(2,3,figsize=(15,8))
        for ri,rs in enumerate((4,25)):
            rows=take(self.rows['head_pairs'],rs=rs,heads=h,subset='common')
            for ci,field in enumerate(('edge_pearson','js_nats','output_pearson')):
                mat=np.full((h,h),np.nan)
                for r in rows:
                    i,j=int(r['head_a']),int(r['head_b']);mat[i,j]=mat[j,i]=num(r[field])
                ax=axes[ri,ci];im=ax.imshow(mat,cmap='viridis' if ci==1 else 'coolwarm',vmin=0 if ci==1 else -1,vmax=None if ci==1 else 1)
                ax.set(title=field+'\n'+self.title(rs,h,'common'),xlabel='head index',ylabel='head index',xticks=range(h),yticks=range(h));fig.colorbar(im,ax=ax,shrink=.8,label='nats' if ci==1 else 'Pearson [1]')
        self.save(fig,'head_specialization','接続Pearson・JS divergenceと、Value/Gate/出力射影後のhead寄与Pearson。接続の類似性だけでは出力機能の同一性を判定できない。','Screening off-diagonal symmetric edges; JS on pooled unit-mass edges; outputs include diagonal Value response')
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ri,rs in enumerate((4,25)):
            rows=take(self.rows['head_metrics'],rs=rs,heads=h,subset='common')
            for j,source in enumerate(SOURCE):
                rr=take(rows,source=source);axes[ri,0].plot([int(r['head']) for r in rr],[num(r['spin_selectivity']) for r in rr],list(STYLES.values())[j][0],marker=list(STYLES.values())[j][1],color=SC[source],label=SOURCE[source])
            axes[ri,0].set(title=self.title(rs,h,'common'),xlabel='head index',ylabel='(mean same - mean opposite)/(sum) [1]');axes[ri,0].legend(fontsize=8)
            for subset,ls in [('native','-'),('common','--')]:
                rr=sorted(take(self.rows['metrics'],rs=rs,subset=subset),key=lambda r:int(r['heads']))
                for key,mk in [('edge_effective_heads','o'),('output_effective_heads','s')]:axes[ri,1].plot([int(r['heads']) for r in rr],[num(r[key]) for r in rr],ls,marker=mk,label=subset+' '+key.split('_')[0],color=COLORS[rs] if mk=='o' else '#8856a7')
            axes[ri,1].set(title=f'rs={rs}; B={self.config["samples"]}/model',xlabel='number of heads H',ylabel='Gram participation ratio [1]');axes[ri,1].legend(fontsize=8)
        self.save(fig,'head_roles','左は距離未調整の同・異spin選択性、右は接続と出力それぞれの有効head数。正確な距離調整は条件付き残差・相関と既存representation解析を参照。','Head-order invariant Gram statistics; each H changes parameter count; not seed uncertainty')
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ri,rs in enumerate((4,25)):
            for ci,field in enumerate(('dlogabs','dphase')):
                ax=axes[ri,ci]
                for H,(ls,mk) in STYLES.items():
                    rows=take(self.rows['wavefunction'],rs=rs,heads=H,subset='common',kind='head_ablation')
                    heads=sorted({int(r['head']) for r in rows})
                    ax.plot(heads,[mean([abs(num(r[field])) for r in take(rows,head=i)]) for i in heads],ls,marker=mk,label=f'H={H}')
                ax.set(title=f'rs={rs}; common; B={self.config["ablation_samples"]}',xlabel='head index (within each model)',ylabel='mean |delta '+('log|Psi|' if ci==0 else 'phase [rad]')+'|');ax.legend()
        self.save(fig,'head_ablation','出力射影W_Oの対応行を0にしてheadを除去し、同一配置の波動関数を再計算。自己項も含む除去。再最適化なしの有限応答で、headの独立エネルギー寄与ではない。','Fixed checkpoint/input; shared output bias retained; phase difference wrapped to (-pi, pi]')

    def cycles(self):
        h=self.hrep;fig,axes=plt.subplots(2,3,figsize=(16,8))
        for ri,rs in enumerate((4,25)):
            a=self.banks.get((rs,h,'common'))
            if a is not None:
                probe_path=self.root/'cache'/take(self.rows['metrics'],rs=rs,heads=h,subset='common')[0]['model_id']/'common/probe.npz'
                with np.load(probe_path,allow_pickle=False) as p:
                    s=p['S'][:,0];i,j=np.triu_indices(s.shape[-1],1);asym=(s-s.swapaxes(-1,-2))[...,i,j]
                axes[ri,0].hist(asym.ravel(),bins=60,density=True,color=COLORS[rs],histtype='step',label='Sij - Sji')
                vals=np.concatenate([a[k] for k in a if k.startswith('circulation_qk_signed_')]);axes[ri,1].hist(vals,bins=70,density=True,color=COLORS[rs],histtype='step',label='QK product cycle')
            for subset,ls,mk in [('native','-','o'),('common','--','s')]:
                rows=sorted(take(self.rows['metrics'],rs=rs,subset=subset),key=lambda r:int(r['heads']));axes[ri,2].plot([int(r['heads']) for r in rows],[num(r['circulation_rms']) for r in rows],ls,marker=mk,color=COLORS[rs],label=subset)
            axes[ri,0].set(title=self.title(rs,h,'common'),xlabel='QK antisymmetric component [1]',ylabel='probability density [1]')
            axes[ri,1].set(title=self.title(rs,h,'common'),xlabel='J = forward product - reverse product [1]',ylabel='probability density [1]')
            axes[ri,2].set(title=f'rs={rs}; B={self.config["samples"]}/model',xlabel='number of heads H',ylabel='QK product-cycle RMS [1]')
            for ax in axes[ri]:ax.legend(fontsize=8)
        self.save(fig,'qk_circulation','方向付きQKの非対称成分と三角形積循環。周期最短像で可縮かつ非縮退の三角形のみ使用。右は配置・headごとのRMS平均で、結晶やHall応答の証明ではない。','Directed scores retained; no edge symmetrization for cycles; pooled histograms are descriptive, no p-values')

    def phase(self):
        h=self.hrep;fig,axes=plt.subplots(2,3,figsize=(16,8))
        for ri,rs in enumerate((4,25)):
            a=self.banks.get((rs,h,'common'))
            if a is None:continue
            for ci,source in enumerate(('qk_affine','window','screening')):
                keys=[k for k in a if k.startswith('sensitivity_'+source+'_phase_l0_h')]
                if not keys:continue
                x=np.concatenate([a[k][0].ravel() for k in keys]);y=np.concatenate([a[k][1].ravel() for k in keys]);v=np.isfinite(x)&np.isfinite(y)
                axes[ri,ci].scatter(x[v][::2],y[v][::2],s=3,alpha=.12,color=SC[source]);axes[ri,ci].set_yscale('symlog',linthresh=1e-6)
                phase=take(self.rows['phase'],rs=rs,heads=h,subset='common',source=source,channel='phase');B=max([int(r['samples']) for r in phase],default=0)
                axes[ri,ci].set(title=f'{SOURCE[source]}; rs={rs}, H={h}, B={B}',xlabel='symmetric pair weight [1]',ylabel='|d phase/d pair log-scale| [rad]')
        self.save(fig,'phase_sensitivity','共通入力での接続重みと内部介入位相感度の散布図。全headを色分けせずまとめた分布の表示。相関係数はhead別にphase.csvへ保存。感度は対称辺のlog倍率に関する微分。','Internal interventions on fixed inputs; not a physical perturbation; y-axis symlog, linear threshold=1e-6 rad')
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ri,rs in enumerate((4,25)):
            for ci,channel in enumerate(('phase','amplitude')):
                ax=axes[ri,ci]
                for subset,ls,mk in [('native','-','o'),('common','--','s')]:
                    vals=[]
                    for source in SOURCE:vals.append(mean([num(r['pearson']) for r in take(self.rows['phase'],rs=rs,heads=h,subset=subset,source=source,channel=channel)]))
                    ax.plot(range(4),vals,ls,marker=mk,color=COLORS[rs],label=subset)
                ax.set(xticks=range(4),xticklabels=list(SOURCE.values()),ylabel='mean head Pearson [1]',title=f'rs={rs}, H={h}, {channel}; B<= {self.config["phase_samples"]}');ax.legend()
        self.save(fig,'phase_correlations','内部介入の振幅・位相感度とのPearson相関を比較。head平均は記述量であり、headや辺を独立runとして誤差推定しない。','Within-head edge correlations on aligned derivative inputs; undefined constant correlations = NA')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ri,rs in enumerate((4,25)):
            ax=axes[ri]
            for source in SOURCE:
                rr=take(self.rows['conditional'],rs=rs,heads=h,subset='common',source=source,channel='phase',control='distance')
                bins=sorted({int(r['label']) for r in rr});ax.plot(bins,[mean([num(r['pearson']) for r in take(rr,label=b)]) for b in bins],label=SOURCE[source],color=SC[source],marker='o',ms=3)
            ax.set(title=f'rs={rs}, H={h}, common; B<= {self.config["phase_samples"]}',xlabel='network distance bin',ylabel='conditional Pearson [1]');ax.legend()
        self.save(fig,'phase_distance','距離binを固定した位相感度との相関。辺数が不足するbinは欠損。spin固定・距離×spin固定のPearson/Spearmanもconditional.csvに保存。','Correlations within bins (>=10 finite edges); no seed-level uncertainty; head mean for visualization')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ri,rs in enumerate((4,25)):
            ax=axes[ri]
            for field,ls,mk in [('pearson_positive_support','-','o'),('pearson_scale_adjusted','--','s')]:
                vals=[mean([num(r[field]) for r in take(self.adjusted,rs=rs,heads=h,subset='common',source=source,channel='phase')]) for source in SOURCE]
                ax.plot(range(4),vals,ls,marker=mk,color=COLORS[rs],label='positive support' if field.endswith('support') else 'derivative / Screening')
            ax.set(xticks=range(4),xticklabels=list(SOURCE.values()),ylabel='mean head Pearson [1]',title=f'rs={rs}, H={h}, common, B<= {self.config["phase_samples"]}');ax.legend(fontsize=8)
        self.save(fig,'phase_scale_control','log辺倍率への微分にはScreening重み自体が掛かる。この機械的な要因を点検するため、正重み辺のみの相関と、感度を実Screening対称重みで割った相関を比較。後者も内部診断であり物理摂動ではない。','Active Screening > 1e-10; dphi/dw_sym at fixed directed ratio = (dphi/dlog-scale)/w_sym; no uncertainty bars')
        self.phase_edges()

    def phase_edges(self):
        h=self.hrep;fig,axes=plt.subplots(2,2,figsize=(11,9))
        for ri,rs in enumerate((4,25)):
            a=self.banks.get((rs,h,'common'))
            if a is None or 'sensitivity_screening_phase_l0_h0' not in a:continue
            x=a['positions'][0].reshape(-1,2);n=len(x);ij=np.triu_indices(n,1)
            # Draw original cell endpoints; periodic branches are not inferred
            # from a line crossing. The figure is a connectivity illustration.
            xx=a['sensitivity_screening_phase_l0_h0'];weights=xx[0,0];response=xx[1,0]
            for ci,(values,label) in enumerate(((weights,'Screening weight'),(response,'phase sensitivity'))):
                ax=axes[ri,ci];count=max(1,int(.1*len(values)));ids=np.argsort(-values)[:count]
                segments=np.stack((x[ij[0][ids]],x[ij[1][ids]]),axis=1);ax.add_collection(LineCollection(segments,color=COLORS[rs],linewidths=.7,alpha=.35))
                ax.scatter(x[:,0],x[:,1],c=a['spins'][0],cmap='coolwarm',vmin=-1,vmax=1,s=18,edgecolors='k',linewidths=.3)
                ax.autoscale();ax.set_aspect('equal');ax.set(title=f'rs={rs}, H={h}, common, B=1, head=0\nTop 10% '+label,xlabel='x [rs aB]',ylabel='y [rs aB]')
        self.save(fig,'phase_edges','重みの大きい辺と位相感度の大きい辺の可視化例。点の色は擬スピン。セル内の端点を直線で結んだ接続図であり、線の長さはPBCの距離を表さない。厳密なtop辺の閾値に同順位がある場合、一致率はNAとする。','One common input, representative head 0; top ceil(0.1*N*(N-1)/2) edges; no physical current interpretation')

    def eigen(self):
        h=self.hrep;fig,axes=plt.subplots(2,3,figsize=(16,8))
        for ri,rs in enumerate((4,25)):
            a=self.banks.get((rs,h,'native'))
            if a is not None:
                vals=np.stack([a[k] for k in a if k.startswith('eigen_eigenvalues_')]);axes[ri,0].plot(np.arange(vals.shape[1]),avg(vals,axis=0),'-o',ms=3,color=COLORS[rs]);axes[ri,0].fill_between(np.arange(vals.shape[1]),np.min(vals,axis=0),np.max(vals,axis=0),color=COLORS[rs],alpha=.15)
                x=a['positions'][0].reshape(-1,2);vec=a['eigen_eigenvectors_b0_l0_h0'][:,-1];im=axes[ri,2].scatter(x[:,0],x[:,1],c=vec,cmap='coolwarm',s=40);fig.colorbar(im,ax=axes[ri,2],label='eigenvector amplitude [1]')
                axes[ri,2].set(title=self.title(rs,h,'native')+'\nB=1 spatial example, head=0, minimum mode',xlabel='x [rs aB]',ylabel='y [rs aB]');axes[ri,2].set_aspect('equal')
            for subset,ls in [('native','-'),('common','--')]:
                rr=sorted(take(self.rows['metrics'],rs=rs,subset=subset),key=lambda r:int(r['heads']))
                for field,mk in [('spin_rayleigh','o'),('crystal_rayleigh','s')]:axes[ri,1].plot([int(r['heads']) for r in rr],[num(r[field]) for r in rr],ls,marker=mk,label=subset+' '+field.split('_')[0],color=COLORS[rs] if mk=='o' else '#8856a7')
            axes[ri,0].set(title=self.title(rs,h,'native'),xlabel='eigenvalue rank (descending)',ylabel='normalized adjacency eigenvalue [1]')
            axes[ri,1].set(title=f'rs={rs}; B={self.config["samples"]}/model',xlabel='number of heads H',ylabel='normalized Rayleigh [1]');axes[ri,1].legend(fontsize=8)
        self.save(fig,'eigenmode_projection','次数規格化対称行列の固有値と物理ベクトルへのRayleigh射影。固有値の帯は配置×headの最小〜最大で、信頼区間ではない。右の単一固有ベクトルの符号は任意で、縮退時の向きには意味を与えない。','Same matrix/configuration for Ritz bounds and Rayleigh; zero degree -> zero inverse; no cross-model eigenvector matching')

    def comparison(self):
        keys=list(METRICS);fig,axes=plt.subplots(1,2,figsize=(19,8))
        for ci,subset in enumerate(('native','common')):
            rows=sorted(take(self.rows['metrics'],subset=subset),key=lambda r:(int(num(r['rs'])),int(r['heads'])))
            if not rows:continue
            x=np.array([[num(r[k]) for r in rows] for k in keys]);center=avg(x,axis=1);sd=np.nanstd(x,axis=1)
            z=np.divide(x-center[:,None],sd[:,None],out=np.zeros_like(x),where=sd[:,None]>1e-12);z[~np.isfinite(x)]=np.nan
            im=axes[ci].imshow(z,cmap='coolwarm',aspect='auto',vmin=-2.5,vmax=2.5)
            axes[ci].set(xticks=range(len(rows)),xticklabels=[f'rs={int(num(r["rs"]))}\nH={r["heads"]}' for r in rows],yticks=range(len(keys)),yticklabels=[METRICS[k][0] for k in keys],title=f'{subset}; B={self.config["samples"]}/model')
            fig.colorbar(im,ax=axes[ci],label='descriptive z-score across 8 models [1]')
        self.save(fig,'phase_comparison','8モデルの横断比較。各指標を8モデル間の平均・標準偏差で標準化した記述的な表示で、相分類精度や統計的有意性ではない。元の単位・数値はmetrics.csv。','Each row standardized within subset; constant metrics shown as 0; no classifier fitted')
        fig,axes=plt.subplots(2,3,figsize=(16,8))
        for ri,rs in enumerate((4,25)):
            rows=sorted(take(self.rows['metrics'],rs=rs,subset='native'),key=lambda r:int(r['heads']));H=[int(r['heads']) for r in rows]
            B=','.join(sorted({r['physical_samples'] for r in rows}))
            axes[ri,0].errorbar(H,[num(r['structure_factor']) for r in rows],yerr=[num(r['structure_factor_block_sem']) for r in rows],fmt='o-',color=COLORS[rs],capsize=3,label='native block SEM')
            axes[ri,2].errorbar(H,[num(r['energy_per_electron_Ha']) for r in rows],yerr=[num(r['energy_block_sem'])/(36*rs**2) for r in rows],fmt='o-',color=COLORS[rs],capsize=3)
            for r in rows:
                h=int(r['heads']);a=self.banks.get((rs,h,'native'))
                if a is None:continue
                edges=a['physics_r_edges'];x=(edges[:-1]+edges[1:])/2;ls,mk=STYLES[h];axes[ri,1].plot(x,avg(a['physics_gr'],axis=0),ls,marker=mk,ms=3,label=f'H={h}')
            axes[ri,0].set(title=f'rs={rs}, H=2/3/4/10, native B={B}',xlabel='number of heads H',ylabel='S(G) crystal-shell mean [1]');axes[ri,0].legend(fontsize=8)
            axes[ri,1].set(title=f'rs={rs}, native B={B}',xlabel='minimum-image distance [rs aB]',ylabel='g(r) [1]');axes[ri,1].legend(fontsize=8)
            axes[ri,2].set(title=f'rs={rs}, native B={B}',xlabel='number of heads H',ylabel='energy per electron [Ha]')
        self.save(fig,'physics_baselines','通常S(G)・g(r)・エネルギー。Nativeの保存済み1024配置を再利用し、8 chain×16時間blockの暫定SEMを表示。rsが違うenergyを直接の相判定に使わない。g(r)は有限N(N−1)規格化。','Error bars: SEM of 16 contiguous time-block means (chains pooled); g(r) mean only; E_Ha/N=E_code/(36*rs^2)')

    def contrast(self,key,subset='native'):
        diffs=[]
        for h in STYLES:
            x=take(self.rows['metrics'],rs=4,heads=h,subset=subset);y=take(self.rows['metrics'],rs=25,heads=h,subset=subset)
            if x and y:diffs.append(num(y[0][key])-num(x[0][key]))
        a=np.array(diffs);a=a[np.isfinite(a)]
        if not len(a):return 'データ不足'
        signs='全head数で正' if np.all(a>1e-12) else '全head数で負' if np.all(a < -1e-12) else 'head数で符号が変化／差が0'
        return f'Δ(rs25−rs4)={fmt(a.min())}〜{fmt(a.max())}（{signs}）'

    def image(self,name):
        if name not in self.captions:return '\n未実施：必要な入力データがない。\n'
        return f'\n![{name}](figures/{name}.png)\n\n図：{self.captions[name]}\n'

    def text(self):
        metrics=self.rows['metrics'];native=take(metrics,subset='native');common=take(metrics,subset='common')
        def span(rows,key):
            vals=np.array([num(r[key]) for r in rows]);vals=vals[np.isfinite(vals)]
            return f'{fmt(vals.min())}〜{fmt(vals.max())}' if len(vals) else 'NA'
        adjusted_examples=[]
        for rs in (4,25):
            rr=take(self.adjusted,rs=rs,heads=self.hrep,subset='common',source='screening',channel='phase')
            adjusted_examples.append(f'rs={rs}で{fmt(mean([num(r["pearson_scale_adjusted"]) for r in rr]))}')
        facts=[f'通常の密度構造因子S(G)は液体候補rs=4で{span(take(native,rs=4),"structure_factor")}、WC候補rs=25で{span(take(native,rs=25),"structure_factor")}。この8条件のH間範囲は分離している。相の確定には独立runと有限サイズ検証が必要。',
            f'実Screeningと内部位相絶対感度のPearsonは、Nativeでrs=4が{span(take(native,rs=4),"phase_pearson")}、rs=25が{span(take(native,rs=25),"phase_pearson")}。ただしlog倍率微分には重み自体が掛かる。H={self.hrep}共通入力の重み因子を除いた相関は'+ '、'.join(adjusted_examples)+'となり、この例で強い正相関は残らない。元の相関だけで物理的機能を証明できない。',
            f'Screeningの結晶波数重みはNativeで{span(native,"crystal_peak")}、幾何対照からの増分は{span(native,"crystal_increment")}。符号を含めhead数で変わり、一律のBragg強調という説明は支持されない。',
            f'Screening残差分散はNativeで{span(native,"residual_variance")}だが、Zを定義できた辺の割合は{span(native,"residual_coverage")}にとどまる。ゼロに近い重みの条件SD=0や疎binはNAとした。分散が約1になる標準化の影響もあり、残差分散だけの相識別は弱い。',
            f'QK積循環RMSは液体候補で{span(take(native,rs=4),"circulation_rms")}、WC候補で{span(take(native,rs=25),"circulation_rms")}。head数依存が大きく、幾何・容量・HFの交絡を除く必要がある。同じcommon配置の通常S(G)にモデル差はない。']
        ritz=all(c.get('rayleigh_ritz_pass',False) for c in self.validation['checks'])
        summary=['# 非PH物理相診断：要約','',
            f'実行対象はN=36、λ=0.72、rs=4/25、H=2/3/4/10、seed=101、1層、80000更新後の8モデル。完了は{len(metrics)} / 16 subset。内部接続は各{self.config["samples"]}配置、通常物理量はNativeの各1024保存配置を使った。相ラベルは液体候補・WC候補であり、相を確定する学習や検定はしていない。','',
            '## 重要な観測','']+['- '+f for f in facts]+['',self.image('physics_baselines'),
            '## どの指標に差が出たか','',
            markdown_table(['指標','Native差','Common-input差'],[[METRICS[k][0],self.contrast(k),self.contrast(k,'common')] for k in ('structure_factor','r90','crystal_increment','phase_pearson','spin_rayleigh','circulation_rms')]),'',
            'ここで「全head数で同符号」は4つの記述的対比であり、有意差を意味しない。学習seed間変動は測定できない。共通入力でS(G)に差がないのは定義による。内部指標で差が小さいか、符号がhead数で変わる項目は上表・総合表に明記した。','',
            '## 幾何・head数と物理の切り分け','',
            '接続の波数ピークそのものは入力配置からも生じる。等重み・距離のみ・spinのみの対照、および距離×spin条件平均との差を使って増分を見る。残差は絶対周期座標、条件平均のbin近似、MC分布ずれも含み、未知の多体相関と断定できない。head数を増やすと総パラメータ数も増え、同じhead番号にモデル間の物理対応はない。集約値はhead順序に依存しない。','',
            '## AHC/WCへの発展と次の実験','',
            '密度秩序にはS(G)・g(r)を基準とし、接続の幾何残差、位相応答との条件付き相関、head除去応答を補助指標にする。方向付き積循環は探索量であり、Berry曲率・Hall電流・Chern数ではない。rs=12のAHC候補checkpointは存在するが今回の均一な8条件比較には追加していない。twist状態族や独立Hall応答を伴うAHC証拠が不足している。','',
            '次に必要なのは、独立学習seedとパラメータ数をそろえた比較、両候補のNative配置を相互入力する交差評価、MCのburn/block長とESS確認、head除去後の再サンプリングとエネルギー評価、rs=12候補の同じ診断、独立したtwist/Hall検証である。','',
            '各指標の定義・詳しい図・数値・注意点は[詳細レポート](report.md)、再生成手順は[README](README.md)を参照。']
        (self.root/'summary.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
        report=['# Screening Transformer：非PH物理相診断','',
            '## 対象と比較の読み方','',
            '\n'.join(facts),'',
            'Nativeは各モデルの既存fresh MCMC測定。全物理量1024配置は[時間block、block内時間、chain]順、内部量はそこから等間隔で32配置を選ぶ。Commonは全モデルに同じ合成一様配置を入力する診断で、平衡分布ではない。commonのS(q)/g(r)はモデルで変わらず、energyは評価していない。共通配置の偏極・random spinの混合も物理的平衡spin分布ではない。','',
            'この8条件では単一学習seedのみ。配置SD、暫定MC block SEM、head間変動を別々に扱い、独立seed誤差・相識別精度・p値を報告しない。head有効数・head平均・min/maxは順序に不変。異なるモデルのhead番号は対応しない。','',
            markdown_table(['rs','H','入力','内部B','物理B','param数','S(G)','block SEM','E/N [Ha]','HF初期'],[[r['rs'],r['heads'],r['subset'],r['samples'],r['physical_samples'],r['parameters'],fmt(r['structure_factor']),fmt(r['structure_factor_block_sem']),fmt(r['energy_per_electron_Ha']),r['initialization']] for r in metrics]),'',
            '## 実装調査と厳密な定義','',
            'Q=XW_Q、K=XW_K、S_ij=qbar_i·kbar_j。qbar=q/sqrt(sum(q²)+10⁻¹²)であり、softmaxは使わない。r=TRIM_MAX sigmoid(rho)、Alpha=[relu(S−(1−r))/r]^p。Windowを掛け、各種介入後の実集約重みで正規化Valueを足し、TanhNorm→Gate→W_O→exp(out_scale)を通る。解析では正式Diagnostic APIのS/Alpha/Window/aggregation_weightsを利用する。通常Valueは自己項を含むが、pair解析では自己辺を除く。','',
            'この8条件の保存実効設定はWIN_SHAPE=smooth7、TRIM_POW=4。Windowはm=1−S7(clip(d/w,0,1))、S7(t)=35t⁴−84t⁵+70t⁶−20t⁷、w=dref softplus(omega)。wはネットワーク周期ノルムでの有限カットオフであり、物理的最小像距離のカットオフではない。各runの設定はmanifestで確認できる。','',
            '符号付きQKの確率解釈を避けるため、距離確率と重み付きcosineのQK対照は明示的にQK affine=(S+1)/2と呼ぶ。Alphaとは異なる。符号付きQKそのものの波数分布はsum|S|を分母にする。標準化残差Zは正の線形変換に対して同じ（epsilonを除く）なので、QK affineのZはQKスコアの幾何残差を表す。','',
            '格子は列ベクトルL。q=m@(2π L⁻¹)、exp(iq·r)。長さ1=rs a_B。Windowのdはsin/cosによる周期ノルムであり、最小像ユークリッド距離と区別する。Window幅とR90の比較は同じnetwork距離を用いる。rs間ではcode長さが同じでもa_B単位では異なる。','',
            markdown_table(['解析','追加前の実装状況','再利用と追加'],[
                ['A 波数','一部実装','lj_mapsの逆格子/結晶shell、通常S(q)を再利用。重み付き対分布と幾何対照を追加'],
                ['B 距離','既存実装','connection_studyの距離bin/R90を再利用、R50/R95/CDF・QK由来正重みを追加'],
                ['C 残差','未実装','配置単位crossfit、疎binのNA、同一有効支持上のFourier増分を追加'],
                ['D head','一部実装','probeの出力寄与を再利用、JS/有効数/head除去応答を追加'],
                ['E 循環','一部実装','既存CCW三角形を再利用。辺差の和と異なる積循環を追加'],
                ['F 位相','既存実装','既存保存介入微分とphase_sensitivityを再利用。条件付き相関と座標応答を追加'],
                ['G 固有','一部実装','attention_modesを再利用し、同一行列でRitz・結晶/擬スピン射影を追加']]),'',
            'ソースと検証対象の詳細はリポジトリ内 `screening_analysis/PHYSICAL_PHASE_DIAGNOSTICS.md`。格子系は参考として確認し、今回の状態や統計に混ぜない。','']
        sections=[
            ('A：Attentionの波数空間',
             '入力の結晶ピークに対し、学習した接続重みが何を強調するか調べる。',
             'S_A(q)=配置平均[sum(i<j) A_sym cos(q·dr)/sum A_sym]。通常S(q)=|sum exp(iq·r)|²/N。基準GはN=36三角結晶に対応する既存first_shell。ピーク/背景はG平均をq≠0,G以外の|S_A|平均で割る。距離×spin条件平均を別配置で推定し、同じ有効支持上の差をcrystal_incrementとする。',
             'S_Aは接続重み付き粒子対の波数分布であり、通常の構造因子や散乱強度ではない。',
             self.contrast('crystal_increment')+'。Commonでは'+self.contrast('crystal_increment','common')+'。',
             '等重み対照にもピークがあれば、その部分は入力幾何で説明できる。増分は条件平均からのずれを表すが、学習された物理秩序を単独で証明しない。',
             '基準Gは相を仮定する参照波数で、実際の全Braggベクトルを探索して確定したものではない。負のcosine重み値や比の小分母に注意。',
             'Native配置の交差入力、角度分解と異なるG shell、bin幅の感度を調べる。', ['attention_fourier','fourier_profiles']),
            ('B：Window・QKの接続距離',
             'Windowの広さと実際に使われる接続距離を区別する。',
             '対称化・自己辺除去後の重みを配置/辺でプールしCDFを求める。R50/R90/R95は正確な重み付き分位点。平均距離=sum w d/sum w。同・異spinを別々に規格化。network距離とminimum-image距離を両方保存。',
             'R90は重みの90%が届く距離であり、Windowのカットオフや因果的な相関長とは異なる。',
             self.contrast('r90')+'。Commonでは'+self.contrast('r90','common')+'。',
             '幅とnetwork R90の大小関係から、窓内でQKが実際に選ぶ範囲を読める。minimum-image R90は実空間での到達範囲を与える。',
             '総重みゼロはNA。pool方式なので高い総重みを持つ配置の寄与が大きい。H間の範囲を統計誤差に読み替えない。',
             '距離依存の滑らかな内部介入と、物理的な再サンプリングを比較する。',['distance_profiles','width_r90','spin_distance']),
            ('C：QKの幾何学的残差',
             '距離と擬スピン組合せだけでは説明できない接続情報を抽出する。',
             'μ_h(d,σ_i,σ_j), s_hを2-foldの別配置で推定しZ=(A−μ)/(s+1e-8)。undirected測定なので++/--/mixedの3群。推定binの辺数20未満、SD<=1e-8はZ=NA。全配置の交互foldで評価し、NA率も保存。残差スペクトルは有効辺平均 Z cos(q·dr)で、確率規格化ではない。',
             '単純な距離/spin説明からの残りで、波動関数への因果効果そのものではない。',
             'Screening残差分散: '+self.contrast('residual_variance')+'。Commonでは'+self.contrast('residual_variance','common')+'。',
             '異なるモデルを同じ配置で評価しても残る差と、各モデルのNativeでだけ生じる差を表で分けた。',
             '交互foldはMCで相関し得る。絶対周期座標、方向、条件binの粗さ、境界外予測も残差となる。標準化残差の分散が1付近なのは推定方法自体の影響を受ける。',
             '独立chainによるfold、物理距離/角度/絶対座標を条件に加えたcrossfitとbin統合を比較する。',['geometric_residuals','residual_profiles','residual_fourier']),
            ('D：Head間の機能分化',
             '接続が似たheadが、実際にも似た出力や波動関数応答を持つか調べる。',
             'Screening辺のPearson、JS(nats)、同異spin選択性を計算。出力寄与c_h=exp(out_scale)*gate_h W_O,hの相関、Gram参加率tr(G)²/tr(G²)、相殺比sum G/tr G。対応するW_O行を0にして全波動関数を再評価。',
             '接続選択とValue/Gate/出力射影後の機能の違いを測る。head除去は固定checkpointの有限診断。',
             '出力有効head数: '+self.contrast('output_effective_heads')+'。相殺比: '+self.contrast('cancellation_ratio')+'。',
             '出力Gramは符号を持つ相殺を含み、辺のJSとは違う情報を与える。head寄与を足した物理的エネルギー分解とは解釈しない。',
             '共有biasを各headへ割り振らない。通常Valueの自己項も除去される。エネルギー/S(q)のhead ablation後評価は未実施。H増加でパラメータ数も変わる。',
             '同容量・複数seedで再比較し、除去モデルで再サンプリングしてenergy/S(q)を評価する。',['head_specialization','head_roles','head_ablation']),
            ('E：QKの方向性・積循環',
             '非対称なQK接続の方向付き構造を探索する。',
             'S_ij−S_ji、J_ijk=A_ij A_jk A_ki−A_ik A_kj A_ji。最短像で可縮かつ非共線の三角形をCCWにそろえる。QK signed/affineと実Screeningを別計算し、all/same/mixed spinで保存。',
             'ネットワークの方向付き3辺積の差。非量子化の内部表現量。',
             self.contrast('circulation_rms')+'。Commonでは'+self.contrast('circulation_rms','common')+'。',
             '受動的な鏡映は幾何学的向きを反転するのでJの符号を反転し、回転・粒子置換・格子並進は維持することを人工グラフで検証した。',
             'QK循環をBerry曲率/Hall電流/Chern数と同一視しない。実モデルへの能動鏡映ではQKも変わり、単なる符号反転は要求しない。active-reflectionの再評価値はvalidation.jsonへ保存。',
             '独立runと対称性をそろえたnull、AHC候補および独立Hall/twist応答と比較する。',['qk_circulation']),
            ('F：接続と波動関数位相・振幅応答',
             '接続が最終波動関数の位相・振幅へ影響するかを、内部介入と座標応答に分けて調べる。',
             '内部介入A: 対称辺倍率exp(z_ij)への∂log|Ψ|/∂z、∂φ/∂zを既存APIで求め、絶対感度と各接続重みのPearson/Spearman、top10%一致率、距離/spin/両方を固定した相関を計算。log倍率微分は重みを含むため、正重み支持のみ、および実Screening対称重みで割った感度との相関もphase_adjusted.csvへ保存する（固定した順逆重み比でのdφ/dw_sym）。物理的座標応答B: 座標を動かし全波動関数を再計算する∂φ/∂r_i、∂log|Ψ|/∂r_iを求め、粒子の接続強度と比較。両方で有限差分検証を行う。',
             'Aは任意の内部辺への感度、Bは反対称波動関数を保つ座標変形への応答。Bには一意の粒子対分解がない。',
             'Screeningと内部位相絶対感度のPearson: '+self.contrast('phase_pearson')+'。Commonでは'+self.contrast('phase_pearson','common')+'。',
             'Window単独や距離/spin条件付き相関との比較により、単なる近距離重み由来の相関かを点検する。相関は非線形性やValueの働きも含むため、重みだけの因果寄与ではない。',
             'Aは一般に粒子交換反対称性を保存する物理摂動ではない。位相は実装で固定されたゲージでのangle。定数全体位相には不変だが、座標依存ゲージに対して座標位相勾配は不変ではない。磁場のベクトルポテンシャルを含むゲージ共変比較は未実施。内部感度はNative/Common各最大4配置（実数はphase.csv）、座標応答は各2配置で、誤差推定/相確定には不足。',
             '配置数・独立runを増やし、Window/距離で説明される成分を評価。ゲージ共変な物理摂動とtwist状態族で検証する。',['phase_sensitivity','phase_correlations','phase_distance','phase_scale_control','phase_edges']),
            ('G：Attention固有モードと物理秩序',
             '接続固有モードが密度波・擬スピン分割と対応するか調べる。',
             'A=(W+Wᵀ)/2、diag A=0。raw AとM=D^(-1/2) A D^(-1/2)を別扱い。v_G=e^(iG·r)、s=σ−mean σに対しR=v†Mv/(v†v)。同じ行列/同じ配置でλmin<=R<=λmaxを検査する。固有値、空間固有ベクトルとmode overlapをcache保存。固有値差1e-8以内の固有空間について射影の二乗和を取り、縮退内の基底に依存しないprojector overlapをeigenspace_metrics.csvに保存。',
             '規格化行列の特定物理ベクトル方向への結合強度。秩序パラメータそのものではない。',
             '擬スピンRayleigh: '+self.contrast('spin_rayleigh')+'。結晶Rayleigh: '+self.contrast('crystal_rayleigh')+f'。Ritz検証={ritz}。',
             'raw固有値の下限と規格化行列のRayleigh値を混ぜると不等式を破ったように見える。今回の同一行列検証はこの混同を避ける。',
             'ゼロ次数は逆平方根を0。偏極配置で中心化spin=0ならRayleighはNA。固有ベクトルの符号や縮退内の回転は任意。個別ベクトルを別配置/別headに直接対応づけない。',
             '固有空間projectorと複数物理モードとの重なりを、別shell・独立seedで検証する。',['eigenmode_projection'])]
        for title,purpose,method,meaning,result,interpretation,limits,next_step,figures in sections:
            report += [f'## {title}','',f'**目的：** {purpose}','',f'**方法：** {method}','',f'**物理的意味：** {meaning}','',f'**結果：** {result}']
            for name in figures:report.append(self.image(name))
            if title.startswith('B：'):
                rr=take(self.rows['head_metrics'],subset='native',source='screening')
                report += ['\nNativeのheadごとの幅と到達距離（両距離定義を併記）。\n',markdown_table(
                    ['rs','H','head','幅 network [rs a_B]','R50 min-image','R90 min-image','R95 min-image','R90 network'],
                    [[r['rs'],r['heads'],r['head'],*[fmt(r[k]) for k in ('width','r50','r90','r95','r90_network')]] for r in rr]),'']
            if title.startswith('F：'):
                table=[]
                for r in metrics:
                    rr=take(self.adjusted,rs=num(r['rs']),heads=int(r['heads']),subset=r['subset'],source='screening',channel='phase')
                    table.append([r['rs'],r['heads'],r['subset'],fmt(r['phase_pearson']),
                        fmt(mean([num(x['pearson_positive_support']) for x in rr])),fmt(mean([num(x['pearson_scale_adjusted']) for x in rr]))])
                report += ['\n実Screeningと位相感度の相関。head平均の記述量。\n',markdown_table(
                    ['rs','H','入力','元Pearson','正支持のみPearson','重み因子除去後Pearson'],table),'']
                wf=[]
                for r in metrics:
                    rr=take(self.rows['wavefunction'],model_id=r['model_id'],subset=r['subset'])
                    co=take(rr,kind='coordinate');ha=take(rr,kind='head_ablation')
                    wf.append([r['rs'],r['heads'],r['subset'],
                        fmt(mean([num(x['phase_gradient_rms']) for x in co])),fmt(mean([num(x['amplitude_gradient_rms']) for x in co])),
                        fmt(max([num(x['fd_scaled_error']) for x in co],default=np.nan)),
                        fmt(max([abs(num(x['dphase'])) for x in ha],default=np.nan)),
                        fmt(max([abs(num(x['dlogabs'])) for x in ha],default=np.nan))])
                report += ['\n物理的座標応答と内部head除去の有限応答（各2配置の例）。\n',markdown_table(
                    ['rs','H','入力','座標∇φ RMS [rad/(rs a_B)]','座標∇log振幅 RMS [(rs a_B)^−1]','座標FD相対誤差最大','除去|Δφ|最大 [rad]','除去|Δlog振幅|最大'],wf),'']
            report += [f'**解釈：** {interpretation}','',f'**注意点：** {limits}','',f'**次の検証：** {next_step}','']
        report += ['## 横断比較と統計上の限界','',self.image('phase_comparison'),
            '配置IDは既存fingerprint契約による位置/spin/atoms/chargesの一致を表す。各CSVはmodel_id/subset/rs/heads/seedを保持する。配置にひもづく表はconfiguration_idも持つ。head集約表に特定配置IDを付けず、使用した配置bankと選択indicesをmanifestに記録した。','',
            'NativeのS(G)とenergyのSEMは保存順の16連続時間block平均間から計算し、chain間を独立配置として水増ししない。ただしblock長の収束やESSは未確認で、MC自己相関の除去は保証しない。内部解析の選択32配置には配置SD/分布を使い、MC SEMを付けない。head間分散も学習seed間誤差ではない。今回の8条件でrun単位分類器を評価するには独立runが不足するため精度は未実施。','',
            '## 総合評価表','',self.evaluation_table(),'',
            '## 未実施・データ不足','',
            '- 独立学習seed変動、等パラメータ容量対照、相識別精度：データ不足。\n- head除去後のエネルギー・再サンプルS(q)：未実施。現在は固定入力log振幅/位相だけ。\n- MC混合・ESS・burn/block長収束：未検証。block SEMは暫定値。\n- rs=12 AHC候補の同一手順解析：未実施（checkpointはあるが今回の8条件には含めない）。\n- 多体Chern：既存phase_evidenceには供給twist状態族からの計算器があるが、適切なtwist状態族/重なり/ギャップの入力が今回ない。実施不可。\n- 能動的座標変形のゲージ共変位相応答：実装未対応。固定実装ゲージの座標位相微分のみ。\n- 距離×spinを保存したランダム化の混合・有意差：今回未実施。既存matched-nullはPH対照であり、今回の非PH指標の有意差証拠へ転用しない。\n- 対応headの物理的matching：未実施。順序不変なhead集約とモデル内ヒートマップを用いる。','',
            '## 検証と由来','',
            f'期待8モデル×2入力、完了{len(metrics)} subset、解析失敗{len(self.validation["failures"])}件。定義一致、実checkpointでの粒子置換/PBC、有限差分、Ritz結果は[validation.json](diagnostics/validation.json)。追加の振幅・位相両方の有限差分とcheckpoint/HF/config不変検証は[verification.json](diagnostics/verification.json)。checkpoint/config/HF/input/sourceのhashとrevision/dirty状態は[input_manifest.json](diagnostics/input_manifest.json)。描画・追加派生統計のソースhashは[report_validation.json](diagnostics/report_validation.json)に分けて記録する。','',
            '追加テストは人工グラフでの規格化、QK/Alpha/Window/Screening公式API一致、粒子置換/PBC、受動鏡映・回転、Ritz、crossfit漏洩、ゼロ辺/空群/疎bin/位相不安定性、再現性を検証する。実行結果はlogsに保存。checkpointや学習ロジックは変更しない。','',
            '## 次の実験の優先順位','',
            '1. S(G)/g(r)とMC誤差を独立seed・長いblockで確認し、Native配置をモデル間で相互入力する。\n2. Hと総パラメータ数を切り分け、HF初期軌道とrsの交絡を同じ条件で検証する。\n3. 位相感度の配置数を増やし、head除去モデルを再サンプリングしてenergy/S(q)を測る。\n4. rs=12候補に同じ非PH診断を行い、独立twist/Hall解析と対応を調べる。']
        (self.root/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        self.readme()

    def evaluation_table(self):
        meaning={'structure_factor':'明確：密度秩序','r90':'明確：重み到達距離','crystal_increment':'中：幾何対照に依存',
                 'residual_variance':'中：条件モデル依存','residual_peak':'中：残差のG射影','edge_effective_heads':'内部機能',
                 'output_effective_heads':'内部出力の多様性','cancellation_ratio':'内部出力の相殺','phase_pearson':'中：内部介入との関連',
                 'spin_rayleigh':'明確な射影・秩序確定不可','crystal_rayleigh':'明確な射影・秩序確定不可','circulation_rms':'探索的・非トポロジカル'}
        rows=[]
        for key,(label,definition) in METRICS.items():
            hspan=[];ispan=[]
            for rs in (4,25):
                vals=[num(r[key]) for r in take(self.rows['metrics'],rs=rs,subset='common')];vals=np.array(vals);vals=vals[np.isfinite(vals)]
                hspan.append(f'rs{rs}: '+(fmt(np.ptp(vals)) if len(vals) else 'NA'))
            for H in STYLES:
                vals=[]
                for rs in (4,25):
                    a=take(self.rows['metrics'],rs=rs,heads=H,subset='native');b=take(self.rows['metrics'],rs=rs,heads=H,subset='common')
                    if a and b:vals.append(abs(num(a[0][key])-num(b[0][key])))
                ispan+=vals
            future='単独では不可、密度基準' if key=='structure_factor' else '位相応答の補助候補・独立Hall検証必須' if key=='phase_pearson' else '探索候補・独立Hall検証必須' if key in ('crystal_increment','residual_peak','circulation_rms') else '補助量、単独識別は未検証'
            rows.append([label,meaning[key],self.contrast(key),'; '.join(hspan),f'平均|Native−Common|={fmt(mean(ispan))}',
                '暫定block SEM・単一seed' if key=='structure_factor' else '記述統計・単一seed（未確立）',future])
        result=markdown_table(['指標','物理的解釈','候補間差の大きさ（Native）','H依存：Commonの範囲','入力配置依存','統計的信頼性','AHC/WC発展性'],rows)
        definitions='\n\n'+ '\n'.join(f'- **{label}：** {definition}' for label,definition in METRICS.values())
        return result+definitions

    def readme(self):
        text='''# 非PH物理相診断の成果物

まず[要約](summary.md)、続いて[詳細レポートと総合評価表](report.md)を読む。

- `figures/`: 英語軸ラベル、単位、H/rs/配置数/規格化を記載したPNG。日本語キャプションはreport.md。
- `data/metrics.csv`: 8モデル×Native/Commonの統合指標。Native energy/S(G)は全1024配置、内部量は選択32配置。
- `data/head_metrics.csv`: sourceごとのhead指標。QK affineをAlphaと混同しない。
- `data/configuration_metrics.csv`: 通常S(G)/g(r)/local energyと正確な配置ID。
- `data/phase_comparison.csv`: rs25−rs4のhead数別対比。単一runの記述統計。
- その他CSV: 距離分布/R50/R90/R95、条件付き残差、循環、固有値射影、内部感度相関、座標応答/head除去。
- `cache/MODEL/{native,common}/`: 正式probe、解析array、CSV、必要時の感度微分。checkpointは複写/変更しない。
- `diagnostics/`: 設定、hash/入力選択/revision/dirty状態、検証と失敗理由。
- `logs/`: 実行とテストの記録。

## 再生成

リポジトリのrootで実行（描画環境にはJAX不要）:

```bash
/home/tomax/venvs/pyrite-bands/bin/python -m screening_analysis.physical_phase_diagnostics \\
  --config configs/physical_phase_diagnostics.json --mode report
```

解析実行はscreening-gpu環境で同じCLIの`--mode all`。中断再開は`--resume`。変更した設定で既存cacheを再利用しない。初回出力が存在する場合は停止する。詳細はリポジトリの`screening_analysis/PHYSICAL_PHASE_DIAGNOSTICS.md`。

AHCの確定、Hall応答、Chern数は今回の内部指標から結論しない。単一学習seedの配置を独立学習runへ数え替えない。Native/Commonは異なる統計母集団。
'''
        (self.root/'README.md').write_text(text,encoding='utf-8')

    def run(self):
        if not self.banks:
            raise ValueError('No completed analysis caches: inspect logs before generating a report')
        self.derived_tables()
        failures=[]
        for method in (self.fourier,self.distances,self.residuals,self.heads,self.cycles,self.phase,self.eigen,self.comparison):
            try:method()
            except Exception as error:
                import traceback
                failures.append(dict(figure_group=method.__name__,error=str(error)))
                (self.root/'logs'/('plot-'+method.__name__+'.txt')).write_text(traceback.format_exc())
        self.text()
        import hashlib
        (self.root/'diagnostics/report_validation.json').write_text(json.dumps(dict(figures=list(self.captions),failures=failures,matplotlib=matplotlib.__version__,numpy=np.__version__,
            report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            derived_statistics='positive-support / weight-factor-adjusted Pearson; degenerate eigenspace projector sums from cached arrays'),indent=2)+'\n')
        print(f'Report: {len(self.captions)} figures, {len(failures)} failures: {self.root}',flush=True)
        if failures:raise RuntimeError(str(failures))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    Report(parser.parse_args().output).run()


if __name__=='__main__':main()
