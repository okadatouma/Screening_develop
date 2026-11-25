"""Descriptive, matched-checkpoint findings from actual saved runs only."""
import json
import numpy as np
from ..experiments.store import csv_save
from ..experiments.report import rows,table,f
from ..experiments.retries import resolve


def findings(root,supplement=None,report_folder=None):
    root=root.resolve();cases=json.loads((root/'plan.json').read_text());observations=[];interventions=[];physical=[]
    for case in cases:
        folder=resolve(root,case['name'])
        if not (folder/'COMPLETE.json').exists():continue
        config=case['config'];summary=json.loads((folder/'summary.json').read_text());task=config['experiment']['task']
        if task=='continuous_electrons' and case['group']=='B_embedding':
            crystal=np.array([[1,0],[0,1],[1,-1]])@np.asarray(config['physics']['crystal_supercell'])
            values={}
            for stage in ['before','after']:
                fourier=rows(folder/'diagnostics'/f'{stage}_fourier.csv')
                shell=[float(r['QK_Fourier']) for r in fourier if (int(r['qx']),int(r['qy'])) in {tuple(x) for x in crystal}]
                spin=json.loads((folder/'diagnostics'/f'{stage}_spin_selection.json').read_text())
                values[stage+'_crystal_QK']=float(np.mean(np.abs(shell)))
                values[stage+'_spin_selectivity']=spin['same']-spin['opposite'] if spin['same'] is not None and spin['opposite'] is not None else None
            observations.append(dict(run=case['name'],embedding=config['embedding']['type'],seed=config['experiment']['seed'],**values))
        if task=='qwz_topology':
            saved=rows(folder/'qwz_metrics.csv')
            for stage in ['before','after']:
                selected=[r for r in saved if r['stage']==stage]
                physical.append(dict(run=case['name'],mode=config['attention']['type'],output=config['wavefunction']['type'],stage=stage,
                                     chern_correct=sum(abs(float(r['chern'])-float(r['target_chern']))<1e-5 for r in selected),
                                     cases=len(selected),projector_mse=float(np.mean([float(r['projector_mse']) for r in selected]))))
        path=folder/'interventions.json'
        if path.exists():
            data=json.loads(path.read_text())
            corrected=folder/'interventions_width_corrected.json'
            if corrected.exists():data=[r for r in data if r['intervention']['type']!='window_width']+json.loads(corrected.read_text())
            for r in data:
                interventions.append(dict(run=case['name'],task=task,intervention=json.dumps(r['intervention'],sort_keys=True),
                     delta_energy=r.get('delta_energy'),delta_projector_mse=r.get('delta_projector_mse'),
                     fidelity=r.get('fidelity'),odd_parity_probability=r.get('odd_parity_probability'),
                     conditional_interpretation_screen=r.get('conditional_interpretation_screen'),resampled=r.get('resampled')))
    folder=report_folder or root/'reports';folder.mkdir(exist_ok=True)
    csv_save(folder/'embedding_physical_features.csv',observations)
    csv_save(folder/'matched_checkpoint_interventions.csv',interventions)
    csv_save(folder/'qwz_before_after_chern.csv',physical)
    text='''# 研究上の問いへの実測回答

記述的なpilotであり、有意差や一般的な相識別精度を主張しない。異なるseedは別行として保持した。

## Q1・Q2: 結晶波数とspin入力

同一の固定probe分布で、結晶shell上のQK Fourier成分の絶対値平均と、同spin−異spinのQK平均差を測定した。
これはQKへの入力情報の利用を調べるもので、実空間秩序の発見を直接示す指標ではない。

'''
    text+=table(['embedding','seed','結晶QK 前','結晶QK 後','spin選択性 前','spin選択性 後'],
                [[r['embedding'],r['seed'],f(r['before_crystal_QK']),f(r['after_crystal_QK']),f(r['before_spin_selectivity']),f(r['after_spin_selectivity'])] for r in observations])
    text+='''
6θ系の波数は今回の小系ではS=diag(2,2)に対応する。6×6の実N36の結晶を今回学習したわけではない。
変化の方向は上表の各seedで確認し、入力次元・初期化の差を含むことを考慮する。WC/AHCのone-hot識別改善は未検証。

## Q3: 学習する候補波数

主pilotのlearned Fourierは低い計算セル波数3つを用いる対照で、結晶shellを候補として含まない。
この条件から結晶波数を自発選択したとは言えない。
別途、低波数3つ＋結晶shell3つの計6候補を含む固定20成分・名目同一parameter数の比較を定義し、実行済みの場合は下表で報告する。
追加planは結果の良い波数を後から選ぶものではなく、候補集合に結晶波数を含める設計上の対照として、全条件を実行前に保存した。
ただし主pilot開始後の追加対照であるため、研究全体を事前登録したとは呼ばない。

'''
    if supplement and (supplement/'plan.json').exists():
        data=[]
        for case in json.loads((supplement/'plan.json').read_text()):
            path=resolve(supplement,case['name'])
            if not (path/'COMPLETE.json').exists():continue
            s=json.loads((path/'summary.json').read_text());amp=None
            if case['config']['embedding']['type']=='learned_fourier':
                history=[json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
                amp=history[-1]['fourier_amplitudes']
            data.append([case['config']['embedding']['type'],case['config']['experiment']['seed'],s['parameter_count'],f(s['energy']),f(s['block_sem']),'' if amp is None else ', '.join(f(v) for v in amp)])
        text+=table(['embedding','seed','parameter数','E','SEM','学習振幅(候補順)'],data)
        text+=f'\n追加対照の保存先: {supplement.resolve()}\n'
    text+='''
N2・自由系・12更新・短いMCの実験であり、物理的な結晶秩序の自発的な波数選択は認定しない。

## Q4・Q5: Windowと相関・CMI・S2・パリティ

各checkpointの幅、実効接続距離、ED/NNの有限鎖相関長、Born CMI、S2、p_-を同時に測定した。
エネルギーだけからp_-が小さくなる条件を観測したが、狭いhard Windowや短い学習で残る場合もある。
同じgの固定Windowと学習可能Windowを比較する。正則化で幅が縮んだ結果を相関長の発見とは呼ばない。
この小規模研究では、幅と相関長・CMI・S2に普遍的な一対一対応は立証していない。

## Q6・Q7・Q8: Gauge輸送・複素波動関数・障害

Gauge-covariant輸送の出力までの変換則、零overlapの除外、ProjectorのHermiticity/冪等性を数値検証した。
native complexと実数2channelは位相を保持するが、自動的に局所U(1)共変にはならない。
charged embeddingとリンク輸送を組み合わせると、ランダムゲージでも出力Projectorの差が丸め誤差程度に抑えられる。
これは与えたBerryリンクと構造制約の効果であり、ネットワークが物理リンクを発見した証拠ではない。
複素Slater/twistは元の行列式を使うため反対称性を保つ。単なる複素指数readoutへ置き換えていない。
「障害が最も明確な構成」をpilotのP誤差だけで順位付けしてはいない。

## Q9: 内部PHと物理的Chern

'''
    text+=table(['run','stage','C一致数','P MSE'],[[r['run'],r['stage'],f"{r['chern_correct']}/{r['cases']}",f(r['projector_mse'])] for r in physical])
    text+='''
未学習の共変構成でもChernを保持することがある。これは入力bundleの情報を継承する構造から説明可能で、学習した内部PHとの新しい対応の発見ではない。
Attention flag PH・特徴VR PH・物理phase-lift VR PHは保存したが、それらをChernと同一視しない。

## Q10: 内部介入の機能的寄与

共通APIからhead、幅、距離、特徴、輸送、patchを変更し、同じcheckpointの状態を再評価した。
E/Fidelity/Projector誤差/曲率/局所位相が変わる介入は機能的依存の証拠になるが、特定の物理概念だけを符号化するheadであると断定しない。
連続系では再サンプリングのSEMを伴うため、差の符号だけを有意な改善/悪化とは呼ばない。
QWZではgapや連続性を保証せず、screenを通ったChernだけを条件付きで解釈する。
詳細な対応行はmatched_checkpoint_interventions.csv。未通過や不変だった介入も削除しない。
'''
    (folder/'RESEARCH_FINDINGS_JA.md').write_text(text)
    extra_figures(root,cases,observations,interventions,None if report_folder is None else root/'figures_corrected')


def extra_figures(root,cases,observations,interventions,figure_folder=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=figure_folder or root/'figures';out.mkdir(exist_ok=True)
    if observations:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for index,row in enumerate(observations):
            axes[0].plot([index-.15,index+.15],[row['before_crystal_QK'],row['after_crystal_QK']],'o-')
            if row['after_spin_selectivity'] is not None:axes[1].scatter(index,row['after_spin_selectivity'])
        labels=[r['embedding']+' s'+str(r['seed']) for r in observations]
        for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=75,fontsize=7)
        axes[0].set_ylabel('mean |crystal-shell QK Fourier|: before -> after');axes[1].set_ylabel('QK same-spin minus opposite-spin')
        fig.tight_layout();fig.savefig(out/'embedding_crystal_spin.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,task,key in zip(axes,['continuous_electrons','lattice_tfim','qwz_topology'],['delta_energy','delta_energy','delta_projector_mse']):
        selected=[r for r in interventions if r['task']==task and r[key] is not None]
        ax.scatter(range(len(selected)),[r[key] for r in selected],s=8);ax.axhline(0,c='k',lw=.6);ax.set(title=task,xlabel='matched-checkpoint intervention index',ylabel=key)
    fig.tight_layout();fig.savefig(out/'functional_interventions.png',dpi=150);plt.close(fig)
    qwz_cases=[c for c in cases if c['config']['experiment']['task']=='qwz_topology' and c['config']['experiment']['seed']==101 and c['group']=='D_geometry']
    if qwz_cases:
        from ..paths import wannier_source
        wannier_source()
        from wannier_ph.geometry import berry_data
        fig,axes=plt.subplots(len(qwz_cases),3,figsize=(10,2.3*len(qwz_cases)))
        for row,case in enumerate(qwz_cases):
            path=resolve(root,case['name'])
            if not (path/'COMPLETE.json').exists():continue
            arr=np.load(path/'diagnostics/after_internal.npz')
            for col,u in enumerate(arr['states']):
                axes[row,col].imshow(berry_data(u)['curvature'],origin='lower',cmap='RdBu_r',vmin=-.5,vmax=.5)
                axes[row,col].set_title(case['config']['experiment']['name']+f' mass{col}',fontsize=7)
        fig.suptitle('Predicted Berry plaquette phases; fixed kx,ky orientation');fig.tight_layout();fig.savefig(out/'qwz_berry_curvature.png',dpi=150);plt.close(fig)
