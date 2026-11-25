"""Reports use completed runs only; failed/unrun rows remain visible in status CSV."""
import csv
import json
from pathlib import Path
import numpy as np
from .store import csv_save,json_save


def read_json(path):return json.loads(Path(path).read_text())
def rows(path):
    if not Path(path).exists():return []
    with Path(path).open() as f:return list(csv.DictReader(f))
def table(headers,data):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in data)
def f(value):return 'NA' if value is None else f'{value:.5g}'


def report(root,supplement=None):
    from ..cli import collect
    root=Path(root);status=collect(root);cases=read_json(root/'plan.json');complete=[]
    for case in cases:
        from .retries import resolve
        folder=resolve(root,case['name'])
        if (folder/'COMPLETE.json').exists():complete.append((case,read_json(folder/'summary.json'),folder))
    if not complete:raise ValueError('no completed calculations to report')
    report_dir=root/'reports';report_dir.mkdir(exist_ok=True)
    output=report_dir/'COMPARISON_REPORT_JA.md'
    if output.exists():raise FileExistsError('preserve prior report; use a new report directory for a later collection')
    groups={task:[r for r in complete if r[0]['config']['experiment']['task']==task] for task in ['continuous_electrons','lattice_tfim','qwz_topology']}
    text=f'''# Embedding・Screening・Wavefunction比較: 実行結果

計画 {len(cases)} 条件、完了 {len(complete)} 条件。失敗/途中/未実行はcomparison.csvに残した。
この報告の数値表・図はCOMPLETE.jsonのある実計算だけから作成している。
小さいCPU pilotであり、WC/AHC認定、熱力学極限、一般的な最良モデルの決定は行わない。
既存の長時間計算や以前のQWZ/Wannier結果を上書き・置換していない。

## 同条件比較の定義

Hamiltonian、N、境界twist、seed、step、optimizer、sampling設定をconfig.jsonに保存した。
task間は異なるHilbert空間であり、エネルギーやPHの値を横断して同じ量として比較しない。
連続系では自然な入力次元の比較に加え、埋め込み入力を12成分へ揃えた比較を実施した。
後者ではbaseline側に能動的feature gainを置き、learned Fourier振幅と名目パラメータ数も合わせる。
ただしパラメータ数の一致は表現能力や最適化条件の完全一致を意味しない。
TFIMではR=4とFullはN=8のring上で同じsupportであり、独立な物理的範囲とは数えない。
head数の比較ではheads×head_dimを揃えるが、headごとのTrim/幅のパラメータ数は変わる。
深さ比較は自然な容量の比較で、容量差を除去した因果比較ではない。

## 連続電子系

Screenformerのcosine QK、Trim、Window、分母なしValue集約、TanhNorm、Gate、射影を再利用した。
通常のsoftmax Attentionへ置き換えていない。以下のEはfresh Bornサンプリングによる推定。
MC誤差・学習不足が大きく、単一seedの小さい差を改善と認定しない。
非相互作用pilotの参照は合成plane-wave表であり、自己無撞着HF解とは呼ばない。
Coulomb/Jastrow比較は同じCoulomb Hamiltonianの独立groupで行う。
twistあり条件は境界条件の検証用であり、θ=0条件とのE差をAnsatz改善効果とは扱わない。

'''
    text+=table(['run','seed','E','block SEM','parameter数','Jastrow'],[[c['config']['experiment']['name'],c['config']['experiment']['seed'],f(s['energy']),f(s['block_sem']),s['parameter_count'],s['jastrow_active']] for c,s,p in groups['continuous_electrons']])
    text+='''
Fourier波数は整数逆格子候補だけを用いる。結晶モードはL=A Sの整数Sから導出するので、6×6以外へ固定6を持ち込まない。
初期・学習後のQK/Screening Fourier、同異spin選択性、Spin×Fourier、学習振幅、Born構造因子を保存した。
振幅には即時のa_q W_q再スケール自由度を抑える入力射影の正規化を使うが、後段との冗長性は残る。
振幅の大きさだけを物理秩序の自発的発見とは解釈しない。
uniform probeの構造因子とBorn状態の構造因子は別ファイル・別測定として扱う。
今回N=2の自由系はWC/AHC候補ではないので、one-hotがWC/AHC識別を改善したという検証にはなっていない。

## TFIM: 物理量とWindow

Hamiltonian・局所エネルギー・ED・S2・CMI・相関長・p_-は既存Lattice_screeningから再利用した。
小規模pilotは全配置の厳密なBorn重みでエネルギー勾配を計算し、有限鎖のEDと比較した。
損失にはパリティやFidelityを加えていない。通常Born MI、全残り変数で条件付けたBorn CMI、Quantum MIは別量。
相関長は有限周期鎖のsecond-momentまたは有限鎖fitであり、熱力学的な相関長ではない。

'''
    text+=table(['run','seed','ΔE/(NJ)','Fidelity','p_-','S2 half','xi(second moment)','幅'],[[c['config']['experiment']['name'],c['config']['experiment']['seed'],f(s['epsilon_energy']),f(s['fidelity']),f(s['odd_parity_probability']),f(s['S2_half']),f(s['xi_second_moment']),','.join(f(v) for v in np.asarray(s['widths']).ravel())] for c,s,p in groups['lattice_tfim']])
    text+='''
幅の学習軌跡、実効接続距離、ED/NNのCMI、S2、p_-を同じcheckpointで比較する図を生成した。
正則化ありで幅が縮むことを、自発的な物理的相関長の発見とは呼ばない。
pilotの短い学習で残るp_-や誤差は長時間学習で再確認が必要。

## QWZ: 複素表現とGauge transport

QWZ状態、Projector、局所atlas、FHS、phase-liftは既存wannier_phを再利用した。
教師はProjector/局所フレーム/遷移位相であり、Chern数をlossに使用していない。
以下のゲージ誤差は、同じcheckpoint・同じ入力にランダム局所U(1)を適用したProjector出力の最大差。
実数2channelとbaselineの生Bloch実虚成分は同じ情報・同じ演算になり、差がないことも対照の一部である。

'''
    text+=table(['run','seed','P MSE','gauge誤差','C一致','parameter数'],[[c['config']['experiment']['name'],c['config']['experiment']['seed'],f(s['projector_mse']),f(s['gauge']['projector_max_difference']),f"{sum(abs(r['chern']-r['target_chern'])<1e-5 for r in s['metrics'])}/{len(s['metrics'])}",s['parameter_count']] for c,s,p in groups['qwz_topology']])
    text+='''
Gauge-covariant設計はcharged入力、物理Berryリンク、atlas coverを入力として受け取る。
共変な輸送と出力射影によるゲージ不変性は設計の検証であり、Berryリンクやトポロジーをネットワークが発見した証拠ではない。
特に非零で連続なcharged出力が入力bundleのChernを保つことは、未学習でも起こり得る。
projectorとatlasは異なる出力型で、atlasには2つの局所フレームだけを有効領域で教師として与える。
大域的に滑らかな周期Blochフレームは要求しない。ProjectorのHermiticity・冪等性・rank残差も各行へ保存した。
最小overlap以下のリンクは0にして接続を無効化し、有効リンクmaskを記録する。

## PHと内部量

Attention/QK/Alpha/Window/実集約重み、特徴点群、phase-liftのPHを同じモデルの学習前後で保存した。
Screenformer/TFIMのPHは既存F2 weighted flag reducerを使い、重み0の辺は欠落させる。
QWZのQKはtanhで有界化して同じflag化を行う。QWZ Attentionはsoftmaxであり、CSV/図のScreening graphという共通キーは「実集約重み」の意味で使う。
特徴PHはEuclidean VR、物理phase-lift PHも8次元Euclidean VR。異なる対象・距離を同じ不変量とは扱わない。
既存研究で確認した標準phase-liftのplateauと、距離・尺度を変えた際の失敗条件は回帰テストの基準にした。
本比較だけで内部PHとWannier障害の普遍的対応を認定しない。

## 介入

共通apply_intervention APIから、head/接続/距離帯/幅/Fourier/6θ/spin/位相/transport/patchを適用可能。
各taskに存在しない介入は拒否する。連続電子の番号だけで特定辺を除く物理的介入も反対称性を壊すため拒否する。
TFIMは介入後の全状態を再正規化して厳密に再評価。連続系は変更後の波動関数からfresh Born chainsで再評価し、元分布の局所Eを流用しない。
連続系のzero_phase_channelはNN軌道の虚部を除く介入で、HF軌道の複素位相は残る。
QWZは介入後のProjector誤差・Berry曲率・Chern・atlas位相を再評価し、元/倍メッシュ、復号norm、overlap、plaquette branchのscreenを記録した。
NNの復号normは物理エネルギーgapではない。実gapや連続性が証明されたとは報告しない。
介入で誤差が増えても、そのheadが位相障害を選択的に担うことの証明にはならない。

## 大型実験・未検証

- 小さい連続系のMCは未収束の可能性がある。WC/AHCの識別改善、学習Fourierによる真の結晶波数の自発選択は未検証。
- N=12/16の長期TFIM、実N=36のembedding比較はproduction planだけを作成し、既存Slurm/GPUジョブには変更・追加を行っていない。
- 連続系twistの小さな矩形でperiodic partのoverlapを測れるが、全twistトーラスのsewing・gappedな固有状態追跡・Hall/Chern認定は未実施。
- 連続系のエンタングルメント、N>2の多体overlap推定、厳密な同一表現能力比較は未実装。
- Stage Fの複合改良は、独立seedと誤差に対して単独効果が確認された後に限る。このpilotだけから最良条件を確定して追加学習しない。
- 結果は相関・介入感度として報告し、内部構造と物理量の一般的な因果対応を断定しない。

全runにconfig、入力hash、checkpoint、optimizer/RNG/MC状態、学習ログ、内部配列、コードrevisionとsource snapshotを保存した。
comparison.csvには失敗・途中・未実行も残している。未実行の条件を実行済みの平均へ混ぜない。
'''
    output.write_text(text)
    figures(root,groups)
    from ..diagnostics.findings import findings
    findings(root,supplement)
    # Required top-level deliverable links to this actual completed study.
    top=Path(__file__).resolve().parents[1]/'COMPARISON_REPORT_JA.md'
    if not top.exists():top.write_text(text+f'\n[研究上の問いへの実測回答]({root.resolve()}/reports/RESEARCH_FINDINGS_JA.md)\n\n元データ・図・全設定の保存先: {root.resolve()}\n')
    json_save(report_dir/'report_manifest.json',dict(complete=len(complete),planned=len(cases),runs=[c['name'] for c,s,p in complete]))
    return output


def figures(root,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    if groups['continuous_electrons']:
        data=[v for v in groups['continuous_electrons'] if v[0]['group']=='B_embedding']
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        names=[]
        for index,(case,s,path) in enumerate(data):
            names.append(case['config']['embedding']['type']+f" s{case['config']['experiment']['seed']}")
            axes[0].errorbar(index,s['energy'],yerr=s['block_sem'] or 0,fmt='o',c=f'C{index%5}')
            vals=rows(path/'diagnostics/after_fourier.csv')
            axes[1].plot(range(len(vals)),[float(r['QK_Fourier']) for r in vals],'.-',label=names[-1])
        axes[0].set(xticks=range(len(names)),xticklabels=names,ylabel='fresh Born energy +/- block SEM');axes[0].tick_params(axis='x',labelrotation=70)
        axes[1].set(xlabel='integer reciprocal mode index (CSV)',ylabel='QK Fourier component');axes[1].legend(fontsize=6)
        fig.suptitle('N=2 free-electron pilot: no WC/AHC labels');fig.tight_layout();fig.savefig(folder/'embedding_energy_fourier.png',dpi=150);plt.close(fig)
        learned=[v for v in data if v[0]['config']['embedding']['type']=='learned_fourier']
        if learned:
            fig,ax=plt.subplots(figsize=(7,4))
            for case,s,path in learned:
                h=[json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
                amp=np.asarray(h[-1]['fourier_amplitudes']);ax.plot(range(len(amp)),amp,'o-',label=f"seed{case['config']['experiment']['seed']}")
            ax.set(xlabel='allowed integer q index',ylabel='learned amplitude (not an order parameter)');ax.legend();fig.tight_layout();fig.savefig(folder/'learned_fourier_amplitudes.png',dpi=150);plt.close(fig)
    if groups['lattice_tfim']:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for case,s,path in groups['lattice_tfim']:
            width=float(np.mean(s['widths']));label=case['config']['attention']['window']['mode']
            for ax,key in zip(axes.ravel()[:4],['xi_second_moment','S2_half','odd_parity_probability','epsilon_energy']):
                if s[key] is not None:ax.scatter(width,s[key],s=15,label=label);ax.set(xlabel='mean head width (site units)',ylabel=key)
            info=np.load(path/'diagnostics/NN_information.npz');n=s['n'];cmi=info['Born_CMI'][np.triu_indices(n,1)].mean()
            axes[1,1].scatter(width,cmi,s=15);axes[1,1].set(xlabel='mean width',ylabel='mean Born conditional MI')
            if case['config']['experiment']['seed']==101 and case['config']['physics']['g']==1.:
                h=[json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
                axes[1,2].semilogy([r['step'] for r in h],[max(r['odd_parity_probability'],1e-12) for r in h],label=case['config']['experiment']['name'])
        axes[1,2].set(xlabel='energy-only step',ylabel='p_-');axes[1,2].legend(fontsize=5);fig.tight_layout();fig.savefig(folder/'window_physics_parity.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        for case,s,path in groups['lattice_tfim']:
            if case['config']['attention']['window']['mode']=='trainable_per_head' and case['config']['experiment']['seed']==101:
                h=[json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
                for head in range(case['config']['model']['heads']):ax.plot([r['step'] for r in h],[r['widths'][0][head] for r in h],label=case['config']['experiment']['name']+f' h{head}')
        ax.set(xlabel='energy-only step',ylabel='width (site units)');ax.legend(fontsize=5);fig.tight_layout();fig.savefig(folder/'head_window_trajectories.png',dpi=150);plt.close(fig)
    if groups['qwz_topology']:
        fig,axes=plt.subplots(1,2,figsize=(13,5));labels=[]
        for index,(case,s,path) in enumerate(groups['qwz_topology']):
            labels.append(case['config']['experiment']['name'].replace('qwz_','')+f" s{case['config']['experiment']['seed']}")
            axes[0].scatter(index,max(s['projector_mse'],1e-12));axes[1].scatter(index,max(s['gauge']['projector_max_difference'],1e-9))
        for ax,name in zip(axes,['Projector MSE','local U(1) output difference']):ax.set(xticks=range(len(labels)),xticklabels=labels,yscale='log',ylabel=name);ax.tick_params(axis='x',labelrotation=90,labelsize=6)
        fig.tight_layout();fig.savefig(folder/'qwz_reconstruction_gauge.png',dpi=150);plt.close(fig)
