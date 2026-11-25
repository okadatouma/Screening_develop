"""Combine completed energy, PH and matched-wavefunction evidence."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    base=Path(__file__).resolve().parents[1]/"results/2026-11-06_followup"
    out=base/"n36_structure_all18_20261109_v2"
    wave=base/"n36_wavefunction_all18_20261109"
    def rows(p):return list(csv.DictReader(p.open()))
    summary=json.loads((out/"summary.json").read_text());groups=summary["groups"]
    kinds=[x["kind"] for x in groups];seeds=[101,202,303]
    ablations=rows(wave/"wavefunction_ablations.csv")
    cka=rows(wave/"feature_cka.csv")
    runs=rows(out/"run_structure.csv");ph=rows(out/"ph_changes.csv")
    # Each of the six learned runs has largest amplitude AND largest probe
    # log-amplitude response for candidate 2 = (1,-1).
    amps=rows(out/"fourier_final_amplitudes.csv")
    for name in {r["name"] for r in amps}:
        assert max([r for r in amps if r["name"]==name],key=lambda r:abs(float(r["amplitude"])))['candidate']=='2'
        candidates=[r for r in ablations if r["name"]==name and r["intervention"].startswith("remove_fourier_")]
        assert max(candidates,key=lambda r:float(r["logabs_centered_rms"]))['intervention']=='remove_fourier_2'
    colors={s:f"C{i}" for i,s in enumerate(seeds)}
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.2})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,kind in enumerate(kinds):
        for s in seeds:
            r=next(x for x in runs if x["kind"]==kind and int(x["seed"])==s)
            axes[0,0].scatter(j,float(r["last1000"]),color=colors[s])
            c=next(x for x in ph if x["kind"]==kind and int(x["seed"])==s and x["graph"]=="QK" and x["variant"]=="rank")
            axes[0,1].plot([j-.13,j+.13],[float(c["before_H1_area"]),float(c["after_H1_area"])],"o-",color=colors[s])
    for ax in axes[0]:ax.set_xticks(range(6),kinds,rotation=35,ha="right",fontsize=8)
    axes[0,0].set_ylabel("Last 1000 training energy (stored units)")
    axes[0,0].set_title("A. Baseline remains a strong training reference")
    axes[0,1].set_ylabel("Rank-QK integrated beta1; before -> after")
    axes[0,1].set_title("B. PH changes differ across representations")
    for i,r in enumerate([x for x in ablations if x["intervention"]=="remove_head_all"]):
        x=float(r["logabs_centered_rms"]);y=float(r["phase_centered_rms"])
        axes[1,0].scatter(x,y,color=f'C{kinds.index(r["kind"])}',marker=['o','s','^'][seeds.index(int(r["seed"]))])
        if r['name']=='n36_six_theta_onehot_l10_s303':
            axes[1,0].annotate('six_theta_onehot / 303\nweak attention dependence',(x,y),xytext=(.07,.08),arrowprops=dict(arrowstyle='->'),fontsize=8)
    axes[1,0].set(xscale="log",yscale="log",xlabel="Log-amplitude shape response RMS",ylabel="Phase shape response RMS (radians)",title="C. Removing all heads: most models change strongly")
    labels=['(1,0)','(0,1)','(1,-1)','(6,0)','(0,6)','(6,-6)']
    for ki,kind in enumerate(kinds[-2:]):
        for s in seeds:
            v=[x for x in ablations if x['kind']==kind and int(x['seed'])==s and x['intervention'].startswith('remove_fourier_')]
            axes[1,1].plot(range(6),[float(x['logabs_centered_rms']) for x in v],marker='o' if ki==0 else 's',color=colors[s],linestyle='-' if ki==0 else '--',alpha=.7,label=f'{s}'+(' L1' if ki else ''))
    axes[1,1].set_xticks(range(6),labels);axes[1,1].set(ylabel='Log-amplitude shape response RMS',xlabel='Removed Fourier candidate',title='D. Candidate (1,-1) has largest response in all 6 runs')
    axes[1,1].legend(fontsize=7,ncol=2)
    fig.suptitle('All 18 completed N36 runs: training and functional structure on matched probes',fontsize=12)
    fig.tight_layout();fig.savefig(out/'assessment_overview.png',dpi=170,bbox_inches='tight');fig.savefig(out/'assessment_overview.pdf',bbox_inches='tight');plt.close(fig)
    text=['# 全18条件の現時点の評価：何を学び、何を使っているか','',
          '全18条件が完了。エネルギー履歴、固定8配置での学習前後PH、辺シャッフル対照、同じ配置での波動関数の振幅・位相変化、head除去とFourier候補除去を組み合わせた。', '',
          '**現時点でも学習の仕方と機能的な内部構造は区別できる。最も説明しやすい特徴利用はlearned Fourierの低波数(1,-1)。baselineは学習エネルギーの比較基準として安定。six_theta+one-hotにはAttention依存が非常に弱いseedがあった。**','',
          '独立評価鎖は全18条件で未収束なので、以下の評価は真の変分エネルギー・正しい基底状態・物理相の順位ではない。固定配置の関数応答には再サンプリング誤差が入らないが、8配置の外への一般化は保証しない。','',
          '## 方式ごとの評価','',
          '|方式|学習末尾1000更新のエネルギー平均±seed標準偏差|現時点での評価|','|---|---:|---|']
    interpretations={
        'baseline':'3 seedともsix_thetaより低い学習エネルギー。head除去で出力も変化し、比較基準として安定。',
        'six_theta':'QKのrank-PHは3 seedとも増加。ただし結晶特徴CKAは大幅減少。結晶表現が改善したとの支持は弱い。',
        'spin_onehot':'学習エネルギーの改善はseedで不一致。入力の追加効果は安定していない。',
        'six_theta_onehot':'QKのrank-PHは3 seedとも減少。seed303はスピン優勢・Attention依存が非常に弱く、同じ仕組みを安定して学ぶ構成とは評価しにくい。',
        'learned_fourier':'全3 seedで(1,-1)を一貫して強く利用。除去による波動関数の振幅変化でも確認。何を使ったか説明しやすい。',
        'learned_fourier_L1':'baselineに近い学習末尾エネルギーを全3 seedで維持し、(1,-1)の利用も再現。ただし最適な物理状態の証明ではない。'}
    for g in groups:text.append(f'|{g["kind"]}|{g["last1000_mean"]:.2f} ± {g["last1000_seed_sd"]:.2f}|{interpretations[g["kind"]]}|')
    text += ['', '![評価の要点](assessment_overview.png)','',
             '## 1. 波動関数が実際に利用している特徴','',
             'learned Fourierの正則化なし3 seed・L1あり3 seedの全6条件で、(1,-1)の学習振幅が最大だった。さらに同じ8配置で各候補のcos/sin対だけを個別に消し、波動関数のlog|ψ|変化から全体規格化を除いたRMSを比較すると、全6条件で(1,-1)の除去応答が最大だった。そのRMSは0.585〜1.010。振幅だけの観察を、出力関数への介入で裏付けた。','',
             'この候補集合・初期化・モデルにおける対角方向の低波数の機能的利用は支持される。物理的なストライプ秩序・結晶化や、真のエネルギーへの改善寄与はまだ示していない。(1,0),(0,1)は共通入力にも含まれるため、候補の重要性には入力の重複も影響する。','',
             '## 2. 内部に構造があってもAttentionをほとんど使わない例','',
             'six_theta_onehot / seed303では、最終粒子特徴と擬スピンとの線形CKAが0.267→0.974に増え、結晶特徴とのCKAは0.601→0.048に低下した。同じ入力上で内部表現が擬スピンに強く偏ったと解釈できる。CKAは中心化した特徴行列の類似度であり物理秩序の観測量ではない。','',
             '実際に4 headすべての出力射影をゼロにすると、この条件の出力変化は振幅形状RMS=0.0224、位相形状RMS=0.0462 radだけだった。他17条件の振幅形状RMSは0.758以上。したがって、このseedは共通8配置上でAttention通信をほとんど使わない解に近い。局所経路を含む波動関数全体の変化は非ゼロなので、未学習とは呼ばない。','',
             '## 3. PHで区別できたこと','',
             '重みの順位でスケールを揃えたQKのH1面積は、six_theta・learned_fourier・learned_fourier_L1で3 seedすべて増加し、six_theta_onehotでは3 seedすべて減少した。単なる重みの大きさ以外に、方式ごとに異なる接続構造への変化がある。固定WindowのPHは全条件で前後完全一致した。','',
             'QK、実集約重み、最終特徴のPHは異なる空間の指標。QKに多数のループがあっても、それが実際の集約や出力にそのまま使われるとは限らない。PHの穴の多さを性能点にはしていない。','',
             '199回の辺シャッフル対照では、QK17/18条件、実集約重み18/18条件でH1面積が対照と異なった（36検定のBH q<0.05）。これは保存グラフに条件付けた「辺の重み分布だけでは説明できない構造」の検出。シャッフルは次数・距離・擬スピン関係を保存しないので、固定幾何や入力構造の効果も含む。方式間の有意な優劣とは区別する。','',
             '## 4. 結晶構造が強く学習された、とは今は読めない','',
             'six_thetaで結晶特徴と最終表現のCKAは3 seed平均0.552→0.053、six_theta_onehotでは0.604→0.077。結晶波数のQK成分も平均0.0687→0.0069、0.0702→0.0168へ低下した。与えた結晶周期をさらに強調した、という結果ではない。特徴が非線形に再符号化された可能性は残るため、情報が完全に消えたとは断定しない。','',
             '## 5. エネルギーの落ち方からの評価','',
             'baselineの末尾1000更新はsix_thetaより同一seedで7.48、10.42、8.56低かった。L1ありlearned Fourierとbaselineとの差は各seedで+0.33、+0.79、−0.24で、この学習予算ではほぼ同じ水準。one-hot化や結晶周期の追加による安定した学習エネルギー改善は観察されなかった。','',
             '学習中のエネルギーはwalker分布にも依存するため、これは最適化過程の比較。3 seedの両側符号反転検定は最小p=0.25であり、通常の5%有意差を主張できる繰り返し数ではない。独立評価の未収束と合わせ、基底状態の優劣には転用しない。','',
             '## 成果物と検証','',
             '[PHの手法・全結果](REPORT_JA.md)、[PH前後比較の図](persistent_homology_before_after.png)、[構造の集計](group_structure.csv)、[辺シャッフル検定](conditional_edge_nulls.csv)。','',
             '[全18条件のエネルギー・混合診断](../n36_all18_energy_20261109/REPORT_JA.md)、[波動関数の介入表](../n36_wavefunction_all18_20261109/wavefunction_ablations.csv)、[特徴CKA](../n36_wavefunction_all18_20261109/feature_cka.csv)。','',
             '実行に使ったモデルの現ソース114ファイルはジョブの凍結ソースとSHA256一致。初期波動関数は固定入力上で全18条件が完全一致。PH reducerを既存実装と照合し、読み込んだ完了成果物の不変性も確認した。新規学習・再サンプリングは行っていない。','']
    (out/'ASSESSMENT_JA.md').write_text('\n'.join(text))
    (out/'assemble_n36_assessment.py').write_bytes(Path(__file__).read_bytes())
    print(out/'ASSESSMENT_JA.md')


if __name__=='__main__':main()
