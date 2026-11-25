# PHと擬スピン・波動関数重要度・物理量を切り分ける解析

今回の実装は、通常PH、感度PH、物理相の証拠を別に扱う。新しいジョブは自動投入しない。
従来の `rs-sweep-20261031/plan.json` は変更していない。

## 1. 擬スピンを保存した条件付き対照

`screening_analysis/constrained_null.py` の二辺交換を使用する。

選ばれた上位k辺の重み付きグラフについて、4個の異なる頂点の辺(a,b),(c,d)を(a,d),(c,b)へ交換する。
擬スピンはs_a=s_c、s_b=s_dを要求する。各頂点の同擬スピン／異擬スピンへの次数が個別に保存される。
新しい辺の距離帯の組が元の辺の距離帯の組と一致する場合のみ採択し、重みを同じ距離帯へ移す。

厳密対照が保存するもの：

- 辺数、頂点ラベルごとの次数、同／異擬スピン別の次数。
- 各距離帯の重みの多重集合と非ゼロ辺数。
- 元のQK／screening／感度行列で正の重みが存在する許容支持。

保存しないもの：距離の個々の値、各PH閾値での次数、連結成分数、各頂点の重みの総和。
次数保存は基準の上位kグラフでの条件であり、フィルトレーション全体で次数を一致させるものではない。
距離はネットワークの周期ノルム。距離帯は全候補ペアから定めた分位点。

交換の試行数・採択数・目標交換数・変わった辺の割合を必ず保存する。有限の交換鎖を一様なグラフ標本とは認定しない。
擬スピンと次数を保存する二辺交換が存在するかも全列挙し、0なら動けないことを明示する。

制約階層を併記する：

1. `spin_degree_distance`：擬スピン・次数・距離帯を厳密に保存。
2. `spin_degree_only`：擬スピン・次数を保存、距離帯は制約しない。
3. `spin_distance_only`：擬スピンの++/--/+-ブロックと距離帯内でゼロを含めて重みを置換。次数は保存しない。次数のRMSE・最大変化も保存する。

全対照は条件付きのネットワークグラフであり、別の波動関数や物理状態を生成したものではない。

**二部グラフの重要な制約**：三角形がないため、flag複体の終端β1=E−V+C（Cは連結成分数）。
E,V,Cが同じなら終端の穴数も同じになる。違いを探す場合は、重みを加える途中の穴・誕生時刻・持続長を調べる必要がある。
上位kで打ち切ったPHの無限deathは、元の全グラフまで辺を追加すれば消える可能性がある。物理的な無限寿命とは解釈しない。

## 2. 波動関数感度を用いる別のフィルトレーション

接続をW_ij,W_ji→exp(ε)W_ij,exp(ε)W_jiと変化させたとき、

`d log ψ/dε = d log|ψ|/dε + i dφ/dε`

を使う。振幅微分の絶対値、位相微分の絶対値、両者の複素絶対値hypotの3種類を別々に保存する。
微分には既にWの係数が含まれるため、Wをもう一度掛けない。
PH入力へは各グラフの最大値で割って[0,1]へ写す。元の最大重要度を `importance-scales.csv` に保存し、順位変換PHも別に計算する。
全微分0の場合は情報のないゼログラフとして記録する。

これは固定されたネットワークの規格化・位相規約での接続感度で、規格化済み状態のゲージ不変なトポロジーやHall応答ではない。
粒子ペアの対応を保つため、位置・擬スピン・atoms・chargesをすべてhash照合してprobeと勾配を結び付ける。近似的な最近傍マッチングはしない。

## 3. 同じ配置での物理量

同じsample IDに対して、第一Bragg殻の各S(q)、殻平均、canonical finite-Nのg(r)、平均粒子間距離を計算する。
オプションで設定通りのハミルトニアンを復元し、局所エネルギーの実部・虚部も計算する。
学習ログの別時刻のエネルギーを配置へ割り当てることはしない。

`within-run-correlations.csv` は同じ配置のPH指標と殻平均S(q)、g(r)の各ビン、局所エネルギーのPearson相関。
各層・ヘッドを分け、5配置未満や定数列では相関を未定義として残す。walkerやヘッドを独立な学習試行と数えない。
保存walker／共通入力に混合・ESSの保証はない。これらは記述的な相関で、有意差・平衡物理量の証明ではない。

複数runの集計は `ph_research_report collect` を用いる。学習run単位の特徴量を出し、λ・rs・初期化を固定した層と、混ぜた探索的相関を区別する。
同じseedの初期化違いは相関し得る。run数とunique seed数を別々に記録し、p値は出さない。
`head=-1` は各ヘッドのPHの平均で、ヘッドのラベル交換に不変。平均重みのPHとは異なる。

## 4. 実行入口

array-2475の既存監査データを使用する場合：

```bash
PYTHON=/home/tomax/venvs/screening-gpu/bin/python
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 "$PYTHON" -m screening_analysis.ph_research audit \
 results/manual/array-2475/connection-audit-validated \
 --output NEW_DIRECTORY --samples 8 --repeats 8 --k 60 90 --bins 6 --energy-samples 8
```

今回の実行済み結果は `results/manual/array-2475/ph-research-v2`（8配置、k=90、4対照反復、局所エネルギーも8配置）。

```bash
env MPLCONFIGDIR=/tmp/rs-sweep-mpl /home/tomax/venvs/pyrite-bands/bin/python \
 -m screening_analysis.ph_research_report audit RESULTS_ROOT \
 --audit-source results/manual/array-2475/connection-audit-validated --output NEW_REPORT
```

rs掃引のnative/commonフォルダだけを再解析する場合：

```bash
"$PYTHON" -m screening_analysis.ph_research subset CASE/analysis/native \
 --output NEW_DIRECTORY --samples 8 --repeats 8 --k 60 90 --energy-samples 8
```

主な出力：`ph-features.csv`、`diagrams.npz`、`betti-curves.npz`、`constrained-null.csv`、`constraint-hierarchy.csv`、
`switch-diagnostics.csv`、`configuration-physics.csv`、`physical-arrays.npz`、`within-run-correlations.csv`。
全PH図の無限deathはnpzでそのまま保持する。

## 5. 複数λ・seedへの接続

`rs_sweep.research.example.json` と準備済み `results/rs-sweep-research-20261031/plan.json` を追加した。

- λ：0, 0.33, 0.50, 0.60, 0.64, 0.72。
- λ=0.50,0.60,0.64ではWC初期化とAHC初期化の両方。
- rs：10,15,20。seed：101,202,303。合計81ケース。
- `analysis.research.enabled=true` の計画だけで追加解析を実行。従来の54ケースの計画は変更しない。
- AHC参照表は係数の初期値として使い、各λ・rsでHFを再計算する。相を強制・認定しない。

新しい投入コマンド（今回は実行していない）：

```bash
sbatch --array=0-80%2 \
 --export=ALL,SWEEP_PLAN=/home/tomax/projects/Screening_topology_gpu/results/rs-sweep-research-20261031/plan.json \
 /home/tomax/projects/Screening_topology_gpu/rs_sweep.job
```

集計：

```bash
/home/tomax/venvs/pyrite-bands/bin/python -m screening_analysis.ph_research_report collect \
 results/rs-sweep-research-20261031/plan.json --output NEW_COLLECTION
```

追加計算は通常の内部診断より重い。`samples/repeats/ks/bins/energy_samples`をJSONで調整する。
既存計画を再利用する場合は新規出力先へprepareし、進行中の計画や結果を上書きしない。

## 6. 独立したWC/AHC相の確認は、まだ未達

現在のzero-twistチェックポイント、HFのChern診断、結晶構造因子、接続PHだけから、学習後の多体Chern数やHall応答を求めることはできない。
この実装は境界ねじれを掛けただけの同じ状態を、最適化済みの状態族と偽って使わない。

既存の検証済みBerry実装を使う `screening_analysis.phase_evidence` を追加した。[Fukui–Hatsugai–Suzukiの方法](https://arxiv.org/abs/cond-mat/0503172)に基づき、独立に用意されたねじれ境界状態族を評価する。
既知QWZの多体状態で、粗密格子・ゲージ変更・直接ベクトルとoverlap入口の一致を検証した。これはλ-jelliumの相確認ではない。

```bash
"$PYTHON" -m screening_analysis.phase_evidence twist-manifest.json --output NEW_EVIDENCE
```

manifest例：

```json
{
  "physical_system": "lambda-jellium",
  "basis_id": "documented common physical Hilbert basis / overlap estimator",
  "state_origin": "independently_optimized_twisted_ground_states",
  "grids": [{"path": "twist-grid-coarse.npz"}, {"path": "twist-grid-fine.npz"}]
}
```

NPZは2方式に対応する。

1. `states[nx,ny,D]`：共通の物理Hilbert基底上の多体状態ベクトル。必要なら `sewing_x/y[D,D]`。任意walkerでの波動関数値の列はこの形式に該当しない。
2. `overlap_x/y[nx,ny]`：隣接ねじれ点間の実際の多体overlap。境界のsewingを含めたことを `overlap_includes_boundary_sewing=true` で明示。Monte Carloで推定する場合は `overlap_error_x/y` も保存する。N36で巨大なHilbertベクトルを列挙せず使うための入口。

共通キーは `theta_x/y`（[0,2π)の一様格子）。独立に計算された `gap_lower_bounds[nx,ny]` も任意で受け付ける。
現在は単一の追跡された状態が対象で、縮退状態群の非Abelian計算は未対応。

格子細分化でChernが安定するか、overlapが小さくないか、plaquette位相が分岐に近くないか、供給されたgapが正かを出力する。
overlapの誤差がChernの安定性に与える影響、ねじれた基底状態の学習収束、熱力学極限の結晶秩序、独立Hall応答は別途必要。
本番のλ-jelliumについて、そのねじれ状態族を学習・サンプリングする工程はまだ実装・実行していないため、`phase_verified=false`を維持する。
