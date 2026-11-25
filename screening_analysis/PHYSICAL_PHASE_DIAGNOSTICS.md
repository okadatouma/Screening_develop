# 非PH物理相診断

調査日: 2026-11-02。対象: 連続電子系N=36、λ=0.72、rs=4/25、H=2/3/4/10、seed=101、1層、80000更新（保存step=79999）。既存の未コミット変更・checkpointは保存する。

## 現状調査（今回の追加前）

|解析|状況|確認した再利用先・今回補う点|
|---|---|---|
|A 波数空間|一部実装|`lj_maps.sk/first_shell/structure_factor`にS(q)。接続重みスペクトル、幾何対照と条件付き増分を追加|
|B 距離|既存実装|`connection_study.analyze_connections`の周期ノルム/最小像距離、同異spin、R90を再利用。R50/R95、CDF、QK由来の明示的正重みを追加|
|C 幾何残差|未実装|距離×spin binで配置を分離したcrossfit、欠損支持の一致した波数対照を追加|
|D head分化|一部実装|`head_contrib.projected_heads`、probeのhead_vectors/Gram、既存head相関を再利用。JS、有効head数、全head出力除去を追加|
|E 循環|一部実装|`directed_loops.oriented_triangles`はPBC最短像の可縮CCW三角形。既存は辺差の**和**。今回は要求された積の差を別関数で追加|
|F 位相応答|既存実装|`rs_observables.phase_sensitivity`の対称辺log重み介入微分、既存保存勾配の入力fingerprint照合を再利用。重みとの条件付き相関、座標微分を追加|
|G 固有モード|一部実装|`rs_observables.attention_modes`。raw/規格化行列で別々のRayleigh値と同一行列のRitz検査、結晶波数射影を追加|

前回中断時点では `physical_phase_diagnostics.py` と `physical_phase_metrics.py` が未追跡の途中コードとして存在した。これを修正・完成し、レポート、テスト、設定を追加する。

## 定義を確認したソース

- `periodicwave/attention.py:screening_attention`: Q=XW_Q、K=XW_K、qbar=q/sqrt(sum(q²)+1e-12)、同様にkbar。S=qbar·kbarは符号付き類似度。**softmaxなし**。r=TRIM_MAX sigmoid(rho)、Alpha=[relu(S-(1-r))/r]^p、Windowを掛ける。介入/clip等の後、実集約重みでValueを集約する。通常Valueは正規化vで自己項を含み、固定Valueのみ対角除去。TanhNorm→head結合→tanh(SiLU(XW_G)) Gate→W_O→exp(out_scale)。共有biasをheadへ割り振らない。
- `periodicwave/attention.py:screening_window`: この8条件の実効設定は入力manifestに記録。cos窓なら(1+cos(pi clip(d/w,0,1)))/2、w=dref softplus(omega)。保存名だけから推測しない。
- `periodicwave/networks.py:make_periodic_construct_input_features`: 格子は列ベクトル、周期ノルムはsin/cos埋め込みの計量。ユークリッド最小像距離とは異なる。code長さ1=rs a_B、q単位=(rs a_B)^-1。逆格子行ベクトルm@(2π L^-1)。
- `screening_analysis/graph_wind.py:minimum_image_edges`: 傾いたセルにも対応する真の最短像。三角形の非可縮・像の縮退・共線性は除外。
- `periodicwave/CustomScreenformer.py`、`lj_maps.py`: 単層入力は周期sin/cos座標と擬スピン±1。この単層QKは距離だけでなく絶対周期座標にも依存し得る。幾何残差を未知の多体相関と断定しない。
- `checkpoint_model.load_model`: 保存configと実効設定、元精度の格子、HF表から既存factoryで復元。再初期化・再学習はしない。HFはplane/WC初期軌道の交絡となる。
- `ph_research.configuration_physics`: 自己項を含む通常S(q)=|rho_q|²/N、g(r)はN(N-1)有限粒子規格化。既存energy測定を再利用。局所energyはcode単位、electronあたりHartree=E/(N rs²)。
- `phase_evidence.py`: 供給されたtwist状態族の重なりからのChern診断は存在。ただし今回の8 checkpointはその入力でないため実行不可。内部循環からChern/Hallを推測しない。
- `Lattice_screening/README.md`: 格子系も本体Attentionを共有するが、今回の解析には格子系の状態・checkpointを混ぜない。

既存出力を調査: `comparison-20261102`、`representation-20261102`、`matched-null-20261102`。common bankは合成一様座標でspin配列の条件も平衡分布と異なる。Nativeは既存connection-studyの8 chain×16 block×8測定=1024配置（burn128、保存間20 step）。各モデルの32配置で新規内部解析、通常物理量とenergyは全1024配置で集計する。

## 実行と再生成

```bash
JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/home/tomax/venvs/screening-gpu/bin/python -m screening_analysis.physical_phase_diagnostics \
  --config configs/physical_phase_diagnostics.json --mode all

# 中断から再開: 同一設定・checkpoint・入力のキャッシュのみ再利用
# 上と同じコマンドに --resume を付加

# 図・本文だけ再生成（JAX/GPU不要）
/home/tomax/venvs/pyrite-bands/bin/python -m screening_analysis.physical_phase_diagnostics \
  --config configs/physical_phase_diagnostics.json --mode report

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/home/tomax/venvs/screening-gpu/bin/python -m unittest tests.test_physical_phase_diagnostics -v
```

GPU環境はPython3.12/JAX0.7.2/NumPy/SciPy、描画は既存pyrite-bands環境のMatplotlibを使用する。依存を更新しない。nvidia-smiはNVMLの版不一致だがJAXのCUDA device評価は可能。解析は1モデルずつ、probeは配置ごと、逆伝播は少数配置、JAXキャッシュはモデル間で解放する。checkpointを上書きしない。成果物cacheから図と本文を再生成できる。MC block SEMは暫定的で、ESS・平衡化・独立seed誤差は未検証。

## 今回の実行結果

8モデル×Native/Commonの16 subsetが完了。図20枚、summary.md/report.md、head/配置ID付きCSV、cache、入力manifestを生成した。内部接続32配置/モデル/入力、内部感度4配置、座標応答・head除去2配置、通常物理量はNative全1024配置を評価した。

追加テスト15件（数値13、描画側派生統計2）と既存の関連回帰18件が合格した。数値13件には合成checkpoint→正式probe→非PH解析の小規模統合試験、描画側2件には縮退固有空間の基底不変性とlog重み微分の重み因子対照を含む。16 subsetで同一行列のRitz検証、QK/Alpha/Window/Screening定義、実checkpointでの粒子置換/PBCを検証した。

追加のオフライン検証はGPU不要:

```bash
/home/tomax/venvs/pyrite-bands/bin/python -m screening_analysis.physical_phase_diagnostics \
  --config configs/physical_phase_diagnostics.json --mode verify
/home/tomax/venvs/pyrite-bands/bin/python -m unittest tests.test_physical_phase_report -v
```

`verification.json`は保存済みの内部介入微分について、振幅・位相の両方の有限差分、checkpoint/config/HF/モデルソースの不変性、bankファイルhashを検証・記録する。解析時点のsource hashと、その後追加した検証/描画コードのsource hashは別々に残る。

AHC候補rs=12のcheckpointは利用可能だが今回の8条件には追加していない。独立seed・容量一致、MC収束/ESS、head除去後energy/S(q)、twist状態族のHall/Chern検証は未実施。重みとlog辺介入感度の相関には重み因子が含まれるので、`phase_adjusted.csv`と`phase_scale_control.png`も読む。
