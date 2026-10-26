# G：正則化とseed分離ベンチマーク

実測結果・図・検証記録は [TOPOLOGY_G_RESULTS.md](TOPOLOGY_G_RESULTS.md)。A〜Fの定義は
[README_topology.md](README_topology.md)、[README_topology_cdef.md](README_topology_cdef.md)を参照。
本追加では通常の波動関数forward、Hamiltonian、局所エネルギーの微分式を変更していない。

## 有効化と対象

`cfg.optim.topology_regularization` の既定値は以下。`mode='off'` または `gamma=0` の場合、
損失ラッパーは元のcallableそのものを返し、trainerはdiagnosticを構築しない。

```python
cfg.optim.topology_regularization = dict(
    mode='off', gamma=0., perturb_sigma=.03, max_samples=8,
    ph_floor=1e-4, landscape_levels=3, landscape_grid=21,
    sampling_score=True)
```

実装は `periodicwave/topology_regularization.py`。有効時はCustomScreenformerと
Adam/LAMB/optimizerなしを受け付ける。実学習で検証したのはAdam。
KFACとの同時利用は未検証なので有効時に明示的エラーにする。無効時の既存KFAC経路は保持。
真のPHモードは参照実装の計算量を制限するためN≤36。複数deviceの実学習は未検証。
`jax.pure_callback`によるCPUペアリングを含み、アクセラレータ上の効率的PH実装ではない。

既存entry pointからの例（出力先には存在しないディレクトリを指定）：

```sh
PW_OPTIMIZER=adam PW_TOPO_MODE=persistence PW_TOPO_GAMMA=.1 \
 PW_TOPO_SAMPLES=4 PW_SEED=11 PW_BATCH=16 PW_OUTPUT_DIR=/tmp/new-lambda-topology-run \
 .venv-pw/bin/python -m periodicwave.configs.run_lambda_jellium 9 15 .33 24 topo-test
```

`PW_TOPO_SIGMA`、`PW_MCMC_STEPS`、`PW_WIDTH`、`PW_ATTN_DIM`、`PW_VALUE_DIM`も指定可能。
既存デフォルトは維持。新規出力先の指定時は配列を構造化した`runtime_config.json`を保存し、
実際のseed、窓設定、JAX精度、Hamiltonian、ansatzを復元できる。既存出力先は拒否して暗黙の再開を防ぐ。
有効時の`train_stats.csv`では`energy`は物理エネルギー、`objective`は正則化込み、
`topology_penalty`は係数を掛ける前の値。既存のauxiliary energy/varianceは物理量のまま。

## 三つの異なる損失

各配置でDiagnostic APIから実際の`weights_window`を取得する。Head平均のグラフを層ごとに作り、
自己辺を除く。Head番号を特徴として固定しない。近傍配置は同じspinで
`R'=R+sigma*Normal(0,I)`。周期性は既存の周期特徴層が処理する。
各モードとも `ell = mean((D(R)-D(R'))**2)/sigma**2`。

| mode | D | 意味 |
|---|---|---|
| persistence | 対称グラフのH0/H1 Persistence Landscape | 真のflag PHに基づく区分的微分可能な正則化 |
| laplacian | 対称グラフの `Tr[(L/N)^k]/N`, k=1,…,4 | 微分可能なスペクトル代替指標。PHではない |
| weights | 向き付きHead平均重み | 重みの滑らかさ。トポロジカル損失とは呼ばない |

PHではゼロ辺による無限寿命の変動を避けるため、対角を除いて
`B_eps=eps+(1-eps)*B`, eps=1e-4 と完成し、`f(edge)=1-B_eps`のsublevel flag complexを使う。
H0/H1のcritical edgeをF2境界行列簡約で選び、その辺の値を通してJAXで微分する。
共通のessential H0一本は除外し、対角上のゼロ寿命paddingはLandscapeに寄与しない。
離散的なペアリング・argmaxそのものは微分しない。タイやペアリング変更点では通常の微分は保証されない。
したがって「全域で滑らかなPH損失」ではない。Aのゼロ辺を欠落と扱う疎グラフPHとは別の定義で、
解析にはraw/completed双方を保存する。Landscapeは既定で3段・[0,1]の21点の離散化。
有限差分検証はペアリングが安定した例で行う。

目的は `E_VMC + gamma*E_{R~|psi|²,noise}[ell]`。
既定では直接のパラメータ微分と、配置分布の微分
`2 cov(ell, grad log|psi|)`を含む。score項は有限バッチの中心化共分散推定。
`sampling_score=False`は固定入力分布に条件付けた勾配であり、同じ目的の完全な微分ではない。
両方の配置に対するネットワーク微分は通る。ノイズとMCMC座標の履歴は微分しない。
三つの損失は単位・尺度が異なるため、同じgammaが同じ正則化強度を意味しない。

## checkpoint、入力分布、ラベル

`checkpoint_catalog.py`は再帰探索し、N、lambda、rs、seed、学習step、ansatz、精度、
収束履歴、ファイル/パラメータhashと重複を記録する。パラメータやファイル名から物理相は推定しない。
復元は`checkpoint_model.py`が既存factoryと保存パラメータを利用し、独自の波動関数forwardは作らない。
`attn_probe.probe(..., evaluation_data=...)`で明示的な入力を渡せる。
入力4配列（位置、spin、原子、電荷）のhashを保存する。

* A：固定した各波動関数から新しくサンプリング。学習用MCMCの保存配置をそのまま再利用しない。
  4 chain、burn32、16 sweep、標本間2 step。seedごとに独立の測定乱数、同じseedのvariant間は対応付け。
  短い鎖であり、平衡・収束の証明ではない。エネルギーは既存Hamiltonianで64配置を評価。
* B：モデル・ラベルとは独立の一様トーラス座標とランダムspin、seed19021。
  同じN/cell/Hilbert空間には全く同じ32配置を入力。物理的な期待値をこの分布で評価したとは扱わない。

新たなlambdaベンチマークのラベルは`lambda=0.33/0.72`という**パラメータ条件**。
WC/AHCラベルはunknownのまま。零twist C3の期待値・固有状態残差も別に測定し、
固有値やChern数へ丸めてラベルに変換しない。

## 識別と統計

8種類の特徴群：密度4×4、有限N規格化g(r)、構造因子8モード、周期距離の統計、
幅/Trimの分位点、重み分布、raw PH、completed PH。
g(r)は最短格子ベクトル長の0.45未満の完全な環状領域で評価。
PHは層・Head平均グラフのdiagram統計とLandscape。個別Headを固定識別子にしない。
A1/A2/A3/QKや密度・重み整合対照はA〜Fの別解析にも保存してある。

主評価には最終checkpointのcore24件のみを使用。各正則化につき2条件×3seedで6判断。
同じrunの配置、途中checkpoint、窓stressを独立した訓練標本として水増ししない。
leave-one-seed-out、訓練foldのみで標準化する最近傍重心分類器、調整なし。
一般APIはseedとfactorの両方を同時に除外する分割、任意の多クラスラベルにも対応。
今回は独立の物理ラベル不足のため、stressに対する物理相汎化性能は算出しない。

Balanced accuracy、binary/macro F1、AUROC、MCC、混同行列を保存。
95%区間はOOF予測をseed群単位で1000回bootstrapする条件付き区間で、モデル再fitを含むnested CIではない。
3群しかないので粗く不安定。ラベル置換は各seed内の対の交換を全8通り列挙し、
標準化・学習を毎回やり直す。全体ラベル反転も同じ精度となり、この設計で最小pは0.25。
全192探索比較にはHolm補正も保存する。

PH距離はH0/H1のBottleneckまたはWasserstein-1の和。
Aは8×8配置の積分布、Bは同じ8配置を対応付け、異なるseed同士のrunを比較する。
within/betweenは物理相でなくパラメータ条件内/間。
essential interval数の不一致による無限距離は有限値へ切り詰めず、JSONのnullとinfinite件数で表す。
同じrunを共有する距離対を独立としたCIは作らない。

## 再実行

出力ディレクトリはすべて新規のものを指定する。現在の結果は上書きしない。

```sh
.venv-pw/bin/python -m screening_analysis.train_topology_pilot --output /tmp/new-topo-g
.venv-pw/bin/python -m screening_analysis.lambda_benchmark --output /tmp/new-lambda-bench
.venv-pw/bin/python -m screening_analysis.checkpoint_catalog . --output /tmp/new-inventory.json
.venv-pw/bin/python -m screening_analysis.analyze_lambda_benchmark /tmp/new-lambda-bench --output /tmp/new-lambda-analysis
.venv-pw/bin/python -m screening_analysis.benchmark_symmetry /tmp/new-lambda-bench /tmp/new-lambda-analysis --output /tmp/new-symmetry
.venv-pw/bin/python -m screening_analysis.report_topology_benchmark /tmp/new-topo-g /tmp/new-lambda-analysis --symmetry /tmp/new-symmetry --output /tmp/new-report
```

保存データを含むcheckpointは信頼できるローカル生成物のみ読み込む（NumPy pickle形式）。
解析・測定の乱数、設定、源コードhash、checkpoint hash、局所エネルギー配列を保存。
SVGとCSVは追加の描画ライブラリなしで生成する。
