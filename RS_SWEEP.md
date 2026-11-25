# λ系列ごとにrsを変えた学習・内部構造解析

本番ジョブは自動投入しない。準備コマンドが計画と共通入力を保存し、投入コマンドを表示する。

## 設定と実行

`rs_sweep.example.json` を編集する。既定はN=36、λ=0（液体候補）、0.33（WC候補）、0.72（AHC候補）、rs=5,10,15,20,25,30、seed=101,202,303、8万更新で54ケース。
λ・rs・seed・学習率・ヘッド数・層数・解析サンプル数をJSONで変更できる。候補名は初期条件の系列名であり、学習後の相ではない。

```bash
cd /home/tomax/projects/Screening_topology_gpu
PYTHON=/home/tomax/venvs/screening-gpu/bin/python
"$PYTHON" -m screening_analysis.rs_sweep prepare \
  --config rs_sweep.example.json --output results/rs-sweep-20261031

# 実行内容だけを確認する。学習・HF・解析は実行しない。
"$PYTHON" -m screening_analysis.rs_sweep run \
  results/rs-sweep-20261031/plan.json --index 0 --dry-run
```

準備済み計画：`results/rs-sweep-20261031/plan.json`。ケース範囲は0〜53。
次の投入は利用者が行う。`rs_sweep.job` のpartition・メモリ・時間等は使用クラスタに合わせる。

```bash
sbatch --array=0-53%2 \
  --export=ALL,SWEEP_PLAN=/home/tomax/projects/Screening_topology_gpu/results/rs-sweep-20261031/plan.json \
  rs_sweep.job
```

各タスクはHF/平面波の準備→既存VMC学習→6項目の解析を順に行う。出力先は新規ディレクトリで、既存runへ暗黙に再開・上書きしない。
タスクはplanのrepository情報を参照して作業ディレクトリを決めるため、Slurmがスクリプトをspoolしてもリポジトリを見失わない。

## rsと初期状態

- ハミルトニアンのCoulomb係数はrs、Δ=rs、Jastrowカスプもそのrsから再計算する。λを固定しrsのみを変えるときのコード既定規約。
- WC候補は各rsでHFを自己無撞着に解き直す。
- AHC候補は指定した参照表の係数を初期値とし、各rsでHFを解き直す。参照表のhashとk/G基底を照合する。既定参照はN36・λ0.72・gcut2.01の既存表。
- HFが収束しない場合は`reference/failed.json`を保存し、そのケースを停止する。rsを変えてHFのChern診断が変わっても元の相へ強制しない。HF診断と学習後の相判定は区別する。
- 液体候補はスーパーセルの低運動量平面波を用いた下バンド参照。自己無撞着HF解とは称しない。`PW_INIT_LOCALIZE=0`でwalkerを結晶点へ強制配置しない。HF＋NN補正＋Jastrowという波動関数の構成は他系列と揃える。
- 平面波の有限Nで開殻の同順位があり得る。初期状態を液体相の証明にしない。
- AHCの別λ・別N・別gcutを使う場合は、その基底に対応する参照表を明示する。

## nativeとcommon

各チェックポイントの保存walkerをnativeとして解析する。さらに、計画作成時に一度だけ作った共通入力を全rs・λ・seedで使用する。
既定の共通入力は一様な位置と、全+擬スピン／ランダム擬スピンを交互に含む固定データ。全runで位置・擬スピン・atoms・charges・順番が同じ。
これは分布を固定したモデル比較用であり、各波動関数の平衡サンプルではない。

既存の共通入力を使う場合：

```bash
"$PYTHON" -m screening_analysis.rs_sweep prepare --config rs_sweep.example.json \
  --reference-inputs results/manual/array-2475/connection-audit-validated/shared-inputs.npz \
  --output results/rs-sweep-from-reference
```

必要キーは `positions [B,2N]`, `spins [B,N]`, `atoms [B,A,2]`, `charges [B,A]`, `lattice [2,2]`。
`analysis.common_samples`までを共通で使用する。相ごとの入力を混ぜる場合は元データの順序・構成にも注意する。準備後に共通入力や参照表が変わればhash照合で停止する。

## 6項目の定義と保存先

各ケースの `analysis/native/` と `analysis/common/` に保存する。

1. **PH**：QK、Alpha（ファイル内ではtrim）、Window、Alpha×Windowのflag PH。birth/death図・landscape・Betti曲線・順位変換・shuffle対照を既存PH実装で算出。`ph/diagrams.npz`, `ph/curves.npz`, `ph/summary.json`。`density-betti.csv`は正の上位k本で接続数を揃えた比較。ゼロ辺は追加せず、同順位を分割する境界を記録する。
2. **Window幅**：`analysis/window-curves.csv`。本体の窓関数を401点で評価。幅・半値距離とコード単位、rs a_B換算を保存。既定は単調smooth7窓で、帯窓・輪切り窓はこの掃引では未対応として停止する。
3. **Alpha距離依存**：`weight-distance.csv`。距離をdrefで揃えた24ビン、各層・ヘッド・擬スピン同/異別に平均と10–90分位。距離は本体の周期ノルム。rs a_B換算も保存。
4. **Attention固有モード**：実際の有向集約重みWを対象に、有向スペクトル（複素数）、非対角の対称化行列B=(W+Wᵀ)/2とD^(-1/2)BD^(-1/2)の固有値・固有ベクトルを保存。参加数1/Σv_i⁴、擬スピンとの重なり、固有値間隔・縮退フラグも記録。`modes.csv`, `attention-modes.npz`。粒子のラベルが違う配置間で固有ベクトルを直接平均しない。縮退時の基底・各ベクトルの参加数は任意性がある。
5. **接続エントロピー**：q_ij=W_ij/Σ_jW_ij、H_i=−Σ_j q_ij log q_ij。一次指標は自己項を除き、H/log(N−1)、有効隣接数exp(H)、ゼロ行率を保存。重みへ新たにsoftmaxをかけない。ゼロ行はエントロピー未定義として区別する。自己項を含む値も別に保存。`entropy.csv`, `row-entropies.npz`。
6. **波動関数の位相感度**：各層・ヘッド・粒子ペアの実際の重みを両方向ともexp(ε)倍し、∂φ/∂εと∂log|ψ|/∂εを自動微分。HF・Jastrowを含む元の波動関数を使う。全非対角／同擬スピン／異擬スピンの接続を10%弱めた応答も保存。`connection-gradients.npz`, `phase-responses.csv`, `finite-difference.csv`。位相APIは既に実数の角度を返す。恒等介入、中央差分1e−3/5e−4で検証し、失敗・非有限の入力はsummaryに明示する。

PHの穴・Attention固有値・接続エントロピーはネットワーク内部の診断で、物理的なChern数ではない。擬スピンに基づく二部グラフ構造を物理的トポロジーと混同しない。

## 集計・作図

作図にはMatplotlibが必要。既存の描画用環境を使用できる。

```bash
env MPLCONFIGDIR=/tmp/rs-sweep-mpl /home/tomax/venvs/pyrite-bands/bin/python \
  -m screening_analysis.rs_sweep_report results/rs-sweep-20261031/plan.json \
  --output results/rs-sweep-20261031-report
```

未完ケースを`status.json`に列挙し、完了分を集計する。

- `observables.csv`：ケース・seed・native/common別の幅、Alpha、固有値、エントロピー、中心化位相応答、固定密度のβ1。
- `native-rs-trends.png/.svg`, `common-rs-trends.png/.svg`：6指標のrs依存。seed別の線を保持し、headを跨ぐ平均・幅のmin/maxなどラベル置換に強い集約を用いる。
- ケース別 `native/common-ph-diagrams.png`：H0/H1のパーシステンス図。無限deathは上端の三角記号で区別し、有限寿命へ変換しない。
- ケース別 `native/common-radial.png`：Alphaの擬スピン別距離曲線・Window曲線。
- ケース別 `native/common-mode-snapshot.png`：最初の配置・第0層・最大4ヘッドについて、最大・第2・最小固有値のモードを粒子座標上に表示。符号は任意、縮退時は基底も任意。全層・ヘッドの数値データはnpzに保存。二部グラフの特徴を見るため、rs依存図には正規化行列の最小固有値も使う。

固定密度のβ1はp≈0.1で全配置・全ヘッドに正の辺が十分あり、同順位を分割しない場合だけ表示する。部分的な配置だけで平均して欠測を隠さない。
位相の中心化応答は入力間で共通のグローバル位相を除く診断。エネルギー変化や平衡分散ではない。幅のrs a_B換算とコード単位はCSVで別列。

## 失敗後の解析再実行

学習が済んでいれば再学習せず、新しい解析先を指定できる。

```bash
"$PYTHON" -m screening_analysis.rs_sweep analyze results/rs-sweep-20261031/plan.json \
  --index 0 --output results/rs-sweep-20261031/CASE_ID/analysis-recomputed
```

全ケースで同じ解析先名を使った場合、集計側に `--analysis-dir analysis-recomputed` を指定する。

## 検証と制約

単体テストは`tests.test_rs_sweep`。計画の重複防止、入力hash、平面波周期性、エントロピー、粒子置換不変性、擬スピン非破壊、複数層の位相勾配の既知解を検証する。
小規模CPUのN4・2更新・rs2点で準備→学習→native/common解析を確認し、作図も検証する。本番VMCやSlurm投入は行わない。
rs15・N36でWC/AHCのHF再作成も確認する。

nativeも保存walkerからの診断で、混合・ESSの認証はない。共通入力には分布外の配置もあり得る。相識別には別途、エネルギー・構造因子・多体位相などの物理量が必要。

## A・B・C：接続距離、波動関数感度、エネルギー

入口は `screening_analysis/connection_study.py` の1ファイル。主要関数は
`analyze_connections`（A）、`analyze_response`（BとCの勾配）、
`sample_state`（位置・擬スピンの追加サンプリング／再学習）、
`scan_window`（幅ごとの比較）、`main`（実行・保存・作図）の5つ。
既存のチェックポイント読込、接続probe、個別辺感度、MCMC、局所エネルギー、VMC損失を再利用する。

今回の3ジョブが学習と既存解析を終えた後、プロジェクト直下で以下を実行する。
ジョブ投入や既存学習の変更は行わない。

```bash
cd /home/tomax/projects/Screening_topology_gpu
PYTHON=/home/tomax/venvs/screening-gpu/bin/python
"$PYTHON" -m screening_analysis.connection_study \
  --plan results/rs-sweep-lambda072-three-points-20261031/plan.json \
  --mode energy
```

既定の保存先は `results/manual/rs-sweep-lambda072-three-points-20261031/CASE_ID/connection-study/`。
`--index 0` を加えると1ケースだけ実行できる。
GPUで実行する場合も、既存のジョブを停止・変更する必要はない。別プロセスの実行に必要な資源を確保する。

| モード | 実施内容 |
| --- | --- |
| `diagnostic` | 保存walker・共通配置でA/B。物理的な状態計量・エネルギーは計算しない |
| `energy` | 上記に追加MCMCを加え、距離帯の状態計量とエネルギー勾配を計算 |
| `scan` | energyに加え、各ヘッドのWindow幅を変更し、その状態から追加サンプリングして評価 |
| `retrain` | scanに加え、すべてのWindow幅を固定し他のパラメータをAdamで再学習して評価 |

幅変更の倍率は `--factors 0.7 1 1.3`（既定）。1ヘッドずつ変更し、他の幅は元のまま。
再学習は各幅で同じ元チェックポイントから始める。`--retrain-steps 1000 --learning-rate 0.0001`
が既定で、これは有限回の最適化結果であり厳密な最小エネルギーではない。
再学習後のパラメータ、構成、学習履歴、評価配置も保存する。
モードを変えて再実行するときは `--output results/manual/NEW_DIRECTORY` など新しい出力先を指定する。
planでの `--output` にはケースごとのサブディレクトリを作る。既存出力を上書きしない。

距離帯介入は `attention.py` の `_radial_epsilon` フックを使う。
各距離帯で `C_ij → C_ij exp(epsilon * band(d_ij))` とし、両方向へ同じ倍率を掛ける。
bandは中心1、帯の外0の滑らかな関数 `(max(1-u²,0))⁴`。自己接続は変更しない。
ネットワーク内部で距離を再計算するので、運動エネルギーの座標微分にも介入が含まれる。
既存チェックポイント・学習ではフックが存在しないため従来の計算経路になる。

出力は以下。

- `CONDITIONS.md`, `conditions.json`：λ、rs、電子数、seed、初期化、チェックポイントhash、解析・サンプリング条件。
- `A-connections.csv`：ヘッド／擬スピン別の平均重みと総量。個別辺の相対摂動感度も別列に記載。
- `A-distances.csv`：重み付き平均距離と90%重み半径。network周期距離と最短ユークリッド距離を区別。`*_aB` はコード距離×rs。
- `B-response.csv`：距離帯全体の振幅・位相感度。個別辺の絶対感度平均とは異なる量。
- `C-energy.csv`：追加MCMCから求めた距離帯のエネルギー勾配と時間ブロックSEM。
- `C-window.csv`, `window-energy.png`：幅変更／再学習後の全系エネルギーと誤差（scan/retrainのみ）。
- `*-measurements.npz`：配置・擬スピン、距離帯の複素感度、局所エネルギー、有限差分検証。
- `overview.png`, `summary.md`, `summary.json`：A/B/C比較図と読み方。

エネルギー単位は元のハミルトニアンの**全系のコード単位**。
初期診断は8配置、追加MCMCは4 chains × 16 blocks × 4配置/chain/blockが既定。
保存walkerからburn-inして位置と擬スピンを更新するが、平衡化を自動認証はしない。
`--burn`, `--blocks`, `--block-size`, `--steps`, `--chains` を変えて安定性を確認する。
時間ブロックSEMもブロック長が十分な場合の暫定推定であり、独立サンプル数ではない。
規格化・全体位相を除いた状態計量とエネルギー勾配は、追加MCMCの行だけに記載する。

作図は既定で `/home/tomax/venvs/pyrite-bands/bin/python` を使う。
別環境なら `--plot-python PATH`。既存CSVからの作図だけなら、Matplotlib環境で
`python -m screening_analysis.connection_study --plot-only OUTPUT_DIRECTORY` を実行する。

## WindowをGaussian・Algebraic・Hybrid・Fullへ切り替える

設定例は `window_variants.example.json`。λ=0.72、1層・2ヘッド、N=36の3条件
（液体候補rs=4、AHC候補rs=12、WC候補rs=25）を含み、窓の初期設定はGaussian。
`training_environment` の以下の値を変更する。各ヘッドの長さ尺度ξは個別に学習する。

| `PW_WIN_SHAPE` | Window m(d) | 設定する値 |
| --- | --- | --- |
| `gaussian` | exp(-(d/ξ)²) | ξのみ |
| `algebraic` | 1 / (1+(d/ξ)^p) | `PW_WIN_POWER`：2・4・6…の固定偶整数。既定4 |
| `hybrid` | (1-η) exp(-(d/ξ)²) + η | `PW_WIN_ETA`：η初期値、既定0.1 |
| `full` | 1 | ξは使わない |
| `cos` / `smooth7` | 従来の有限範囲の窓 | 後方互換のため維持 |

Hybridのηはヘッドごとに `sigmoid(mask_eta_logit)` として学習し、0～1に制限する。
`PW_FREEZE_ETA=1` にすると `PW_WIN_ETA` の値へ固定でき、この場合はη=0、1も指定可能。
学習するηの初期値は0より大きく1未満にする。η=0固定はGaussian、η=1固定はFullと同じ窓になる。
`PW_FREEZE_W=1` は従来どおりξだけを固定し、ηの固定は別に指定する。

Algebraicのpは学習せず、実験条件として変更する。原点での座標微分・ラプラシアンを安定に扱うため、
2以上の偶数に限定し、内部では距離の二乗の整数累乗として計算する。

**FullはWindowを外す比較**であり、既存のQK→Trim→Value集約を維持する。
Trimで0になる辺は残るので、「全ペアの接続重みが必ず非ゼロ」やsoftmax Attentionへの変更ではない。
Fullでも保存形式を揃えるため `mask_omega` は残すが、前向き計算に使わず勾配は0。
幅の解析値はNaN（意味のある幅なし）として保存する。

例：Hybridを試す設定部分（既存JSONの残りの項目はそのまま）。

```json
"PW_WIN_SHAPE": "hybrid",
"PW_WIN_POWER": "4",
"PW_WIN_ETA": "0.1",
"PW_FREEZE_ETA": "0"
```

編集したJSONから、既存と同じ方法で計画を作る。

```bash
/home/tomax/venvs/screening-gpu/bin/python -m screening_analysis.rs_sweep prepare \
  --config window_variants.example.json \
  --output results/window-gaussian-20261101
```

`prepare` はジョブを投入せず、対応するsbatchコマンドを表示する。
形・p・η初期値・η固定設定をruntime_config.jsonへ保存し、解析時は保存値から復元する。
従来のcos/smooth7チェックポイントには影響しない。

ξは全方式に共通の「半値半径」ではない点に注意する。
Gaussianの半値半径はξ√log2、Algebraicはξ、従来のcos/smooth7は幅wの1/2。
Hybridの絶対重み1/2の半径はη<1/2のときだけ有限で、η≥1/2では存在しない。
`window-curves.csv` に実際の `half_weight_radius`、床から半分まで減衰する `half_decay_radius`、
`tail_floor`、`cutoff_radius`、η、p、形を保存する。
Gaussian・Algebraic・Hybridは有限距離で強制的に0へ切らない。
同じ初期半値半径を比較したい場合は、この換算に合わせて `PW_W_LO` / `PW_W_HI` のξ範囲を設定する。

A/B/C解析は新しい窓にも対応する。Fullには幅の自由度がないため、
`connection_study --mode scan/retrain` は使わず `diagnostic/energy` で評価する。
Hybridの幅を固定した再学習では、ηを固定しない限りηも他のパラメータとともに再学習する。
