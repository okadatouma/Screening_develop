# Screenformer内部トポロジーによるWC/AHC識別：研究報告

2026-10-26。対象は`okadatouma/Screening_develop`、developを起点とする作業ブランチ
`research/screenformer-topology`。既存診断整備を保持して実装した。

**本研究仮説は現時点で未検証である。** 実装、人工系検証、既存checkpoint解析、
小規模VMC、seed分離評価、正則化比較は実行した。一方、独立にWC/AHCと認定された
収束状態がなく、頑強な物理相識別やScreenformerのChern数との対応は確認できていない。
「lambda条件の識別」「人工幾何の識別」「物理相の識別」を区別して以下に報告する。

## 成果物と実行段階

指定保存先は[results/screening_topology](../../results/screening_topology/README.md)。
checkpointは元の場所を維持し、既存測定の主要JSONを
[report-release/measurements](../../results/screening_topology/report-release/measurements/)へ複写した。
新しいA〜E、物理1RDM、統計、ログ、実行設定はこのresults配下に保存している。
既存の`.gitignore`により`results/`とcheckpointはGit追跡対象外。コード・manifest・本報告は追跡可能。

| Stage | 実施内容 | 状態 |
|---|---|---|
| 1 | 人工グラフ、人工波動関数、周期・置換・ゲージ・勾配・CLI検証 | 実行済み、最終51テストPASS |
| 2 | 既存自由N1の12 checkpoint、未学習N9の42条件のA〜F検証 | 実行済み、過去の数値成果物を保全 |
| 2追加 | 実学習N9候補6状態×A/B、各8配置でA〜E、2層stressも8配置 | 今回CLIから実行済み |
| 3 | lambda系35短時間run、3seedの条件比較、実N9候補1RDM | 小規模実行済み、WC/AHC相は未検証 |
| 3拡大 | 独立6seed×2lambda、2000更新の設定・CPU/GPUジョブ | dry-runのみ。学習は未実行 |
| 4 | 自由N4の21run、lambda系24 core runで3種類の正則化 | 既に実行済み。物理相改善は未検証 |
| 5 | 群分割・bootstrap・置換・全比較表・図・保存先統合 | 実行済み |

段階の順序について：1〜11の依頼で実施した正則化実験は、この統合CLI作成より前の探索実験である。
Stage 2・3で有意な最良指標が確立した後の、事前選択された確認実験とは扱わない。
今回、まず単体テストを実行し、保存された候補の再解析、物理推定器の適用、集計へ進めた。
利用可能なJAX deviceは`CpuDevice(id=0)`のみ。GPUでの追加学習は未実行。

## 再利用した実装と変更範囲

`CustomScreenformer`の正式Diagnostic API、`attn_probe`、`graph_wind`、
`head_contrib`、`head_cancel`、既存Network factory、MCMC、Hamiltonian、VMC lossを再利用した。
波動関数の独自forwardや別Hamiltonianを複製していない。
QK、Trim、Window、集約後の重み、Hidden、Head射影・相殺は診断経路で取得する。
固定Valueとcos/smooth7/帯域/sin・Gaussian ringにも既存定義で対応。
通常forwardの戻り値を変えていない。

コードの入口は以下。

| 機能 | ファイル |
|---|---|
| 統合CLI、manifest・provenance・再開 | `topology/cli.py`, `workflow.py`, `train_job.py` |
| PH | `topology/persistence.py`（以前の`topology.py`を内容不変で移動） |
| A/B checkpoint解析 | `topology_run.py`, `graph_wind.py` |
| C/D/E | `multiscale_topology.py`, `directed_loops.py`, `hidden_topology.py`, `topology_cde_run.py` |
| Fと独立検証 | `physical_loops.py`, `berry_validation.py`, `rotation_probe.py`, `topology/physical.py` |
| G | `periodicwave/topology_regularization.py`, trainerの明示オプション |
| 群分割統計・全内部特徴比較 | `phase_metrics.py`, `topology/evaluate.py` |
| checkpoint探索と復元 | `checkpoint_catalog.py`, `checkpoint_model.py` |
| 図・既存結果の集約 | `topology/plots.py`, `report_topology_benchmark.py` |

類似処理を持つ新たな解析フレームワークは作らず、package側は薄い実行・報告層とした。
従来の`from screening_analysis import topology`の関数APIを保っている。
`topology/__init__.py`は同じPH実装を再公開する。

## 定義と不変性

A1=Window、A2=Trim、A3=その積、比較QKは`(S+1)/2`で[0,1]へ明示写像。
`B=(W+W.T)/2`、自己辺を除去。全N頂点を保持し、辺は厳密に`B>t`で追加する。
PHは`u=1-B`のsublevel flag complex、三角形までのF2簡約でH0/H1を計算する。
通常解析では重み0の辺は常に欠落、同値は同時刻、essential intervalは無限死亡として保持する。
Landscape等の有限寿命統計からessentialを除き、個数を別記する。
PH距離でessential個数が違う場合、無限距離を有限値に置換しない。
正の辺のrank化と、重み多重集合・support数を厳密に保つedge shuffleを別対照にした。
rank化だけではsupport密度を一致させない点も明記する。

Bの巻き付きは最短像辺の埋め込みから閉路の整数格子変位を取り、独立方向数0/1/2を求める。
斜交セルでも最近接格子点を完全探索し、縮退最短像が活性辺にある場合は曖昧として除外する。
閾値で同値の辺は同時に追加。出現閾値はその値の直下でrankが現れる上限値。
電子置換、周期並進、各電子への格子ベクトル加算で不変性を検査した。
これは実空間グラフの巻き付きで、多体Chern数ではない。

CはQK cutoff×最短像距離cutoffのBetti/巻き付き曲面。
1変数PHのdiagramとは同一視せず、多変数rank invariantは未実装・未実行。
Dは可縮で非退化な三角形を幾何学的CCWに向けた循環。
非可縮三角形、縮退最短像、面積0を除外し、正逆・周長・epsilon依存を保存する。
QKの共通直交変換、Head置換で不変性を確認したが、一般GL不変性は仮定しない。
実数の非対称QK循環は物理的TR破れや量子化不変量を意味しない。

Eはmean、mean+std、moments、Gram spectrumによる置換不変点群を層ごとに解析。
層間の包含写像がないためzigzag persistenceは未実行。
Hidden PHとScreeningグラフPHは別に集計する。
表現の距離分位点を含む特徴とPHのみの8統計を分離し、単なる距離情報の混入を検査する。

Fの固定物理基底は有限Fourier部分空間をユニタリ変換した局在DVR軌道。
電子トークン番号とは異なる。既存の検証済み推定器
`G_ab = E_|Psi|² sum_i phi_a(x_i) integral dy phi_b(y)* Psi(R_i→y)/Psi(R)`を用いる。
複素共役、同じ電子slotへの置換、全電子和、pseudospinの明示和を保持する。
基底局所位相に対して`Im(G_ab G_bc G_ca)`は不変だが、量子化Chernではない。
平均後のHermitize・trace再規格化・比のclipは行わない。
固定有限部分空間のtraceは射影粒子数で、一般にはNと等しくない。
3因子には異なる独立chainを使うU統計を併記し、同じ推定行列を3回使うバイアスを避ける。
詳しい導出と独立検証は[README_topology_cdef.md](../README_topology_cdef.md)。

## 計算データと標本数

| データ | N・Hamiltonian | seed/run | 配置・規模 | 解釈 |
|---|---|---|---|---|
| 元の実学習checkpoint | N1、正方L4、自由電子、相互作用0 | complex/real、複数checkpoint、独立seed不明 | 12 checkpoint、各8配置 | 動作確認。WC/AHCではない |
| 人工幾何と未学習モデル | N9、規則/一様配置、学習Hamiltonianなし | 11/23/37、7窓/Value条件 | 42条件、各12配置 | 人工分類。相ラベルに転用しない |
| 解析的連続Slater | N2、既知有限Fourier軌道 | 構成乱数19、学習seedなし | 固定25軌道、625二粒子求積配置から非零振幅を使用 | 1RDM検算 |
| 学習N1の物理1RDM | N1自由電子の固定checkpoint | 測定seed1701、8chain | 8192 fresh配置、求積精度細分 | 元checkpointへのF適用 |
| 独立QWZ | 半充填、m=-1,+1,+3 | 厳密対角化、学習seedなし | 2/3/5角セル、6/10角twist格子、Kubo48角 | C=-1,+1,0。Screenformerとは別模型 |
| Gの自由系 | N4、L4、相互作用0 | 3seed×7損失係数条件=21run | 64更新、batch16、測定16chain×32sweep | 1344更新、PH等を直接比較 |
| lambda core | N9、rs15、lambda.33/.72、Delta15 | 3seed×4正則化×2条件=24run | 各24更新、batch16、fresh4chain×16sweep | 物理相は未検証 |
| lambda stress等 | N9/N4、同rs、lambda.33/.72/0 | seed11、窓等8＋基準ansatz2＋lambda0の計11run | 各12更新 | 一因子stressと初期ansatz依存 |
| 今回のA〜E再解析 | 上のoff core6状態、A/B | 3seed、12 view | 各8配置、1層2Head | 96配置入力評価。独立runは6 |
| 2層のHidden再解析 | N9、rs15、lambda.33、2層 | seed11、1状態 | Bの8配置 | 層変化の実データ。相比較ではない |
| 今回の物理1RDM | N9、rs15、lambda.33/.72 | seed11のみ、各4chain | 各16配置、3×3空間×2spin基底、4/6角求積 | ノイズ大。相やChernは未検証 |
| 継続学習テスト | lambda.33、off core seed11 | 同じ元checkpointを2回、独立seedではない | 24→25更新を各1回、新規出力 | 2更新。主比較には含めない |

合計の主要学習は56run・2052更新、今回の継続動作確認を含め2054更新。
探索一覧は1336 checkpoint、読込エラー0、独立に検証済みの物理相ラベル0。
件数には試験用・人工モデル・途中checkpointを含み、独立物理状態数ではない。

Aは各固定波動関数からのfresh MCMC、Bは全モデルと独立な同じ一様座標・spinプール。
同じN/cellのB入力hashが一致することを確認した。
Aの測定乱数はseed群間で独立、同seedのvariant間は対応付ける。
Bの密度・距離・g(r)・S(k)は同じ入力なので全モデルで同一。
Aの短いMCで平衡・収束が保証されたとは扱わない。

## 要求された方法比較表

全行のWC/AHC列は独立物理ラベル不足により「未検証」。
seed頑強性の検討は3群のlambda条件評価であって、物理相頑強性の認定ではない。

| 方法 | WC/AHC識別精度 | seed頑強性 | 収束依存性 | Chern数との相関 | 実装状況 |
|---|---|---|---|---|---|
| QK類似度 | 未検証 | 条件評価3seed、有意差なし | 途中checkpoint解析、収束精度別は未実行 | 未検証 | 診断・グラフ対照・直交不変性、実行済み |
| Windowのみ | 未検証 | raw/rank/shuffle、有意差なし | 小規模窓stress、収束状態は未検証 | 未検証 | A1、窓形状・幅・リング比較、実行済み |
| Trimのみ | 未検証 | Bで5/6例あり、補正p=1 | 指数stress、収束状態は未検証 | 未検証 | A2、実行済み |
| 完全Screening | 未検証 | Aで5/6、Bで3/6、有意差なし | 1/12/24更新の一部、収束未検証 | 未検証 | A3と実集約重みを区別、実行済み |
| Persistent Homology | 未検証 | 条件内/間距離の分離なし | PH距離・checkpoint変化、長時間未実行 | 未検証 | H0/H1、Landscape、距離、実行済み |
| トーラス巻き付き | 未検証 | A/Bとも3/6の例、有意差なし | 小系閾値曲線、収束未検証 | 実空間量であり対応未検証 | 0/1/2rank・最短像・幾何baseline、実行済み |
| Multiscreen距離分解 | 未検証 | A4/6、B3/6、有意差なし | 窓stress、長時間未実行 | 未検証 | 2D曲面・Window統合・冗長性、実行済み |
| 向き付きループ | 未検証 | Screening A5/6、B3/6、有意差なし | epsilon依存・短期のみ | Chern符号と未比較 | CCW・正逆・距離帯・epsilon検査、実行済み |
| Hidden representation | 未検証 | pooling依存、補正p=1 | 実2層・checkpoint、収束未検証 | 未検証 | 層別点群PH、zigzagは未実行 |
| 物理的な複素ループ | 未検証 | N9はseed11のみ、群評価未実行 | MC/求積誤差を測定、学習収束未検証 | 独立QWZのみ検算。実モデルは未検証 | 固定基底1RDM・ゲージ・chain積、実行済み |
| トポロジカル正則化 | 未検証 | E/ばらつきの一貫した改善なし | 64/24更新、長期比較未実行 | 未検証 | 真のPH/スペクトル/平滑化、default off、実行済み |

## 全内部特徴の新しい条件識別比較

[72比較のJSON](../../results/screening_topology/internal-comparison-final/metrics.json)と
[CSV](../../results/screening_topology/internal-comparison-final/metrics.csv)。各モデル8配置、
3seed群、同じ6状態、leave-one-seed-out、訓練foldだけで標準化する最近傍重心、調整なし。
表のQK/Window/Trim/Screeningは同じPH・Betti・巻き付き統計でグラフ入力だけを変えた比較。

| 特徴 | A自然分布 BA | B共通入力 BA |
|---|---:|---:|
| QKグラフ | .1667 | .5000 |
| Windowグラフ | .6667 | .5000 |
| Trimグラフ | .1667 | .8333 |
| 完全Screeningグラフ | .8333 | .5000 |
| Screening PHのみ | .8333 | .5000 |
| 巻き付き確率のみ | .5000 | .5000 |
| QK×距離Multiscreen | .6667 | .5000 |
| Window統合 | .6667 | .5000 |
| QK向き付き循環 | .3333 | .5000 |
| Screening向き付き循環 | .8333 | .5000 |
| Hidden PHのみ：mean | .3333 | .6667 |
| Hidden PHのみ：mean+std | .5000 | .5000 |
| Hidden PHのみ：moments | .8333 | .5000 |
| Hidden PHのみ：Gram spectrum | .6667 | .5000 |

全比較でHolm補正p=1。raw/rank/shuffle、Hidden距離のみ・PH+距離の混合特徴も省略せず保存した。
BでTrimが5/6だった新しい例も含め、探索後に「成功した特徴」を確定的に選択していない。
各BAは6判断なので1件で.1667変わる。配置のbootstrapを独立seedの信頼区間には使っていない。
群単位1000 bootstrapは固定OOFモデルに条件付ける粗いCI。
3つのbalanced paired seedでは置換が8通りしかなく最小p=.25。
F1、AUROC、MCC、混同行列、fold番号、null分布もJSONに保存した。

前回の16配置・平均Head completed PHではA BA=.8333/AUROC1、B BA=.5/AUROC.6667。
[前回G結果](../TOPOLOGY_G_RESULTS.md)と今回の特徴定義・配置数は異なるので、表を取り混ぜない。
同lambda/異lambdaのPH距離はA .152118/.150542、B .146973/.147257。
seed間差より条件間差が大きいとは確認できない。
従来baselineとの比較・小摂動・8/16/32配置も前回の全192比較に保存、Holm補正はすべて1。
72追加比較と192前回比較は別々に補正した探索群で、通算の事前登録確認検定ではない。

## 正則化の実測と失敗条件

既定off、gamma0では元loss callableそのもの。物理Eと正則化込みobjectiveをログで分離する。
真のPHはepsilon完成グラフのLandscapeの近傍一致。critical edgeを通る区分的微分で、
離散ペアリングはstop gradient。タイ・切替点の古典微分は保証しない。
スペクトルはLaplacianの1〜4次trace moment、平滑化は重みMSE。三者を同一損失と呼ばない。
score項を含む配置分布の勾配も独立2状態例で検査した。能動PHはN≤36、KFAC有効化は未検証のため拒否。
実学習はAdamのみ、GPU・多device・LAMB能動学習は未実行。

自由N4の3seed平均：off E=3.77128、seed SD=.08093、局所分散1.67956。
PH gamma.1は3.77479/.08655/1.60945、gamma1は3.82336/.11299/1.31362。
分散が減る一方で平均Eとseedばらつきは悪化した。
一様共通入力でのPH安定性もoff .010485→PH gamma1 .014733で悪化。
スペクトルgamma1は同指標.009360だがE=3.81914。重みgamma1は自身の平滑度を改善するがE改善は一貫しない。
三損失の尺度は違うため、同じgammaの単純比較から最適手法を決めない。

lambda系gamma.1の差は概ね1e-4で、数千〜1万超の局所分散に比べ非常に小さい。
24更新のモデルを収束状態と呼べない。初期ansatzを変えた12更新対照ではEが大幅に変わるため、
初期状態や波動関数構成の違いを内部トポロジーが識別している可能性を排除できていない。
実測の全2052更新にNaN/非有限値はなく、今回の継続2更新も有限。
未発生という観測は大規模学習のNaN率0を保証しない。

## 物理ループとChern：独立検証と実候補を分離

解析的N2の固定基底1RDMは最大誤差3.89e-16、trace2。
既存N1の実checkpointにも8192 fresh配置で適用した。
独立QWZ模型では実際のtwist Hamiltonian、Slater/Fock重なり、独立Kubo積分、
格子細分、U(1)/占有空間ゲージ検査からC=-1,+1,0を確認した。
このChern値はlambda候補の値ではない。

今回N9の実候補へ適用した1RDM（18基底、16配置、4chain）の結果：

| lambda | 求積辺数 | Re trace(G) | Hermiticity残差 Frobenius | 行列SEM Frobenius | 独立chainループRMS | 平均ループjackknife SEM |
|---|---:|---:|---:|---:|---:|---:|
| .33 | 4 | 8.28829 | 3.44704 | 2.26534 | .0026281 | .0048263 |
| .33 | 6 | 8.25937 | 3.42696 | 2.29668 | .0026030 | .0045967 |
| .72 | 4 | 8.13505 | 3.93128 | 2.67751 | .0037109 | .0070347 |
| .72 | 6 | 8.14439 | 3.83991 | 2.64857 | .0033328 | .0068546 |

各144幾何三角形。求積による変化に加えMC誤差が大きい。traceを9へ強制していない。
loop平均もSEMより小さく、差の認定には使えない。
この粗い3×3基底では保持された空間三角形の距離スケールが限られ、
今回のWindow重み付き値はHead間でほぼ一致した。距離分解の成功例とは扱わない。
Window重みは固定基底中心間で再評価し、電子トークンのWを物理軌道へそのまま移していない。

38状態の零twist C3も候補固有状態残差1.077〜3.492で小さくなく、固有値へ丸められない。
EwaldのC3整合誤差最大7.39e-13はHamiltonian側の検算で、状態の相認定ではない。
Screenformerでの正しいtwist-Hamiltonian族と収束状態追跡は未実行。
したがって内部特徴と物理Chern数の相関係数や符号対応は**未検証**。

## 図とデータ由来

[図一覧](../../results/screening_topology/report-release/FIGURES.md)。主要な要求図を以下へ対応付ける。

| 図 | 保存先（results/screening_topology配下） | 由来と限界 |
|---|---|---|
| 1 結合グラフ | `report-release/01-candidate-graphs.svg` | 実候補2条件、共通配置。WC/AHC認定なし |
| 2 Window/Trim/Screening PD | `report-release/02-components-PD.svg` | 同配置・層・Head、QK対照とessential表示 |
| 3 rankと閾値 | `report-release/03-winding.svg` | 実候補、4入力グラフ・幾何対照数値 |
| 4 二次元曲面 | `report-release/04-multiscale.svg` | 実候補。1D persistenceではない |
| 5 向き付き循環 | `report-release/05-loop-comparison.svg` | 2実候補の距離帯RMS。空bin除外 |
| 6 Hidden層変化 | `report-release/06-hidden-layers.svg` | 左：実2層stress、右：人工未学習、分離表示 |
| 7 checkpoint変化 | `report-release/07-checkpoints.svg` | 実候補1/12/24更新の一部、PH特徴ノルム |
| 8 seedばらつき | `report-release/08-seeds.svg` | 実候補3seedのfresh E |
| 9 識別性能 | `report-release/09-condition-classification.svg` | lambda条件のみ。WC/AHC図は未検証で作らない |
| 10 物理loop/Chern | `report-release/10-independent-physical-loops.svg` | 独立QWZのみ。実候補Chern相関図は未実行 |
| 10補助 実1RDM | `physical-candidates/actual-projected-trace.svg` | 2実候補の有限求積・射影trace |
| 11 収束曲線 | `report-release/11-regularized-training.svg` | 自由N4の7条件、エネルギー/分散 |

別途、全内部特徴の比較図は`internal-comparison-final/methods.svg`、
人工配置のHidden対照は`report-release/synthetic-hidden-controls.svg`。
実データ、人工配置、解析波動関数、独立模型を図題・凡例・indexで分けた。
SVGのXMLを検査したが、raster化した目視レイアウト検査は未実施。

## 保全・失敗・再現性の監査

[最終検証](../../results/screening_topology/validation.json)、
[smokeログ](../../results/screening_topology/smoke-release/tests.txt)、
[探索一覧](../../results/screening_topology/scan/inventory.json)。
既存563 checkpoint、前回までの全1334 checkpointのhashが不変。
新規継続2ファイルを加えた1336件は読込可能。旧checkpointのoptimizer・配置にも書き込みなし。
既存のテストは変更してPASSさせていない。最終smokeは従来45＋CLI6の51件。
これは全リポジトリのテストがPASSしたという意味ではない。既存[tests/RESULTS.md](../../tests/RESULTS.md)は
元33メソッドについて30 PASS / 1 FAIL / 2 ERROR（KFAC別実行を含む）を記録しており、
K10・定数folx・Psiformer点微分・倍精度KFAC・Envelope不整合等を既知事項としている。
その既存全suiteは今回は再実行しておらず、今回の新規変更による失敗と混同しない。
既存probe6、float64 snapshot902配列/21651要素、float32+KFAC1319配列/31142要素の
前回通過ログも残し、通常計算を変える変更は今回はしていない。

継続ジョブの初回は、runtime JSONをJAX配列へ変換した後に保存し、
シリアライズできず停止した。数値更新前に失敗した。JSONを変換前に保存するよう修正し、
元checkpointを変更せず、別の新規出力先で継続学習に成功した。失敗出力とログも保全。
監査の初回は、まだ生成していない監査JSONへの自己参照リンクを先に検査して停止した。
生成予定の自己参照のみを先行検査から除き、出力生成後に存在を確認する順序に修正した。
既知の既存SyntaxWarning（図用文字列のescape）は今回の数値失敗とは区別する。

各新コマンドの`provenance.json`にGit commit/branch、dirty状態、tracked diff hash、
未追跡を含む全Python源のhash、Python/NumPy/SciPy/JAX/folx/Optax/KFAC等の版、
乱数seed、PW/JAX環境設定を記録した。manifestにN、rs、lambda、層/窓等の実設定を保存する。
過去の成果物は当時のsource/checkpoint hash・設定・乱数情報を保持し、収集時環境を別記した。
元の古いcheckpointの学習seed等が欠落している場合はnullのままで、現在の情報を過去の実行時情報と偽装しない。
過去実行時の完全なdirty-tree/Git環境を遡及復元することは未完了。

## 8つの研究上の問いへの回答

1. **最も正確にWC/AHCを識別した内部指標は何か。** 未検証なので決められない。
   lambda条件では複数の指標が探索的に5/6になったが、全て有意でなく、配置分布・特徴定義で変わる。
2. **seedや収束精度に最も頑強な指標は何か。** 確認できていない。
   入力摂動に対するPHの数学的安定性を、異なる学習seedや未収束状態への頑強性に読み替えない。
3. **通常の密度・構造因子・距離分布以上の情報を使えているか。** 実証できていない。
   人工幾何では距離のみで6/6の条件があり、内部特徴の成功例の説明になり得る。
   BでTrimが5/6だったことは追加情報の候補だが、3seed・多重比較・未検証ラベルでは証拠不足。
4. **Screening特有の構造を利用する意味はあったか。** 診断上はWindow/Trimの役割と
   幾何依存・零辺・Head冗長性を分離できた。物理相分類の追加価値は未確認。
5. **内部トポロジーと物理Chern数の対応は見つかったか。** 見つかっていない。
   実モデルのChern数を信頼して計算できておらず、対応の検定自体が未実行。
6. **不足は位相情報か、学習・統計か。** 分離不能。明確に不足しているのは学習収束、
   独立物理ラベル、seed数、物理1RDMのMC精度。実数QK/Windowが必要な複素位相を保持する保証もないが、
   今回の否定的結果だけから情報論的に不可能だとは結論しない。
7. **論文化の可能性が高い方向は何か。** 現段階では、検証可能な固定基底複素ループと
   距離スケール分解を独立Chern計算と接続する方向が有望。
   内部PHだけでChernを読むという主張より、独立に認定した状態での追加情報の有無を
   密度・距離整合対照とともに検証する設計に研究価値がある。現結果だけの相識別成功論文は支持できない。
8. **次に優先する実験は何か。** 同一Hamiltonianで独立認定したWC/AHC競合ansatzを用意し、
   エネルギー・分散・C3等で収束を確認する。続いて6以上の独立seedで共通配置/自然配置を分け、
   group holdoutと事前固定特徴で追加情報を検定する。物理1RDMはchain長・求積・基底を細分し、
   正しいtwist-Hamiltonian族で状態を追跡してChernを検算する。
   その後に有望指標を選び、gammaを訓練群内で選択した正則化の確認実験へ進む。

## 再現コマンドと継続方法

リポジトリrootで実行。下記の新規パスは存在しないものを使う。
標準の4コマンドは実際に上記保存先で実行済み。

```sh
PY=.venv-pw/bin/python
$PY -m screening_analysis.topology.cli scan --output /tmp/new-topology-scan
$PY -m screening_analysis.topology.cli smoke --output /tmp/new-topology-smoke
$PY -m screening_analysis.topology.cli benchmark \
  --manifest screening_analysis/topology/experiments.yaml --output /tmp/new-topology-benchmark
# 完了済みtaskは入力hashと数値コードhashを照合してskip
$PY -m screening_analysis.topology.cli benchmark \
  --manifest screening_analysis/topology/experiments.yaml --output /tmp/new-topology-benchmark --resume
$PY -m screening_analysis.topology.cli report \
  --input results/screening_topology --output /tmp/new-topology-report
```

`report`はこの研究の実測archiveを統合するコマンドで、任意形式の未解析checkpointだけから図を作る汎用処理ではない。
`--input`には`benchmark`、`layer-stress`、`physical-candidates`を含む今回と同じ構造を渡す。
新しい学習checkpointを解析するには既存`attn_probe --collect-internal`でprobeを保存し、
manifestの`runs[].probe`をそのファイルへ変更する。パスはmanifest基準。
物理ラベルを付ける場合は`phase_verified=true`と独立`phase_evidence`が必要。
このフラグは独立診断を自動計算したという意味ではなく、利用者が提供する証拠を記録する契約である。

```sh
# 大規模設定：planのみ。CPU/GPU、seed、N、反復、PW環境はtraining.yamlで編集
$PY -m screening_analysis.topology.cli train --manifest screening_analysis/topology/training.yaml \
  --output /tmp/new-scaleup-plan --dry-run
# 実際の追加学習（今回未実行）
sh screening_analysis/topology/jobs/run.sh screening_analysis/topology/training.yaml /tmp/new-scaleup
# 物理1RDMの小規模計算（今回実行済みと同じ設定）
$PY -m screening_analysis.topology.physical --output /tmp/new-candidate-1rdm --samples 16
# 全内部特徴の統計
$PY -m screening_analysis.topology.evaluate --input results/screening_topology/benchmark --output /tmp/new-internal-metrics
```

YAMLファイルは追加依存なしで読むためJSON構文（有効なYAML1.2）で記載。
通常のYAML記法を使う場合のみPyYAMLを追加する。
学習jobsは途中失敗フォルダを自動で上書きしない。
`restore`へ元run（`runtime_config.json`とcheckpointを含む）のパスを指定し、
**新しいid・出力先**で総更新数`iterations`を増やす。`resume_smoke.yaml`は実行済みの1更新継続例。
`--resume`は完了ジョブの設定一致を確認してskipする機能。
元trainerはPRNGをcheckpointに保存しないため、継続はパラメータ/optimizer/walkerの統計的継続であり、
ビット単位に同じ乱数軌跡を再開する機能ではない。device数・batch数の互換性も既存restoreが検査する。
