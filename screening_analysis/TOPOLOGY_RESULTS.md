# Screenformerトポロジー解析：実装・数値検証・対照実験（2026-10-26）

C〜Fの追加実装・物理検証は[追補結果](TOPOLOGY_CDEF_RESULTS.md)を参照。本書はA/B実施時点の記録を保持する。

**PH・巻き付き解析と診断取得の実装、既存checkpointでの検証、幾何交絡の対照実験を実施した。
WC/AHCの頑強な相識別、Chern/Berryとの関係、正則化によるVMC改善は未検証である。**
ローカルには物理的に判定済みのWC/AHC checkpoint・configがなく、この段階を成功と認定できない。

今回得た主な研究上の結果は、**未学習ネットワークのWindowだけでも規則配置と一様配置を識別でき、
完全Screeningでは辺をシャッフルしても高い識別率が残る**ことである。
トポロジー特徴の分類率が高いという理由だけでは、学習されたWC/AHC相識別の証拠にならない。

実装の定義・使い方：[README_topology.md](README_topology.md)。
最終数値：[pilot.json](topology_results_2026-10-26_final/pilot.json)、
検証：[verification.json](topology_results_2026-10-26_final/validation/verification.json)。

## 作業範囲と保護

- 作業開始時のtreeはclean。`develop=038730e`を含む`okada_dev=410ea7a`に、既存の診断API・G2b検証がある。
  `develop`起点の`research/screenformer-topology`を作り、`okada_dev`までfast-forwardした。
- 通常forward、波動関数、Hamiltonian、loss、trainer、JAX/folx微分の実装は変更していない。
  変更は`screening_analysis`の診断保存拡張・解析追加、独立テスト・文書である。
- 当初存在した**521 checkpointをSHA-256で全件照合し、不変**。
  既存Snapshotを更新・再採取していない。旧cacheも上書きしていない。
- 開発途中のpilotが生成した未学習checkpoint・配置を固定し、最終コードでもう一度全条件を評価した。
  開発途中の出力は`topology_results_2026-10-26/`、採用する数値は`topology_results_2026-10-26_final/`。
- 依存パッケージは更新・追加していない。PHはNumPy/SciPy参照実装、図は単体SVG。
  SVG XMLの整合を検査した。利用可能なブラウザ／描画環境がなく、ラスタ化した目視確認は未実施。

## 実装したもの

1. `attn_probe.probe(..., collect_internal=True)`：Q/K/V、正規化後の量、固定Valueの対特徴、
   層入出力、Head出力寄与・Gram相殺・共有bias、既存S/Alpha/Window/W、距離・幅・checkpoint由来を保存。
   同一の正式診断APIと`head_contrib.projected_heads`を使用し、Transformerのforwardを複製していない。
2. `topology.py`：対称重みのflag complex、H0/H1 Diagram、Betti、有限Total/最大Persistence、
   3段Landscape、Bottleneck／Wasserstein。零辺・同値・孤立頂点・essentialを明示処理。
3. `graph_wind.weighted_winding`：実空間最短像を使う重み付きグラフのRank 0/1/2、
   出現閾値、同値イベント、斜交格子・周期表現の不変性。旧幾何CLIは保持。
4. `topology_run.py`：Window／Trim／完全Screening／QKのraw・順位変換・辺シャッフル比較。
   配置・層・Head・checkpointごとのCSV/NPZ、Rank確率曲線、Diagram距離、幾何baseline、SVG。
   phase根拠と各相3group以上がある場合だけ、探索的な学習group holdout分類を実行できる。
5. `topology_pilot.py`と`topology_evaluate.py`：実際の既存checkpoint監査、固定した合成配置の実験、
   PH／巻き付き／density・weightの特徴を分離したseed holdout評価。

## 検証結果

|検査|結果|
|---|---|
|新規PH・巻き付き・runner|10テストPASS|
|追加内部量保存|1テストPASS、cos/smooth7/band/sin/gauss/fixed/nomaskの7条件|
|既存attn_probe/head_contrib/head_cancel|16テストPASS、float32/64・旧解析比較を含む|
|現行G2b float64 Snapshot|**902配列・21,651要素、厳密一致**|
|現行G2b float32、Adam/KFAC込みSnapshot|**1,319配列・31,142要素、厳密一致**|
|既存checkpoint|当初521件すべてhash不変、途中pilotの入力checkpointも不変|

PHは全4頂点グラフ64通りを、独立なF2境界ランク式`beta1=E−rank(d1)−rank(d2)`で照合した。
重み・同値閾値・零辺・孤立頂点・四角形の対角によるH1消滅、LandscapeとDiagram距離も検算。
巻き付きは既知の1方向／2方向ループ、電子置換、周期並進、各電子を別のセルへ移した表現、
斜交格子の総当たり最短像との一致を確認した。最短像が縮退するactive edgeは黙って選ばず拒否する。

内部量保存では、追加保存なしのS/W/Alpha/距離との要素一致、Qbar・KbarからSの再構成、
Head和＋biasとAttention出力の一致、層間hiddenの接続、固定Value、pickle不要の再読込を確認した。
通常forwardの物理量・微分・学習更新は、上記の固定Snapshotで保護した。

最初に古い`cpu_validation/snapshot-v1`／float32版も比較したが、
既にG2bで変更済みの複素局所エネルギーdtypeで失敗した。
これは今回の解析変更による不一致ではなく、現行本体に対応するG2b基準の比較は上記のとおりPASS。
失敗ログも別名で保存し、旧基準を変更していない。

検証ログは[validation](topology_results_2026-10-26_final/validation/verification.json)から同ディレクトリの
`unit.txt`、`internal-probe.txt`、`existing-probes.txt`、`snapshot64.txt`、`snapshot32-kfac.txt`を参照。
これは本体の全既知不具合を解決したという意味ではない。既存の物理検証に記載された問題は保持している。

## 利用可能な学習済みcheckpointでの実験

`tests/g3a_validation/{screen-complex64,screen-default64,screen-complex32,screen-kfac32}`について、
初期・中間・最終の**計12 checkpoint、各8 walker**を評価した。
保存された`effective-config.json`、当時の環境記録、変更されていないconfig constructorを照合し、
未保存runtime値は明示設定として渡した。checkpointから設定を自動復元したとは扱っていない。

これらは自由1電子系である。自己結合除去後のグラフは1頂点・0辺となり、全条件・全時点で
`beta0=1, beta1=0, winding rank=0`、有限Persistence=0、H0 essential=1だった。
初期から最終へのBottleneck／Wasserstein距離も0。
読込・解析・保存の経路が実checkpointで動くことは確認できるが、相識別能力の評価にはならない。

例：[初期・最終を含む数値](topology_results_2026-10-26_final/screen-complex64-000015/metrics.csv)、
[Diagram距離](topology_results_2026-10-26_final/screen-complex64-000015-drift.json)。

## 未学習ネットワークによる幾何交絡の対照実験

条件はN9、三角セル、2層・3Head、hidden幅16、Q/K/V幅4、float64。
初期化seedは11/23/37。窓／Value条件はcos、wide、smooth7、band、sin、gauss、fixedの7つ。
各条件に同じ12個の規則配置（3×3格子＋揺らぎ＋一様並進）と12個の一様乱数配置を与えた。
合計**42条件**。これは`|psi|²`によるMonte Carloでも学習済み状態でもなく、WC/AHCラベルではない。

各重みについてraw、正の辺の順位変換、厳密に同じ重みmultisetのランダム辺配置2本を比較した。
配置・層・Headを独立実験と数えず、各ネットワークseedの平均特徴でnearest-centroidをfitする。
seedを丸ごとholdoutし、標準化もtrain側だけで行った。各窓の6判定は3seedに依存し、
同じ幾何標本を全条件で共用するため、6つの独立な物理実験ではない。
shuffle2本・seed3本なのでp値や頑強性の有意性は主張しない。

### 完全Screeningの分類結果

数値は規則配置／一様配置の正答数、分母6。

|条件|PH raw|PH 順位|PH 辺shuffle|density・重み平均/標準偏差のみ|巻き付きのみ|
|---|---:|---:|---:|---:|---:|
|cos|6/6|5/6|5/6|6/6|3/6|
|wide|5/6|5/6|6/6|5/6|5/6|
|smooth7|6/6|6/6|6/6|6/6|3/6|
|band|6/6|6/6|6/6|5/6|3/6|
|sin|6/6|6/6|6/6|6/6|5/6|
|gauss|6/6|6/6|6/6|5/6|5/6|
|fixed Value|6/6|5/6|6/6|6/6|3/6|

図：[分類結果](topology_results_2026-10-26_final/geometry-feature-evaluation.svg)。
全4重み・特徴・窓・対照の数値と混同行列：
[geometry-feature-evaluation.json](topology_results_2026-10-26_final/geometry-feature-evaluation.json)。

rawのPH特徴のみで比較すると、Windowは全7条件6/6だった一方、Trimは6条件3/6・fixedだけ4/6、
QKは5条件3/6・sin/gaussだけ4/6だった。
この未学習pilotでは、完全Screeningの識別は主に窓の幾何情報で説明できる。
density・重み分布のみでも5〜6/6、辺shuffleのPHでも5〜6/6であり、
**グラフの辺配置に固有のトポロジーが識別に不可欠だという証拠は得られていない。**
順位変換だけではsupportの零数が残るため、交絡が完全に除去されたとは言えない。
これは手法の不成立を証明する結果でもなく、真のWC/AHCで同じ結論になるとは限らない。

閾値曲線の例：
[cos・規則配置](topology_results_2026-10-26_final/synthetic-cos-seed11-ordered/curves.svg)、
[cos・一様配置](topology_results_2026-10-26_final/synthetic-cos-seed11-uniform/curves.svg)。
層・Headごとの値は同ディレクトリの`metrics.csv`／`curves.npz`、
幾何baselineの巻き付き出現距離は`geometry.json`、DiagramとLandscapeは`diagrams.npz`に保存した。
同じseed・窓における2幾何分布のDiagram距離は
[比較例](topology_results_2026-10-26_final/synthetic-cos-seed11-geometry-distances.json)。
別々の配置indexを対応させた記述的距離であり、量子状態間の物理的距離とは解釈しない。

## 研究の問いに対する到達点

|問い|今回の回答|
|---|---|
|WC/AHCでScreeningトポロジーが異なるか|判定できない。ラベル付き研究checkpointが未入手|
|QK/Trim/Window/Headのどこに情報があるか|合成未学習pilotではWindowが支配的。Head寄与・相殺を保存する経路は検証済み。実相では未判定|
|seed/checkpoint/収束/窓に頑強か|3seed×7条件の幾何対照、4自由1電子run×3時点を実行。物理的相識別の頑強性・収束依存は未判定|
|Chern数・Berry位相との関係|未測定。実空間グラフの巻き付きをChern数として扱っていない|
|正則化でVMCが改善するか|正則化の学習組込・比較は未実施。改善を主張しない|

物理的に検証された標的がないまま、PHを増減させる損失を既存VMCへ追加してはいない。
今回の対照では幾何・densityへの依存が強く、そのような損失は窓や零辺を動かすだけで
数値上の目標を達成する可能性がある。物理的な正則化効果とは別に評価する必要がある。

## 続行に必要なデータと判定手順

必要なのは、同一N・格子・Hamiltonian条件で独立にWC/AHCと判定したcheckpointとconfig、
当時のruntime設定、seed、学習時点、エネルギー／分散／標本化の収束記録である。
λやfolder名だけからphaseを割り当てない。twistごとの状態・重なり、または独立の多体Chern判定があれば、
グラフ特徴と物理量の関係を別に照合できる。

まず同条件の両相でWindow/Trim/完全Screening/QKとdensity・重み分布対照を比較し、
独立seed、checkpoint系列、収束精度、窓条件をholdoutする。
同一runのwalker・Head・checkpointがtrainとtestへ漏れない単位で評価し、
ブロック化MC誤差と学習seed間の誤差を別に報告する。
その後に初めて、物理指標を悪化させずseed間の失敗率／変分エネルギー／分散を改善するかを、
正則化あり・なしの同一予算実験で比較する。

再実行：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m screening_analysis.topology_pilot --output NEW_PILOT
# 既にある固定合成入力の診断から再評価する場合
JAX_PLATFORMS=cpu .venv-pw/bin/python -m screening_analysis.topology_pilot \
  --reuse-inputs screening_analysis/topology_results_2026-10-26/inputs --output NEW_REEVALUATION
.venv-pw/bin/python -m screening_analysis.topology_evaluate NEW_REEVALUATION/pilot.json --output NEW_FEATURE_RESULTS.json
```
