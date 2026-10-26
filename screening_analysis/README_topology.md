# ScreeningグラフのPHとトーラス巻き付き

通常のforward、VMC、Hamiltonian、JAX/folx微分、既存checkpoint形式を変更せず、
正式Diagnostic APIから内部量を再評価する。独立JIT診断の配列はProduction内部の
ビット列そのものではない。研究結果と実行記録は[TOPOLOGY_RESULTS.md](TOPOLOGY_RESULTS.md)。

## 内部量の取得

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m screening_analysis.attn_probe RUN \
  --settings known-settings.json --collect-internal --output NEW-probe.npz
```

旧configの格子が文字列の場合は、既存probeと同様に元精度の`--lattice '[[...],[...]]'`
が必要。保存されていないTrim上限、指数、窓形状、FREEZE、clip、JAX精度を推測しない。
`--config-path`で外部の構造化configを明示でき、`effective-config.json`の`config`ラッパーも読む。
保存先は新規ファイル限定。checkpointとoptimizer stateは書き込まない。

既存のS/Alpha/Window/W_window/W/d/positions/spins/r/wに加えて、オプションで以下を保存する。

|量|shape・意味|
|---|---|
|q,k,q_normalized,k_normalized|[B,L,N,H,K]、正式APIの射影・eps付き長さ正規化|
|v,v_normalized,v_effective|[B,L,N,H,D]。固定Value時は存在せず、pair_features [B,L,N,N,P]を保存|
|hidden_input,hidden_output|[B,L,N,E]、各層の入出力|
|head_vectors|[B,L,N,H,E]、既存projected_heads関数によるexp(out_scale) z_h W_h|
|head_bias|[B,L,E]、共有biasをHeadと分離|
|head_gram|[B,L,H,H]、電子平均のc_h・c_k、負の交差項が相殺|
|head_cancellation_ratio|[B,L]、||sum c||²の電子平均／sum||c||²の電子平均、分母0はNaN|

metadataにcheckpoint番号・hash、config hash、明示seed（なければnull）、debug設定、
Hamiltonianパラメータ、実効設定、窓のrawパラメータを保存する。
リングの`w_sigma`は既存probeの表示上の半幅であり、平滑化幅とは区別する。
リング境界・平滑化係数等の実際の値は`metadata.window_parameters`と`dref`から読める。

## PHの定義

`topology/persistence.py`はNumPy/SciPyだけを利用するF2境界行列の参照実装。
計算量のためN≤64に制限し、H0/H1だけを扱う。clique complexの三角形までを作る。
辺から作るflag complexであり、重みから作った距離に三角不等式は仮定しない。

- A1=`Window`（無効なら恒等窓1）、A2=`Alpha`、A3=`W_window=alpha*m`、比較=`(S+1)/2`。
- A3はclip等の介入前。実際の集約`W`もprobeに残す。行和正規化はしない。
- B=(W+W.T)/2、対角を除く。全N頂点を最初から保持する。
- 辺のfiltration値u=1-B、三角形は3辺のuの最大。uの増加で追加する。
- 零重み辺は最後まで存在しない。正の同値重みは同時に追加し、寿命0のintervalは除く。
- グラフは厳密にB>t。t=Bではその辺はない。`analyze_graph`は同じグラフを与える閾値区間内でBettiを評価し、閾値変換の丸めを避ける。
- Diagramは上三角の(birth u,death u)。消滅しない成分はdeath=+inf。
- Total/max PersistenceとLandscapeは有限intervalだけで算出。essential数を必ず併記。
  無限intervalを有限な寿命へ切り詰めない。Landscapeはu∈[0,1]の51点、上位3段。
- BottleneckはL∞の最小最大対応、WassersteinはL∞をground costとするp=1（APIはp≥1）。
  対角への対応を含む。essential数が違えば距離はinf、等しければbirthを順序対応する。
  JSONのnullはNaN/infを表すため、有限値0と同一視しない。NPZにはinfを保持する。

標準ライブラリの定義との対応：[GUDHI Rips/flag complex](https://gudhi.inria.fr/python/3.6.0/)、
[Wasserstein・essential parts](https://pages.saclay.inria.fr/vincent.rouvreau/gudhi/sklearn_cubical/wasserstein_distance_user.html)。
今回GUDHIを依存に追加していない。全4頂点グラフ64通りと独立F2ランク式、
重み・同値・孤立頂点・既知の四角形で検算する。

## 巻き付きの定義

`graph_wind.weighted_winding`にpositions [N,2]、列ベクトル格子L [2,2]、重みを渡す。
SVDによる探索範囲保証付きの最近格子点探索で、実空間最短像の辺を埋め込む。
ネットワークの周期ノルム距離と、辺埋込みに使う最短像Euclidean距離は別物。
周期表現変更・並進・電子置換でRankが変わらないことを斜交セルでも検査する。

同じ長さの最短像が複数あるactive edgeは埋込みが一意でないため、明示エラーにする。
runnerはPHを保持し、巻き付きだけをNaN・ambiguousとして保存する。
縮退を電子番号依存の規則で破って、偽の不変性を主張しない。

同値重みをまとめてunion-findへ加え、全連結成分の閉路ベクトルが張るRank=0/1/2を測る。
`critical_rank1/2`は出現する重み閾値の**上限**であり、等号ではまだ出現しない。
辺が存在しなければNaN。閾値ごとのRank確率は[B,repeat]上で平均し、層・Headは保持する。
旧`thresholds`とCLIは履歴baselineとしてそのまま残す。斜交セルの新解析には新APIを使う。

幾何baselineも独立保存：`exp(-d_periodic/dref)`で同じ手順を使う。
これをcutoff=-dref log(t)へ戻せば、従来の距離グラフd<cutoffの出現閾値が得られる。
**この巻き付きは実空間グラフの量で、多体Chern数やBerry位相ではない。**
また、clique PHは三角形を抽象的に充填する一方、巻き付きは埋め込まれたグラフの閉路を測る。
トーラスを巻く3辺の閉路でもclique PHのH1は0になり得るため、2つの指標は一致を要求しない。

## 対照実験と集計

raw、正の辺のmidrank/M（同値と零を保持）、辺重みシャッフルの3条件を独立保存する。
順位変換は正の重み分布の尺度を除くが、support densityは揃わない。
シャッフルは各配置・層・Headのedge density、零数、全重みmultisetを厳密に保持する。
同じdensity曲線でもトポロジーが異なるかを調べる対照であり、物理的な対照相ではない。
同値を勝手に分割して固定edge数を作らない。実際のdensity曲線と重みの平均・標準偏差を保存する。

```json
{
  "null_repeats": 8,
  "runs": [
    {
      "id": "wc-seed11-step6000",
      "group": "seed11",
      "phase": "WC",
      "phase_evidence": "独立な物理判定を記載",
      "folder": "path/to/run",
      "checkpoint": "path/to/run/qmcjax_ckpt_006000.npz",
      "settings": "known-settings.json",
      "nconf": 64
    }
  ]
}
```

`probe`に保存済みNPZを指定してもよい。パスはmanifestからの相対パス。
Head/layer/walkerを跨ぐ分布は保存するが、独立標本数を水増ししない。

```sh
.venv-pw/bin/python -m screening_analysis.topology_run manifest.json --output NEW_DIRECTORY
```

各runへ`metrics.csv`、`diagrams.npz`、`curves.npz`、`geometry.json`、`summary.json`、
通常のSVG折れ線グラフを出力。curvesは[B,L,H,repeat,...,t]、巻き付き確率は[L,H,rank,t]。
run間の同じ配置index/layer/headのDiagram距離も保存するが、異なるrun間の配置indexは
物理的な対応を保証しない。距離は分布を調べる記述量として扱う。

WC/AHCの独立判定根拠・group・各相3group以上があれば、Head/layer平均、checkpoint平均の
特徴を使うleave-group-out nearest-centroidを探索的に実行する。
標準化はtrain groupだけでfit。特徴は有限H0/H1 persistence、essential数、Betti曲線平均、Rank確率。
独立seedの他に収束精度・窓・Hamiltonianのholdoutが必要であり、少数seedの正解率だけで
頑強性や物理的な位相との関連を認定しない。異なる窓を同じmanifestへ入れた場合は窓ごとの
比較も別途行い、設定差を相ラベルの代用にしない。

## 検証・pilot再実行

```sh
.venv-pw/bin/python -m unittest tests.test_topology tests.test_topology_run -v
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_topology_probe -v
JAX_PLATFORMS=cpu .venv-pw/bin/python -m screening_analysis.topology_pilot --output NEW_DIRECTORY
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.baseline compare tests/g2b_validation/snapshot-float64-g2b-v1 --exact
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/g2b_validation/snapshot-float32-g2b-v1 --exact
```

pilotは既存の自由1電子checkpoint12個と、未学習N9ネットワークの3seed×7窓/Value条件×2幾何分布を扱う。
後者の規則配置／一様配置は量子MCではなく、WC/AHCとは呼ばない。既存checkpoint全件のhashを前後照合する。
物理的なphase・Chern検証、正則化によるVMC改善はこのpilotで検証できない。
