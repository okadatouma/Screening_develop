# 最終チェックポイントの接続解析（PHを追加しない）

対象は `array-2475/lam*/training/qmcjax_ckpt_079999.npz`。各64 walkerをプールした128入力（位置・擬スピン・atoms・charges）を両モデルに同じ順序で与える。
自分のrun由来の64入力がnative、128入力がshared。入力由来のラベルを全集計に保存する。walkerは独立試行・平衡測定とはみなさない。

```bash
env JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/tomax/venvs/screening-gpu/bin/python -m screening_analysis.connection_audit \
  results/manual/array-2475 --output NEW_DIRECTORY
env MPLCONFIGDIR=/tmp/array2475-mpl \
  /home/tomax/venvs/pyrite-bands/bin/python -m screening_analysis.connection_audit_report NEW_DIRECTORY
```

解析先は新規ディレクトリにする。作図にはMatplotlibが必要。実データの完成版は `connection-audit-validated`。
最初の `connection-audit` は恒等介入の検証で停止した途中出力であり、位相を誤解釈した未検証データを含むため使用しない。

## 計算内容

- Window対距離：本体の `screening_window` による501点の曲線と、実入力の距離ビン内分布。smooth7窓なので指数減衰率はfitしない。既存の全チェックポイントから幅とTrimパラメータの履歴を読む。
- Alpha対距離：方向平均した非対角辺を、距離・擬スピン一致/不一致・入力由来・ヘッド別に集計する。
- Window→screeningの入れ替え：k=30,60,90の正の上位辺。同順位の境界や支持不足は省略し、利用できたグラフ数を保存。promoted=screeningだけ、demoted=Windowだけ、retained=両方。粒子i,jをCSVに保存する。
- 角度：格子ベクトルを列とする正確なユークリッド最小像を求め、実空間x軸からの角度をmod πで扱う。最短像が一意でない辺・ゼロ長は角度集計から除外。Windowの距離自体は既存の周期ノルムであり、両距離を保存する。
- 選択率：昇格の分母は「Windowが正でWindow top-kに入らなかった候補」、降格の分母は「Window top-k」。擬スピン比較は入力由来・距離・角度を共通分布へ標準化する。角度比較は入力由来・距離・擬スピンを標準化する。各セル最低5候補、共通支持領域のカバー率を記録。因果推定・有意差検定ではない。
- ヘッド：各配置で630辺に対するPearson/Spearman相関、20距離ビンの平均を差し引いた相関、出力射影後のヘッド寄与ベクトルのcosine、相殺比を求める。重みの負相関は出力ベクトルの相殺とは区別する。
- 波動関数：HF・Jastrowを含む元のfactoryから復元する。実際の有向集約重みの両方向を exp(ε) 倍する（自己項は固定）。新しい任意の `_edge_scale` フックは非指定時に元の経路を保つ。

1層・2ヘッドの各630無向ペアに対して `∂log|ψ|/∂ε`、`∂φ/∂ε` を自動微分する。複素Network APIの第1返り値は複素数 exp(iφ) ではなく、既に実数の位相角φである。
位相角0は正常な値なので、零波動関数判定には使わない。非有限のlog振幅・勾配はエラーにする。

選んだ接続群を10%弱め、振幅・位相の実応答と線形予測を記録する。位相差はmod 2π。
全入力に共通の振幅倍率・グローバル位相を除いた、中心化応答RMSも集計する。ただし平衡分布による重み付け・再最適化・エネルギー測定はしていない。

## 検証と出力

```bash
env JAX_PLATFORMS=cpu /home/tomax/venvs/screening-gpu/bin/python \
  -m unittest tests.test_connection_audit
/home/tomax/venvs/pyrite-bands/bin/python -m unittest tests.test_connection_audit_report
```

実チェックポイントでは恒等介入と元のネットワークを照合し、4配置×2ヘッドで感度最大の辺を選び、ε=1e-3と5e-4の中央差分で勾配を確認する。
パラメータをJITの動的引数として両経路に渡し、混合精度の定数畳み込みの差を避ける。

`shared-inputs.npz` と各モデルの `shared-probe.npz`、`edge-data.npz`、`connection-sensitivity.npz` が元の数値配列。
`edge-replacements.csv` は粒子ペアの具体的な入れ替え、`selection-strata.csv` は分母を含む層別件数、`wavefunction-responses.csv` は配置ごとの介入応答。
`aggregate-summary.json` と各summaryに検証結果、チェックポイント・HF・共通入力のhashを保存する。
`top-edge-composition.csv` は固定kグラフの擬スピン構成・三角形数・通常のcycle rankであり、新規PH計算ではない。

初期化直後は未保存。各条件1学習でλ・HF基底も異なるため、差を物理相や学習効果だけに帰属させない。128配置の単純平均に物理的な確率分布としての意味は与えない。
