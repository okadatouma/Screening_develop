# 接続数を揃えたPH解析

`density_ph.py` は、既存の最終 `probe.npz` を読み、学習・チェックポイントを変更せずに解析する。

```bash
/home/tomax/venvs/screening-gpu/bin/python -m unittest tests.test_density_ph tests.test_topology
/home/tomax/venvs/screening-gpu/bin/python -m screening_analysis.density_ph \
  results/manual/array-2475 --output NEW_DENSITY_DIRECTORY --null-repeats 8
env JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/tomax/venvs/screening-gpu/bin/python -m screening_analysis.density_ph_history \
  results/manual/array-2475 --output NEW_HISTORY_DIRECTORY
```

出力先は原則として新規ディレクトリにする。履歴の `--resume` は既存probeと完成済み解析を再利用する。解析条件やソースを変更した場合は新しい出力先を使う。

## 第1・第2段階

各配置・層・ヘッドについて QK、Alpha (`trim`)、Window、Alpha×Window (`screening`)、距離基準を同じkで比較する。
方向を平均して自己接続を除外し、正の辺を重みの降順で追加する。最大kは各グラフの正の辺数の最小値。
三角形は最後の辺が入ると同時に追加するflag複体。PHはF2上の既存実装を使用する。

同順位は同じ乱数優先順位で処理する。ただし同順位を分割する境界は記録し、主集計と図から除外する。
ゼロ辺は追加しない。[-1e-12,0) および (1,1+1e-12] の丸め誤差のみclipし、個数を記録する。
その範囲を超える不正な重みはエラーにする。

距離基準は保存probeの周期ノルム `d` に対する `exp(-d/dref)`。ユークリッド最小像距離とは区別する。
距離対照は距離の分位点で最大10帯に分け、各帯内でゼロを含む重みを並べ替える。
各帯内の重みの多重集合・正の辺数を厳密に保存する。個別の辺長・次数・連結性は保存しない。

- `curves.csv`: 各配置・層・ヘッド・kのβ0/β1、対照反復、同順位フラグ。
- `edge-overlap.csv`: k=30,60,90での辺集合のJaccard一致度（サポートが足りる場合）。
- `summary.json`: 入力・ソースhash、支持辺数、共通k範囲、丸め誤差処理、集計値。
- `primary_mean_beta1_global_common_k`: 全配置・全ヘッドで利用可能な共通kの平均。条件ごとに上限が違うため、そのまま条件間比較には使わない。
- `mean_beta1_per_graph_shared_untied_k`: 各配置・ヘッドの利用可能範囲での平均。範囲が異なるので補助指標。
- `diagrams-edge-count.npz`: 誕生・消滅を辺数単位で保存。比較領域の最大kで打ち切るため、無限寿命は打ち切り後に消滅する可能性がある。

配置・ヘッド・対照反復は独立学習ではない。標準誤差や相分類精度の根拠にしない。

## 第3段階

各runの最終probeに保存された位置・擬スピン等を固定し、約4千、2万、4万、6万、8万ステップの既存チェックポイントを再評価する。
これはパラメータ変化を調べる診断で、各時点の波動関数からの平衡サンプリングではない。
初期化直後のチェックポイントは存在しないので再構成しない。

`history.csv` の `fixed_k_*` は、全時点・全配置・全ヘッドに共通のk範囲で集計する。時系列比較にはこの列を使う。
接頭辞のない列は各時点の別々の支持範囲で集計した補助値で、時系列比較には使わない。
エネルギー列は保存された学習ログの直前最大2千更新の平均であり、独立エネルギー測定ではない。

描画にはMatplotlibのある既存環境を使える。

```bash
env MPLCONFIGDIR=/tmp/array2475-mpl /home/tomax/venvs/pyrite-bands/bin/python \
  -m screening_analysis.density_ph_plot results/manual/array-2475
```

この描画コマンドの既定入力は `density-ph-final` と `density-ph-history`。密度解析の入力名は `--density-dir` で指定できる。

## 第4・第5段階

この入口は再学習や相ラベルの推定を実行しない。同一λ・複数初期化・複数seedの学習、独立な物理相の根拠、十分に混合したサンプリングが必要。
学習runを単位に分割し、単純な物理量や距離特徴のベースラインと比較してから分類性能を評価する。
