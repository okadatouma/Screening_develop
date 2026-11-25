# Embedding・Screening・Wavefunction比較基盤

このディレクトリは既存Screening_topology_gpu内の研究用packageです。
連続電子、格子TFIM、QWZは共通設定・保存形式・介入APIを使いますが、異なる物理状態を同じWavefunctionへ押し込めません。
既存の学習・解析経路は既定値のまま保持しています。

## 環境と入口

リポジトリroot `/home/tomax/projects/Screening_topology_gpu` から実行します。
専用CPU環境は `research_framework/.venv` に作成済みです。既存GPU環境・Slurmジョブは変更していません。

別環境を作る場合はvenvを作成してから `pip install -r research_framework/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu` を実行します。
このホストはensurepipがないため、環境作成に `python3 /home/tomax/virtualenv.pyz research_framework/.venv` を使用しました。

```bash
PYTHON=research_framework/.venv/bin/python
JAX_PLATFORMS=cpu "$PYTHON" -m research_framework.cli validate research_framework/configs/continuous_baseline.yaml
JAX_PLATFORMS=cpu "$PYTHON" -m research_framework.cli run research_framework/configs/continuous_baseline.yaml research_framework/results/my_electron_run
```

設定ファイルの `embedding`、`attention`、`wavefunction` を変更します。他の設定を変える必要はありません。
未知のキー、不適合なtask/ansatz、周期境界に適合しないFourier波数、連続系の微分不適切なhard Windowなどは実行前に拒否します。
型・選択肢の検証は `config.py`。不適合な設定を黙って別モデルへ落としません。

```python
from research_framework import load_config, make_task
from research_framework.factory import make_embedding, make_attention, make_wavefunction

config = load_config("research_framework/configs/continuous_baseline.yaml")
embedding = make_embedding(config["embedding"], config["experiment"]["task"], config["physics"], config["model"])
attention = make_attention(config["attention"], config["experiment"]["task"])
task = make_task(config)
```

Factoryが返す型はtaskごとに異なります。連続系はJAX init/apply、QWZはTorch module、TFIMは既存LatticeWavefunctionです。
共通の実行はCLI/adapterが担当します。

## モデル選択

| task | Embedding | Attention | Wavefunction |
|---|---|---|---|
| continuous_electrons | baseline / six_theta / spin_onehot / six_theta_onehot / learned_fourier / complex(real2channel) | 元のScreening＋baseline/fixed/full/shared trainable/per-head trainable Window | baseline / complex / hf / hf_jastrow / hf_nn_jastrow / twist_dependent |
| lattice_tfim | 既存site embedding | 既存Screening、hard R=1/2/4/Full、sigmoid固定/各head学習 | 既存positive real lattice wavefunction |
| qwz_topology | 生Bloch実虚 / projector / complex / gauge_covariant | real / native complex / gauge_covariant、固定/共有/各head学習Window | projector / local_atlas |

Gauge-covariant構成は、charged embedding・輸送・複素出力を一緒に使います。Projector入力とcharged輸送を不適切に混在させません。
QWZの通常Attentionはsoftmaxです。連続系とTFIMのScreening集約にはsoftmaxを入れていません。

## 周期とFourier

格子ベクトルは列。L=A Sの整数supercell行列を `physics.crystal_supercell` に指定します。
`six_theta` はS=diag(6,6)の場合に6θとなり、他のSでは対応する結晶逆格子波数へ変わります。
`embedding.modes` は計算セル逆格子に対する整数係数だけです。連続した不適合波数は学習しません。
learned Fourierは係数a_qを学習します。入力射影の正規化で直近のスケール冗長性を抑えますが、a_qだけを秩序変数とは解釈しません。
自然な次元比較と `model.capacity_mode: padded` の固定入力次元・名目parameter数比較を分けています。

## 比較計画

```bash
"$PYTHON" -m research_framework.cli plan research_framework/results/my_pilot --tier pilot
"$PYTHON" -m research_framework.cli suite research_framework/results/my_pilot/plan.json --timeout 600
"$PYTHON" -m research_framework.cli collect research_framework/results/my_pilot
"$PYTHON" -m research_framework.cli report research_framework/results/my_pilot
```

suiteは各条件を独立子プロセスで実行し、途中のエラー後も他条件の結果を失いません。
必要なら `--task lattice_tfim` 等で3つのadapterを独立実行できます。
全組合せの直積は実行しません。pilot planは単一変更・必要な互換性を満たす構成・容量対照のみです。
Stage Fの複合改良は、単独効果を独立seedと誤差で確認してから追加します。

```bash
"$PYTHON" -m research_framework.cli plan research_framework/results/my_production_plan --tier production
```

大型計画を作るだけでSlurmへsubmitしません。N36では元候補の実格子・HF表・Hamiltonian設定を参照しますが、Hessian推定器の計算負荷を測るまでは実行不可としています。
元の長時間checkpointをこのpackage形式へ自動変換する機能はありません。既存checkpointは既存 `screening_analysis.checkpoint_model.load_model` の正式経路で読む設計を維持しています。

## Checkpointと再開

```bash
"$PYTHON" -m research_framework.cli run research_framework/configs/lattice_tfim.yaml research_framework/results/resume_demo --until 50
"$PYTHON" -m research_framework.cli run research_framework/configs/lattice_tfim.yaml research_framework/results/resume_demo --resume
```

各step checkpointは不変で、latest pointerだけを原子的に更新します。optimizer、乱数、連続系/TFIMのwalkerも保存します。
同じconfig・同じsource hashの場合だけ再開できます。sourceが変わったrunは保持して、新規出力先で実行してください。
JAX/Torch両方の中断・再開を、非中断計算と一致するかテストします。
中断でcheckpointより先に書かれた履歴は、再開時にそのrunの最新durable stepまで戻します。
ローカルで自分が生成したpickleだけを読み、外部由来のpickleは読み込まないでください。

## 介入と物理的評価

```python
from research_framework.interventions import apply_intervention
changed_model, changed_params = apply_intervention(config, model, params,
    {"type": "head_ablation", "layer": 0, "head": 0})
```

介入名は `head_ablation`, `remove_connections`, `distance_band`, `window_width`, `remove_fourier`, `remove_six_theta`, `remove_spin`, `replace_spin`, `zero_phase_channel`, `disable_gauge_transport`, `patch_connections`。
存在しない機能や範囲外head/layerはエラーです。
連続電子では粒子番号だけで接続を切ると反対称性を壊すため、物理距離に対する滑らかなkernelを使います。
介入後のEは変更後の状態から再サンプリングします。TFIMは全配置を再正規化して再評価します。
QWZではnorm/overlap/branch/メッシュ一致screenを付け、Chernを無条件に物理gapのある相として扱いません。

## 保存物と診断

各runにconfig、入力hash/NPZ、revision、変更patch、source snapshot、環境、履歴、checkpoint、summary、介入、diagnosticsを保存します。
内部量はQ/K/QK/Alpha/Window/実集約重み/Value/head寄与を取得可能なtaskで保存します。
PHは既存reducerとRipserを再利用します。phase-liftは独立プロセスで時間上限を付け、JAX初期化後のforkを避けます。
TFIMではconnected correlation、有限鎖xi、Born MI/CMI、Quantum MI、S2、Fidelity、p_-を解析します。
QWZではProjector/atlas、Gauge応答、Berry曲率、Wilson/seam巻き付き、Chern、内部PHと物理phase-liftを比較します。
twist依存の多体波動関数には座標の準周期境界と元Hamiltonianの微分を使い、N<=2で共通periodic基底のoverlap求積も実装しています。
小矩形のholonomyを多体Chern数とは呼びません。全twistトーラスのsewing・gap・収束は別途必要です。

## テスト

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PYTHON" -m pytest -q research_framework/tests
```

従来factoryとの一致、周期/結晶周期/微分/交換共変、Window勾配、U(1)輸送の共変性、複素出力射影、零overlap、反対称性、twist局所E、Projector条件、PH、ED/MI/CMI/パリティ、中断再開を検証します。

## 今回の実行結果

主pilotは `results/2026-11-06_pilot`、結晶波数を候補集合に含む追加対照は `results/2026-11-06_fourier_full_candidates` に保存しています。
主pilot73条件、追加6条件を実際に計算しました。大型7条件は `results/2026-11-06_production_plan` の設定だけで未実行です。

```bash
"$PYTHON" -m research_framework.cli report research_framework/results/2026-11-06_pilot --supplement research_framework/results/2026-11-06_fourier_full_candidates
```

完了済みreportの上書きは拒否します。新しい計算は新しい保存先で報告してください。
実装上のエラーを修正して再試行するときは、同一config・新しい出力先で失敗履歴を保持します。

```bash
"$PYTHON" -m research_framework.cli suite research_framework/results/my_pilot/plan.json --retry-failed
```

元のrunは保持され、`retries.json` が新しい試行を指します。集計は対応する最新試行を使います。

最終の幅介入の集計には `results/2026-11-06_pilot/reports_corrected/` を使います。
修正前のreports・診断は履歴として保持しました。詳しい経緯はIMPLEMENTATION_REPORT_JA.md。
図の入口は [figures/README_JA.md](figures/README_JA.md) です。
