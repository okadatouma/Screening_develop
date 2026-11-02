# lj_reference：λ-jellium（lj3）の計算を段ごとに書き直した参照用コード

本体（`periodicwave/`）のうち、いまの λ-jellium の run（lj3 の設定）で実際に通る部分だけを、処理の段ごとに 1 ファイルずつ書き直したもの。
本体には 2DEG・モアレの時期の選択肢が多く残っていて、lj3 の計算がどの関数で行われているかを追いにくいので、読むために作った。
学習には使わない（本体は変えていない）。同じパラメータとウォーカーから本体と同じ値が出ることを `check.py` で確かめた（下の「検算」）。

段の番号は `doc/code_map/code_map.pdf` と同じ：③ 設定、④ 学習のループ（1 step の ④-1〜④-4）、⑤ 1 回の $\psi$ の計算（⑤-1〜⑤-4）。
地図の各段の箱に、このフォルダの対応するファイルを「参照」として書いてある。

## ファイルと段

| 段 | ファイル | 計算すること | 本体 |
|---|---|---|---|
| ③ | `model.py` | run のフォルダ（`config.json`、チェックポイント）と ① の HF の表を読み、定数・パラメータ・ウォーカーを返す | `configs/run_lambda_jellium.py`（設定を作る）、`checkpoint.py`、`lj_orbitals.load_hf_table` |
| ⑤-1 | `inputs.py` | 1 電子の入力 12 成分、対の距離 $d_{ij}$（周期ノルム） | `networks.make_periodic_construct_input_features`、`pbc/feature_layer.make_pbc_feature_layer`、`CustomScreenformer.one_electron_inputs` |
| ⑤-2 | `attention.py` | 埋め込み → 遮蔽アテンション → MLP。各電子の特徴 $h_i$ | `CustomScreenformer` の `make_screenformer_layers`、`make_self_attention_block`、`make_screening_attention`、`make_mlp`、`make_layer_norm` |
| ⑤-3 | `orbitals.py` | 軌道 $\phi^k_j=\phi^{\rm HF}_j+\varepsilon\,[W^k_{s}h]_j$ と、擬スピンの組ごとの Jastrow $J$ | `lj_orbitals.make_lj_orbitals` |
| ⑤-4 | `wavefunction.py` | 行列式 4 枚の和 → (位相 $\varphi$, $\ln\lvert\psi\rvert$) | `CustomScreenformer.make_fermi_net`、`network_blocks.logdet_matmul` |
| ④-1 | `sampling.py` | $\lvert\psi\rvert^2$ に従う Metropolis（全電子の移動 1 回と擬スピン反転 3 回を 10 回） | `mcmc.make_mcmc_step`（`_make_mcmc_step_with_flips`、`mh_update`、`mh_spin_flip`、`mh_accept`） |
| ④-2 | `hamiltonian.py` | 局所エネルギー $E_L=H\psi/\psi$ と項ごとの値 | `pbc/lambda_jellium.local_energy`、`pbc/hamiltonian.make_2DCoulomb_potential` |
| ④-3、④-4 | `gradient.py` | $E=\langle E_L\rangle$ と勾配（$E_L$ の切り詰めを含む）、更新の流れ | `loss.make_loss`、`loss.clip_local_values`、`train_screen.py`（KFAC） |
| | `check.py` | 本体との突き合わせ | |

$\phi^{\rm HF}_j$：① の HF の Bloch 軌道（固定）、$\varepsilon=0.05$、$W^k_{s}$：擬スピン $s$ ごとの線形写像（学習）、$k=1,\dots,4$：行列式の番号。
Jastrow の因子は本体と同じく、軌道の行列の各成分に $e^{J/N}$ として掛ける（`orbitals.py`）。

読む順：`wavefunction.py`（⑤ のつながり）→ `inputs.py` → `attention.py` → `orbitals.py`、つぎに `hamiltonian.py` → `sampling.py` → `gradient.py`。
各ファイルの冒頭に式と本体の対応する関数を書いた。⑤ のファイルには、使うパラメータの `params` の中の位置も書いた。

## 使い方

関数は 1 つの配置について書いてある：位置 `x`（長さ $2N$、$(x_1,y_1,x_2,y_2,\dots)$、長さの単位 $r_s a_B$）、擬スピン `s`（長さ $N$、$\pm1$）。
ウォーカーの軸は `jax.vmap` で足す。エネルギーの単位は $2/r_s^2$ Ry。

```python
from lj_reference import model as M, wavefunction, hamiltonian
model, params, walkers = M.load_run('<run のフォルダ>', hf_dir='<HF の表のフォルダ>')
x, s, atom = walkers['x'][0], walkers['s'][0], walkers['atom']
phase, logabs = wavefunction.log_psi(params, model, x, s, atom)          # ⑤：ψ = exp(logabs + i phase)
e_l, terms = hamiltonian.make_local_energy(model)(params, x, s, atom)     # ④-2：E_L と項ごとの値
```

`atom` は本体の入力の形に合わせたダミーの原点（いまは 0）。`hf_dir` を省くと `config.json` の `lj_hf_path` をそのまま使う。

検算（リポジトリの根で）：

```
JAX_PLATFORMS=cpu python -m lj_reference.check <run のフォルダ> [...] --hf-dir <HF の表のフォルダ> [--out 結果.json]
```

本体の窓と Trim の形を決める環境変数（`PW_WIN_SHAPE=smooth7`、`PW_TRIM_POW=4`、`PW_FREEZE_W=1`）は `check.py` が lj3 の値に設定する。
CPU 2 コアで 1 run あたり約 2.5 分（ほとんどがコンパイル）。

## 検算（2026-11-02）

lj3 の 4 本（λ = 0.33・0.72 の WC 型と AHC 型、最後のチェックポイント）で、同じパラメータ・ウォーカーについて参照用コードと本体を倍精度で計算した。
差と値の大きさは 4 本での最大。

| 段 | 比べた量 | ウォーカー | 差の最大 | 値の大きさ |
|---|---|---|---|---|
| ⑤-1 | 1 電子の入力 12 成分 | 64 | $5.7\times10^{-14}$ | 1 |
| ⑤-1 | 対の距離 $d_{ij}$ | 64 | $2.5\times10^{-14}$ | 3.86 |
| ⑤-2 | 特徴 $h_i$ | 64 | $5.0\times10^{-13}$ | 43.3 |
| ⑤-3 | 軌道の行列 $\Phi^k_{ij}$ | 64 | $1.2\times10^{-13}$ | 8.12 |
| ⑤-4 | $\ln\lvert\psi\rvert$ | 64 | $8.5\times10^{-14}$ | 100 |
| ⑤-4 | 位相 $\varphi$（$2\pi$ を法として） | 64 | $4.4\times10^{-14}$ | $\pi$ |
| ④-1 | MCMC 後の位置 | 16 | 0 | 82.7 |
| ④-1 | MCMC 後に擬スピンが違う電子の数 | 16 | 0 | 576 個中 |
| ④-1 | 受理率（位置、反転） | 16 | 0 | 0.28〜0.58 |
| ④-2 | $E_L$（本体は学習と同じ folx） | 4 | $1.3\times10^{-11}$ | 460 |
| ④-2 | 運動 $-\sum_i c_i\nabla_i^2\psi/\psi$ | 4 | $6.2\times10^{-12}$ | 468 |
| ④-2 | 擬スピン反転の項 | 4 | $1.1\times10^{-11}$ | 674 |
| ④-2 | $\Delta n_-$ | 4 | 0 | 315 |
| ④-2 | クーロン（Ewald） | 4 | $1.0\times10^{-12}$ | 550 |
| ④-3 | $E=\langle E_L\rangle$ | 8 | $6.5\times10^{-12}$ | 449 |
| ④-3 | 勾配 $\partial E/\partial\theta$（全パラメータ） | 8 | $7.3\times10^{-12}$ | 50.3 |
| ④-3 | 勾配（パラメータの組ごとの相対差） | 8 | $5.8\times10^{-12}$ | （相対） |
| ④-3 | 窓の幅 $\omega$ の勾配 | 8 | 0 | 両方とも 0 |

差はすべて倍精度の丸めの大きさ（相対 $10^{-11}$ 以下）。

- MCMC は同じ乱数の鍵と幅から、同じ配置と同じ受理率を出す（乱数を使う順を本体とそろえてある）。
- $E_L$ は、学習で使う folx のラプラシアンの本体と比べた。項ごとの値は本体の `lambda_jellium.make_energy_components` と比べた。
- 勾配は、本体の `loss.make_loss` を学習のコードと同じく `jax.value_and_grad(..., has_aux=True)` で微分したもの（KFAC で前処理する前）と比べた。
  窓の幅 $\omega$ の勾配は両方とも 0（lj3 では窓を固定）。

## 本体と書き方が違うところ（値は同じ）

- ラプラシアン：本体は folx の前進ラプラシアン。ここは $\ln\psi$ の 2 階微分の行列（`jax.hessian`、$2N\times2N$）の対角。遅いが式どおり。
- 擬スピン反転の項の $D_i\ln\psi(S^{(i)})$：本体は `jax.linearize` の 2 方向、ここは `jax.jacrev`。
- MCMC のくりかえし：本体は `lax.fori_loop`、ここは Python の `for`。
- 更新：本体は KFAC（学習率 10、減衰つき）。`gradient.vmc_step` は単純な勾配降下で、1 step の流れを示すためのもの（本体とは比べていない）。
- 書いていないもの：複数の GPU（`pmap`）、MCMC の幅の調整、チェックポイントとログの書き出し、lj3 以外の選択肢（余弦の窓、帯窓、学習する Trim、入力 `sincos_s` など）。
  `model.load_run` は lj3 の設定であることを確かめ、違えば止める。
- 精度：本体の学習は単精度。参照用コードはどちらの精度でも動く（`check.py` は倍精度）。
