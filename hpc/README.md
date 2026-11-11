# hpc：λ-jellium の多点学習を GPU の計算機・スパコンで走らせる

## 何を走らせるか

1 本のネットワークで、点の表の (r_s, λ) の基底状態を同時に学習する（`periodicwave/lj_multi.py`）。HF は使わない。
点ごとに変えるのは (r_s, λ) だけで、次の 3 か所に入る。相（WC か AHC か）はネットワークに渡さない。

- ネットワークの 1 電子の入力の最後の 2 成分 ((r_s − 中心)/幅, (λ − 中心)/幅)
- ハミルトニアン：Δ = r_s（2/r_s Ry）、c₊ = ½ + r_s λ²、擬スピン反転の項 i r_s λ、Δ n₋、クーロン r_s × Ewald
- Jastrow のカスプ γ₊₊ = r_s/(6c₊)、γ₋₋ = r_s/3、γ₊₋ = r_s/(c₊ + c₋)

流れ：ウォーカーをセルの中に一様に置く → **事前学習**（ネットワークの軌道を、相互作用のない λ-jellium の基底状態の軌道
φ⁰_j(r, s) = e^{ik_j·r} u_{k_j}(s; λ) に合わせる。k_j は結晶の第 1 ブリルアン域の 6×6 の k 点、u は下のバンドのスピノル）
→ MCMC の焼きなまし → VMC（KFAC）。勾配で E_L から引く平均と E_L の切り詰めは点ごとに取る。

点の表：`periodicwave/configs/lj_multi_points_32.txt`（32 点。境界から離した 3 つの領域、表の注釈を参照）。

1 ノードの GPU を全部使う（ウォーカーを GPU に分ける。各 GPU に全部の点のウォーカーを同じ数ずつ置く）。
**PW_BATCH は GPU の数 × 点の数 の倍数**（32 点で GPU 2 枚なら 64 の倍数、4 枚なら 128 の倍数）。

## 準備（計算機ごとに 1 回）

リポジトリを置いて、その根で：

```
bash hpc/setup_env.sh            # $HOME/venv_pw に jax[cuda12]==0.7.2 ほか（hpc/requirements_gpu.txt）
```

Python 3.11 以上が要る（jax 0.7.2）。別の python を使うときは `PYTHON=<python> bash hpc/setup_env.sh`。
CUDA のライブラリは pip で入るので、計算機に要るのは NVIDIA のドライバだけ。

TSUBAME4.0 は標準の python3 が 3.9 なので、TSUBAME の案内のとおり miniconda で 3.11 の環境を作ってから使う
（案内では、インストールは計算ノードの対話ジョブ（qrsh・iqrsh）で）：

```
module load miniconda
eval "$(conda shell.bash hook)"
conda create -n pw311 -c conda-forge --override-channels python=3.11 -y
conda activate pw311
PYTHON=python bash hpc/setup_env.sh      # $HOME/venv_pw の python は pw311 の python を指す
```

ジョブの中（`hpc/ljm_tsubame.sh`）では `module purge` のあと `$HOME/venv_pw/bin/python` をそのまま使う（conda を有効にしなくてよい）。

## 走らせ方

| 計算機 | 新しい run | 続き |
|---|---|---|
| Slurm の計算機（1 ノード） | `sbatch hpc/ljm_slurm.sh` | 同じコマンド |
| TSUBAME4.0（H100 × 4 の 1 ノード、最大 24 時間） | `qsub -g <グループ> hpc/ljm_tsubame.sh` | 同じコマンド |
| 研究室の ws（GPU 2 枚） | PC の `ws_pw_kit\_log_ljm.bat` | `ws_pw_kit\_log_ljm_resume.bat` |

同じ TAG の結果のフォルダがあれば、最後のチェックポイントから自動で再開する（事前学習はしない）。
`PW_MAX_MINUTES` を過ぎたら、チェックポイントを書いて自分で止まる（ジョブの時間切れで消えないように）。
値を変えるときは `sbatch --export=ALL,TAG=m32b,PW_BATCH=8192 hpc/ljm_slurm.sh`、`qsub -g <グループ> -v TAG=m32b hpc/ljm_tsubame.sh` のように。
ログは `jobs/logs/run_<TAG>_<ジョブ番号>.log`（事前学習の loss、止まった step、エラー）。

### ほかの計算機で走らせるとき（GPU が 1〜2 枚の Slurm、Slurm のない PC）

GPU の数・パーティション・時間は sbatch の引数で変える（`hpc/ljm_slurm.sh` の `#SBATCH` の行より優先される）。
`PW_MAX_MINUTES` はジョブの時間の上限より 1 時間ほど短く（7 日 = 10080 分）：

```
sbatch --gres=gpu:2 --partition=<パーティション> --time=7-00:00:00 \
       --export=ALL,TAG=m32a,PW_BATCH=2048,PW_MAX_MINUTES=10000 hpc/ljm_slurm.sh
```

Slurm のない Linux の PC（GPU 付き）では、同じスクリプトをそのまま走らせる（`#SBATCH` の行はただのコメント。PC の GPU を全部使い、
選ぶときは前に `CUDA_VISIBLE_DEVICES=0` などを付ける）。Windows は WSL2 の中で（jax の GPU 版は Linux だけ）：

```
TAG=m32a PW_BATCH=1024 nohup bash hpc/ljm_slurm.sh > ljm_m32a.out 2>&1 &
```

- `PW_BATCH` を 1024・2048・4096 にすれば、32 点で GPU 1・2・4 枚のどれでも割り切れる。割り切れないときは始めにエラーで止まる。
- GPU のメモリが足りない（ログに `RESOURCE_EXHAUSTED`）ときは `PW_EL_CHUNK` を 64、32 と小さくする（遅くなるが、値は丸めの範囲で同じ）。
- 単精度で走らせる（ジョブのスクリプトで `JAX_ENABLE_X64=0`、lj3 と同じ）。
- このブランチ（wada_dev）のコードで走らせる。ほかのブランチのコードとは環境変数の名前などが違うので、別のフォルダに clone する。

はじめての計算機では、まず短い試しで 1 step の時間とメモリを確かめる（事前学習 4000 step のあと 60 分学習して止まる）：

```
sbatch --gres=gpu:<GPU の数> --time=02:00:00 --export=ALL,TAG=t1,PW_MAX_MINUTES=60 hpc/ljm_slurm.sh
```

ログの最後の「PW_MAX_MINUTES = 60 に達したので step T で止めた」から 1 step ≈ 3600/T 秒。
`train_stats.csv` の energy と `multi_stats.csv` の点ごとの E が下がっていて、NaN と Traceback がなければよい。

## 出力

`results/lambda-jellium/CustomScreenformer/N36_multi32_<TAG>/`

| ファイル | 中身 |
|---|---|
| `multi_stats.csv` | 各 step の、点ごとの `E{p}`（Re⟨E_L⟩、コード単位）と `nm{p}`（⟨n₋⟩/N）。p は点の表の順（0 から） |
| `train_stats.csv` | 全ウォーカーの平均（点をまぜた値）、受理率など |
| `pretrain_stats.csv` | 事前学習の loss（⟨\|Φ_net − Φ⁰\|²⟩） |
| `qmcjax_ckpt_*.npz` | チェックポイント。`data['charges']` に各ウォーカーの (r_s, λ, 点の番号) |
| `config.json` | 設定（`system.multi_points` が点の表） |

エネルギーはコード単位（2/r_s² Ry を 1 とする）。点どうしを比べるときは E[Ry] = (2/r_s²) E に直す。
r_s = 15 の lj3（HF から出発）の値：λ = 0.33 の WC −445.204、λ = 0.72 の AHC −402.665（36 電子の合計、コード単位）。

## 主な環境変数（既定は `ljm_slurm.sh`・`ljm_tsubame.sh` の値）

| 変数 | 既定 | 意味 |
|---|---|---|
| `TAG` | m32a | 結果のフォルダの名前の最後 |
| `PW_MULTI` | `periodicwave/configs/lj_multi_points_32.txt` | 点の表（各行 r_s λ） |
| `PW_BATCH` | 4096（ws は 2048） | ウォーカーの数（GPU の数 × 点の数 の倍数） |
| `ITERS` | 200000 | VMC の step 数の上限 |
| `PW_PRETRAIN`、`PW_PRETRAIN_LR` | 4000、0.001 | 事前学習の step 数と学習率（Adam） |
| `PW_LR`、`PW_LR_DELAY` | 10、100000 | KFAC の学習率 lr/(1 + t/delay) |
| `PW_MAX_MINUTES`、`PW_SAVE_MIN` | 1380（ws は 690）、30 | 自分で止まるまでの分、チェックポイントの間隔（分） |
| `PW_EL_CHUNK` | 128 | 局所エネルギーを GPU 1 枚で何ウォーカーずつ計算するか（メモリが足りなければ小さく） |
| `PW_VENV` | `$HOME/venv_pw` | Python の環境（`hpc/setup_env.sh` で作ったもの） |

ネットワークは lj3 と同じ（1 層、16 ヘッド、q·k 16、MLP 64、行列式 4、窓は smooth7 で固定、Trim の幅は r = 2 で固定）に、
入力 (r_s, λ) の 2 成分を足し、軌道は HF なしのネットワークの出力（倍率 1）。

## 規模の目安

lj3 は ws の GPU 1 枚で 1024 ウォーカー・約 0.94 秒/step（ウォーカー 1 本・1 step あたり約 0.9 ms）。
4096 ウォーカー（32 点 × 128）で 20 万 step なら、ws の GPU 換算で約 200 GPU 時間（H100 での速さは未測定）。
必要な step 数はまだ分からない（CPU の N = 9 と ws の短い run で当たりを取ってから決める）。
チェックポイントは 1 個約 6 MB（N = 36、4096 ウォーカー）で、30 分ごとに消さずに溜まる（1 日約 0.3 GB）。Python の環境は数 GB。

## 確かめたこと（クラウドの CPU、jax 0.7.2）

- `test_lj_multi.py`：E_L と差分の力ずく計算（ウォーカーごとに違う (r_s, λ)）、ウォーカーごとの E_L ＝ 同じ (r_s, λ) を定数で渡した E_L、
  点ごとの勾配 ＝ 点ごとの 1 点の勾配の平均、事前学習の的が Coulomb なしの H の固有状態（E_L = Σ k²/2）、事前学習で軌道が的に近づく。
- `run_lambda_jellium.py` の多点（N = 9、2 点）を、CPU で 2 台の装置に分けて 4 step と、チェックポイントからの再開。
- GPU では、まだ走らせていない。
