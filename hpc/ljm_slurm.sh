#!/bin/bash
# λ-jellium の多点学習（HF なし、1 本のネットワークで多くの (r_s, lam)）を、Slurm の計算機の 1 ノードで走らせる。
# リポジトリの根で：
#   sbatch hpc/ljm_slurm.sh                    新しい run（TAG の結果のフォルダがなければ）
#   sbatch hpc/ljm_slurm.sh                    続き（同じ TAG のフォルダがあれば、最後のチェックポイントから自動で再開）
# 変えるときは sbatch --export=ALL,TAG=m32b,PW_BATCH=4096 hpc/ljm_slurm.sh のように。
# ノードの GPU を全部使う（JAX がウォーカーを GPU に分ける）。PW_BATCH は GPU の数 x 点の数 の倍数。
# GPU の数・パーティション・時間はコマンドの引数で変えられる（下の #SBATCH より優先）：
#   sbatch --gres=gpu:2 --partition=<パーティション> --time=7-00:00:00 --export=ALL,PW_MAX_MINUTES=10000 hpc/ljm_slurm.sh
# Slurm のない Linux の PC（GPU 付き）でも、そのまま動く（#SBATCH の行はただのコメント。PC の GPU を全部使う）：
#   TAG=m32a PW_BATCH=1024 nohup bash hpc/ljm_slurm.sh > ljm_m32a.out 2>&1 &
#SBATCH --job-name=pwf_ljm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=ljm_%j.out
#SBATCH --error=ljm_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
VENV="${PW_VENV:-$HOME/venv_pw}"
[ -x "$VENV/bin/python" ] || { echo "ERROR: $VENV がない（bash hpc/setup_env.sh）"; exit 1; }

export LJ_N="${LJ_N:-36}" ITERS="${ITERS:-200000}" TAG="${TAG:-m32a}"
export PW_MULTI="${PW_MULTI:-periodicwave/configs/lj_multi_points_32.txt}"
export PW_BATCH="${PW_BATCH:-4096}" PW_LR="${PW_LR:-10}" PW_LR_DELAY="${PW_LR_DELAY:-100000}"
export PW_LAYERS="${PW_LAYERS:-1}" PW_HEADS="${PW_HEADS:-16}" PW_DET="${PW_DET:-4}" PW_FREEZE_W=1
export PW_FLIPS=3 PW_FLIP_BATCH=0 PW_EL_CHUNK="${PW_EL_CHUNK:-128}" PW_DELTA_MULT=1 PW_INIT_SPIN=random
export PW_WIN_SHAPE=smooth7 PW_TRIM_POW=4 PW_LJ_NETSCALE=1 PW_LJ_JASTROW=1 PW_INPUT12_INIT=random
export PW_PRETRAIN="${PW_PRETRAIN:-4000}" PW_PRETRAIN_LR="${PW_PRETRAIN_LR:-0.001}"
export PW_SAVE_MIN="${PW_SAVE_MIN:-30}" PW_MAX_MINUTES="${PW_MAX_MINUTES:-1380}"   # ジョブの時間より短く（事前学習と焼きなましの時間は数えない）
export JAX_ENABLE_X64=0   # lj3 と同じ単精度（倍精度は GPU によってはずっと遅い）
unset PW_LJ_HF PW_INPUT12 PW_TRIM_FIX
mkdir -p jobs/logs
JOBID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
echo "job $JOBID on $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-all}  tag=$TAG  batch=$PW_BATCH  points=$PW_MULTI  $(date)"
command -v nvidia-smi > /dev/null && { nvidia-smi -L || true; }
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 PYTHONPATH="$PWD" "$VENV/bin/python" -u periodicwave/configs/run_lambda_jellium.py \
  "$LJ_N" "$ITERS" "$TAG" > "jobs/logs/run_${TAG}_${JOBID}.log" 2>&1
echo "exit=$? $(date)   log: jobs/logs/run_${TAG}_${JOBID}.log"
