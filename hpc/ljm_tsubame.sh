#!/bin/sh
# TSUBAME4.0（東京科学大学、Altair Grid Engine）で λ-jellium の多点学習を 1 ノード（H100 x 4）で走らせる。
# リポジトリの根で（グループ名は -g でコマンドに付ける。スクリプトの中には書けない）：
#   qsub -g <TSUBAME のグループ> hpc/ljm_tsubame.sh
# 同じ TAG のフォルダがあれば、最後のチェックポイントから自動で再開する（1 ジョブ最大 24 時間なので、続きは同じコマンドで）。
# 変えるときは qsub -g <グループ> -v TAG=m32b,PW_BATCH=8192 hpc/ljm_tsubame.sh のように。
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N pwf_ljm
#$ -o ljm.out
#$ -e ljm.err
module purge
VENV="${PW_VENV:-$HOME/venv_pw}"
[ -x "$VENV/bin/python" ] || { echo "ERROR: $VENV がない（bash hpc/setup_env.sh）"; exit 1; }

export LJ_N="${LJ_N:-36}" ITERS="${ITERS:-200000}" TAG="${TAG:-m32a}"
export PW_MULTI="${PW_MULTI:-periodicwave/configs/lj_multi_points_32.txt}"
export PW_BATCH="${PW_BATCH:-4096}" PW_LR="${PW_LR:-10}" PW_LR_DELAY="${PW_LR_DELAY:-100000}"
export PW_LAYERS="${PW_LAYERS:-1}" PW_HEADS="${PW_HEADS:-16}" PW_DET="${PW_DET:-4}" PW_FREEZE_W=1
export PW_FLIPS=3 PW_FLIP_BATCH=0 PW_EL_CHUNK="${PW_EL_CHUNK:-128}" PW_DELTA_MULT=1 PW_INIT_SPIN=random
export PW_WIN_SHAPE=smooth7 PW_TRIM_POW=4 PW_LJ_NETSCALE=1 PW_LJ_JASTROW=1 PW_INPUT12_INIT=random
export PW_PRETRAIN="${PW_PRETRAIN:-4000}" PW_PRETRAIN_LR="${PW_PRETRAIN_LR:-0.001}"
export PW_SAVE_MIN="${PW_SAVE_MIN:-30}" PW_MAX_MINUTES="${PW_MAX_MINUTES:-1380}"   # 24 時間より短く（事前学習と焼きなましの時間は数えない）
export JAX_ENABLE_X64=0   # lj3 と同じ単精度
unset PW_LJ_HF PW_INPUT12 PW_TRIM_FIX
mkdir -p jobs/logs
echo "job ${JOB_ID:-local} on $(hostname)  tag=$TAG  batch=$PW_BATCH  points=$PW_MULTI  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 PYTHONPATH="$PWD" "$VENV/bin/python" -u periodicwave/configs/run_lambda_jellium.py \
  "$LJ_N" "$ITERS" "$TAG" > "jobs/logs/run_${TAG}_${JOB_ID:-local}.log" 2>&1
echo "exit=$? $(date)"
