#!/bin/bash
#SBATCH --job-name=pw-window
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=pw-window-%A_%a.out
set -euo pipefail
# partition/account/memory、CUDA module等は大学の設定に従いsbatch引数/起動前に指定する。
# PW_PYTHONはJAX GPU環境のpython。PW_PLANはsuite.pyで保存した絶対パス。
: "${PW_PLAN:?PW_PLAN is required}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
"${PW_PYTHON:-python}" -m periodicwave.window_experiments.suite \
  --plan "$PW_PLAN" --execute-task "$SLURM_ARRAY_TASK_ID"
