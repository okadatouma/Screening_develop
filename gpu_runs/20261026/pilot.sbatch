#!/bin/bash
#SBATCH --job-name=screen-pilot
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/tomax/projects/Screening_topology_gpu/gpu_runs/20261026/logs/pilot-%j.log
set -euo pipefail
cd /home/tomax/projects/Screening_topology_gpu
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PW_OPTIMIZER=adam PW_LR=.002 PW_BATCH=64 PW_EL_CHUNK=8 PW_FLIP_BATCH=6
export PW_HEADS=2 PW_ATTN_DIM=8 PW_VALUE_DIM=8 PW_WIDTH=32 PW_DET=1 PW_LAYERS=1
export PW_MCMC_STEPS=10 PW_BURN_IN=32 PW_SAVE_MIN=15 PW_SEED=101 PW_CHECK_NAN=1
export PW_WIN_SHAPE=smooth7 PW_TRIM_POW=4 PW_TOPO_MODE=off PW_TOPO_GAMMA=0
export PW_LJ_HF="$PWD/gpu_runs/20261026/hf/hf_N36_lam0.33_wc.txt"
export PW_OUTPUT_DIR="$PWD/gpu_runs/20261026/pilot-${SLURM_JOB_ID}"
/home/tomax/venvs/screening-gpu/bin/python -u -m periodicwave.configs.run_lambda_jellium 36 15 .33 3 gpu-pilot
