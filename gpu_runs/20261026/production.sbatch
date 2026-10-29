#!/bin/bash
#SBATCH --job-name=screen80k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --output=/home/tomax/projects/Screening_topology_gpu/gpu_runs/20261026/logs/production-%A_%a.log
set -euo pipefail
cd /home/tomax/projects/Screening_topology_gpu
export PYTHONPATH="$PWD"
/home/tomax/venvs/screening-gpu/bin/python -u gpu_runs/20261026/run_task.py "${SLURM_ARRAY_TASK_ID}"
