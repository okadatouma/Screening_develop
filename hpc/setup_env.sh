#!/bin/bash
# GPU の計算機に Python の環境を作る（1 回だけ）。リポジトリの根で：
#   bash hpc/setup_env.sh [<環境の置き場所、既定 $HOME/venv_pw>]
# Python 3.11 以上が要る（jax 0.7.2）。別の python を使うときは PYTHON=<python> bash hpc/setup_env.sh。
# TSUBAME4.0 は標準の python3 が 3.9 なので、先に miniconda で 3.11 の環境を作る（hpc/README.md の「準備」）。
# jax[cuda12] は CUDA のライブラリを pip で入れるので、計算機に要るのは NVIDIA のドライバだけ。
set -euo pipefail
VENV="${1:-$HOME/venv_pw}"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || { echo "Python 3.11 以上が要る"; exit 1; }
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$(dirname "$0")/requirements_gpu.txt"
"$VENV/bin/python" -c 'import jax; print("jax", jax.__version__, "devices", jax.devices())'
echo "ok: $VENV"
