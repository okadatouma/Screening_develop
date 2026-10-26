#!/bin/sh
# Run from repository root. Manifest controls platform, sizes, seeds, and updates.
set -eu
PYTHON=${PW_PYTHON:-.venv-pw/bin/python}
MANIFEST=${1:-screening_analysis/topology/training.yaml}
OUTPUT=${2:-results/screening_topology/scaleup}
exec "$PYTHON" -m screening_analysis.topology.cli train --manifest "$MANIFEST" --output "$OUTPUT" --resume
