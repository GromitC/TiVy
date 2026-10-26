#!/usr/bin/env bash
# Synthetic example: generate five shape classes, mine them, render the summary.
#
#   examples/run_synthetic.sh [OUT_PREFIX]      default OUT_PREFIX=examples/out/synthetic
#
# Writes OUT_PREFIX.png (line density), OUT_PREFIX_bands.png (quantile bands),
# OUT_PREFIX.json (per-chart numbers) and OUT_PREFIX.pkl (re-renderable result).
set -euo pipefail
cd "$(dirname "$0")/.."
# python: $PYTHON, else a local virtualenv (.venv or tivy_env), else python3
PY=${PYTHON:-}
if [ -z "$PY" ]; then
    for c in .venv/bin/python tivy_env/bin/python; do [ -x "$c" ] && PY=$c && break; done
    PY=${PY:-python3}
fi
OUT=${1:-examples/out/synthetic}

"$PY" tivy_cli.py demo --out "$OUT" --n-cols 4 \
    --labels-order cyclic,increasing,decreasing,upward_shift,downward_shift

echo
echo "done: open ${OUT}.png and ${OUT}_bands.png"
