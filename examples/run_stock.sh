#!/usr/bin/env bash
# Stock example: render the 14 patterns mined in the original "Stock Prices
# Clustering" notebook (270 trading days of normalized prices, sector labels
# per ticker). The mined result ships as examples/stock.pkl, so no mining step.
#
#   examples/run_stock.sh [OUT_PREFIX]          default OUT_PREFIX=examples/out/stock
#
# The JSON's per-chart "label_mix" lists the dominant sectors in each pattern.
set -euo pipefail
cd "$(dirname "$0")/.."
# python: $PYTHON, else a local virtualenv (.venv or tivy_env), else python3
PY=${PYTHON:-}
if [ -z "$PY" ]; then
    for c in .venv/bin/python tivy_env/bin/python; do [ -x "$c" ] && PY=$c && break; done
    PY=${PY:-python3}
fi
OUT=${1:-examples/out/stock}

"$PY" tivy_cli.py render examples/stock.pkl --out "$OUT" \
    --n-cols 4 --chart-width 180 --chart-height 90 --interpolate 200

echo
echo "done: open ${OUT}.png and ${OUT}_bands.png"
