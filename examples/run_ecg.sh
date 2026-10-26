#!/usr/bin/env bash
# ECG example: all 109,446 MIT-BIH heartbeats, 5 AAMI classes, same parameters
# as the "ECG Clustering" notebook. Takes ~5 s end to end.
#
#   examples/run_ecg.sh CSV_DIR [OUT_PREFIX]    default OUT_PREFIX=examples/out/ecg
#
# CSV_DIR must contain mitbih_train.csv and mitbih_test.csv from
# https://www.kaggle.com/datasets/shayanfazeli/heartbeat
# Rows: N normal, SVEB supraventricular ectopic, VEB ventricular ectopic,
# F fusion, Q unknown. The flat tail at zero is the dataset's zero padding.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$([ -x tivy_env/bin/python ] && echo tivy_env/bin/python || echo python)}
CSV_DIR=${1:?usage: examples/run_ecg.sh CSV_DIR [OUT_PREFIX]}
OUT=${2:-examples/out/ecg}
NPZ="$(dirname "$OUT")/ecg.npz"

if [ ! -f "$NPZ" ]; then
    "$PY" examples/prepare_ecg.py "$CSV_DIR" "$NPZ"
fi

"$PY" tivy_cli.py run "$NPZ" --out "$OUT" \
    --chunk-size 18 --nperm 4 --bandwidth 1 --minlen 10 \
    --min-support 100 --min-length 100 --min-size 500 \
    --n-cols 6 --chart-width 140 --chart-height 70 --interpolate 100 \
    --labels-order N,SVEB,VEB,F,Q --opacity 0.85

echo
echo "done: open ${OUT}.png and ${OUT}_bands.png"
