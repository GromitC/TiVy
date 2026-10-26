#!/usr/bin/env bash
# Maintainers only: refresh the standalone copies in this skill folder from the
# repo root (TiVy.py, TSVis/, tivy_cli.py, requirements.txt, example scripts).
# The repo root is the source of truth; never edit lib/ by hand.
set -euo pipefail
SKILL="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SKILL/../.." && pwd)"
mkdir -p "$SKILL/lib/TSVis" "$SKILL/examples"
cp "$REPO/TiVy.py" "$SKILL/lib/"
cp "$REPO/TSVis/"*.py "$SKILL/lib/TSVis/"
cp "$REPO/tivy_cli.py" "$REPO/requirements.txt" "$SKILL/"
cp "$REPO/examples/"run_*.sh "$REPO/examples/prepare_ecg.py" "$REPO/examples/stock.pkl" "$SKILL/examples/"
echo "synced skill copies from $REPO"
