#!/usr/bin/env bash
# One-time setup for the standalone TiVy skill.
# Creates .venv next to this file, installs the pinned requirements, and renders
# the synthetic demo as a smoke test (out/setup_check.png).
#
#   ./setup.sh                 # uses python3
#   PYTHON=python3.12 ./setup.sh
#
# Needs: Python 3.10+, a C compiler (macOS: `xcode-select --install`;
# Debian/Ubuntu: `apt install build-essential`), and a desktop session for
# rendering (glfw opens a hidden OpenGL window; plain SSH without a display
# will not render).
#
# About dtaidistance: TiVy uses its C routine dtw.distance_matrix_fast. The
# PyPI wheel for recent Pythons ships without the C library, and a default
# source build (isolated) compiles against the newest numpy instead of the
# pinned one, which then fails at runtime ("module 'dtaidistance' has no
# attribute 'DTWSeriesMatrix'"). So it is rebuilt here from source with
# --no-build-isolation against the numpy that is actually installed.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

if [ ! -x .venv/bin/python ]; then
    "$PY" -m venv .venv
fi
PIP=".venv/bin/python -m pip"
$PIP install --quiet --upgrade pip
echo "installing pinned requirements ..."
$PIP install --quiet -r requirements.txt

echo "building dtaidistance against the installed numpy ..."
DTAI=$(grep -i '^dtaidistance==' requirements.txt)
$PIP install --quiet --force-reinstall --no-deps --no-build-isolation --no-binary dtaidistance --no-cache-dir "$DTAI"

.venv/bin/python - <<'EOF'
import numpy as np
from dtaidistance import dtw
assert dtw.try_import_c(), "dtaidistance C library not available"
d = dtw.distance_matrix_fast(np.random.rand(6, 20))      # exercises the numpy <-> C path
assert d.shape == (6, 6)
print("dtaidistance C library OK")
EOF

echo "running smoke test ..."
.venv/bin/python tivy_cli.py demo --out out/setup_check --mode lines --n-cols 4 \
    --labels-order cyclic,increasing,decreasing,upward_shift,downward_shift | tail -3
echo
echo "setup OK. Open out/setup_check.png to see the demo summary."
