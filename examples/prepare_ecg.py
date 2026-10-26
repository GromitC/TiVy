"""Build an .npz (data + labels) from the MIT-BIH heartbeat CSVs used in the ECG notebook.

    python examples/prepare_ecg.py CSV_DIR [OUT_NPZ]

CSV_DIR holds mitbih_train.csv and mitbih_test.csv from
https://www.kaggle.com/datasets/shayanfazeli/heartbeat
OUT_NPZ defaults to examples/out/ecg.npz. See run_ecg.sh for the full pipeline.
"""
import os
import sys

import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out", "ecg.npz")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

df = pd.concat((pd.read_csv(os.path.join(src, "mitbih_train.csv"), header=None),
                pd.read_csv(os.path.join(src, "mitbih_test.csv"), header=None)))
data = df.iloc[:, :180].to_numpy(dtype=np.float32)          # first 180 samples of each beat
tag = {0: "N", 1: "SVEB", 2: "VEB", 3: "F", 4: "Q"}
labels = np.array([tag[int(v)] for v in df.iloc[:, -1]])
np.savez(out, data=data, labels=labels)
print(f"wrote {out}: {data.shape[0]} beats x {data.shape[1]} samples,",
      dict(zip(*np.unique(labels, return_counts=True))))
