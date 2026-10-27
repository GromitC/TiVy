---
name: tivy
description: Summarize a large collection of time series into a small grid of visual patterns with TiVy (Time Series Visual Summary), then view the result as a PNG and a JSON description. Use when the user asks to summarize, cluster, find common shapes or motifs in, or "visualize all of" many time series (hundreds to hundreds of thousands of rows), or mentions TiVy, time series summary, or subsequence clustering.
---

# TiVy: time series visual summary

TiVy mines a set of equal-length time series for frequently shared subsequence
patterns and draws each pattern as a small chart: a blue line-density map of the
member series, a black representative ("central") line, and the time interval
the pattern spans. This folder is self-contained: library code in `lib/`, a
CLI, examples, and a parameter guide. All commands below run **from this
skill's folder** (the directory containing this file).

## Setup (once per machine)

```bash
cd <this skill folder>
./setup.sh            # creates .venv, installs pinned requirements, renders out/setup_check.png
```

If `.venv/bin/python` is missing, run `setup.sh` first. It needs a C compiler:
it rebuilds the dtaidistance DTW extension against the pinned numpy. Without
that step TiVy fails with "compiled dtaidistance C library is not available"
or "module 'dtaidistance' has no attribute 'DTWSeriesMatrix'". Rendering
opens a hidden OpenGL window through glfw, so it needs a logged-in desktop
session.

## Run

```bash
.venv/bin/python tivy_cli.py demo --out out/demo                      # sanity check, no data
.venv/bin/python tivy_cli.py run DATA.npy --out out/summary           # real data
.venv/bin/python tivy_cli.py run data.csv --label-col -1 --out out/ecg \
    --chunk-size 18 --min-support 100 --min-length 100 --min-size 500 --n-cols 6
.venv/bin/python tivy_cli.py render out/summary.pkl --out out/v2 --mode lines --opacity 0.8
```

Input: `.npy`, `.npz` (keys `data`, optional `labels`), `.csv/.tsv/.txt`
(rows = series, no header unless `--header`), or `.pkl` (array, or dict with
`data` and `labels`). `--transpose` for time x series layouts, `--label-col N`
or `--labels FILE` for per-series class labels, `--no-labels` to ignore labels
in the file, `--max-length` to trim long series. NaNs are not accepted; fill
or trim first.

**X-axis labels (dates).** Tick labels show real time stamps instead of
indices whenever the data carries them: an `.npz`/pickle key named `dates`,
`x`, `time`, `timestamps` or `x_labels` with one entry per time point, or the
column header of a wide CSV read with `--header`. Otherwise pass
`--x-labels FILE` (one label per line, or `.npy`/`.npz`). Date-like labels
are shortened automatically (`--x-date-format auto`: same year -> `Jan 02`,
several years -> `2026-01`; or give a strftime pattern, or `none`). Labels
are stored in the `.pkl`, so `render` keeps them, and the JSON reports each
chart's `interval_labels` (start and end dates) next to the index interval.
When you prepare data yourself, save the dates: `np.savez(out, data=X,
labels=classes, dates=date_strings)`.

Outputs for `--out out/summary`:

| file | what |
|---|---|
| `out/summary.png` | line-density view, the widget default |
| `out/summary_bands.png` | 90% (blue) and full-range (grey) quantile bands |
| `out/summary.json` | per-chart: label, n_series, interval (+ `interval_labels` when dates are known), y range, central line start/end |
| `out/summary.pkl` | mined result, re-renderable with `render` without re-mining |

After a run, **Read the PNG** to look at the patterns and **Read the JSON** for
the numbers; report both to the user. Charts are ordered left to right, top to
bottom, matching `charts[i].chart` in the JSON. With labels, each label gets its
own row group with the label text at the left, and colors are a fixed
colorblind-safe palette (max 8 classes before colors repeat; the row label is
the primary identity cue).

## Parameters

**Before changing any mining parameter, read `parameters.md` in this folder.**
It explains the mechanism of every knob, gives measured effects on three
datasets, a symptom-to-knob table, and tested starting points per data size.
Side-by-side renders per parameter are in `sweep/sheets/` (Read them).

The short version:

- `--chunk-size` (window, default 50): pick the time scale of the features
  you want, with 4 to 10 windows per series. Patterns start and end on
  window boundaries.
- `--strength` (default 1): keep at 1. Below 1 explodes into one cluster per
  series (no patterns, minutes of runtime) unless LSH is coarse; above 1.5
  merges everything into one blob.
- `--min-support` and `--min-size`: set together, support >= size. Raise
  support first to get fewer, bigger charts without losing coverage.
- `--min-length`: drop short stubs; 2 to 3 windows is a good floor.
- `--nperm 8 --bandwidth 1` is the speed/quality sweet spot for thousands of
  series; use `--nperm 4 --bandwidth 1` for 100k series. `--no-lsh` only for
  a few hundred series.
- `--max-length-pattern` / `--max-size` drop patterns; they do not trim them.
- Chart options: `--n-cols`, `--chart-width/height`, `--opacity` (darkness of
  each chart's dense core; raise toward 0.85 when faint), `--density auto|linear`,
  `--mode lines|bands|both`, `--no-central-line`, `--central-line-px 1.5 3.5`,
  `--labels-order N,SVEB,VEB`, `--x-labels FILE`, `--x-date-format`. These
  never change what is mined; iterate with `render` on the saved `.pkl`.

## Interpreting

- Each chart is one pattern: a set of series that behave alike over the shown
  interval. Series outside the interval are blank; the interval is in the JSON.
- The black line is the DTW-average representative of up to 50 members; it
  gets thicker as the chart holds more series.
- `central_line_change` in the JSON gives the net rise or fall across the
  interval; `n_series` is the pattern's support.
- Cyclic shapes show as a tight oscillating band; level shifts as two
  plateaus; trends as a sloped band.

## Worked examples

```bash
examples/run_synthetic.sh                 # ~3 s, no data needed
examples/run_stock.sh                     # ~1 s, renders the shipped examples/stock.pkl
examples/run_ecg.sh path/to/mitbih_csvs   # ~5 s + CSV load; needs the Kaggle MIT-BIH files
```

Outputs land in `examples/out/`. Stock renders a saved result with per-series
sector labels (see `label_mix` in the JSON); ECG is 109k beats in 5 classes.
Reuse their parameters as starting points for similar data sizes.

## Library use (Python)

```python
import sys; sys.path.insert(0, "lib")
from TiVy import TiVy
tivy = TiVy(chunk_size=50, strength=1, use_LSH=True, nPerm=20, bandWidth=0.5)
tivy.cluster_subsequences(X)                         # X: 2-D array, series x time
scan = tivy.prefix_scan(tivy.clusters, tivy.ts_start, minlen=1)
clusters = tivy.optimize_patterns(scan['sequenceClusterDB'], scan['counter'],
                                  minSupport=10, minLength=50, minSize=1, labels=None)
out = tivy.visualize(clusters, path='out/summary.png', show=False)   # image + summary + result
# lower level: from TSVis.offscreen import TimeSeriesVisOffscreen
```

## Known quirks

- `TSVis.synthetic_data.generateSyntheticData` returns 5 groups (it drops the
  "normal" group it computes); the upstream README expects 6. The `demo`
  subcommand works around this.
- Results are not bit-reproducible run to run (NumPy and Python RNGs); seed
  both when comparing settings.
- More than ~40 charts gets hard to read in one image. Raise `--min-size` /
  `--min-support` or filter by label.
- `lib/` is a copy of the repo root refreshed by `sync_from_repo.sh`; edit
  the repo, not the copy.
