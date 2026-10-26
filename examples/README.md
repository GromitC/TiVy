# TiVy examples

Three ready-to-run examples. Each script mines (or loads) a dataset, renders
the summary offscreen and writes into `examples/out/` (git-ignored):

| file | what |
|---|---|
| `<name>.png` | line-density view: colored density cloud of all member series + black representative line |
| `<name>_bands.png` | quantile-band view: 90% band in color, full range in grey |
| `<name>.json` | per-chart numbers: label, n_series, time interval, y range, central line start/end |
| `<name>.pkl` | the mined result, re-renderable with `tivy_cli.py render` without mining again |

The picture is the same one the Jupyter widget shows, so it can be viewed
anywhere, including by an AI assistant. Scripts use `tivy_env/bin/python`
when that virtualenv exists, otherwise `python`; override with `PYTHON=...`.
Rendering opens a hidden window via glfw, so run from a desktop session.

```bash
examples/run_synthetic.sh                 # ~3 s, no data needed
examples/run_stock.sh                     # ~1 s, uses examples/stock.pkl
examples/run_ecg.sh path/to/mitbih_csvs   # ~5 s + CSV load, needs the Kaggle files
```

## Synthetic (`run_synthetic.sh`)

Generates five groups of 200-point series (cyclic, increasing, decreasing,
upward shift, downward shift), mines them with the README defaults (window
50, support 10, length 50) and lays the patterns out one row per group.
The shift groups split into several patterns because the shift happens at
different times.

## Stock prices (`run_stock.sh`)

`stock.pkl` is the already-mined result from the original "Stock Prices
Clustering" notebook: 14 patterns over 270 trading days, up to 1002 tickers
per pattern. Each pattern carries one sector label per ticker; the JSON
`label_mix` field lists the top sectors in each chart and the CLI prints the
dominant one.

## ECG, MIT-BIH heartbeats (`run_ecg.sh CSV_DIR`)

Download `mitbih_train.csv` and `mitbih_test.csv` from
<https://www.kaggle.com/datasets/shayanfazeli/heartbeat>. The script builds
`out/ecg.npz` (first 180 samples of each of the 109,446 beats plus class
labels) and runs TiVy with the notebook's parameters (window 18, support
100, length 100, size 500). Rows are the AAMI classes N, SVEB, VEB, F, Q.
The flat tail at zero is the dataset's zero padding for short beats. Note
`out/ecg.pkl` is ~150 MB because it stores every member series.

## Re-rendering and options

Any `.pkl` written by `run` or `demo` can be re-rendered with new chart
options, for example a denser, lines-only view:

```bash
tivy_env/bin/python tivy_cli.py render examples/out/synthetic.pkl --out /tmp/s \
    --mode lines --opacity 0.9 --n-cols 5 --central-line-px 1 3
```

Options: `--mode lines|bands|both`, `--opacity` (darkness of each chart's
dense core), `--density auto|linear` (`linear` is the widget's raw count and
saturates on large charts), `--n-cols`, `--chart-width/height`,
`--no-central-line`, `--central-line-px THIN THICK`, `--labels-order`.
Run `tivy_cli.py run --help` for the mining options.

From Python the same picture comes from `tivy.visualize(clusters, path=...)`
after `optimize_patterns`, or from `TSVis.offscreen.TimeSeriesVisOffscreen`
directly.
