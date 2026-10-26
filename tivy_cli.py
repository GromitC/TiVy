#!/usr/bin/env python
"""Command-line front end for TiVy that produces files instead of a notebook widget.

    python tivy_cli.py run  DATA --out PREFIX [mining options] [chart options]
    python tivy_cli.py render RESULT.pkl --out PREFIX [chart options]
    python tivy_cli.py demo --out PREFIX

Outputs (for PREFIX = out/summary):
    out/summary.png          line-density summary (what the widget shows)
    out/summary_bands.png    quantile-band view            (--mode both/bands)
    out/summary.json         machine-readable description of every chart
    out/summary.pkl          the mined result, re-renderable with `render`

DATA may be .npy / .npz, .csv / .tsv / .txt (rows = series, no header unless
--header), or .pkl / .pickle holding a 2-D array or a dict with keys
"data" and optionally "labels".
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo layout: TiVy.py / TSVis next to this file. Standalone skill layout: under lib/.
for _p in (HERE, os.path.join(HERE, "lib")):
    if os.path.exists(os.path.join(_p, "TiVy.py")):
        sys.path.insert(0, _p)
        break


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_data(path, header=False, label_col=None, transpose=False, max_length=None):
    ext = os.path.splitext(path)[1].lower()
    labels = None
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
    elif ext == ".npz":
        z = np.load(path, allow_pickle=True)
        arr = z[z.files[0]]
        if "labels" in z.files:
            labels = z["labels"]
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict):
            arr = np.asarray(obj["data"])
            labels = obj.get("labels")
        else:
            arr = np.asarray(obj)
    elif ext in (".csv", ".tsv", ".txt"):
        import pandas as pd
        sep = "\t" if ext == ".tsv" else (None if ext == ".txt" else ",")
        df = pd.read_csv(path, header=0 if header else None, sep=sep, engine="python")
        arr = df.to_numpy()
    else:
        raise SystemExit(f"unsupported data format: {ext}")

    arr = np.asarray(arr)
    if arr.dtype == object:
        arr = np.array([np.asarray(r, dtype=float) for r in arr])
    if transpose:
        arr = arr.T
    if label_col is not None:
        labels = arr[:, label_col]
        arr = np.delete(arr, label_col, axis=1)
    arr = arr.astype(float)
    if arr.ndim != 2:
        raise SystemExit(f"expected a 2-D array (series x time), got shape {arr.shape}")
    if max_length is not None:
        arr = arr[:, :max_length]
    if labels is not None:
        labels = np.asarray(labels)
        if np.issubdtype(labels.dtype, np.floating) and np.all(np.mod(labels, 1) == 0):
            labels = labels.astype(int)
        labels = labels.astype(str)
    return arr, labels


# ---------------------------------------------------------------------------
# mining
# ---------------------------------------------------------------------------
def mine(data, labels, a):
    from TiVy import TiVy

    if data.shape[1] < a.chunk_size:
        raise SystemExit(f"chunk_size {a.chunk_size} is larger than the series length {data.shape[1]}")
    if np.isnan(data).any():
        raise SystemExit("input contains NaN; TiVy needs complete series (fill or trim first)")

    timings = {}
    t = time.time()
    tivy = TiVy(chunk_size=a.chunk_size, strength=a.strength, use_LSH=not a.no_lsh,
                nPerm=a.nperm, bandWidth=a.bandwidth, random_seed=a.seed)
    tivy.cluster_subsequences(data)
    timings["cluster_subsequences_s"] = round(time.time() - t, 2)

    t = time.time()
    scan = tivy.prefix_scan(tivy.clusters, tivy.ts_start, minlen=a.minlen)
    clusters = tivy.optimize_patterns(scan["sequenceClusterDB"], scan["counter"],
                                      minSupport=a.min_support, minLength=a.min_length,
                                      minSize=a.min_size, maxLength=a.max_length_pattern,
                                      labels=labels)
    timings["scan_optimize_s"] = round(time.time() - t, 2)

    if len(clusters) == 0:
        raise SystemExit("no patterns found; lower --min-support / --min-size / --min-length "
                         "or change --chunk-size")

    t = time.time()
    result = tivy.output_result(clusters, show_main_trend=not a.no_central_line, return_idx=True)
    timings["output_result_s"] = round(time.time() - t, 2)

    result["params"] = {
        "chunk_size": a.chunk_size, "strength": a.strength, "use_LSH": not a.no_lsh,
        "nPerm": a.nperm, "bandWidth": a.bandwidth, "minlen": a.minlen,
        "minSupport": a.min_support, "minLength": a.min_length, "minSize": a.min_size,
        "n_series": int(data.shape[0]), "n_timepoints": int(data.shape[1]),
    }
    result["timings"] = timings
    return result


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render(result, prefix, a):
    from TSVis.offscreen import TimeSeriesVisOffscreen

    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    labels = result.get("labels") or None
    if labels is not None and len(labels) == 0:
        labels = None
    # Some saved results (e.g. the stock example) carry one label per *series*
    # inside each chart rather than one label per chart. The widget only lays
    # out per-chart labels, so summarise those into the JSON instead.
    label_mix = None
    if labels is not None and any(np.ndim(l) > 0 for l in labels):
        label_mix = []
        for l in labels:
            vals, cnt = np.unique(np.asarray(l).astype(str), return_counts=True)
            order = np.argsort(-cnt)[:5]
            label_mix.append({str(vals[i]): int(cnt[i]) for i in order})
        labels = None
    central = result.get("center_line") if not a.no_central_line else None
    labels_order = a.labels_order.split(",") if (a.labels_order and labels is not None) else None

    t = time.time()
    vis = TimeSeriesVisOffscreen(n_cols=a.n_cols, padding=a.padding, n_ticks=a.n_ticks,
                                 chart_width=a.chart_width, chart_height=a.chart_height,
                                 interpolate=a.interpolate, central_line_px=tuple(a.central_line_px))
    vis.fit(result["result"], labels=labels, labels_order=labels_order, central_line=central)
    fit_s = round(time.time() - t, 2)

    files = {}
    t = time.time()
    if a.mode in ("lines", "both"):
        p = prefix + ".png"
        vis.render(p, show_lines=True, opacity=a.opacity, density=a.density)
        files["lines_png"] = p
    if a.mode in ("bands", "both"):
        p = prefix + ("_bands.png" if a.mode == "both" else ".png")
        vis.render(p, show_lines=False)
        files["bands_png"] = p
    render_s = round(time.time() - t, 2)

    summary = vis.summary()
    summary["params"] = result.get("params", {})
    summary["timings"] = dict(result.get("timings", {}), fit_s=fit_s, render_s=render_s)
    summary["files"] = files
    if "idx" in result:
        for row, idx in zip(summary["charts"], result["idx"]):
            row["series_idx_sample"] = [int(i) for i in list(idx)[:20]]
    if label_mix is not None:
        for row, mix in zip(summary["charts"], label_mix):
            row["label_mix"] = mix
    with open(prefix + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
    files["json"] = prefix + ".json"
    return summary


def print_summary(s):
    print(f"TiVy summary: {s['n_charts']} pattern charts, grid {s['n_rows']}x{s['n_cols']}, "
          f"time axis 0..{s['x_limit']}, y range {s['y_limit'][0]:.3g}..{s['y_limit'][1]:.3g}")
    for c in s["charts"]:
        lab = f" [{c['label']}]" if "label" in c else ""
        if "label_mix" in c:
            top = next(iter(c["label_mix"].items()))
            lab = f" [mostly {top[0]} {top[1]}/{c['n_series']}]"
        trend = ""
        if "central_line_change" in c:
            trend = f", central line {c['central_line_start']:.3g} -> {c['central_line_end']:.3g}"
        print(f"  chart {c['chart']:>2}{lab}: {c['n_series']:>5} series, t={c['interval'][0]}..{c['interval'][1]}"
              f", y {c['y_min']:.3g}..{c['y_max']:.3g}{trend}")
    for k, v in s["files"].items():
        print(f"  {k}: {v}")
    print("  timings:", ", ".join(f"{k}={v}" for k, v in s["timings"].items()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_chart_args(p):
    g = p.add_argument_group("chart options")
    g.add_argument("--mode", choices=["lines", "bands", "both"], default="both",
                   help="lines = line density (widget default); bands = 90%% / 100%% quantile bands")
    g.add_argument("--n-cols", type=int, default=5)
    g.add_argument("--chart-width", type=int, default=260)
    g.add_argument("--chart-height", type=int, default=140)
    g.add_argument("--padding", type=int, default=50)
    g.add_argument("--n-ticks", type=int, default=3)
    g.add_argument("--interpolate", type=int, default=500, help="density-map x resolution")
    g.add_argument("--opacity", type=float, default=0.6,
                   help="lines view: darkness of each chart's dense core (auto density)")
    g.add_argument("--density", choices=["auto", "linear"], default="auto",
                   help="auto scales per chart (default); linear is the widget's raw count")
    g.add_argument("--central-line-px", type=float, nargs=2, default=(1.5, 3.5),
                   metavar=("THIN", "THICK"),
                   help="central line thickness in px for the smallest and largest chart")
    g.add_argument("--no-central-line", action="store_true")
    g.add_argument("--labels-order", default=None, help="comma-separated row order for labels")


def add_data_args(p):
    g = p.add_argument_group("input options")
    g.add_argument("--header", action="store_true", help="CSV has a header row")
    g.add_argument("--label-col", type=int, default=None,
                   help="column index holding a per-series label (e.g. -1 for last)")
    g.add_argument("--labels", default=None, help="separate file (.npy/.csv/.txt) with one label per series")
    g.add_argument("--no-labels", action="store_true",
                   help="ignore labels found in the data file (mine all series together)")
    g.add_argument("--transpose", action="store_true", help="input is time x series")
    g.add_argument("--max-length", type=int, default=None, help="truncate series to this many points")


def add_mining_args(p):
    g = p.add_argument_group("mining options (TiVy)")
    g.add_argument("--chunk-size", type=int, default=50, help="window size; must be <= series length")
    g.add_argument("--strength", type=float, default=1.0, help="clustering strength")
    g.add_argument("--no-lsh", action="store_true", help="disable locality-sensitive hashing")
    g.add_argument("--nperm", type=int, default=20, help="LSH permutations")
    g.add_argument("--bandwidth", type=float, default=0.5, help="LSH bandwidth")
    g.add_argument("--minlen", type=int, default=1, help="minimum sequence length in the prefix scan")
    g.add_argument("--min-support", type=int, default=10)
    g.add_argument("--min-length", type=int, default=50, help="minimum pattern length (time points)")
    g.add_argument("--max-length-pattern", type=int, default=None)
    g.add_argument("--min-size", type=int, default=1, help="minimum number of series in a pattern")
    g.add_argument("--seed", type=int, default=42)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="mine patterns from a data file and render them")
    p_run.add_argument("data")
    p_run.add_argument("--out", required=True, help="output prefix, e.g. out/summary")
    add_data_args(p_run); add_mining_args(p_run); add_chart_args(p_run)

    p_ren = sub.add_parser("render", help="re-render a saved .pkl result with new chart options")
    p_ren.add_argument("result")
    p_ren.add_argument("--out", required=True)
    add_chart_args(p_ren)

    p_demo = sub.add_parser("demo", help="generate synthetic data, mine it, render it")
    p_demo.add_argument("--out", required=True)
    p_demo.add_argument("--n-per-group", type=int, default=100)
    p_demo.add_argument("--length", type=int, default=200)
    add_mining_args(p_demo); add_chart_args(p_demo)

    a = ap.parse_args(argv)

    if a.cmd == "render":
        with open(a.result, "rb") as fh:
            result = pickle.load(fh)
        print_summary(render(result, a.out, a))
        return

    if a.cmd == "demo":
        from TSVis.synthetic_data import generateSyntheticData
        np.random.seed(a.seed)
        groups = generateSyntheticData(a.n_per_group, a.length)
        names = ["decreasing", "increasing", "downward_shift", "cyclic", "upward_shift"]
        data = np.concatenate([g / np.max(np.abs(g)) * 30 for g in groups])
        labels = np.concatenate([[n] * len(g) for n, g in zip(names, groups)])
        print(f"demo data: {data.shape[0]} series x {data.shape[1]} points, groups: {names}")
    else:
        data, labels = load_data(a.data, a.header, a.label_col, a.transpose, a.max_length)
        if a.labels:
            lab, _ = load_data(a.labels, False, None, False, None) if a.labels.endswith(".npy") else (None, None)
            if lab is None:
                lab = np.loadtxt(a.labels, dtype=str, delimiter=",").ravel()
            labels = np.asarray(lab).ravel().astype(str)
        if a.no_labels:
            labels = None
        if labels is not None and len(labels) != data.shape[0]:
            raise SystemExit(f"{len(labels)} labels for {data.shape[0]} series")
        print(f"loaded {data.shape[0]} series x {data.shape[1]} points"
              + (f", {len(np.unique(labels))} label classes" if labels is not None else ""))

    result = mine(data, labels, a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".pkl", "wb") as fh:
        pickle.dump(result, fh)
    summary = render(result, a.out, a)
    summary["files"]["result_pkl"] = a.out + ".pkl"
    print_summary(summary)


if __name__ == "__main__":
    main()
