"""Parameter sweep for TiVy on synthetic / stock / ecg.

    .venv/bin/python -I sweep/sweep.py synthetic stock ecg     (from the skill folder)
    .venv/bin/python -I sweep/sweep.py --resheet stock         (rebuild sheets from stats.csv)

For each dataset: one-at-a-time deviations from a baseline (DATASETS below).
Clustering runs once per clustering setting and is reused for all
scan/optimize settings. Outputs go to out/sweep/<ds>/<tag>.png, stats.csv and
per-parameter contact sheets out/sweep/sheets/<ds>__<param>.png.
The ecg dataset needs examples/out/ecg.npz (see examples/run_ecg.sh).
"""
import csv, json, os, random, sys, time, warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SKILL, "lib"))          # standalone skill layout
sys.path.insert(0, os.path.dirname(SKILL))              # repo layout (skills/tivy/sweep -> repo root)
from TiVy import TiVy                               # noqa: E402
from TSVis.offscreen import TimeSeriesVisOffscreen  # noqa: E402

OUT = os.path.join(SKILL, "out", "sweep")
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
except OSError:
    try:
        FONT = ImageFont.truetype("DejaVuSans.ttf", 26)
    except OSError:
        FONT = ImageFont.load_default()

CLUSTER_KEYS = {"chunk_size", "strength", "use_LSH", "nPerm", "bandWidth", "B", "max_iter", "LSH_sample", "steps"}

DATASETS = {
    "synthetic": dict(
        base=dict(chunk_size=50, strength=1, use_LSH=True, nPerm=20, bandWidth=0.5, B=20, max_iter=None,
                  LSH_sample=1, steps=0, minlen=1, minSupport=10, minLength=50, minSize=1, labels=False),
        sweep=dict(chunk_size=[20, 25, 40, 50, 100], strength=[0.5, 0.8, 1, 1.5, 2, 3],
                   nPerm=[2, 4, 8, 20, 40], bandWidth=[0.25, 0.5, 1, 2, 4], use_LSH=[False],
                   LSH_sample=[3], B=[5], max_iter=[2, 4], minlen=[5, 20],
                   minSupport=[1, 5, 20, 40], minLength=[0, 100, 150], minSize=[0, 10, 30],
                   maxLength=[100], maxSize=[30], labels=[True]),
        vis=dict(n_cols=6, chart_width=200, chart_height=100, padding=45, interpolate=300)),
    "stock": dict(
        base=dict(chunk_size=30, strength=1, use_LSH=True, nPerm=20, bandWidth=0.5, B=20, max_iter=None,
                  LSH_sample=1, steps=0, minlen=10, minSupport=50, minLength=60, minSize=50, labels=False),
        sweep=dict(chunk_size=[10, 15, 30, 54, 90], strength=[0.5, 0.8, 1, 1.5, 2, 3],
                   nPerm=[2, 4, 8, 20, 40], bandWidth=[0.25, 0.5, 1, 2, 4],
                   LSH_sample=[3], B=[5], max_iter=[2, 4], minlen=[1, 50],
                   minSupport=[10, 20, 100, 300], minLength=[0, 30, 120, 180], minSize=[0, 10, 200],
                   maxLength=[120], maxSize=[200], labels=[True]),
        vis=dict(n_cols=6, chart_width=200, chart_height=100, padding=45, interpolate=300)),
    "ecg": dict(
        base=dict(chunk_size=18, strength=1, use_LSH=True, nPerm=4, bandWidth=1, B=20, max_iter=None,
                  LSH_sample=1, steps=0, minlen=10, minSupport=100, minLength=100, minSize=500, labels=False),
        sweep=dict(chunk_size=[9, 12, 18, 30, 45, 60], strength=[0.5, 0.8, 1, 1.5, 2, 3],
                   nPerm=[2, 4, 8, 20], bandWidth=[0.25, 0.5, 1, 2, 4],
                   LSH_sample=[3], B=[5], max_iter=[2, 4], minlen=[1, 100],
                   minSupport=[10, 1000, 5000], minLength=[0, 36, 72, 144], minSize=[0, 100, 2000, 10000],
                   maxLength=[90], maxSize=[5000], labels=[True]),
        vis=dict(n_cols=6, chart_width=200, chart_height=100, padding=45, interpolate=150)),
}


def load_stock_raw():
    """Full-interval members of the shipped stock result = complete raw series."""
    import pickle
    r = pickle.load(open(os.path.join(SKILL, "examples", "stock.pkl"), "rb"))
    rows, labs = [], []
    for ts, lab in zip(r["result"], r["labels"]):
        full = ~np.isnan(ts).any(axis=1)
        rows.append(ts[full]); labs.append(np.asarray(lab)[full])
    X, L = np.concatenate(rows), np.concatenate(labs)
    _, idx = np.unique(np.round(X, 6), axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx].astype(float), L[idx].astype(str)


def load(ds):
    if ds == "synthetic":
        from TSVis.synthetic_data import generateSyntheticData
        np.random.seed(0)
        groups = generateSyntheticData(100, 200)
        names = ["decreasing", "increasing", "downward_shift", "cyclic", "upward_shift"]
        X = np.concatenate([g / np.max(np.abs(g)) * 30 for g in groups])
        L = np.concatenate([[n] * len(g) for n, g in zip(names, groups)])
        order = ["cyclic", "increasing", "decreasing", "upward_shift", "downward_shift"]
    elif ds == "stock":
        X, L = load_stock_raw()
        order = None
    else:
        z = np.load(os.path.join(SKILL, "examples", "out", "ecg.npz"), allow_pickle=True)  # examples/run_ecg.sh builds it
        X, L = z["data"].astype(float), z["labels"].astype(str)
        order = ["N", "SVEB", "VEB", "F", "Q"]
    return X, L, order


def cluster(X, cfg):
    random.seed(0); np.random.seed(42)
    t = time.time()
    tivy = TiVy(chunk_size=cfg["chunk_size"], strength=cfg["strength"], use_LSH=cfg["use_LSH"],
                nPerm=cfg["nPerm"], bandWidth=cfg["bandWidth"], B=cfg["B"], max_iter=cfg["max_iter"],
                LSH_sample=cfg["LSH_sample"], steps=cfg["steps"])
    tivy.cluster_subsequences(X)
    dt = time.time() - t
    per_window = [len(np.unique(c[c != 0])) for c in tivy.clusters]
    return tivy, dt, per_window


def mine(tivy, cfg, L):
    random.seed(0); np.random.seed(42)
    t = time.time()
    scan = tivy.prefix_scan(tivy.clusters, tivy.ts_start, minlen=cfg["minlen"])
    n_cand = len(scan["counter"])
    clusters = tivy.optimize_patterns(scan["sequenceClusterDB"], scan["counter"],
                                      minSupport=cfg["minSupport"], minLength=cfg["minLength"],
                                      maxLength=cfg.get("maxLength"), minSize=cfg["minSize"],
                                      maxSize=cfg.get("maxSize"), labels=L if cfg["labels"] else None)
    dt = time.time() - t
    return clusters, n_cand, dt


def render(tivy, clusters, cfg, vis_kw, path, order):
    if len(clusters) == 0:
        img = Image.new("RGB", (1500, 160), "white")
        ImageDraw.Draw(img).text((20, 60), "no patterns", fill="gray", font=FONT)
        img.save(path); return
    res = tivy.output_result(clusters, return_idx=True)
    labels = res.get("labels") or None
    vis = TimeSeriesVisOffscreen(n_ticks=3, **vis_kw)
    vis.fit(res["result"], labels=labels, labels_order=order if labels else None, central_line=res["center_line"])
    vis.render(path, opacity=0.75)


def stats_row(ds, param, value, cfg, n_series, n_win, per_window, n_cand, clusters, X, t_cl, t_mine):
    sizes = [len(c[0]) for c in clusters]
    lens = [c[2] - c[1] for c in clusters]
    covered = set()
    for c in clusters:
        covered |= set(c[0])
    return dict(dataset=ds, param=param, value=str(value), n_windows=n_win,
                clusters_per_window=round(float(np.mean(per_window)), 1),
                max_clusters_window=int(np.max(per_window)), candidates=n_cand,
                n_patterns=len(clusters), coverage=round(len(covered) / n_series, 2),
                size_med=int(np.median(sizes)) if sizes else 0, size_max=int(max(sizes)) if sizes else 0,
                len_med=int(np.median(lens)) if lens else 0, len_min=int(min(lens)) if lens else 0,
                len_max=int(max(lens)) if lens else 0,
                t_cluster=round(t_cl, 1), t_mine=round(t_mine, 1), cfg=json.dumps(cfg))


def run(ds):
    spec = DATASETS[ds]
    X, L, order = load(ds)
    os.makedirs(f"{OUT}/{ds}", exist_ok=True)
    rows, cache = [], {}

    def cluster_key(cfg):
        return tuple((k, cfg[k]) for k in sorted(CLUSTER_KEYS))

    configs = [("baseline", "base", dict(spec["base"]))]
    for p, vals in spec["sweep"].items():
        for v in vals:
            cfg = dict(spec["base"]); cfg[p] = v
            if p in ("chunk_size",) and ds == "ecg" and v != spec["base"][p]:
                pass
            configs.append((p, v, cfg))

    for param, value, cfg in configs:
        tag = f"{param}={value}".replace("/", "_")
        if cfg.get("minLength", 0) and cfg["minLength"] < cfg["chunk_size"] and param == "chunk_size":
            pass
        key = cluster_key(cfg)
        if key not in cache:
            print(f"[{ds}] clustering for {tag} ...", flush=True)
            cache[key] = cluster(X, cfg)
        tivy, t_cl, per_window = cache[key]
        clusters, n_cand, t_mine = mine(tivy, cfg, L)
        path = f"{OUT}/{ds}/{tag}.png"
        try:
            render(tivy, clusters, cfg, spec["vis"], path, order)
        except Exception as e:  # keep sweeping
            print(f"   render failed for {tag}: {e}", flush=True)
        row = stats_row(ds, param, value, cfg, X.shape[0], len(tivy.clusters), per_window, n_cand, clusters, X, t_cl, t_mine)
        rows.append(row)
        print(f"[{ds}] {tag:22s} win={row['n_windows']:3d} cl/win={row['clusters_per_window']:5.1f} cand={row['candidates']:6d} "
              f"patterns={row['n_patterns']:4d} cover={row['coverage']:.2f} size_med={row['size_med']:6d} "
              f"len_med={row['len_med']:4d} t={row['t_cluster']+row['t_mine']:.1f}s", flush=True)

    with open(f"{OUT}/{ds}/stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    make_sheets(ds, rows, spec)


def make_sheets(ds, rows, spec):
    os.makedirs(f"{OUT}/sheets", exist_ok=True)
    base_row = rows[0]
    W, MAXH = 1500, 520
    for param in spec["sweep"]:
        sel = [r for r in rows if r["param"] == param]
        # include the baseline value in its sorted place
        if str(spec["base"].get(param)) not in [r["value"] for r in sel] and param in spec["base"]:
            b = dict(base_row); b["param"] = param; b["value"] = str(spec["base"][param]); b["_tag"] = "baseline=base"
            sel.append(b)
        def sortkey(r):
            try: return float(r["value"]) if r["value"] not in ("None", "True", "False") else {"None": 1e9, "True": 1, "False": 0}[r["value"]]
            except ValueError: return 0
        sel.sort(key=sortkey)
        tiles = []
        for r in sel:
            tag = r.get("_tag") or f"{param}={r['value']}"
            p = f"{OUT}/{ds}/{tag}.png"
            img = Image.open(p).convert("RGB") if os.path.exists(p) else Image.new("RGB", (W, 100), "white")
            s = W / img.width
            img = img.resize((W, max(1, int(img.height * s))), Image.LANCZOS)
            note = ""
            if img.height > MAXH:
                img = img.crop((0, 0, W, MAXH)); note = f"  (cropped: {r['n_patterns']} charts)"
            is_base = str(spec["base"].get(param)) == r["value"]
            title = (f"{param} = {r['value']}{'  [baseline]' if is_base else ''}   |   {r['n_windows']} windows, "
                     f"{r['clusters_per_window']} clusters/window, {r['candidates']} candidates -> {r['n_patterns']} patterns, "
                     f"coverage {r['coverage']}, median size {r['size_med']}, median length {r['len_med']}, "
                     f"{float(r['t_cluster']) + float(r['t_mine']):.1f}s{note}")
            tile = Image.new("RGB", (W, img.height + 44), "white")
            d = ImageDraw.Draw(tile)
            d.rectangle((0, 0, W, 40), fill=(235, 235, 230))
            d.text((12, 7), title, fill=(20, 20, 20), font=FONT)
            tile.paste(img, (0, 44))
            tiles.append(tile)
        H = sum(t.height for t in tiles) + 6 * (len(tiles) - 1)
        sheet = Image.new("RGB", (W, H), (120, 120, 120))
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y)); y += t.height + 6
        sheet.save(f"{OUT}/sheets/{ds}__{param}.png")
    print(f"[{ds}] sheets written", flush=True)


def resheet(ds):
    rows = list(csv.DictReader(open(f"{OUT}/{ds}/stats.csv")))
    make_sheets(ds, rows, DATASETS[ds])


if __name__ == "__main__":
    args = sys.argv[1:] or ["synthetic", "stock", "ecg"]
    if args[0] == "--resheet":
        for ds in args[1:]:
            resheet(ds)
    else:
        for ds in args:
            run(ds)
