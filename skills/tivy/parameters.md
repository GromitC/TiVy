# TiVy parameter guide

How each TiVy parameter changes the mined patterns, with numbers from a sweep
over three datasets (26 Oct 2026). Read this before tuning. Raw numbers are in
`sweep/stats_*.csv`; side-by-side renders per parameter are in `sweep/sheets/`
(open them with Read to see the effect, e.g. `sweep/sheets/stock__minSupport.jpg`).

| dataset | series x points | baseline used in the sweep |
|---|---|---|
| synthetic | 232 x 200, 5 shape groups | chunk 50, strength 1, LSH nPerm 20 / bandWidth 0.5, minlen 1, minSupport 10, minLength 50, minSize 1 |
| stock | 2,939 x 270 daily normalized prices | chunk 30, strength 1, nPerm 20 / bandWidth 0.5, minlen 10, minSupport 50, minLength 60, minSize 50 |
| ecg | 109,446 x 180 heartbeats | chunk 18, strength 1, nPerm 4 / bandWidth 1, minlen 10, minSupport 100, minLength 100, minSize 500 |

## 1. How the pipeline works (what the knobs act on)

```
series (N x T)
  └─ cut into floor(T / chunk_size) non-overlapping windows           chunk_size
       └─ per window: cluster the N window-subsequences                strength, B, max_iter, steps
            (LSH buckets -> one sample per bucket -> gap-statistic     use_LSH, nPerm, bandWidth, LSH_sample
             Ward clustering on DTW distances -> bucket inherits label)
  └─ each series is now a string of cluster IDs, one per window
       └─ prefix_scan: count every contiguous ID sequence shared by     minlen
            >= minlen series  -> "candidates"
       └─ optimize_patterns: greedy pick, longest sequences first,      minSupport, minLength, maxLength,
            drop candidates with < minSupport members, subtract        minSize, maxSize, labels
            overlapping members, then filter by length/size
  └─ output_result -> one chart per pattern (members over its interval)
```

Consequences that explain most behaviour:

- A pattern always starts and ends on a window boundary and spans a whole
  number of windows. Its length is `k * chunk_size`.
- "Clusters per window" is the resolution of the alphabet. Few clusters per
  window means few, big, vague patterns; many clusters per window means many
  candidates, fragmentation, and slow scans.
- `minSupport` acts during the greedy selection; `minSize` filters after it.
  They are not interchangeable (see §3).
- Clustering cost is dominated by the DTW distance matrix of the points that
  actually get clustered (all N without LSH, one sample per LSH bucket with
  LSH) times `B + 1` linkages, and DTW itself is quadratic in `chunk_size`.

## 2. Quick rules

1. Start from `strength = 1`. Below 1 the stopping rule rarely fires: every
   sample becomes its own cluster, there are no shared patterns, and runtime
   explodes (stock: 1,090 clusters/window, 929 s, 0 patterns). Above 1 merges
   aggressively (ECG >= 1.5: one cluster per window, one pattern holding all
   109k beats). The useful range is about 0.8 to 2; 0.8 only helps when LSH
   leaves few samples per window (ECG with nPerm 4: 0.8 gave 20 patterns vs 7).
2. Pick `chunk_size` at the time scale of the features you want to separate,
   with 4 to 10 windows per series. Fewer than 4 windows cannot localise
   anything; more than ~15 fragments patterns and inflates the scan
   (ECG chunk 9: 20 windows, 11,906 candidates, 17 s scan vs 0.6 s).
3. Set `minSupport` and `minSize` together, `minSupport >= minSize`. A low
   `minSupport` with a high `minSize` loses most of the data (stock
   minSupport 10 / minSize 50: 10 patterns, 24% coverage).
4. `minLength` is the cleanest lever on pattern count once clustering is
   fixed (stock: 0 -> 120 patterns, 60 -> 83, 120 -> 29, 180 -> 2).
5. `maxLength` / `maxSize` drop patterns, they do not trim or split them.
   Setting `maxLength` below the typical pattern length gives zero output.
6. LSH (`nPerm`, `bandWidth`) is primarily a speed knob. Too coarse (nPerm <= 4
   or bandWidth >= 2 on ~200-point windows) assigns a series to different
   buckets in consecutive windows and shatters patterns into stray pieces.
   Too fine costs minutes for no gain (stock nPerm 40: 85 s vs 0.9 s at 8,
   same patterns).
7. `labels` splits every pattern by class before the size filters. With many
   classes or small classes, lower `minSize`/`minSupport` or expect nothing
   (stock with 14 sectors and minSize 50: 0 patterns).
8. `minlen` only prunes the scan. It changes runtime and memory, not results
   (stock: 18,521 candidates at 1, 395 at 50, identical 83 patterns).
9. Results are not bit-reproducible: LSH projections and the gap statistic
   use NumPy's RNG, the greedy selection uses Python's `random`. Seed both
   (`np.random.seed`, `random.seed`) when comparing settings, and expect
   pattern counts to move by a few between runs.

## 3. Parameter by parameter

### chunk_size  (TiVy, window length in time points)

Mechanism: windows of `chunk_size`; the last partial window is dropped.
`T` not divisible by `chunk_size` silently loses the tail.

| dataset | values -> patterns (median length) | notes |
|---|---|---|
| synthetic T=200 | 20 -> 21 (120), 25 -> 8 (200), 40 -> 11 (200), 50 -> 5 (200), 100 -> 5 (200) | 20 shatters the cyclic group into half-period pieces; 50 and 100 give one chart per true group |
| stock T=270 | 10 -> 56 (70, coverage 0.75), 15 -> 79, 30 -> 83, 54 -> 49 (162), 90 -> 45 (180) | clustering time 13 s, 17 s, 18 s, 36 s, 93 s: DTW cost grows with window length |
| ecg T=180 | 9 -> 25, 12 -> 20, 18 -> 7, 30 -> 22, 45 -> 13, 60 -> 10 | 9 and 12 blow up the scan (11,906 / 5,157 candidates, 17 s / 7 s) |

Tune: the shortest feature you care about should be >= 1 window and the
series should hold 4 to 10 windows. Patterns can only start where a window
starts, so a shift at t=70 with chunk 50 shows up as a step at 50 or 100.

### strength  (TiVy, gap-statistic stopping threshold)

Mechanism: for k = 1, 2, ... clusters the rule stops at the first k where
`gap(k-1) * strength - gap(k) + s >= 0`. Larger `strength` stops earlier
(fewer clusters per window). `steps` adds to `strength` at every k;
`max_iter` caps k; `B` is the number of reference datasets (default 20).

| dataset | strength -> clusters/window -> patterns |
|---|---|
| synthetic | 0.5 / 0.8 -> 230 (= every series) -> 0, 17 s · 1 -> 4.0 -> 5 · 1.5 -> 2.5 -> 3 · 2, 3 -> 2.0 -> 2 (trends only; cyclic merged into the shifts) |
| stock | 0.5 / 0.8 -> 1,090 -> 0, 929 s · 1 -> 5.1 -> 83 · 1.5 -> 2.3 -> 56 · 2 -> 2.1 -> 43 · 3 -> 1.6 -> 11 (one pattern of 1,622 series) |
| ecg | 0.5 -> 9.5 -> 37 (coverage 0.76) · 0.8 -> 4.4 -> 20 · 1 -> 1.9 -> 7 · 1.5+ -> 1.0 -> 1 pattern of all 109,446 |

Why ECG tolerates < 1: with nPerm 4 only a few dozen bucket samples per
window are clustered, so "one cluster per sample" is still a small alphabet.
With thousands of samples (no LSH or fine LSH) < 1 is catastrophic.

Related: `max_iter = 2` behaves like a large strength (synthetic 2 patterns,
stock 36 with a 679-series blob). `B = 5` is 3 to 4x faster than 20 with the
same result on synthetic/stock but noisier k on ECG (15 patterns vs 7); keep
20 for final runs, use 5 while exploring.

### use_LSH, nPerm, bandWidth, LSH_sample  (TiVy, who gets clustered)

Mechanism: each window-subsequence is projected onto `nPerm` random
directions and binned with width `bandWidth * sqrt(chunk_size)`; identical
bin vectors form a bucket. `LSH_sample` random members per bucket are
clustered by the gap rule; every bucket member inherits the label. More
`nPerm` or smaller `bandWidth` -> more buckets -> more points clustered ->
closer to exact, slower. `use_LSH=False` clusters all N points (only
feasible for N up to a few hundred: it is an N x N DTW matrix times B+1).

| dataset | nPerm -> patterns (cluster time) | bandWidth -> patterns (cluster time) |
|---|---|---|
| synthetic | 2 -> 15, 4 -> 18 (len 50), 8 -> 8, 20 -> 5, 40 -> 7 | 0.25 -> 8 (1.3 s), 0.5 -> 5, 1 -> 12, 2 -> 14, 4 -> 17 (len 50, 0.1 s); no LSH -> 8 (1.3 s) |
| stock | 2 -> 42 (0.1 s), 4 -> 71 (0.2 s), 8 -> 87 (0.9 s), 20 -> 83 (17.6 s), 40 -> 80 (85 s) | 0.25 -> 90 (91 s), 0.5 -> 83 (18 s), 1 -> 77 (1.9 s), 2 -> 65 (0.3 s), 4 -> 31 (0.1 s, one 667-series blob) |
| ecg | 2 -> 30, 4 -> 7, 8 -> 19, 20 -> 35 (28 s, coverage 0.72) | 0.25 -> 25, 0.5 -> 38, 1 -> 7, 2 -> 20, 4 -> 2 (one 105k blob) |

Reading the sheets: coarse hashing does not merge groups so much as make
assignments inconsistent across windows, which shows as many short charts
covering 1 to 2 windows of an otherwise coherent group (synthetic nPerm 4,
bandWidth 4). Very coarse (bandWidth 4) finally merges everything into a blob.
`LSH_sample = 3` triples the clustered points: 8 s vs 1 s on synthetic,
247 s vs 18 s on stock, for no visible gain. Leave it at 1.

Tune: nPerm 8 / bandWidth 1 is a good default for thousands of series
(stock: same patterns as nPerm 20 in 1/20 of the time). For 100k series keep
nPerm around 4 and bandWidth around 1 and accept that strength < 1 may be
needed to get more than a handful of clusters.

### minlen  (prefix_scan, minimum series sharing a prefix)

Prunes the scan tree. No effect on the final patterns in any dataset; large
effect on candidates and scan time (stock: 18,521 candidates at 1, 1,727 at
10, 395 at 50). Set it to roughly `minSupport / 2`; never above `minSupport`.

### minSupport  (optimize_patterns, greedy selection threshold)

Mechanism: candidates with fewer than `minSupport` members are discarded,
then candidates are consumed longest-first; when a chosen pattern overlaps
another in time, the shared series are removed from the other. A low value
lets many long, small candidates win first and strip members from the big
ones, which the later `minSize` filter then throws away.

| dataset | minSupport -> patterns (coverage, median size) |
|---|---|
| synthetic | 1 .. 40 -> 5 or 6 (clusters are large anyway) |
| stock (minSize 50) | 10 -> 10 (0.24, 73) · 20 -> 15 (0.31, 60) · 50 -> 83 (0.98, 65) · 100 -> 49 (1.00, 125) · 300 -> 11 (0.94, 416) |
| ecg (minSize 500) | 10 / 1000 -> 7 · 5000 -> 3 (median size 14,700) |

Tune: `minSupport` ~ the smallest group you want to see, and >= `minSize`.
Raising it is the way to get fewer, bigger, cleaner charts without losing
coverage (stock 100: coverage 1.0 with 49 charts vs 83).

### minSize, maxSize  (optimize_patterns, post-filters on member count)

`size > minSize` (strict) and `size <= maxSize` after the greedy pass.

- stock: 0 / 10 -> 93, 50 -> 83, 200 -> 0 (nothing survived because
  minSupport 50 had already produced ~65-series patterns; raise minSupport
  with minSize).
- ecg: 0 / 100 -> 13, 500 -> 7, 2000 -> 4, 10000 -> 2 (coverage still 0.91:
  two blobs hold most beats).
- `maxSize` removes the dominant blobs: ecg maxSize 5000 -> 5 patterns but
  coverage 0.08. Use it to look at the minority shapes, not for a summary.

### minLength, maxLength  (optimize_patterns, post-filters on time span)

In time points; effective values are multiples of `chunk_size`.

- stock: 0 / 30 -> 120 patterns (includes 1-window stubs), 60 -> 83,
  120 -> 29 (coverage 0.60), 180 -> 2 (0.04).
- synthetic / ecg: no effect because every pattern already spans the whole
  series (aligned data cluster the same way in every window).
- `maxLength` below the typical length -> 0 patterns (synthetic 100, ecg 90).
  stock maxLength 120 -> 71 (dropped the 12 longest).

Tune: `minLength` = 2 to 3 windows removes stubs; raise it to keep only
long-lived behaviour, at the cost of coverage.

### labels  (optimize_patterns, per-series class)

Each candidate is split into one candidate per class, then filtered as
usual, and the output is sorted class by class. It never merges classes.

- synthetic (5 groups): 5 -> 6 patterns, now one row per group.
- ecg (5 classes): 7 -> 12 patterns; this is the notebook's result.
- stock (14 sectors, sizes 2 to 661): 0 patterns at minSize 50 because no
  single-sector slice of a pattern is that large.

Tune: with labels, divide `minSize`/`minSupport` by roughly the number of
classes, or restrict to the few classes you care about. Pass labels to the
renderer too so rows are grouped (the CLI does this automatically).

### Rendering knobs (not mining)

`opacity` (0.6 default, raise to 0.85 for faint charts), `density auto|linear`
(auto rescales per chart so 10 or 100k series both read; linear is the
widget's raw count), `n_cols`, `chart_width/height`, `central_line_px`,
`labels_order`. They never change what is mined; use `tivy_cli.py render` on
the saved `.pkl` to iterate on them for free.

## 4. Symptom -> knob

| what you see | likely cause | try |
|---|---|---|
| 0 patterns, clustering took minutes | strength < 1 (every sample its own cluster) | strength 1 |
| 0 patterns, clustering fast | minSize or minLength above what exists; maxLength/maxSize too low; labels splitting | lower minSize/minLength, remove max*, raise minSupport with minSize |
| one giant chart holding nearly everything | strength >= 1.5, bandWidth >= 4, or max_iter 2 | strength 1 (or 0.8 with coarse LSH), bandWidth 1 |
| dozens of 1-2 window stubs, same shape repeated | LSH too coarse (nPerm <= 4, bandWidth >= 2) or chunk too small | nPerm 8, bandWidth 1, larger chunk, higher minLength |
| too many charts (> 40) but all look reasonable | thresholds too low for N | raise minSupport first (keeps coverage), then minLength |
| low coverage (< 0.7) with few charts | minSupport << minSize, or minLength near T | minSupport >= minSize; lower minLength |
| scan (prefix_scan/optimize) slow, memory heavy | many windows x many clusters | larger chunk, raise minlen, strength 1 |
| patterns start/stop at odd places | boundaries are window edges | choose chunk_size so edges fall where behaviour changes |

## 5. Tested starting points

| regime | settings | result |
|---|---|---|
| few hundred series, 100-500 points (synthetic) | chunk ≈ T/4, strength 1, nPerm 20, bandWidth 0.5, minlen 1, minSupport 10, minLength ≈ T/4, minSize 1 | 5 charts, 1 per true group, ~2 s |
| thousands of series (stock 2,939 x 270) | chunk 30, strength 1, nPerm 8, bandWidth 1, minlen 10, minSupport 100, minLength 90, minSize 100 | 31 charts, coverage ~1, ~2 s |
| 100k series (ecg 109k x 180), unlabeled | chunk 18, strength 0.8, nPerm 4, bandWidth 1, minlen 10, minSupport 500, minLength 100, minSize 500 | 33 charts, ~5 s |
| 100k series with class labels | chunk 18, strength 1, nPerm 4, bandWidth 1, minlen 10, minSupport 100, minLength 100, minSize 500, labels | 12 charts grouped by class (the paper's ECG figure) |

CLI equivalents:

```bash
.venv/bin/python tivy_cli.py run data.npz --no-labels --out out/x \
    --chunk-size 30 --nperm 8 --bandwidth 1 --minlen 10 \
    --min-support 100 --min-length 90 --min-size 100
```

Iterate in this order: (1) chunk_size for the time scale, (2) strength 1 and
LSH for speed/quality, (3) minSupport (with minSize) for chart count,
(4) minLength to drop stubs, (5) rendering options on the saved .pkl.

## 6. Reproducing or extending the sweep

`sweep/sweep.py` runs one-at-a-time deviations from a per-dataset baseline,
caches clustering across the filter parameters, writes `stats.csv` and
per-parameter contact sheets:

```bash
.venv/bin/python -I sweep/sweep.py synthetic      # or stock, ecg (run from the skill folder)
.venv/bin/python -I sweep/sweep.py --resheet stock
```

Stock raw series are rebuilt from `examples/stock.pkl` (full-interval
patterns); ecg needs `examples/out/ecg.npz` from `examples/run_ecg.sh`. Edit the `DATASETS` dict to add datasets or
values. Whole run: ~20 min, dominated by the slow configurations
(strength < 1 on stock, LSH_sample 3, nPerm 40).
