"""Headless (non-Jupyter) rendering of the TiVy summary.

``TimeSeriesVis`` in ``widget.py`` draws through vispy's ``jupyter_rfb``
backend and controls the view with ipywidgets, so it can only be seen inside
a running notebook. ``TimeSeriesVisOffscreen`` reuses the exact same buffer
and drawing code but renders into an offscreen OpenGL context (glfw by
default) and returns the pixels, so the summary can be written to a PNG and
viewed by anything that can open an image: a terminal tool, a report, or an
AI assistant such as Claude.

    vis = TimeSeriesVisOffscreen(n_cols=7, chart_width=260, chart_height=140)
    vis.fit(result['result'], central_line=result['center_line'])
    vis.render('summary.png')               # line-density view
    vis.render('bands.png', show_lines=False)  # quantile-band view
    vis.summary()                           # machine-readable description
"""
import contextlib

import numpy as np
from vispy import app, scene
from vispy.scene import visuals

from . import widget as _widget
from .widget import TimeSeriesVis, CustomPanZoomCamera
from .util import hex2rgb


# ---------------------------------------------------------------------------
# Fast ribbon meshes.
#
# widget.prepare_buffers triangulates the quantile band and the buffered
# central line with an exact-arithmetic constrained Delaunay (sect/ground);
# that costs ~15 s per chart. Both shapes are ribbons between two curves
# sampled at the same x positions, so a strip triangulation is exact and
# runs in microseconds. These replace the slow builders only while the
# offscreen renderer is preparing its buffers; the Jupyter widget is untouched.
# ---------------------------------------------------------------------------
def _ribbon_mesh(x, y_top, y_bot):
    """Triangle strip between two curves sharing x. Returns (verts Nx3, faces Mx3).

    Vertices are emitted three per triangle, in face order, exactly like
    geometry.polygon2mesh does: widget.draw offsets successive meshes with
    ``faces[-1, -1] + 1``, so the last face must end on the highest index.
    """
    n = len(x)
    if n < 2:
        return None, None
    top = np.column_stack((x, y_top))
    bot = np.column_stack((x, y_bot))
    # two triangles per segment: (top_i, top_i+1, bot_i), (top_i+1, bot_i+1, bot_i)
    t1 = np.stack((top[:-1], top[1:], bot[:-1]), axis=1)
    t2 = np.stack((top[1:], bot[1:], bot[:-1]), axis=1)
    tris = np.concatenate((t1, t2)).reshape(-1, 2)
    verts = np.hstack((tris, np.zeros((len(tris), 1))))
    faces = np.arange(len(verts)).reshape(-1, 3)
    return verts, faces


def _runs(mask):
    """Start/stop index pairs of consecutive True runs in a boolean mask."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(r[0], r[-1] + 1) for r in np.split(idx, splits)]


def _fast_area_buffer(d_x, d_y, q=1):
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        y_min = np.nanquantile(d_y, 1 - q, 0)
        y_max = np.nanquantile(d_y, q, 0)
    verts, faces, offset = [], [], 0
    for a, b in _runs(~np.isnan(y_min)):
        v, f = _ribbon_mesh(d_x[a:b], y_max[a:b], y_min[a:b])
        if v is None:
            continue
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    if not verts:
        return None, None
    return np.concatenate(verts), np.concatenate(faces)


def _make_fast_line_mesh(sx, sy):
    """Build a stroke mesher for a polyline whose buffer coords map to pixels
    by (sx, sy). `half_px` is the stroke half-width in pixels; each segment
    becomes a quad offset along its own perpendicular, so steep or jagged
    lines keep a constant on-screen thickness."""
    def mesher(d_x, d_y, half_px):
        verts, faces, offset = [], [], 0
        for a, b in _runs(~np.isnan(d_y)):
            if b - a < 2:
                continue
            x, y = d_x[a:b] * sx, d_y[a:b] * sy          # pixel space
            p0 = np.column_stack((x[:-1], y[:-1]))
            p1 = np.column_stack((x[1:], y[1:]))
            d = p1 - p0
            L = np.hypot(d[:, 0], d[:, 1])
            L[L == 0] = 1.0
            nrm = np.column_stack((-d[:, 1], d[:, 0])) / L[:, None] * half_px
            quad = np.stack((p0 + nrm, p1 + nrm, p0 - nrm,
                             p1 + nrm, p1 - nrm, p0 - nrm), axis=1)   # 2 tris / segment
            pts = quad.reshape(-1, 2) / np.array([sx, sy])             # back to buffer space
            v = np.hstack((pts, np.zeros((len(pts), 1))))
            verts.append(v)
            faces.append(np.arange(len(v)).reshape(-1, 3) + offset)
            offset += len(v)
        if not verts:
            return np.zeros((3, 3)), np.array([[0, 1, 2]])
        return np.concatenate(verts), np.concatenate(faces)
    return mesher


def _silent_tqdm(iterable, *args, **kwargs):
    return iterable


@contextlib.contextmanager
def _fast_buffers(verbose, sx, sy):
    saved = (_widget._make_area_buffer, _widget._make_line_mesh_buffer, _widget.tqdm)
    _widget._make_area_buffer = _fast_area_buffer
    _widget._make_line_mesh_buffer = _make_fast_line_mesh(sx, sy)
    if not verbose:
        _widget.tqdm = _silent_tqdm
    try:
        yield
    finally:
        _widget._make_area_buffer, _widget._make_line_mesh_buffer, _widget.tqdm = saved

# Categorical palette validated for colour-vision deficiency (adjacent pairs
# dE >= 8 in OKLab) on a white surface. Row labels always accompany the colour,
# so identity never rests on hue alone.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _lighten(rgb, amount=0.55):
    return [c + (1.0 - c) * amount for c in rgb]


class TimeSeriesVisOffscreen(TimeSeriesVis):
    """Drop-in replacement for ``TimeSeriesVis`` that renders to pixels."""

    def __init__(self, *args, backend="glfw", label_margin=0.06,
                 central_line_px=(1.5, 3.5), **kwargs):
        """central_line_px: (thinnest, thickest) stroke width in pixels; the
        stroke grows linearly with the number of series in the chart, as in
        the widget."""
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.label_margin = label_margin
        self.central_line_px = central_line_px
        self.show_lines = True
        self._show_central = True
        self.canvas = None

    # -- buffers -------------------------------------------------------------
    def prepare_buffers(self, central_line=None):
        import builtins
        # Central-line buffers are built in widget "buffer space"
        # (x / x_limit, y / y_limit[1]); these factors convert to chart pixels.
        y_top = self.y_limit[1] if self.y_limit[1] != 0 else 1.0
        sx = self.chart_width
        sy = self.chart_height * y_top / (self.y_limit[1] - self.y_limit[0])
        with _fast_buffers(self.verbose, sx, sy):
            if not self.verbose:
                # widget.prepare_buffers prints a progress banner
                saved_print, builtins.print = builtins.print, lambda *a, **k: None
                try:
                    super().prepare_buffers(central_line)
                finally:
                    builtins.print = saved_print
            else:
                super().prepare_buffers(central_line)
        for b in self.buffers:
            b["alpha_raw"] = b["alpha_line"]      # per-bin line counts

    def _apply_density(self, density):
        """Turn per-bin line counts into alphas.

        widget.make_buffer computes alpha = alpha_line * opacity and clamps
        at 1, so with thousands of series every bin saturates into a solid
        block. 'auto' composites the lines properly (alpha = 1-(1-a)^count)
        and picks the per-line alpha `a` per chart so that the chart's
        95th-percentile density reaches `self.opacity`. 'linear' keeps the
        widget's behaviour."""
        op = min(max(float(self.opacity), 0.01), 0.99)
        for b in self.buffers:
            H = b["alpha_raw"]
            if density == "linear":
                b["alpha_line"] = H
                continue
            nz = H[H > 0]
            h95 = max(float(np.percentile(nz, 95)), 1.0) if nz.size else 1.0
            a = 1.0 - (1.0 - op) ** (1.0 / h95)
            alpha = 1.0 - (1.0 - a) ** H
            b["alpha_line"] = alpha / op          # widget multiplies by opacity again

    def set_widthScale(self, data):
        # self.scale(n) -> stroke half-width in pixels for a chart of n series
        lengths = [len(d) for d in data]
        lo, hi = min(lengths), max(lengths)
        thin, thick = self.central_line_px
        if hi == lo:
            self.scale = lambda n: (thin + thick) / 4.0
        else:
            self.scale = lambda n: (thin + (thick - thin) * (n - lo) / (hi - lo)) / 2.0

    # -- colours -----------------------------------------------------------
    def make_canvas_spec(self, data, labels=None, labels_order=None, central_line=None):
        out = super().make_canvas_spec(data, labels, labels_order, central_line)
        if labels is not None:
            keys = list(labels_order) if labels_order is not None else list(self.color.keys())
            keys += [k for k in self.color if k not in keys]
            self.color = {k: hex2rgb(PALETTE[i % len(PALETTE)]) for i, k in enumerate(keys)}
            self.colorLight = {k: _lighten(v) for k, v in self.color.items()}
        return out

    # -- widget state without widgets ------------------------------------
    def make_buttons(self, central_line=None):
        self.show_area = not self.show_lines
        self.show_central_line = central_line is not None and self._show_central
        self.sort_labels = bool(self.make_labels)

    # -- offscreen canvas ---------------------------------------------------
    def init_canvas(self):
        app.use_app(self.backend)
        ca = scene.SceneCanvas(keys=None, show=False,
                               size=(self.canvas_width, self.canvas_height),
                               bgcolor=[1, 1, 1])
        self.canvas = ca
        view = ca.central_widget.add_view()

        self.text = visuals.Text(anchor_x="right")
        view.add(self.text)
        self.label_texts = visuals.Text(anchor_x="right")
        view.add(self.label_texts)

        self.camera = CustomPanZoomCamera([self.text, self.label_texts])
        view.camera = self.camera
        # Leave room on the left for row labels when the charts are grouped:
        # label font is chart_height/5 pt, ~0.6 em per character.
        margin = 0.0
        if self.make_labels:
            longest = max(len(str(l)) for l in self.labels)
            est = 0.6 * (self.chart_height / 5) * longest / self.canvas_width + 0.015
            margin = max(self.label_margin, est)
        self.camera.rect = (-margin, 0, 1 + margin, 1)
        self.view = view

        self.line = visuals.Line()
        self.line.set_data(pos=np.array([[0, 0, 0]]), color=np.array([[0, 0, 0, 0]]))
        self.line.order = 1
        view.add(self.line)

        self.mesh = visuals.Mesh()
        self.mesh.order = 0
        view.add(self.mesh)

    def visualize(self):
        raise RuntimeError("Offscreen renderer: call render(path) instead of visualize().")

    # -- public API ---------------------------------------------------------
    def render(self, path=None, show_lines=True, show_central_line=True,
               opacity=None, size_threshold=0, interval=None, density="auto"):
        """Render the fitted summary and return an RGBA uint8 array.

        show_lines        True  -> line-density view (the widget's default)
                          False -> quantile-band view (90% band + full range)
        show_central_line draw the black representative line when available
        opacity           'auto' density: how dark the dense core of each chart
                          gets (0..1); 'linear': per-line alpha as in the widget
        density           'auto' (scale per chart, works for 10 or 100k series)
                          or 'linear' (widget behaviour, saturates on big charts)
        size_threshold    hide charts with fewer series than this
        interval          (start, end) time window; charts outside are hidden
        path              if given, also write a PNG there
        """
        if not hasattr(self, "buffers"):
            raise RuntimeError("call fit(data, ...) before render()")
        self.show_lines = show_lines
        self._show_central = show_central_line
        if opacity is not None:
            self.opacity = float(opacity)
        self._apply_density(density)
        self.size_threshold = size_threshold
        if interval is not None:
            self.interval_threshold = [int(interval[0]), int(interval[1])]

        self.make_buttons(self.central_line)
        self.init_canvas()
        self.paint()
        img = self.canvas.render()
        self.canvas.close()
        self.canvas = None
        if path is not None:
            from PIL import Image
            Image.fromarray(img).save(path)
        return img

    def summary(self):
        """Machine-readable description of every chart, in display order."""
        rows = []
        for i, (d, b) in enumerate(zip(self.data, self.buffers)):
            start, end = (int(v) for v in b["interval"])
            with np.errstate(all="ignore"):
                seg = d[:, start:end + 1]
                row = {
                    "chart": i,
                    "n_series": int(d.shape[0]),
                    "interval": [start, end],
                    "length": end - start + 1,
                    "y_min": float(np.nanmin(seg)),
                    "y_max": float(np.nanmax(seg)),
                    "y_median": float(np.nanmedian(seg)),
                }
            if self.labels is not None:
                row["label"] = str(self.labels[i])
            if self.central_line is not None:
                cl = np.asarray(self.central_line[i], dtype=float)
                valid = cl[~np.isnan(cl)]
                if valid.size:
                    row["central_line_start"] = float(valid[0])
                    row["central_line_end"] = float(valid[-1])
                    row["central_line_change"] = float(valid[-1] - valid[0])
            rows.append(row)
        return {
            "n_charts": len(rows),
            "n_cols": self.n_cols,
            "n_rows": self.n_rows,
            "x_limit": int(self.x_limit),
            "y_limit": [float(self.y_limit[0]), float(self.y_limit[1])],
            "charts": rows,
        }
