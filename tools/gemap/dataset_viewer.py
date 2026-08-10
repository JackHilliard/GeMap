"""Interactive CARLA dataset viewer: browse tiles by town, render top-down
views of the raw LiDAR point cloud, a density heat map, and/or the
reference-line polylines.

Reads the dataset directly (manifest.json + blocks/*.npz +
reference_lines/*.json) -- no model, no checkpoint, no GPU, and no
mmdet3d/torch import at all. Only needs `flask`, `matplotlib`, `numpy`.

Usage (from the repo root):
  pip install flask matplotlib numpy
  python3 tools/gemap/dataset_viewer.py \
      --data-root /path/to/carla --split train

Then open http://127.0.0.1:5001

--- Coordinate frames (important) ---
Each tile's .npz stores two different origins, and they are NOT the same:
  * `offset`      -- what `features[:, 0:3]` is actually relative to
                     (verified: points - offset == features xyz, exactly)
  * `tile_center` -- the geometric centre of the tile (verified: exactly
                     the midpoint of `tile_bounds`, on every tile)
The reference_lines/*.json polylines are in world coordinates, so something
must be subtracted from them to land in the same frame as the points.

This viewer renders everything in the **tile-centred** frame, matching what
the model trains on: polylines get `tile_center` subtracted, and the point
cloud -- which arrives in the `offset` frame -- gets the difference
`tile_center - offset` subtracted. Both are translated by the same vector,
so their relative alignment is exactly the alignment that was measured
against real driving-surface returns (points whose label == 0), across 40
tiles:
    polyline - offset       -> median 0.038 m to nearest road point
    polyline - tile_center  -> median 0.388 m to nearest road point
i.e. the 0.038 m figure, *not* the 0.388 m one. That second number is what
you get from re-framing the polylines alone and leaving the points where
they were; shifting both is a rigid translation and preserves the first.

What the tile-centred frame buys is that the tile now occupies exactly
[-tile_radius, +tile_radius] on both axes, so these plots' axes bound the
data instead of cropping it. In the raw `offset` frame the tile is
displaced from the origin by a mean of 1.25 m and up to 12.07 m on the 25 m
export (and ~17 m on the 60 m one), which used to push part of every tile
outside the axis limits.

tools/gemap/custom_carla_map_converter.py builds the training pkl in this
same frame, so what you see here is what the model trains and evals
against. (Pkls generated before 2026-08-10 are in the old `offset` frame
and no longer load -- the dataset rejects them rather than training against
a misaligned patch.)
"""
import argparse
import html
import io
import json
import os
import os.path as osp
import threading
import time
from datetime import datetime
from urllib.parse import urlencode

import matplotlib
matplotlib.use('Agg')
# Rendering matplotlib inside a threaded web server needs BOTH of these:
#
#  1. The object-oriented Figure/FigureCanvasAgg API, never pyplot --
#     pyplot is a global state machine and is not thread-safe.
#  2. RENDER_LOCK below, serializing every render.
#
# Neither alone was sufficient here: Flask's dev server is threaded by
# default and a browser requests every tile image on a page concurrently,
# which segfaulted the process outright (no traceback -- SIGSEGV). Switching
# to the OO API still crashed under a concurrent-request stress test; only
# adding the lock made it survive. Sequential curl requests never reproduce
# it, so test this path with genuinely parallel requests if you touch it.
# Rendering is fast enough (~0.1-1 s/tile) that serializing a page's worth
# of images is fine for a local viewer.
import numpy as np
import matplotlib.patheffects as pe
from flask import Flask, abort, make_response, request, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

app = Flask(__name__)
STATE = {}
RENDER_LOCK = threading.Lock()

BG = '#0d1117'
BG_PANEL = '#161b22'
BORDER = '#30363d'
TEXT = '#c9d1d9'
TEXT_MUTED = '#8b949e'
ACCENT = '#58a6ff'

# ITU-R BT.709 luma, matching LoadCarlaPointsFromFile's rgb->strength
RGB2GRAY = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# label ids come from manifest['lane_type_lookup']; -1 == unlabeled
LABEL_COLORS = {
    -1: '#484f58', 0: '#58a6ff', 1: '#d29922', 2: '#3fb950', 3: '#bc8cff',
    4: '#f85149', 5: '#39c5cf', 6: '#db6d28', 7: '#e3b341', 8: '#8b949e',
}

# 'inferno' rather than a hand-rolled dark ramp: an earlier version started
# the colormap at the page background colour (#0d1117), which made every
# low-density cell literally indistinguishable from empty space -- the heat
# map looked blank. inferno is perceptually uniform and its low end is a
# visible dark purple, so occupied-but-sparse cells still read as occupied.
DENSITY_CMAP = 'inferno'


def discover_splits(data_root, only=None):
    """Find every <data_root>/<split>/manifest.json. Returns
    {split_name: manifest}, ordered train-ish first for a stable UI."""
    found = {}
    for entry in sorted(os.listdir(data_root)):
        if only and entry != only:
            continue
        mpath = osp.join(data_root, entry, 'manifest.json')
        if osp.isfile(mpath):
            with open(mpath) as f:
                found[entry] = json.load(f)
    # conventional ordering first, then anything else alphabetically
    order = {'train': 0, 'val': 1, 'test': 2}
    return dict(sorted(found.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))


def build_index(splits):
    """Flatten every split's tiles into one list, tagging each with the
    split it came from, and map town -> split. Towns are assumed unique to
    a split (true for this dataset: town01-05 are train, town10hd is test);
    if a town ever appeared in two splits, the town key is 'split:town' so
    they stay distinct rather than silently merging."""
    tiles, town_split, town_counts = [], {}, {}
    for split, man in splits.items():
        for t in man['tiles']:
            t = dict(t)
            t['_split'] = split
            tiles.append(t)
        for town in man['towns']:
            key = town if town not in town_split else f'{split}:{town}'
            town_split[key] = split
            town_counts[key] = man.get('tiles_per_town', {}).get(town, 0)
    return tiles, town_split, town_counts


def tile_by_name(name):
    return STATE['tiles_by_name'].get(name)


def split_dir(split):
    return osp.join(STATE['data_root'], split)


def load_block(name, split):
    path = osp.join(split_dir(split), 'blocks', f'{name}.npz')
    if not osp.isfile(path):
        return None
    return np.load(path)


def load_polylines(name, origin, split):
    """Returns list of (N,2) arrays in the tile-centred frame, given the
    origin to subtract (see module docstring)."""
    path = osp.join(split_dir(split), 'reference_lines',
                     f'{name}_reference_lines.json')
    if not osp.isfile(path):
        return []
    with open(path) as f:
        rl = json.load(f)
    out = []
    for p in rl['polylines']:
        pts = np.asarray(p['points'], dtype=np.float32)
        if pts.shape[0] < 2:
            continue
        out.append(pts[:, :2] - origin[:2])
    return out


def discover_results(work_dir):
    """Find every prediction-results json under a training work_dir.

    Two locations matter, because they're written by different code paths:
      * <work_dir>/**/carlamap_results.json -- what you get from
        `tools/test.py --format-only --eval-options jsonfile_prefix=...`
        when you point it inside the work_dir.
      * val/<work_dir>/<ctime>/carlamap_results.json -- what the *training*
        eval hook writes. Note mmdet_train.py hardcodes
        `osp.join('val', cfg.work_dir, <ctime>)`, i.e. a path relative to
        the CWD training ran from, OUTSIDE the work_dir. If training ran in
        a container without that path bind-mounted, those results were
        discarded when the container exited.

    Returns {label: path}, newest first.
    """
    found = {}
    roots = [work_dir, osp.join('val', work_dir)]
    for root in roots:
        if not osp.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith('.json') and 'result' in fn.lower():
                    p = osp.join(dirpath, fn)
                    label = osp.relpath(p, work_dir if root == work_dir
                                         else osp.dirname(work_dir))
                    found[label] = p
    return dict(sorted(found.items(),
                        key=lambda kv: osp.getmtime(kv[1]), reverse=True))


def load_results(path):
    """Parse a carlamap_results.json into {sample_token: [(pts, score, cls)]}.

    Predicted points are already in the model's tile-local BEV frame (the
    same frame as the LiDAR points fed in, i.e. the tile-centred frame this
    viewer draws in), so unlike the GT reference lines they need no origin
    subtraction -- they're plotted as-is.
    """
    cached = STATE['results_cache'].get(path)
    if cached is not None:
        return cached
    with open(path) as f:
        blob = json.load(f)
    out = {}
    for entry in blob.get('results', []):
        token = entry.get('sample_token')
        vecs = []
        for v in entry.get('vectors', []):
            pts = np.asarray(v.get('pts', []), dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            vecs.append((pts[:, :2],
                          float(v.get('confidence_level', 1.0)),
                          v.get('cls_name', '?')))
        out[token] = vecs
    STATE['results_cache'][path] = out
    return out


def style_axes(ax, radius):
    ax.set_facecolor(BG)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def render_tile(*args, **kwargs):
    """Thread-safe wrapper -- see RENDER_LOCK's comment at the top."""
    with RENDER_LOCK:
        return _render_tile(*args, **kwargs)


def _render_tile(name, mode, show_polylines,
                  point_size, max_points, log_density=True, split=None,
                  results_path=None, score_thresh=0.3):
    if split is None:
        t = tile_by_name(name)
        split = t['_split'] if t else None
    block = load_block(name, split)
    if block is None:
        return None
    feat = block['features']
    xy = feat[:, :2]
    labels = block['labels'] if 'labels' in block else None
    radius = float(block['tile_radius']) if 'tile_radius' in block else 12.5

    # The tile-centred frame, matching the converter and the model (see
    # this file's module docstring). `origin` is subtracted from the
    # world-frame polylines; the point cloud is already in the `offset`
    # frame, so it gets the difference instead.
    origin = block['tile_center'] if 'tile_center' in block \
        else block['offset']
    recenter_shift = np.asarray(origin, dtype=np.float32) \
        - np.asarray(block['offset'], dtype=np.float32)
    xy = xy - recenter_shift[:2]

    fig = Figure(figsize=(6, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.patch.set_facecolor(BG)
    style_axes(ax, radius)

    n_raw = xy.shape[0]
    subsampled = False
    if mode != 'density' and n_raw > max_points:
        # Only for scatter rendering -- density uses ALL points so the
        # histogram stays quantitatively correct.
        sel = np.random.default_rng(0).choice(n_raw, max_points, replace=False)
        xy_plot = xy[sel]
        labels_plot = labels[sel] if labels is not None else None
        subsampled = True
    else:
        xy_plot = xy
        labels_plot = labels

    if mode == 'density':
        # points per 1 m^2 cell -- uses every point, never the subsample,
        # so the counts stay quantitatively correct
        nbins = max(1, int(round(2 * radius)))
        edges = np.linspace(-radius, radius, nbins + 1)
        H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=[edges, edges])
        # Log norm by default: density spans a huge dynamic range on this
        # dataset (the degenerate 5,000,000-point tiles put ~99.9% of their
        # points in a single cell, so a linear scale maps every other cell
        # to the colormap's zero end and the map reads as blank).
        norm = None
        if log_density and H.max() > 0:
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=max(H[H > 0].min(), 1) if (H > 0).any() else 1,
                            vmax=max(H.max(), 2))
        im = ax.imshow(H.T, origin='lower', extent=[-radius, radius, -radius, radius],
                        cmap=DENSITY_CMAP, aspect='equal', norm=norm,
                        interpolation='nearest')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('points / m²' + (' (log)' if norm is not None else ''),
                      color=TEXT, fontsize=8)
        cb.ax.tick_params(colors=TEXT_MUTED, labelsize=7)
        cb.outline.set_edgecolor(BORDER)
        occ = int((H > 0).sum())
        title_extra = (f'max {H.max():,.0f} pts/m², median {np.median(H):,.0f}, '
                        f'{occ}/{H.size} cells occupied')
    elif mode == 'rgb':
        # features[:, 3:6] are per-point RGB in 0..1
        rgb = np.clip(feat[:, 3:6], 0.0, 1.0)
        rgb_plot = rgb[sel] if subsampled else rgb
        ax.scatter(xy_plot[:, 0], xy_plot[:, 1], c=rgb_plot, s=point_size,
                    linewidths=0)
        title_extra = 'true RGB colour'
    elif mode == 'intensity':
        strength = feat[:, 3:6] @ RGB2GRAY
        s_plot = strength[sel] if subsampled else strength
        ax.scatter(xy_plot[:, 0], xy_plot[:, 1], c=s_plot, s=point_size,
                    cmap='cividis', linewidths=0, alpha=0.9)
        title_extra = 'intensity (BT.709 luma of RGB)'
    elif mode == 'label' and labels_plot is not None:
        for lab in np.unique(labels_plot):
            msk = labels_plot == lab
            lname = STATE['lane_types'].get(str(int(lab)), 'unlabeled'
                                             if lab == -1 else str(lab))
            ax.scatter(xy_plot[msk, 0], xy_plot[msk, 1],
                        s=point_size, linewidths=0, alpha=0.9,
                        c=LABEL_COLORS.get(int(lab), '#8b949e'),
                        label=f'{lname} ({msk.sum():,})')
        leg = ax.legend(fontsize=6, loc='upper right', facecolor=BG_PANEL,
                         edgecolor=BORDER, labelcolor=TEXT, framealpha=0.9,
                         markerscale=3)
        leg.get_frame().set_linewidth(0.5)
        title_extra = 'coloured by lane label'
    else:  # 'points', and 'label' on a block with no labels array
        ax.scatter(xy_plot[:, 0], xy_plot[:, 1], s=point_size,
                    c='#58a6ff', linewidths=0, alpha=0.6)
        title_extra = ('top-down' if mode != 'label'
                        else 'top-down (no labels in this block)')

    outline = [pe.Stroke(linewidth=3.0, foreground='#000000', alpha=0.6),
                pe.Normal()]
    overlay_handles = []

    if show_polylines:
        # red reads well on the dark scatter views but disappears against
        # inferno's orange/yellow mid-range, so switch to cyan there; the
        # black outline keeps it legible over inferno's bright cells too.
        pl_color = '#00e5ff' if mode == 'density' else '#f85149'
        gt = load_polylines(name, origin, split)
        for pl in gt:
            ax.plot(pl[:, 0], pl[:, 1], color=pl_color, linewidth=1.8,
                     alpha=0.98, zorder=5, path_effects=outline)
        if gt:
            overlay_handles.append(
                Line2D([], [], color=pl_color, linewidth=1.8,
                        label=f'GT ({len(gt)})'))

    n_pred = 0
    if results_path:
        # Predictions are deliberately styled to be unmistakable against
        # GT: bright yellow, dashed, thicker, drawn on top (higher zorder).
        preds = load_results(results_path).get(name, [])
        for pts, score, _cls in preds:
            if score < score_thresh:
                continue
            n_pred += 1
            ax.plot(pts[:, 0], pts[:, 1], color='#ffd60a', linewidth=2.0,
                     alpha=0.95, zorder=6, linestyle='--',
                     path_effects=outline)
        overlay_handles.append(
            Line2D([], [], color='#ffd60a', linewidth=2.0, linestyle='--',
                    label=f'pred ({n_pred} @ score≥{score_thresh:g})'))

    if overlay_handles:
        # ax.legend() REPLACES any existing legend, so the label-mode class
        # legend has to be re-added as a standalone artist first, otherwise
        # adding this overlay legend silently removes it.
        existing = ax.get_legend()
        if existing is not None:
            existing.set_zorder(20)
            ax.add_artist(existing)
        leg2 = ax.legend(handles=overlay_handles, fontsize=6, loc='lower left',
                          facecolor=BG_PANEL, edgecolor=BORDER,
                          labelcolor=TEXT, framealpha=0.95)
        leg2.get_frame().set_linewidth(0.5)
        # legends default to zorder 5, but predictions are drawn at 6 and
        # would otherwise scribble straight over the legend box
        leg2.set_zorder(20)

    sub_note = f', showing {max_points:,}' if subsampled else ''
    ax.set_title(f'{name}  [{split}]\n{n_raw:,} pts{sub_note} — {title_extra}',
                  color=TEXT, fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, facecolor=BG)
    # no plt.close() needed -- Figure created via the OO API isn't held by
    # pyplot's global registry, so it's garbage-collected normally
    buf.seek(0)
    return buf


PAGE = """<!doctype html>
<html><head>
<title>CARLA dataset viewer</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1.5em;
          background: {bg}; color: {text}; }}
  h1 {{ font-size: 1.25em; margin: 0 0 0.2em; }}
  .sub {{ color: {muted}; font-size: 0.85em; margin-bottom: 1.2em; }}
  form {{ background: {panel}; border: 1px solid {border}; border-radius: 6px;
          padding: 1em; margin-bottom: 1.5em; display: flex;
          flex-wrap: wrap; gap: 1.2em; align-items: flex-end; }}
  fieldset {{ border: none; margin: 0; padding: 0; }}
  label.top {{ display: block; font-size: 0.7em; text-transform: uppercase;
               letter-spacing: 0.06em; color: {muted}; margin-bottom: 0.35em; }}
  select, input[type=number] {{
    background: {bg}; color: {text}; border: 1px solid {border};
    border-radius: 4px; padding: 0.4em 0.5em; font-size: 0.9em; }}
  .checks label {{ display: block; font-size: 0.85em; margin: 0.15em 0; }}
  button {{ background: {accent}; color: #0d1117; border: none;
            border-radius: 4px; padding: 0.55em 1.4em; font-size: 0.9em;
            font-weight: 600; cursor: pointer; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  figure {{ margin: 0; background: {panel}; border: 1px solid {border};
            border-radius: 6px; padding: 8px; }}
  figure img {{ display: block; width: 420px; border-radius: 3px; }}
  figcaption {{ font-size: 0.72em; color: {muted}; margin-top: 5px;
                text-align: center; word-break: break-all; }}
  a {{ color: {accent}; }}
</style>
</head><body>
<h1>CARLA dataset viewer</h1>
<div class="sub">{data_root} &mdash; splits: <code>{split}</code> &mdash;
  {n_tiles:,} tiles across {n_towns} towns<br>
  showing <b>{town}</b> (<b>{town_split}</b>) tiles {start}&ndash;{end} &mdash;
  representation: <b>{mode}</b></div>

<form method="get">
  <fieldset>
    <label class="top">Town</label>
    <select name="town">{town_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Tiles to show</label>
    <input type="number" name="count" value="{count}" min="1" max="60" step="1">
  </fieldset>
  <fieldset>
    <label class="top">Start index</label>
    <input type="number" name="start" value="{start}" min="0" step="1">
  </fieldset>
  <fieldset>
    <label class="top">Representation</label>
    <select name="mode">{mode_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Point size</label>
    <input type="number" name="point_size" value="{point_size}" min="0.1"
           max="20" step="0.1">
  </fieldset>
  <fieldset class="checks">
    <label class="top">Overlays</label>
    <label><input type="checkbox" name="polylines" value="1" {pl_checked}>
      reference-line polylines</label>
    <label><input type="checkbox" name="linear_density" value="1" {lin_checked}>
      linear density scale (default: log)</label>
  </fieldset>
  {pred_fields}
  <button type="submit">Render</button>
</form>

<div class="gallery">
{gallery}
</div>
</body></html>
"""


PRED_FIELDS = """
  <fieldset>
    <label class="top">Predictions (work-dir)</label>
    <select name="results">{result_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Pred score &ge;</label>
    <input type="number" name="score_thresh" value="{score_thresh}"
           min="0" max="1" step="0.05">
  </fieldset>
"""

NO_WORKDIR_NOTE = """
  <fieldset>
    <label class="top">Predictions</label>
    <div class="meta" style="max-width:22em">
      pass <code>--work-dir &lt;dir&gt;</code> at startup to overlay predicted
      polylines from a results json
    </div>
  </fieldset>
"""

NO_RESULTS_NOTE = """
  <fieldset>
    <label class="top">Predictions</label>
    <div class="meta" style="max-width:30em">
      No predictions found in <code>{work_dir}</code>. Training only writes
      checkpoints and logs there &mdash; the training-time eval hook writes its
      results to <code>val/&lt;work_dir&gt;/&lt;timestamp&gt;/</code> relative
      to the CWD training ran from, which for a container run is usually not
      bind-mounted, so they were discarded.
      {howto}
    </div>
  </fieldset>
"""

HOWTO_WITH_CKPT = """
      <br><br>Generate them from the checkpoint already in this work-dir:
      <br><code style="display:block;white-space:pre-wrap;margin-top:.4em">python3 tools/test.py \\
  {config} \\
  {ckpt} \\
  --format-only --eval-options jsonfile_prefix={work_dir}/results</code>
      then reload this page (results are re-scanned per request).
"""

HOWTO_NO_CKPT = """
      <br><br>There's no <code>.pth</code> checkpoint here either, so train
      first, then run <code>tools/test.py --format-only --eval-options
      jsonfile_prefix=&lt;work_dir&gt;/results</code>.
"""


def no_results_note(work_dir):
    """Build the 'no predictions' note, filled in with the checkpoint and
    config actually present in this work_dir so the suggested command is
    copy-pasteable rather than a template with placeholders."""
    ckpts = sorted(glob_pth(work_dir))
    cfgs = sorted(f for f in os.listdir(work_dir) if f.endswith('.py')) \
        if osp.isdir(work_dir) else []
    if ckpts:
        # prefer latest.pth / best_*, else whatever's newest
        preferred = next((c for c in ckpts if osp.basename(c) == 'latest.pth'),
                          None) or ckpts[-1]
        cfg = (osp.join(work_dir, cfgs[0]) if cfgs
               else 'projects/configs/carla/gemap_carla_r50_24ep_lidar.py')
        howto = HOWTO_WITH_CKPT.format(config=cfg, ckpt=preferred,
                                        work_dir=work_dir)
    else:
        howto = HOWTO_NO_CKPT
    return NO_RESULTS_NOTE.format(work_dir=work_dir, howto=howto)


def glob_pth(work_dir):
    out = []
    if not osp.isdir(work_dir):
        return out
    for dirpath, _d, filenames in os.walk(work_dir):
        for fn in filenames:
            if fn.endswith('.pth'):
                out.append(osp.join(dirpath, fn))
    return out


def _safe_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def resolve_results_path(label):
    """Map a UI label back to a real path, and refuse anything not in the
    discovered set -- the label reaches us straight from a query string, so
    it must never be joined onto a filesystem path unchecked."""
    if not label or not STATE.get('work_dir'):
        return None
    return discover_results(STATE['work_dir']).get(label)


def _opts(values, current, labels=None):
    out = []
    for i, v in enumerate(values):
        lab = labels[i] if labels else v
        sel = ' selected' if str(v) == str(current) else ''
        out.append(f'<option value="{v}"{sel}>{lab}</option>')
    return '\n'.join(out)


@app.route('/')
def index():
    towns = list(STATE['town_split'].keys())
    town = request.args.get('town', towns[0])
    if town not in STATE['town_split']:
        town = towns[0]
    town_split = STATE['town_split'][town]
    count = max(1, min(60, int(request.args.get('count', 6))))
    start = max(0, int(request.args.get('start', 0)))
    mode = request.args.get('mode', 'rgb')
    point_size = request.args.get('point_size', '1.5')
    polylines = '1' if request.args.get('polylines') else ''
    linear_density = '1' if request.args.get('linear_density') else ''
    results = request.args.get('results', '')
    score_thresh = request.args.get('score_thresh', '0.3')
    # first load defaults to polylines on -- it's the most useful view and
    # makes the frame issue above immediately visible
    if not request.args:
        polylines = '1'

    # re-scan on each request so results produced while the viewer is
    # running show up without a restart
    result_files = (discover_results(STATE['work_dir'])
                     if STATE.get('work_dir') else {})
    if results and results not in result_files:
        results = ''

    # town keys may be 'split:town' if a town name ever collides across
    # splits (see build_index); match on the bare town name either way
    bare_town = town.split(':', 1)[-1]
    tiles = [t for t in STATE['tiles']
             if t['town'] == bare_town and t['_split'] == town_split
             ][start:start + count]

    # Unconditional cache-buster. The per-mode query params already make
    # each image a distinct URL, so caching *shouldn't* be able to serve a
    # stale render -- but in practice it did anyway (server verified to
    # return four different PNGs per mode, byte-for-byte, while the browser
    # kept showing one of them). A token that changes every page load makes
    # every image URL unique, which no cache layer can defeat. Rendering is
    # cheap and local, so always re-fetching costs nothing that matters.
    cachebust = f'{time.time():.6f}'

    figs = []
    for t in tiles:
        params = {
            'name': t['name'],
            'split': t['_split'],
            'mode': mode,
            'point_size': point_size,
            'v': cachebust,
        }
        if polylines:
            params['polylines'] = '1'
        if linear_density:
            params['linear_density'] = '1'
        if results:
            params['results'] = results
            params['score_thresh'] = score_thresh
        # urlencode + html.escape is REQUIRED here, not cosmetic. A raw "&"
        # separator in an HTML attribute starts an entity reference, and
        # browsers resolve known entity names even without the closing ";"
        # -- so "&mode=density&gt_frame=offset" was being parsed as
        # mode="density>_frame=offset" (&gt -> ">"), silently corrupting the
        # mode and dropping gt_frame entirely. Every request then fell
        # through to the default branch and rendered "top-down" no matter
        # what was selected. curl never reproduced it because nothing was
        # HTML-parsing the URL; only a real browser triggers it.
        q = '?' + urlencode(params)
        q_attr = html.escape(q, quote=True)
        cap = (f'{t["name"]} — {t["n_points"]:,} pts, '
               f'{t["n_polylines"]} GT polylines')
        if results:
            preds = load_results(result_files[results]).get(t['name'])
            if preds is None:
                cap += ' — <span style="color:#d29922">no preds for this tile</span>'
            else:
                kept = sum(1 for _p, s, _c in preds if s >= float(score_thresh))
                cap += f', {kept} preds'
        figs.append(
            f'<figure><a href="/tile.png{q_attr}" target="_blank">'
            f'<img src="/tile.png{q_attr}"></a>'
            f'<figcaption>{cap}</figcaption></figure>')

    # NB: not named `html` -- that shadows the stdlib `html` module used
    # above for attribute escaping.
    page_html = PAGE.format(
        bg=BG, panel=BG_PANEL, border=BORDER, text=TEXT, muted=TEXT_MUTED,
        accent=ACCENT,
        data_root=osp.abspath(STATE['data_root']),
        split=', '.join(STATE['splits'].keys()),
        n_tiles=len(STATE['tiles']), n_towns=len(towns),
        town=town, town_split=town_split, mode=mode,
        end=start + len(tiles),
        town_opts=_opts(towns, town,
                         [f'{t}  ({STATE["town_split"][t]}'
                          f', {STATE["town_counts"][t]:,} tiles)'
                          for t in towns]),
        count=count, start=start, point_size=point_size,
        mode_opts=_opts(['rgb', 'label', 'points', 'density', 'intensity'],
                         mode,
                         ['true RGB colour', 'lane label',
                          'top-down (flat colour)',
                          'density heat map (1 m² bins)', 'intensity']),
        pl_checked='checked' if polylines else '',
        lin_checked='checked' if linear_density else '',
        pred_fields=(
            NO_WORKDIR_NOTE if not STATE.get('work_dir')
            else no_results_note(STATE['work_dir'])
            if not result_files
            else PRED_FIELDS.format(
                score_thresh=score_thresh,
                result_opts=_opts([''] + list(result_files.keys()), results,
                                   ['(none)'] + list(result_files.keys())))),
        gallery='\n'.join(figs) or
                f'<p style="color:{TEXT_MUTED}">no tiles for that town/range</p>',
    )
    resp = make_response(page_html)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/tile.png')
def tile_png():
    name = request.args.get('name')
    if not name or tile_by_name(name) is None:
        abort(404)
    try:
        point_size = float(request.args.get('point_size', 1.5))
    except ValueError:
        point_size = 1.5
    buf = render_tile(
        name,
        mode=request.args.get('mode', 'rgb'),
        show_polylines=bool(request.args.get('polylines')),
        point_size=point_size,
        max_points=STATE['max_points'],
        log_density=request.args.get('linear_density') != '1',
        split=request.args.get('split'),
        results_path=resolve_results_path(request.args.get('results')),
        score_thresh=_safe_float(request.args.get('score_thresh'), 0.3),
    )
    if buf is None:
        abort(404)
    resp = send_file(buf, mimetype='image/png')
    # Every render is cheap and always reflects on-disk data; caching only
    # ever causes confusion here (e.g. "changing the representation does
    # nothing" when the browser is quietly serving a stale image).
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', default=None,
                    help='restrict to a single split; default is to load '
                         'every <data-root>/*/manifest.json found (so towns '
                         'from train and test both appear in the picker)')
    p.add_argument('--work-dir', default=None,
                    help='training work_dir; enables overlaying predicted '
                         'polylines from any *result*.json found under it '
                         '(or under val/<work-dir>, where the training eval '
                         'hook writes them)')
    p.add_argument('--port', type=int, default=5001)
    p.add_argument('--max-points', type=int, default=150000,
                    help='cap on points drawn in scatter modes (density mode '
                         'always uses every point so the histogram stays exact)')
    return p.parse_args()


def main():
    args = parse_args()
    STATE['data_root'] = args.data_root
    STATE['max_points'] = args.max_points
    STATE['work_dir'] = args.work_dir
    STATE['results_cache'] = {}
    if args.work_dir is not None and not osp.isdir(args.work_dir):
        hint = ''
        if osp.isfile(args.work_dir):
            hint = (f'\n  That is a file, not a directory -- did you mean '
                    f'its parent?\n    --work-dir {osp.dirname(args.work_dir)}')
        elif osp.isdir(osp.dirname(args.work_dir) or '.'):
            siblings = [d for d in os.listdir(osp.dirname(args.work_dir) or '.')
                        if osp.isdir(osp.join(osp.dirname(args.work_dir) or '.', d))]
            if siblings:
                hint = ('\n  Directories that do exist there: '
                        + ', '.join(sorted(siblings)[:8]))
        raise SystemExit(f'--work-dir is not a directory: {args.work_dir}{hint}')
    if not osp.isdir(args.data_root):
        raise SystemExit(f'no such data root: {args.data_root}')

    splits = discover_splits(args.data_root, only=args.split)
    if not splits:
        where = args.split or '<any subdirectory>'
        raise SystemExit(
            f'no manifest.json found under {args.data_root}/{where}')
    STATE['splits'] = splits
    tiles, town_split, town_counts = build_index(splits)
    STATE['tiles'] = tiles
    STATE['tiles_by_name'] = {t['name']: t for t in tiles}
    STATE['town_split'] = town_split
    STATE['town_counts'] = town_counts
    # lane_type_lookup is dataset-wide; take it from whichever split has it
    STATE['lane_types'] = next(
        (m['lane_type_lookup'] for m in splits.values()
         if 'lane_type_lookup' in m), {})

    # Flask runs with debug=False (the reloader is unreliable when the
    # process is backgrounded), so edits to this file do NOT take effect
    # until it's restarted. Print the file's mtime so a stale process is
    # obvious rather than looking like "my changes did nothing".
    mtime = datetime.fromtimestamp(osp.getmtime(osp.abspath(__file__)))
    for split, man in splits.items():
        print(f"  {split:>6}: {man['n_tiles']:>6,} tiles  "
              f"({', '.join(man['towns'])})")
    print(f"Loaded {len(tiles):,} tiles across {len(town_split)} towns "
          f"from {len(splits)} split(s)")
    print(f'Code version: {osp.basename(__file__)} last modified '
          f'{mtime:%Y-%m-%d %H:%M:%S}')
    print(f'Viewer on http://127.0.0.1:{args.port}')
    app.run(host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
