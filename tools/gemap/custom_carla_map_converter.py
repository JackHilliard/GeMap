"""Convert the CARLA road-polyline tile dataset into GeMap's map-annotation
pkl format.

Unlike the nuScenes/AV2 converters, CARLA tiles are static square patches
(not per-timestamp driving-log frames), so there is no ego pose/SE3
transform and no need to clip polylines against a moving patch. What this
script does need to do is put the polylines and the point cloud in one
common frame, and that frame is the **tile centre** -- see below.

The tile-centred frame
----------------------
A block's ``features[:, 0:3]`` is stored relative to the block's own
``offset``, and ``points - offset == features[:, :3]`` holds exactly. But
``offset`` is *not* the tile centre: it is displaced from ``tile_center``
by a mean of 1.25m and up to 12.07m on the 25m export, and by up to ~17m
on the 60m one -- the displacement scales with tile size. Because the
model's patch (``pc_range``) is symmetric about the origin, an
offset-framed tile does not line up with it: measured on the 25m train
split, 16% of GT points and 82% of divider instances fell outside
+/-12.5, and 651/4103 tiles kept under 90% of their LiDAR points.

So this converter subtracts ``tile_center`` from the world-frame polylines
and records ``recenter_shift = tile_center - offset``, which
``LoadCarlaPointsFromFile`` subtracts from the point cloud. Both are
translated by the same vector, so their relative alignment -- a median
0.038m from real driving-surface returns -- is preserved exactly, while
the tile now sits squarely in ``[-tile_radius, +tile_radius]`` on both
axes. Verified on a real tile: the offset frame spans xy
[-13.85, 11.40] against a 12.5m radius, the re-centred frame spans
[-12.4999, 12.4999] with 100% of points inside.

This is what makes tile size a pure configuration number: the patch is
exactly the tile at any radius, so nothing downstream has to know about
25m specifically.

Nothing here assumes a particular tile size. The exporter has produced at
least 25m tiles (``tile_radius`` 12.5) and 60m ones (``tile_radius`` 30.0),
and both layouts it writes are accepted::

    <data_root>/<split>/manifest.json            # split export
    <data_root>/<split>/blocks/<tile>.npz
    <data_root>/<split>/reference_lines/<tile>_reference_lines.json

    <data_root>/grid_manifest.json               # single-town grid export
    <data_root>/blocks/<tile>.npz                # (no <split> level, and
    <data_root>/reference_lines/<tile>_...json   #  tile names carry no town)

Where a directory holds both files, ``manifest.json`` wins and
``grid_manifest.json`` only backfills keys it lacks -- the same rule
``tools/gemap/dataset_viewer.py`` uses. ``manifest.json`` is written by a
later packaging step and is the *curated* view: in the Town10HD grid export
it lists 30 tiles where grid_manifest.json lists 33, and the 3 it omits
contain only sidewalk_edge polylines and no driving centerline at all
(``gt_source: driving_lanes``). Its dataset-level *counts* are still not
trustworthy -- the same file reports ``n_tiles: 4103`` while listing 30 --
so everything here is derived from the tile entries, never from those
counts.

Each tile's LiDAR block is also scanned for points that would actually
survive the training pipeline's filters (``z <= --z-max`` and inside
``--lidar-point-cloud-range``). Tiles with fewer than ``--min-lidar-points``
such points would voxelize to *zero* voxels and crash ``extract_lidar_feat``,
so they are dropped from the pkl and listed in a sidecar report. Every kept
sample records its own count so the dataset can re-check cheaply. The
default range follows the manifest's tile size rather than a hardcoded
+/-12.5, so the scan describes the tiles being converted.

Usage, from the GeMap root::

    python tools/gemap/custom_carla_map_converter.py \\
        --data-root data/carla/ --out-dir data/carla/ --split train

The pkl records each tile's path relative to --data-root, and the dataset
rejoins it against `raw_data_root` from the config (which defaults to
`data/carla/`, repo-relative). Keeping --data-root repo-relative and equal
to that default is what makes the resulting pkl portable across machines and
containers -- passing an absolute --data-root here still works, but then
raw_data_root must be overridden to match wherever that was.
"""

import argparse
import json
import os

import mmcv
import numpy as np

# z half of the range the training config's LiDAR branch uses
# (`lidar_point_cloud_range` in projects/configs/carla/
# gemap_carla_r50_24ep_lidar.py) and `z_max` from
# projects/configs/carla/carlasim_map.py. The xy half is NOT a constant:
# it follows the tile size read from the manifest (see default_pc_range),
# because a range narrower than the tile silently crops it and a wider one
# just wastes BEV cells. The whole range is recorded into the pkl so the
# dataset can warn if config and pkl ever drift apart.
DEFAULT_Z_RANGE = (-72.0, 96.0)
DEFAULT_Z_MAX = 96.0
# Only used when a manifest carries no tile geometry at all; matches the
# original 25m export this converter was written against.
FALLBACK_TILE_RADIUS = 12.5
# Slack allowed before a polyline is reported as leaving its own tile. The
# exporter samples arcs at a fixed arc_gap, so a vertex can land slightly
# past the boundary without anything being wrong.
BOUNDS_MARGIN = 1.0

MANIFEST_NAMES = ('manifest.json', 'grid_manifest.json')

# Written into the pkl so a reader can tell which frame the annotations are
# in. Pkls produced before the 2026-08-10 re-centring carry no such key and
# are in the block `offset` frame; CustomCarlaLocalMapDataset warns on them
# rather than silently training against a misaligned patch.
ANNOTATION_FRAME = 'tile_center'


def parse_args():
    parser = argparse.ArgumentParser(
        description='CARLA map data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='root of the CARLA tile dataset (contains <split>/manifest.json, '
        'or a manifest.json/grid_manifest.json directly)')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/carla/',
        help='output directory for the generated pkl')
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split subdirectory under --data-root to convert; also used as '
        'the label in the output filename. If no such subdirectory exists, '
        '--data-root itself is read and this is only the output label')
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        metavar='CLASS',
        help="optional subset of polyline classes to keep as divider "
        "instances, as class_lookup ids or names (e.g. `0` or "
        "`driving_centerline`). Default: keep every polyline. Only exports "
        "whose polylines carry a class can be filtered")
    parser.add_argument(
        '--lane-types',
        type=str,
        nargs='+',
        default=None,
        metavar='CLASS',
        help='deprecated alias for --classes (it never matched anything: it '
        "compared lane_type_lookup ids against each polyline's `type`, which "
        "holds a geometry kind such as 'arc'/'straight')")
    parser.add_argument(
        '--lidar-point-cloud-range',
        type=float,
        nargs=6,
        default=None,
        metavar=('X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX'),
        help='range the LiDAR voxelizer will use; points outside it are '
        'dropped before voxelization, so a tile with none inside produces '
        'zero voxels. Default: +/-tile_radius in xy (read from the manifest) '
        f'and {DEFAULT_Z_RANGE[0]}..{DEFAULT_Z_RANGE[1]} in z')
    parser.add_argument(
        '--z-max',
        type=float,
        default=DEFAULT_Z_MAX,
        help="LoadCarlaPointsFromFile's early z filter (default: matches "
        'the training config)')
    parser.add_argument(
        '--min-lidar-points',
        type=int,
        default=1,
        help='drop tiles with fewer than this many in-range LiDAR points; '
        '1 drops only genuinely zero-voxel tiles (default: 1)')
    parser.add_argument(
        '--no-lidar-check',
        action='store_true',
        help='skip the per-tile LiDAR scan entirely (faster, but no tile is '
        'dropped and no point counts are recorded)')
    return parser.parse_args()


def load_manifest(data_root, split):
    """Locate and load the manifest describing one set of tiles.

    Accepts either export layout: ``<data_root>/<split>/`` (the split
    export) or ``<data_root>`` itself (a grid export, which has no split
    level). Returns ``(tile_dir, manifest)``; `tile_dir` is where `blocks/`
    and `reference_lines/` live.
    """
    for candidate in ([os.path.join(data_root, split)] if split else
                      []) + [data_root]:
        present = [
            n for n in MANIFEST_NAMES
            if os.path.isfile(os.path.join(candidate, n))
        ]
        if not present:
            continue
        manifest = {}
        # reversed so the preferred name (first in MANIFEST_NAMES) is applied
        # last and its keys win, while the other only backfills. A null value
        # never overwrites a real one -- grid_manifest.json spells absent
        # options as `null` rather than omitting them.
        for name in reversed(present):
            with open(os.path.join(candidate, name), encoding='utf-8') as f:
                loaded = json.load(f)
            manifest.update({
                k: v
                for k, v in loaded.items()
                if v is not None or k not in manifest
            })
        if not manifest.get('tiles'):
            raise ValueError(f'{candidate}: manifest lists no tiles')
        return candidate, manifest
    raise FileNotFoundError(
        f'no {" or ".join(MANIFEST_NAMES)} under {data_root!r}'
        f'{f" or {os.path.join(data_root, split)!r}" if split else ""}')


def manifest_tile_radius(manifest):
    """Half a tile's side, in metres, from whichever key the export used.

    `tile_radius` (25m export) and `tile_side` (some grid manifests) are
    dataset-level; failing both, the widest per-tile `bounds` gives the same
    answer for a uniform grid and an upper bound otherwise, which is the
    safe direction for sizing a point-cloud range. Returns None only when an
    export states its geometry nowhere -- the per-tile bounds check can
    still fall back to each reference-lines json, so that is not fatal.
    """
    if manifest.get('tile_radius') is not None:
        return float(manifest['tile_radius'])
    if manifest.get('tile_side') is not None:
        return float(manifest['tile_side']) / 2.0
    radii = []
    for tile in manifest.get('tiles', []):
        bounds = tile.get('bounds')
        if bounds is None:
            continue
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:
            radii.append(np.max(bounds[2:] - bounds[:2]) / 2.0)
        elif bounds.size == 6:
            radii.append(np.max(bounds[3:5] - bounds[:2]) / 2.0)
    return float(max(radii)) if radii else None


def tile_footprint(tile, ref, tile_radius):
    """World-frame xy footprint of one tile, as ``(lo, hi)`` arrays.

    Prefers the tile's own `bounds` (exact, and the only source that would
    describe a non-square tile), then its centre plus the export's tile
    radius. The reference-lines json repeats all three keys, so it backs up
    a manifest whose tile entries are sparse. Returns None when nothing
    supplies a footprint, which only disables the bounds warning below.
    """
    bounds = tile.get('bounds') or ref.get('tile_bounds')
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:  # [x_min, y_min, x_max, y_max]
            return bounds[:2], bounds[2:]
        if bounds.size == 6:  # [x_min, y_min, z_min, x_max, y_max, z_max]
            return bounds[:2], bounds[3:5]

    center = tile.get('center') or ref.get('tile_center')
    radius = ref.get('tile_radius')
    radius = float(radius) if radius is not None else tile_radius
    if center is None or radius is None:
        return None
    center = np.asarray(center, dtype=np.float64)[:2]
    return center - radius, center + radius


def default_pc_range(tile_radius):
    """The range a training config for this tile size would use: square in
    xy, matching the tile, with the config's z span.

    Exact now that annotations and points are tile-centred -- the tile
    occupies precisely [-radius, +radius] on both axes, so this neither
    crops the tile nor wastes BEV cells on empty margin.
    """
    radius = tile_radius if tile_radius is not None else FALLBACK_TILE_RADIUS
    z_min, z_max = DEFAULT_Z_RANGE
    return [-radius, -radius, z_min, radius, radius, z_max]


def resolve_classes(selected, manifest):
    """Map ``--classes`` entries (ids or names) onto class_lookup ids.

    Returns a set of int ids, or None for "keep everything". Raises if the
    export has no class taxonomy or a name/id is not in it -- silently
    keeping nothing is how the old --lane-types flag behaved, and it
    produced an empty pkl with no indication why.
    """
    if not selected:
        return None
    lookup = manifest.get('class_lookup') or {}
    if not lookup:
        raise ValueError(
            'this export has no `class_lookup`, so its polylines carry no '
            'class and cannot be filtered; re-run without --classes')
    by_name = {name: int(cid) for cid, name in lookup.items()}
    out = set()
    for entry in selected:
        entry = str(entry)
        if entry in by_name:
            out.add(by_name[entry])
        elif entry.lstrip('-').isdigit() and str(int(entry)) in lookup:
            out.add(int(entry))
        else:
            known = ', '.join(
                f'{cid}={name}' for cid, name in sorted(lookup.items()))
            raise ValueError(
                f'unknown class {entry!r}; this export has: {known}')
    return out


def polyline_class_id(poly):
    """The polyline's class_lookup id, or None on a class-free export.

    Deliberately does not fall back to `type`: that key exists in every
    export and holds a geometry kind ('arc'/'straight'), not a class.
    """
    if poly.get('class_id') is not None:
        return int(poly['class_id'])
    return None


def count_points_in_range(lidar_path, recenter_shift, pc_range, z_max):
    """Count a block's points that would survive the training pipeline.

    Mirrors what the model actually sees, in order: the ``recenter_shift``
    ``LoadCarlaPointsFromFile`` applies, then its ``z <= z_max`` filter,
    then the LiDAR ``Voxelization``'s own range filtering.
    ``GridSamplePoints`` sits between the last two but only *clamps* grid
    coordinates -- it never drops points and never moves them -- so it
    cannot change this count.

    Returns:
        tuple[int, int]: raw point count, and count surviving both filters.
    """
    with np.load(lidar_path) as block:
        xyz = np.asarray(block['features'][:, :3], dtype=np.float32)
    n_raw = int(xyz.shape[0])
    xyz = xyz - np.asarray(recenter_shift, dtype=np.float32)
    if z_max is not None:
        xyz = xyz[xyz[:, 2] <= z_max]
    lo = np.asarray(pc_range[:3], dtype=np.float32)
    hi = np.asarray(pc_range[3:], dtype=np.float32)
    # `>= lo` / `< hi` mirrors hard_voxelize's own
    # `floor((p - lo) / voxel) in [0, grid)` convention rather than
    # PointsRangeFilter.in_range_3d's strict-both-sides test. The two differ
    # only for points exactly on a boundary plane, which can never flip a
    # zero/non-zero verdict in practice.
    n_in_range = int(np.all((xyz >= lo) & (xyz < hi), axis=1).sum())
    return n_raw, n_in_range


def convert_carla_tiles(data_root,
                        split,
                        classes=None,
                        pc_range=None,
                        z_max=DEFAULT_Z_MAX,
                        min_lidar_points=1,
                        lidar_check=True):
    tile_dir, manifest = load_manifest(data_root, split)
    tile_radius = manifest_tile_radius(manifest)
    keep_classes = resolve_classes(classes, manifest)
    if pc_range is None:
        pc_range = default_pc_range(tile_radius)
        if tile_radius is None:
            print(f'[warn] this export states no tile size; assuming '
                  f'tile_radius={FALLBACK_TILE_RADIUS}m for the point-cloud '
                  'range. Pass --lidar-point-cloud-range if that is wrong')
    print(f'{tile_dir}: {len(manifest["tiles"])} tiles, tile_radius='
          f'{tile_radius if tile_radius is not None else "unknown"}, '
          f'pc_range={[round(v, 2) for v in pc_range]}')

    samples = []
    dropped = []
    out_of_bounds = []
    coverage = []
    total_instances = 0
    prog_bar = mmcv.ProgressBar(len(manifest['tiles']))
    for idx, tile in enumerate(manifest['tiles']):
        prog_bar.update()
        name = tile['name']

        # Stored relative to data_root (joined back in
        # CustomCarlaLocalMapDataset.get_data_info), not absolute -- an
        # absolute path bakes in wherever this conversion happened to run,
        # which breaks the moment the pkl is read from a different
        # container/mount (e.g. converted under /MapTR, trained under
        # /workspace/GeMap). Matches CarlaSegDataset's pts_path convention.
        lidar_path = os.path.relpath(
            os.path.join(tile_dir, 'blocks', f'{name}.npz'), data_root)
        abs_lidar_path = os.path.join(data_root, lidar_path)
        if not os.path.isfile(abs_lidar_path):
            raise FileNotFoundError(abs_lidar_path)

        # `offset` is the frame features[:, 0:3] is stored in; `tile_center`
        # is the frame everything downstream works in. The difference is
        # what LoadCarlaPointsFromFile subtracts from the point cloud, so
        # that points and the polylines below -- shifted by the same
        # tile_center -- stay in exact relative alignment while the tile
        # lands squarely inside +/-tile_radius. See this module's docstring
        # for why the offset frame is not usable as-is.
        #
        # np.load is lazy, so this reads only the small `offset` array --
        # it does not pull the full point cloud into memory. (The separate
        # count_points_in_range() call below does read the full `features`
        # array when --no-lidar-check is off; that is the expensive part of
        # this loop, ~14ms for a 110K-point tile.)
        with np.load(abs_lidar_path) as block:
            offset = np.asarray(block['offset'], dtype=np.float32)
        origin = np.asarray(tile['center'], dtype=np.float32)
        recenter_shift = origin - offset

        n_raw = n_in_range = None
        if lidar_check:
            n_raw, n_in_range = count_points_in_range(abs_lidar_path,
                                                      recenter_shift, pc_range,
                                                      z_max)
            if n_in_range < min_lidar_points:
                # This tile would voxelize to zero (or near-zero) voxels and
                # crash extract_lidar_feat mid-run. Drop it before its
                # polylines are counted, so the reported instance totals
                # describe what training will actually see.
                dropped.append(
                    dict(
                        name=name,
                        town=tile.get('town'),
                        n_points=n_raw,
                        n_points_in_range=n_in_range))
                continue
            if n_raw:
                coverage.append((n_in_range / n_raw, name))

        ref_path = os.path.join(tile_dir, 'reference_lines',
                                f'{name}_reference_lines.json')
        with open(ref_path, encoding='utf-8') as f:
            ref = json.load(f)

        divider = []
        for poly in ref['polylines']:
            if keep_classes is not None and \
                    polyline_class_id(poly) not in keep_classes:
                continue
            pts = np.array(poly['points'], dtype=np.float32)
            if pts.shape[0] < 2:
                continue
            divider.append(pts - origin)
            total_instances += 1

        # Sanity check only (tiles are asserted to already be the patch, so
        # no clipping is applied) -- collect, don't crash, if this fires.
        #
        # Compared against the tile's own footprint shifted into the
        # annotation frame. Now that `origin` is `tile_center`, that shift
        # reduces to +/-tile_radius for a square tile, but going through
        # `bounds` keeps the check exact and would still describe a
        # non-square tile. Before re-centring this had to be loosened by
        # |tile_center - offset| and still fired on 1750/4103 train tiles.
        footprint = tile_footprint(tile, ref, tile_radius)
        if footprint is not None and divider:
            lo, hi = (footprint[0] - origin[:2] - BOUNDS_MARGIN,
                      footprint[1] - origin[:2] + BOUNDS_MARGIN)
            pts = np.concatenate([p[:, :2] for p in divider])
            overshoot = float(
                np.max(np.maximum(lo - pts, pts - hi), initial=0.0))
            if overshoot > 0:
                out_of_bounds.append(dict(name=name, overshoot=overshoot))

        samples.append(
            dict(
                lidar_path=lidar_path,
                sample_idx=name,
                token=name,
                timestamp=idx,
                town=tile.get('town'),
                tile_center=tile['center'],
                # The origin the annotation below is relative to. Equal to
                # tile_center since the 2026-08-10 re-centring; kept as its
                # own key because the frame must never be ambiguous when
                # reading the pkl back.
                annotation_origin=origin.tolist(),
                # What LoadCarlaPointsFromFile subtracts from
                # features[:, 0:3] to bring the point cloud into that same
                # frame. Precomputed here so the loader needs no second
                # np.load and no knowledge of how the frame is defined.
                recenter_shift=recenter_shift.tolist(),
                tile_bounds=tile.get('bounds'),
                tile_radius=tile_radius,
                # None when --no-lidar-check was passed; the dataset treats
                # a missing count as "unknown" and keeps the sample.
                num_lidar_points=n_raw,
                num_lidar_points_in_range=n_in_range,
                annotation=dict(divider=divider),
            ))

    n = max(len(samples), 1)
    print()
    if lidar_check:
        print(f'{split}: {len(samples)} tiles kept, {len(dropped)} dropped '
              f'(<{min_lidar_points} in-range LiDAR point'
              f'{"" if min_lidar_points == 1 else "s"}), {total_instances} '
              f'divider instances ({total_instances / n:.1f} per tile)')
        for entry in dropped[:20]:
            print(f'  [drop] {entry["name"]}: {entry["n_points"]} raw points, '
                  f'{entry["n_points_in_range"]} in range')
        if len(dropped) > 20:
            # A long list usually means the range is wrong, not the data --
            # the sidecar json below has the rest.
            print(f'  [drop] ... and {len(dropped) - 20} more')
    else:
        print(f'{split}: {len(samples)} tiles, {total_instances} divider '
              f'instances ({total_instances / n:.1f} per tile) '
              '[LiDAR check skipped]')
    report_coverage(coverage, pc_range)
    report_out_of_bounds(out_of_bounds)
    return samples, dropped, dict(
        tile_dir=tile_dir,
        tile_radius=tile_radius,
        tile_side=manifest.get('tile_side'),
        pc_range=list(pc_range),
        class_lookup=manifest.get('class_lookup') or {},
        classes_kept=sorted(keep_classes) if keep_classes else None,
        n_out_of_bounds=len(out_of_bounds))


def report_coverage(coverage, pc_range):
    """How much of each tile the configured range actually keeps.

    A range narrower than the tile crops it. With tile-centred annotations
    this should sit at ~100% for every tile, so anything lower means the
    range was overridden with one copied from a differently-sized export.
    (In the old `offset` frame it was structurally below 100%: the range
    was square around `offset` while the tile was displaced from it, which
    put 651/4103 train tiles under 90%.)
    """
    if not coverage:
        return
    fracs = np.array([c for c, _ in coverage])
    worst_frac, worst_name = min(coverage)
    poor = int((fracs < 0.9).sum())
    print(f'  [range] median {np.median(fracs):.1%} of raw points fall inside '
          f'{[round(v, 2) for v in pc_range]}; worst tile {worst_name} '
          f'{worst_frac:.1%}')
    if poor:
        print(f'  [range] {poor}/{len(fracs)} tiles keep <90% of their points '
              '-- check the range matches this export\'s tile size')


def report_out_of_bounds(out_of_bounds):
    if not out_of_bounds:
        return
    print(f'  [warn] {len(out_of_bounds)} tile(s) have polyline points '
          f'outside their own tile bounds (+{BOUNDS_MARGIN}m margin):')
    for entry in sorted(out_of_bounds, key=lambda e: -e['overshoot'])[:10]:
        print(f'  [warn]   {entry["name"]}: {entry["overshoot"]:.2f}m past '
              'the boundary')
    if len(out_of_bounds) > 10:
        print(f'  [warn]   ... and {len(out_of_bounds) - 10} more')


def main():
    args = parse_args()
    lidar_check = not args.no_lidar_check
    classes = args.classes
    if args.lane_types:
        print('[warn] --lane-types is deprecated and never matched anything; '
              'treating it as --classes (matched against class_lookup)')
        classes = (classes or []) + args.lane_types
    samples, dropped, meta = convert_carla_tiles(
        args.data_root,
        args.split,
        classes,
        pc_range=args.lidar_point_cloud_range,
        z_max=args.z_max,
        min_lidar_points=args.min_lidar_points,
        lidar_check=lidar_check)
    # Derived from the manifest's tile size inside convert_carla_tiles when
    # not given on the CLI, so read it back rather than re-deriving it here.
    pc_range = meta['pc_range']
    mmcv.mkdir_or_exist(args.out_dir)
    out_path = os.path.join(args.out_dir, f'carla_map_infos_{args.split}.pkl')
    mmcv.dump(
        dict(
            samples=samples,
            split=args.split,
            data_root=args.data_root,
            # Which frame `annotation` is expressed in. A pkl without this
            # key predates the re-centring and is in the block `offset`
            # frame; the dataset refuses to train against one silently.
            annotation_frame=ANNOTATION_FRAME,
            # The tile geometry these samples were built from, so a later
            # reader can tell a 25m export from a 60m one without reopening
            # the source manifest. CustomCarlaLocalMapDataset checks its
            # configured pc_range against tile_radius on load.
            tile_geometry=dict(
                tile_dir=meta['tile_dir'],
                tile_radius=meta['tile_radius'],
                tile_side=meta['tile_side']),
            class_lookup=meta['class_lookup'],
            classes_kept=meta['classes_kept'],
            # Records the geometry the per-sample counts were measured
            # against, so CustomCarlaLocalMapDataset can warn if the config
            # it is running under no longer matches.
            lidar_check=dict(
                enabled=lidar_check,
                point_cloud_range=list(pc_range),
                z_max=args.z_max,
                min_lidar_points=args.min_lidar_points),
            dropped_tiles=dropped),
        out_path)
    print(f'Saved {out_path}')

    if lidar_check:
        # Sidecar report, so a cluster run can be inspected without
        # unpickling the (large) infos file.
        report_path = os.path.join(
            args.out_dir, f'carla_map_infos_{args.split}_dropped.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(
                dict(
                    split=args.split,
                    data_root=args.data_root,
                    tile_radius=meta['tile_radius'],
                    point_cloud_range=list(pc_range),
                    z_max=args.z_max,
                    min_lidar_points=args.min_lidar_points,
                    n_kept=len(samples),
                    n_dropped=len(dropped),
                    dropped=dropped),
                f,
                indent=2)
        print(f'Saved {report_path}')


if __name__ == '__main__':
    main()
