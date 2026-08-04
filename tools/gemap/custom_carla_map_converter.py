"""Convert the CARLA road-polyline tile dataset into MapTR's map-annotation
pkl format.

Unlike the nuScenes/AV2 converters, CARLA tiles are static 25m x 25m patches
(not per-timestamp driving-log frames), so there is no ego pose/SE3 transform
and no need to clip polylines against a moving patch -- this script only
needs to shift each tile's world-frame reference-line polylines into the
same frame as its point cloud, by subtracting the block's own `offset` (the
frame `LoadCarlaPointsFromFile` reads, and NOT the same as `tile_center` --
see the comment on the np.load call below).

Expected input layout (see projects/mmdet3d_plugin/datasets/carla_utils.py)::

    <data_root>/<split>/manifest.json
    <data_root>/<split>/blocks/<tile_name>.npz
    <data_root>/<split>/reference_lines/<tile_name>_reference_lines.json

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='CARLA map data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='root of the CARLA tile dataset (contains <split>/manifest.json)')
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
        'the label in the output filename')
    parser.add_argument(
        '--lane-types',
        type=int,
        nargs='+',
        default=None,
        help='optional subset of lane_type_lookup ids to keep as divider '
        'instances (default: keep all types)')
    return parser.parse_args()


def convert_carla_tiles(data_root, split, lane_types=None):
    split_dir = os.path.join(data_root, split)
    with open(os.path.join(split_dir, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    tile_radius = float(manifest.get('tile_radius', 12.5))

    samples = []
    total_instances = 0
    for idx, tile in enumerate(manifest['tiles']):
        name = tile['name']

        # Stored relative to data_root (joined back in
        # CustomCarlaLocalMapDataset.get_data_info), not absolute -- an
        # absolute path bakes in wherever this conversion happened to run,
        # which breaks the moment the pkl is read from a different
        # container/mount (e.g. converted under /MapTR, trained under
        # /workspace/GeMap). Matches CarlaSegDataset's pts_path convention.
        lidar_path = os.path.join(split, 'blocks', f'{name}.npz')
        abs_lidar_path = os.path.join(data_root, lidar_path)
        if not os.path.isfile(abs_lidar_path):
            raise FileNotFoundError(abs_lidar_path)

        # The polylines below are in WORLD coordinates and must be shifted
        # into the same frame as the LiDAR points the model actually sees.
        # That frame is the block's `offset`, NOT its `tile_center`:
        # LoadCarlaPointsFromFile reads `features[:, 0:3]`, and
        # `points - offset == features[:, :3]` holds exactly, while
        # tile_center differs from offset by a mean of ~2.4m (max >7m)
        # across the train split. Measured against real driving-surface
        # returns (label == 0), `- offset` puts polylines a median 0.038m
        # from the road vs 0.388m for `- tile_center`. Using tile_center
        # here (as this converter originally did) misaligns every GT
        # polyline against its own point cloud, which matters a lot given
        # chamfer eval thresholds of 0.5/1.0/1.5m.
        #
        # np.load is lazy, so this reads only the small `offset` array --
        # it does not pull the full point cloud into memory.
        with np.load(abs_lidar_path) as block:
            origin = np.asarray(block['offset'], dtype=np.float32)

        ref_path = os.path.join(split_dir, 'reference_lines',
                                f'{name}_reference_lines.json')
        with open(ref_path, encoding='utf-8') as f:
            ref = json.load(f)

        divider = []
        for poly in ref['polylines']:
            if lane_types is not None and poly['type'] not in lane_types:
                continue
            pts = np.array(poly['points'], dtype=np.float32)
            if pts.shape[0] < 2:
                continue
            divider.append(pts - origin)
            total_instances += 1

        # Sanity check only (tiles are asserted to already be the patch, so
        # no clipping is applied) -- warn, don't crash, if this ever fires.
        #
        # The bound is tile_radius *plus* how far `origin` sits from the
        # tile's geometric centre: coordinates here are relative to the
        # block's `offset`, which is not the centre, so a polyline spanning
        # the full tile legitimately reaches tile_radius + |tile_center -
        # origin| from the origin. The point cloud is displaced by exactly
        # the same amount (measured: |xy| up to ~18.4m against a 12.5m
        # radius), so comparing against a bare tile_radius here just fired
        # on most tiles and drowned out any real out-of-bounds case.
        tile_center = np.array(tile['center'], dtype=np.float32)
        margin = 1.0
        bound = tile_radius + float(np.abs(tile_center[:2] - origin[:2]).max()) \
            + margin
        for pts in divider:
            if np.any(np.abs(pts[:, :2]) > bound):
                print(f'[warn] {name}: polyline point(s) outside tile '
                      f'radius ({bound:.2f}m incl. origin offset + {margin}m '
                      f'margin), max abs xy = {np.abs(pts[:, :2]).max():.2f}')
                break

        samples.append(
            dict(
                lidar_path=lidar_path,
                sample_idx=name,
                token=name,
                timestamp=idx,
                town=tile.get('town'),
                tile_center=tile['center'],
                # the origin the annotation below is actually relative to --
                # recorded explicitly so the frame isn't ambiguous when
                # reading the pkl back (tile_center above is NOT it)
                annotation_origin=origin.tolist(),
                tile_bounds=tile.get('bounds'),
                annotation=dict(divider=divider),
            ))

    n = max(len(samples), 1)
    print(f'{split}: {len(samples)} tiles, {total_instances} divider '
          f'instances ({total_instances / n:.1f} per tile)')
    return samples


def main():
    args = parse_args()
    samples = convert_carla_tiles(args.data_root, args.split, args.lane_types)
    mmcv.mkdir_or_exist(args.out_dir)
    out_path = os.path.join(args.out_dir, f'carla_map_infos_{args.split}.pkl')
    mmcv.dump(
        dict(samples=samples, split=args.split, data_root=args.data_root),
        out_path)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
