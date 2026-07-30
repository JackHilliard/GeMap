"""Convert the CARLA road-polyline tile dataset into MapTR's map-annotation
pkl format.

Unlike the nuScenes/AV2 converters, CARLA tiles are static 25m x 25m patches
(not per-timestamp driving-log frames), so there is no ego pose/SE3 transform
and no need to clip polylines against a moving patch -- each tile's
reference-line polylines are already tile-local; this script only needs to
subtract the tile center to get tile-relative coordinates (matching the
`offset` convention `LoadCarlaPointsFromFile` already uses for point clouds).

Expected input layout (see projects/mmdet3d_plugin/datasets/carla_utils.py)::

    <data_root>/<split>/manifest.json
    <data_root>/<split>/blocks/<tile_name>.npz
    <data_root>/<split>/reference_lines/<tile_name>_reference_lines.json

Usage::

    python tools/maptrv2/custom_carla_map_converter.py \\
        --data-root /path/to/carla --out-dir data/carla/ --split test
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
        tile_center = np.array(tile['center'], dtype=np.float32)

        lidar_path = os.path.abspath(
            os.path.join(split_dir, 'blocks', f'{name}.npz'))
        if not os.path.isfile(lidar_path):
            raise FileNotFoundError(lidar_path)

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
            divider.append(pts - tile_center)
            total_instances += 1

        # Sanity check only (tiles are asserted to already be the patch, so
        # no clipping is applied) -- warn, don't crash, if this ever fires.
        margin = 1.0
        for pts in divider:
            if np.any(np.abs(pts[:, :2]) > tile_radius + margin):
                print(f'[warn] {name}: polyline point(s) outside tile '
                      f'radius ({tile_radius}+{margin}m margin), max abs xy '
                      f'= {np.abs(pts[:, :2]).max():.2f}')
                break

        samples.append(
            dict(
                lidar_path=lidar_path,
                sample_idx=name,
                token=name,
                timestamp=idx,
                town=tile.get('town'),
                tile_center=tile['center'],
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
