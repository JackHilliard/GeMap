_base_ = [
    '../_base_/default_runtime.py'
]
#
# CARLA simulator LiDAR + vectorized map-GT data config.
#
# Map-annotated counterpart to carlasim_lidar.py: wires up
# `CustomCarlaLocalMapDataset` against the pkl produced by
# tools/maptrv2/custom_carla_map_converter.py, instead of `CarlaSegDataset`'s
# raw (GT-free) directory scan.
#
# Generate the pkl first:
#   python tools/maptrv2/custom_carla_map_converter.py \
#       --data-root /path/to/carla --out-dir data/carla/ --split test
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

dataset_type = 'CustomCarlaLocalMapDataset'
data_root = 'data/carla/'

# Only a `test` split subset is available locally; reused for train/val/test
# purely to exercise the pipeline (see maptrv2_carla_r50_24ep_lidar.py for
# the same caveat). For the full remote-cluster dataset, re-run the
# converter with --data-root pointed at the full dataset and update these
# to real train/val pkls -- no code changes needed.
ann_file_train = data_root + 'carla_map_infos_train.pkl'
ann_file_val = data_root + 'carla_map_infos_test.pkl'
ann_file_test = data_root + 'carla_map_infos_test.pkl'
map_ann_file = data_root + 'carla_map_gt.json'

# Matches the real 25m x 25m square CARLA tile (tile_radius=12.5); z-range
# is generous since map GT (divider polylines) is XY-only for code_size=2
# and this bound only needs to comfortably contain the LiDAR point cloud.
point_cloud_range = [-12.5, -12.5, -2.0, 12.5, 12.5, 24.0]
map_classes = ['divider']

# LiDAR points are [x, y, z, strength].
load_dim = 4
use_dim = 4
# Some tiles (confirmed: 6 in the full remote train set, all town03) contain
# LiDAR returns spanning 100+ meters in z within a single 25m x 25m tile --
# almost certainly a highway overpass/multi-level structure, not a flat
# driving surface. z_max=15.0 (tuned only against the flat local town10hd
# test subset) drops every point in those tiles, crashing
# extract_lidar_feat's voxelize() with zero surviving points. Raised to
# comfortably cover the observed range (z in [-66.90, 90.52] across the 6
# confirmed tiles) rather than filtering those tiles out -- see
# lidar_point_cloud_range in maptrv2_carla_r50_24ep_lidar.py, which must
# stay in sync with this value (kept >= its z upper bound so this isn't
# the tighter constraint).
z_max = 96.0

input_modality = dict(use_lidar=True, use_camera=False)

train_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        z_max=z_max),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]

# MultiScaleFlipAug3D isn't doing multi-scale/flip TTA here (flip=False) --
# it's required regardless because MapTRv2.forward_test() unconditionally
# indexes img_metas as a nested list (img_metas[0][0][...]), which only
# mmcv's test-time collation produces when the pipeline is wrapped this way.
test_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        z_max=z_max),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=map_classes),
            dict(type='CustomCollect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        # `classes` here is Custom3DDataset's generic per-object class list
        # (unused; we have no box-level classes), distinct from `map_classes`
        # below (the actual MapTRv2 head's map-element class list).
        classes=[],
        map_classes=map_classes,
        pc_range=point_cloud_range,
        modality=input_modality,
        box_type_3d='LiDAR',
        filter_empty_gt=True,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        classes=[],
        map_classes=map_classes,
        pc_range=point_cloud_range,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        classes=[],
        map_classes=map_classes,
        pc_range=point_cloud_range,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=True),
)
