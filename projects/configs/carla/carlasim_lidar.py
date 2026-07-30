_base_ = [
    '../_base_/default_runtime.py'
]
#
# CARLA simulator LiDAR-only data config.
#
# This is a *dataloader* config: it wires up `CarlaSegDataset` +
# `LoadCarlaPointsFromFile` to load LiDAR point clouds (xyz + strength) from the
# CARLA road-block `.npz` files. There is no model/task head yet, so this config
# is meant for exercising the data path, e.g.:
#
#   python tools/misc/browse_dataset.py \
#       projects/configs/carla/carlasim_lidar.py --output-dir vis_carla
#
# or programmatically:
#
#   from mmcv import Config
#   from mmdet3d.datasets import build_dataset
#   cfg = Config.fromfile('projects/configs/carla/carlasim_lidar.py')
#   ds = build_dataset(cfg.data.train)
#   sample = ds[0]                 # {'img_metas': ..., 'points': DataContainer}
#
# Load the plugin so the custom dataset + transform get registered.
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

dataset_type = 'CarlaSegDataset'
# Local dev/test path (only a `test` split subset exists here so far).
# For the full remote-cluster dataset, point this at `data/carla/` (with
# `train`/`val` split subdirectories, each with its own manifest.json) instead.
data_root = '/home-local/johil9.nobkp/Documents/Code/carla/'

# LiDAR points are [x, y, z, strength]; the z<=15.0 filter matches the source
# Pointcept dataset.
load_dim = 4
use_dim = 4
z_max = 15.0

input_modality = dict(use_lidar=True, use_camera=False)

# No task classes while the dataset is LiDAR-only.
class_names = []

train_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        z_max=z_max),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_gt=False,
        with_label=False),
    dict(type='CustomCollect3D', keys=['points'])
]

test_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        z_max=z_max),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_gt=False,
        with_label=False),
    dict(type='CustomCollect3D', keys=['points'])
]

# Only a `test` split subset is available locally; reused for train/val/test
# here purely to exercise the data path (see carlasim_map.py for the same
# caveat on the map-annotated MapTRv2 training config).
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        box_type_3d='LiDAR',
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        box_type_3d='LiDAR',
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        box_type_3d='LiDAR',
        test_mode=True),
)
