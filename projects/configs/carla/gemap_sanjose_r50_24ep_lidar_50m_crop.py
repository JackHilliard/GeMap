"""HRDX San Jose 50 m tiles -> rotated 30 m crops, one class (curb).

GeMap's side of the San Jose benchmark: a thin overlay on
`gemap_carla_r50_24ep_lidar_50m_crop.py` mirroring the MapTRv2 repo's
`maptrv2_sanjose_r50_24ep_lidar_50m_crop.py`. The MODEL and the crop recipe
are the CARLA 50 m ones unchanged (30 m crops, uniform yaw + <= 3.79 m shift
+ sigma 0.02 m noise at train; frozen, unrotated, aimed crops at eval;
colour-free 3-channel points). What changes is the data:

* **The export is `san_jose_tiles_50m`** (HRDX San Jose, 17723 tiles of
  real LiDAR, `tile_radius` 25, world-frame points). Its layout is FLAT --
  `blocks/` and `reference_lines/` sit directly under the root, no
  `<split>/` level -- and the split manifests live in a separate directory,
  `san_jose_splits_v1/v1/{train,val,test}.json` (12073 / 1312 / 1978 tiles;
  a leakage-free 750 m-block spatial split). `CustomSanJose50mCropDataset`
  reads both.
* **One class, `curb`.** The export's taxonomy is `{1: curb,
  2: curb_parametric}`; the parametric class re-traces the same curbs, so
  only `curb` is trained. `map_classes = ['curb']`, `aux_seg.seg_classes =
  1`; `num_vec_one2one` 50 x 1 class keeps the inherited `max_num=50` legal.
* **The splits were finalised with the MapTRv2 loader's crop code** (the
  same code as this repo's `CustomCarla50mCropDataset`): every val/test
  tile has a non-empty, >= 1000-point crop at the frozen pose
  `default_rng(crop_seed=0 + index)` in manifest order, every train tile a
  non-empty crop for one of four seeds. The dataset checks the manifests'
  `finalize` block against its own kwargs and warns if they drift.
* **val != test here**, so the two frozen splits get their OWN
  `map_ann_file`s -- `_format_gt()` writes a GT json once and never
  regenerates it. Delete both whenever the tile set, crop kwargs or seed
  change.

Data layout (relative to the repo root):

    data/san_jose_50m/blocks/<name>.npz
    data/san_jose_50m/reference_lines/<name>_reference_lines.json
    data/san_jose_splits_v1/v1/{train,val,test}.json
"""

_base_ = ['./gemap_carla_r50_24ep_lidar_50m_crop.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ---- data source -----------------------------------------------------------
dataset_type = 'CustomSanJose50mCropDataset'
data_root = 'data/san_jose_50m/'
splits_root = 'data/san_jose_splits_v1/v1/'
ann_file_train = splits_root + 'train.json'
ann_file_val = splits_root + 'val.json'
ann_file_test = splits_root + 'test.json'
# One GT json PER frozen split (see the docstring).
map_ann_file_val = 'data/carla30/sanjose_map_gt_50m_crop_curb_val.json'
map_ann_file_test = 'data/carla30/sanjose_map_gt_50m_crop_curb_test.json'

# ---- taxonomy --------------------------------------------------------------
map_classes = ['curb']
num_map_classes = len(map_classes)

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=num_map_classes,
    feat_down_sample=32,
    pv_thickness=1,
)

# ---- crop geometry (== the 50 m base; restated for config isolation) -------
tile_radius = 15.0
crop_size = 2 * tile_radius
lidar_point_cloud_range = [
    -tile_radius, -tile_radius, -72.0, tile_radius, tile_radius, 96.0
]
lidar_voxel_size = [0.1, 0.1, 0.4]

crop_common = dict(
    # flat export: no <split>/ level. Stated explicitly because mmcv merges
    # dicts recursively and the base's split='train'/'test' would survive.
    split=None,
    crop_size=crop_size,
    require_gt=True,
    gt_tries=32,
    min_polyline_len=2.0,
    gt_classes=('curb',),
    reference_dir='reference_lines',
)
crop_train = dict(rotate=True, translate=True, noise_std=0.02, **crop_common)
# Frozen by seed, no yaw, no noise; crop_seed=0 is what the splits were
# finalised with.
crop_eval = dict(rotate=False, translate=True, noise_std=0.0, crop_seed=0,
                 **crop_common)

# ---- pipelines (== the 50 m base, restated with this file's class list) ----
train_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(
        type='GridSamplePoints',
        grid_size=lidar_voxel_size,
        point_cloud_range=lidar_point_cloud_range),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]
test_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(
        type='GridSamplePoints',
        grid_size=lidar_voxel_size,
        point_cloud_range=lidar_point_cloud_range),
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

model = dict(
    pts_bbox_head=dict(
        num_classes=num_map_classes,
        aux_seg=aux_seg_cfg,
        bbox_coder=dict(num_classes=num_map_classes)))

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        map_classes=map_classes,
        aux_seg=aux_seg_cfg,
        **crop_train),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        map_ann_file=map_ann_file_val,
        pipeline=test_pipeline,
        map_classes=map_classes,
        **crop_eval),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        map_ann_file=map_ann_file_test,
        pipeline=test_pipeline,
        map_classes=map_classes,
        **crop_eval),
)

evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='CarlaMap_chamfer/mAP', rule='greater')
