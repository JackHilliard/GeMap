"""50 m CARLA tiles -> rotated 30 m crops, two classes (driving / curb).

A thin overlay on `gemap_carla_r50_24ep_lidar_30m.py`. The MODEL is the 30 m
one unchanged -- same `point_cloud_range` (+-15), `sparse_shape`, 100x100 BEV
(0.3 m/cell), coder and assigner ranges, colour-free `SparseEncoder(
in_channels=3)` -- because every crop the network sees IS a 30 m tile. What
changes is where those tiles come from. This is the GeMap side of the 50 m
crop benchmark defined in the MapTRv2 repo
(`maptrv2_carla_r50_24ep_lidar_50m_crop.py`); the crop code is the same, so
the two repos score identical eval crops for a given `crop_seed`.

* **The data are 50 m tiles on disk** (`tile_radius` 25 m), read straight from
  the export's `<split>/manifest.json` by `CustomCarla50mCropDataset` -- there
  is no converter pkl. Each `__getitem__` draws a uniform yaw about the tile
  centre plus a centre shift of up to 3.79 m (25 - 15*sqrt2, the slack a
  rotated 30 m square leaves inside a 50 m one, so every crop is fully
  populated), crops a 30 m square, and adds sigma = 0.02 m Gaussian noise to
  the points. GT polylines get the SAME pose and are clipped to the same box
  with the tiler's own `clip_polyline_to_box`.
* **Val/test crops are frozen by `crop_seed`**, not centred: `_format_gt()`
  caches the eval GT to `map_ann_file` once, so the crop each tile is scored
  on must be identical on every run. A *centred* crop would leave ~24% of the
  test tiles with no GT at all (the manifest was filtered for a *reachable*
  non-empty crop, found by aiming); seeded aiming drops that to <1%. No yaw
  at eval, so map quality is not conflated with rotation robustness.
* **The crop frame replaces the tile_center frame.** The crop centre is the
  origin by construction, so `LoadCarla50mCrop` does the reframing itself and
  there is no `recenter_shift` / `LoadCarlaPointsFromFile` in these pipelines.
  Points' z is re-zeroed per crop on its own 2nd-percentile ground.
* **Two classes, `driving` and `curb`**, in that (label) order -- the export's
  own names. Labels come from `map_classes` order via `VectorizedCarlaLocalMap`
  (the AV2 base's hardcoded CLASS2LABEL would silently vectorise these names
  to zero instances). `aux_seg.seg_classes` is raised to 2 so the auxiliary
  BEV mask is per class; restated on BOTH sides (model `aux_seg` and
  `data.train.aux_seg`) because the head builds its seg conv from one and the
  dataset rasterises from the other.
* **Colour-free points**: `LoadCarla50mCrop` defaults to `load_dim=3,
  use_dim=3`, matching the inherited `in_channels=3`.
* `samples_per_gpu` is inherited (1, from carlasim_map.py). The MapTRv2 side
  pins 16 for an 80 GB H100; raise it with `--cfg-options
  data.samples_per_gpu=N` for a real run.

Data layout expected under `data_root` (the export root, NOT `data/carla30/`):

    <data_root>/<split>/manifest.json               tiles: [{name, center}]
    <data_root>/<split>/blocks/<name>.npz           points (world), features,
                                                    tile_center, tile_radius
    <data_root>/<split>/reference_lines/<name>_reference_lines.json

The eval GT json is written next to the other CARLA GT files and must not be
shared with any other config (delete it whenever the tile set, crop kwargs
or `crop_seed` change -- `_format_gt()` never regenerates an existing file).

Smoke-tested 2026-10-13 on `../carla_test_V2` (2170 test tiles, radius 25):
see tools/gemap/verify_crop_dataset.py.
"""

_base_ = ['./gemap_carla_r50_24ep_lidar_30m.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ---- data source -----------------------------------------------------------
dataset_type = 'CustomCarla50mCropDataset'
data_root = 'data/carla50m/'
ann_file_train = data_root + 'train/manifest.json'
ann_file_val = data_root + 'test/manifest.json'
ann_file_test = data_root + 'test/manifest.json'
# Own GT json: _format_gt() writes it once and never regenerates, and the
# frozen eval crops below are what it records.
map_ann_file = 'data/carla30/carla_map_gt_50m_crop_2cls.json'

# ---- taxonomy --------------------------------------------------------------
map_classes = ['driving', 'curb']
num_map_classes = len(map_classes)

# Per-class auxiliary BEV mask (the 30 m base keeps seg_classes=1). Same dict
# on both sides below.
aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=num_map_classes,
    feat_down_sample=32,
    pv_thickness=1,
)

# ---- crop geometry ---------------------------------------------------------
# Must equal the base's lidar_point_cloud_range / lidar_voxel_size (restated
# because mmcv evaluates each config file in isolation and the pipelines
# below need them by name). crop_size is the base's tile size: +-15 m.
tile_radius = 15.0
crop_size = 2 * tile_radius
lidar_point_cloud_range = [
    -tile_radius, -tile_radius, -72.0, tile_radius, tile_radius, 96.0
]
lidar_voxel_size = [0.1, 0.1, 0.4]

crop_common = dict(
    crop_size=crop_size,
    require_gt=True,
    gt_tries=32,
    min_polyline_len=2.0,
    # export class names, positionally paired with map_classes above
    gt_classes=('driving', 'curb'),
    reference_dir='reference_lines',
)
crop_train = dict(split='train', rotate=True, translate=True, noise_std=0.02,
                  **crop_common)
# Frozen by seed, no yaw, no noise -- see the docstring.
crop_eval = dict(split='test', rotate=False, translate=True, noise_std=0.0,
                 crop_seed=0, **crop_common)

# ---- pipelines -------------------------------------------------------------
# LoadCarla50mCrop replaces LoadCarlaPointsFromFile: it reads the world-frame
# points, applies the pose drawn by the dataset (so points and GT cannot
# decouple) and crops. load_dim/use_dim=3 is the colour-free convention and
# must match SparseEncoder.in_channels=3 in the base.
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
        # num_vec_one2one stays at 50, so the one2one branch flattens to
        # 50 * 2 = 100 and the inherited max_num=50 is within topk's range.
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
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        map_classes=map_classes,
        **crop_eval),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        map_classes=map_classes,
        **crop_eval),
)

# Restated purely to re-point it at THIS test_pipeline; otherwise the eval
# hook inherits the base's copy and loads pkl-style points into a crop
# dataset.
evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='CarlaMap_chamfer/mAP', rule='greater')
