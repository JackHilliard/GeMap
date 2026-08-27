_base_ = [
    './gemap_carla_r50_24ep_lidar.py',
]
#
# The 25m config (tile_radius=12.5) retargeted at a 30m export
# (tile_radius=15.0). Model architecture, schedule, losses and every
# non-geometric setting are inherited unchanged -- only what depends on
# tile size is restated below.
#
# Why anything has to be restated at all: mmcv's per-file config isolation
# means a `_base_` variable is not visible as a plain Python name while
# this file executes, so `tile_radius` below cannot feed the parent's
# derivations -- those already ran, at 12.5. The same reason
# `--cfg-options tile_radius=15.0` does not work: it is a dict merge
# applied after the parent has been exec'd, so it moves the knob without
# moving anything derived from it.
#
# Everything here is therefore derived from this file's own tile_radius
# using the same formulas as the parent, and the values are asserted
# against the pkl at load time by CustomCarlaLocalMapDataset -- a drift
# between the two files fails immediately rather than after a run.
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# THE tile-size knob for this config. See the parent for what each derived
# value means; only the number differs.
tile_radius = 15.0

# --- geometry, all derived exactly as the parent derives it -------------

# Map/coder range. z is the parent's (-30, 20), which differs from the
# *dataset's* z (-2, 24) restated further down -- that asymmetry is
# inherited on purpose, not an oversight.
point_cloud_range = [
    -tile_radius, -tile_radius, -30.0, tile_radius, tile_radius, 20.0
]
voxel_size = [0.15, 0.15, 20.0]

lidar_point_cloud_range = [
    -tile_radius, -tile_radius, -72.0, tile_radius, tile_radius, 96.0
]
lidar_voxel_size = [0.1, 0.1, 0.4]

# round(extent / voxel) + 1, as in the parent. z is unchanged from the 25m
# config (same z range and voxel), so only x/y move: 30/0.1+1 = 301.
sparse_shape = [
    int(round((lidar_point_cloud_range[3] - lidar_point_cloud_range[0]) /
              lidar_voxel_size[0])) + 1,
    int(round((lidar_point_cloud_range[4] - lidar_point_cloud_range[1]) /
              lidar_voxel_size[1])) + 1,
    int(round((lidar_point_cloud_range[5] - lidar_point_cloud_range[2]) /
              lidar_voxel_size[2])) + 1,
]

# 0.3 m/cell -> 100x100, the shared benchmark resolution (was 0.25 ->
# 120x120, inherited from the 25m config). 0.3 is doubly GeMap's own
# number: upstream GeMap trains nuScenes/AV2 at 0.3 m/cell (200x100 over
# 60x30 m), and the 25m config's 0.25 was only ever chosen to make the
# 25 m tile divide into the round 100x100 grid -- which at 30 m is what
# 0.3 gives. It also matches the MapTRv2/PMT/mapdiffusion 30m configs
# exactly (100x100), so BEV token count stops being a cross-repo confound.
bev_resolution = 0.3
bev_h_ = int(round((point_cloud_range[4] - point_cloud_range[1]) /
                   bev_resolution))
bev_w_ = int(round((point_cloud_range[3] - point_cloud_range[0]) /
                   bev_resolution))

map_classes = ['divider']
fixed_ptsnum_per_gt_line = 20
eval_use_same_gt_sample_num_flag = True

# --- data roots --------------------------------------------------------
#
# Kept separate from the 25m dataset's `data/carla/` so both can coexist;
# the converter writes carla_map_infos_<split>.pkl into whatever --out-dir
# it is given, so a shared directory would have the two exports overwrite
# each other. Generate with::
#
#   python tools/gemap/custom_carla_map_converter.py \
#       --data-root data/carla30/ --out-dir data/carla30/ --split train
#
# The converter reads tile_radius from that export's manifest and prints
# it; it must say 15.0 for this config, and the dataset asserts as much.
data_root = 'data/carla30/'
# None = resolve LiDAR paths against the data_root recorded inside the
# annotation pkl (what each lidar_path is relative to), which also lets the
# MapTRv2 benchmark repo's pkls -- shareable since 2026-08-28 -- load here
# unchanged. Set explicitly only when the tile export lives at a different
# path than at conversion time.
raw_data_root = None
ann_file_train = data_root + 'carla_map_infos_train.pkl'
ann_file_val = data_root + 'carla_map_infos_test.pkl'
ann_file_test = data_root + 'carla_map_infos_test.pkl'
map_ann_file = data_root + 'carla_map_gt.json'

# --- model overrides ---------------------------------------------------
#
# Partial nested dicts: mmcv merges these into the parent's `model`
# recursively, so only the leaves named here change.
#
# Deliberately NOT overridden: `transformer.encoder`, whose pc_range still
# reads 12.5-derived. That encoder is structurally required by
# MapTRPerceptionTransformer.__init__ but never invoked on the
# modality='lidar' path (see the parent), so restating its whole
# attn_cfgs list -- mmcv replaces lists wholesale rather than merging them
# -- would copy ~30 lines of dead config for no behavioural change.
#
# Also not overridden: lidar_bev_proj.in_channels. It is a channel count
# (output_channels x z_out), so it follows the z range, not tile size, and
# the LiDAR BEV map is bicubic-resized to (bev_h_, bev_w_) before that
# projection sees it. z is unchanged here, so 3200 still holds.
model = dict(
    lidar_encoder=dict(
        voxelize=dict(point_cloud_range=lidar_point_cloud_range),
        # in_channels=3 pairs with use_dim=3 on the loaders below: the colour
        # ("strength") channel is dropped to match the MapTRv2 30m HM
        # benchmark convention (its tidy-configs branch trains colour-free).
        # The two MUST move together -- a 3-channel input against the
        # parent's 4-channel first conv (or vice versa) fails at the first
        # sparse conv. sparse_shape and lidar_bev_proj.in_channels do not
        # move: neither depends on the input channel width.
        backbone=dict(in_channels=3, sparse_shape=sparse_shape),
    ),
    pts_bbox_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        bbox_coder=dict(
            post_center_range=[-(tile_radius + 2.0)] * 4 +
            [tile_radius + 2.0] * 4,
            pc_range=point_cloud_range,
        ),
        positional_encoding=dict(
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            assigner=dict(pc_range=point_cloud_range),
        )),
)

# --- pipelines ---------------------------------------------------------
#
# Restated in full because they are lists: mmcv replaces a list wholesale
# rather than merging into it, so there is no way to patch only
# GridSamplePoints' point_cloud_range from here.
# --- actor augmentation ---------------------------------------------------
# Vehicles/pedestrians scanned once from CARLA (point2vector_data/
# carla_actor_scan.py) and pasted in at load time, together with the ground
# shadow each one removes. Set actor_catalogue = None to disable. Runs after
# LoadCarlaPointsFromFile (its tile-centred frame) and before GridSamplePoints,
# so pasted points get the same voxel decimation as real ones. GT polylines are
# left untouched on purpose: the model must infer map elements under traffic.
actor_catalogue = None
actor_paste = dict(
    type='CarlaActorPaste',
    catalogue=actor_catalogue,
    n_vehicles=(0, 5),
    n_pedestrians=(0, 6),
    prob=0.8)

train_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        # load_dim stays 4: the loader builds the strength column before
        # selecting, and use_dim=3 keeps only [x, y, z] -- see the
        # in_channels=3 note on the model above.
        load_dim=4,
        use_dim=3,
        z_max=96.0),
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
if actor_catalogue is not None:
    train_pipeline.insert(1, actor_paste)

test_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=3,  # colour-free, matching train_pipeline
        z_max=96.0),
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

# The dataset's own pc_range, whose z (-2, 24) is carlasim_map.py's rather
# than the model's -- only min_z/max_z for vectorization come from it, and
# the tile-geometry assert looks at x/y. Restated here because that grand
# parent config derived it from its own 12.5.
dataset_pc_range = [
    -tile_radius, -tile_radius, -2.0, tile_radius, tile_radius, 24.0
]

data = dict(
    train=dict(
        data_root=data_root,
        raw_data_root=raw_data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        pc_range=dataset_pc_range,
        bev_size=(bev_h_, bev_w_)),
    val=dict(
        data_root=data_root,
        raw_data_root=raw_data_root,
        ann_file=ann_file_val,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        pc_range=dataset_pc_range,
        bev_size=(bev_h_, bev_w_)),
    test=dict(
        data_root=data_root,
        raw_data_root=raw_data_root,
        ann_file=ann_file_test,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        pc_range=dataset_pc_range,
        bev_size=(bev_h_, bev_w_)),
)

# `evaluation` holds its own reference to the test pipeline, so the parent's
# copy (built from the 12.5 lidar_point_cloud_range) has to be replaced too
# -- otherwise eval would grid-sample against a range narrower than the
# tile while training used the right one.
evaluation = dict(
    interval=2,
    pipeline=test_pipeline,
    metric='chamfer',
    save_best='CarlaMap_chamfer/mAP',
    rule='greater')
