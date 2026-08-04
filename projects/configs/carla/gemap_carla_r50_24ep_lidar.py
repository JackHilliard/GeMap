_base_ = [
    '../carla/carlasim_map.py',
]
#
# MapTRv2 LiDAR-only config (CARLA).
#
# Unlike the camera+LiDAR fusion configs, this uses a real LiDAR-only BEV
# path added to MapTRv2 (see projects/mmdet3d_plugin/maptr/detectors/
# maptrv2.py, .../maptr/modules/transformer.py, .../maptr/dense_heads/
# maptrv2_head.py): BEV is built directly from the LiDAR SparseEncoder
# output instead of camera features (LSSTransform), which MapTRv2
# otherwise always requires.
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Matches the real 25m x 25m square CARLA tile (tile_radius=12.5). z-range
# is generous: divider polylines are XY-only for code_size=2 (z gets
# clamped but never reaches the final regression target), and this just
# needs to comfortably contain the LiDAR point cloud's z spread.
point_cloud_range = [-12.5, -12.5, -30.0, 12.5, 12.5, 20.0]
voxel_size = [0.15, 0.15, 20.0]

# LiDAR branch geometry (kept separate from the map/coder point_cloud_range
# above -- same 25m x/y extent). z range was originally [-10, 18] (margin
# around the flat local town10hd subset's observed [-7.7, 14.8]), but the
# full remote train set has 6 confirmed town03 tiles with LiDAR returns
# spanning z in [-66.90, 90.52] within a single tile (a highway
# overpass/multi-level structure, not flat driving surface) -- the old
# range dropped every point in those tiles, crashing extract_lidar_feat.
# Widened to comfortably cover the observed extremes rather than filtering
# those tiles out; z_max in carlasim_map.py (LoadCarlaPointsFromFile's
# early filter) must stay >= this range's z upper bound. This is a ~6x
# increase in z voxel-grid resolution at the same voxel size -- re-measure
# sparse_shape/lidar_bev_proj.in_channels below via a dummy
# extract_lidar_feat() call rather than hand-deriving (see CLAUDE.md
# gotcha #4); sparse convs only compute over occupied voxels so normal
# (non-degenerate) tiles shouldn't see a proportional compute/memory hit,
# but this hasn't been verified against real full-dataset tiles yet.
lidar_point_cloud_range = [-12.5, -12.5, -72.0, 12.5, 12.5, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]

map_classes = ['divider']
num_vec=50
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
# Square, matching the square 25m x 25m point_cloud_range above (was
# 200x100, inherited unchanged from the original asymmetric 30m x 60m
# nuScenes-derived range -- that mismatch caused a shape error between the
# seg head's output and the dataset's gt_seg_mask, which uses bev_size
# below for its canvas).
bev_h_ = 100
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,  # no camera imagery to rasterize into
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

model = dict(
    type='MapTRv2',
    use_grid_mask=True,
    video_test_mode=False,
    modality='lidar',
    lidar_encoder=dict(
        voxelize=dict(
            max_num_points=10,
            point_cloud_range=lidar_point_cloud_range,
            voxel_size=lidar_voxel_size,
            max_voxels=[90000, 120000]),
        backbone=dict(
            type='SparseEncoder',
            in_channels=4,  # CARLA points are xyz + strength (not nuScenes' xyz+intensity+ring=5)
            # (x, y, z) order -- confirmed against the working nuScenes
            # fusion config's sparse_shape=[300,600,41] for a 30x60x8m
            # range at the same voxel resolution. z re-measured for the
            # widened lidar_point_cloud_range above via a dummy
            # extract_lidar_feat() call (was [251,251,71] for the old
            # [-10,18] z range).
            sparse_shape=[251, 251, 421],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                          128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
            block_type='basicblock'
        ),
    ),
    pts_bbox_head=dict(
        type='MapTRv2Head',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec_one2one=50,
        num_vec_one2many=300,
        k_one2many=6,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=aux_seg_cfg,
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            modality='lidar',
            # SparseEncoder outputs C*D channels (output_channels=128 * the
            # z-dim after its internal downsampling); measured empirically
            # via a dummy extract_lidar_feat() call against this exact
            # config -- output shape is (1, 3200, 32, 32), i.e. z
            # downsampled to 25 (128*25=3200) and x/y (250 cells each)
            # downsampled to 32. Was 384 (z downsampled to 3) before
            # lidar_point_cloud_range's z span widened from 28m to 168m
            # (see that config's comment) -- the much larger input z-range
            # downsamples proportionally less, not just linearly, so this
            # was re-measured rather than scaled by hand.
            lidar_bev_proj=dict(
                type='ConvFuser',
                in_channels=[3200],
                out_channels=_dim_,
            ),
            # Structurally required by MapTRPerceptionTransformer.__init__
            # (build_transformer_layer_sequence(encoder) always runs) but
            # never invoked on the modality='lidar' path -- kept minimal.
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DecoupledDetrTransformerDecoderLayer',
                    num_vec=num_vec,
                    num_pts_per_vec=fixed_ptsnum_per_pred_line,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm','cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='GeMapNMSFreeCoder',
            post_center_range=[-14.5, -14.5, -14.5, -14.5, 14.5, 14.5, 14.5, 14.5],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss',
                      loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_seg=dict(type='SimpleLoss',
            pos_weight=4.0,
            loss_weight=1.0),
        loss_pv_seg=dict(type='SimpleLoss',
                    pos_weight=1.0,
                    loss_weight=2.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='GeMapAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost',
                      weight=5),
            pc_range=point_cloud_range))))

file_client_args = dict(backend='disk')

# GridSamplePoints must live here (not in carlasim_map.py's base
# train_pipeline/test_pipeline) because it needs lidar_point_cloud_range,
# which only exists in this derived config -- mmcv's per-file config
# isolation means the base file can't see it. Collapses each tile's raw
# LiDAR points to ~1 representative point per occupied
# (lidar_voxel_size)-sized cell before the (slow, and at extreme point
# counts, silently incorrect) LiDAR voxelizer ever sees them. grid_size
# matches lidar_voxel_size exactly so this adds no spatial precision loss
# beyond what the voxelizer already imposes.
train_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
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

# Matches carlasim_map.py's own test_pipeline shape (inherited at the
# data.test dict level via _base_), plus the same GridSamplePoints step as
# train_pipeline above; redefined here as a bare name partly because
# `evaluation` below needs it directly and _base_ variables aren't visible
# as plain Python names in this file's own execution scope.
test_pipeline = [
    dict(
        type='LoadCarlaPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        # Kept in sync with carlasim_map.py's own z_max and this file's
        # lidar_point_cloud_range z upper bound -- see the comments there.
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

# carlasim_map.py's dataset dicts don't set these (they're model-config
# concerns, matching the nuScenes/AV2 convention of overriding them here).
data = dict(
    train=dict(
        pipeline=train_pipeline,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        bev_size=(bev_h_, bev_w_),
        # Needed so the dataset's own vector_map produces gt_seg_mask,
        # matching aux_seg_cfg's use_aux_seg/bev_seg above (only wired for
        # train -- bev_seg is a training-time auxiliary loss).
        aux_seg=aux_seg_cfg),
    val=dict(
        pipeline=test_pipeline,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        bev_size=(bev_h_, bev_w_)),
    test=dict(
        pipeline=test_pipeline,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        bev_size=(bev_h_, bev_w_)),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='CarlaMap_chamfer/mAP', rule='greater')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(max_keep_ckpts=1, interval=2)
find_unused_parameters=True
