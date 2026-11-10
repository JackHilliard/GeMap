# GeMap — working notes

Project-specific knowledge that is not derivable from the code or git log.
The CARLA LiDAR map pipeline is the active work; everything else is
upstream GeMap.

## The CARLA tile pipeline

```
<raw export>/<split>/{manifest.json, blocks/*.npz, reference_lines/*.json}
   -> tools/gemap/custom_carla_map_converter.py
   -> data/carla/carla_map_infos_<split>.pkl
   -> CustomCarlaLocalMapDataset  (projects/mmdet3d_plugin/datasets/carla_offlinemap_dataset.py)
   -> projects/configs/carla/gemap_carla_r50_24ep_lidar.py
```

`tools/gemap/dataset_viewer.py` renders the raw export directly (no model,
no torch) and is the fastest way to sanity-check a frame or a tile.

## Coordinate frames — the thing to get right (2026-08-10)

Each `.npz` block carries **two** origins, and they are not the same:

* `offset` — what `features[:, 0:3]` is stored relative to
  (`points - offset == features[:, :3]`, exactly).
* `tile_center` — the tile's geometric centre (verified: *exactly* the
  midpoint of `tile_bounds`, on all 4103 train tiles).

They differ by a mean of **1.25 m and up to 12.07 m** on the 25 m export,
and up to ~17 m on the 60 m one — the displacement scales with tile size.

Everything downstream now works in the **tile-centred** frame: the
converter subtracts `tile_center` from the world-frame polylines and
records `recenter_shift = tile_center - offset`, which
`LoadCarlaPointsFromFile` subtracts from the point cloud. Both are
translated by the same vector, so their relative alignment is untouched,
while the tile lands squarely inside `[-tile_radius, +tile_radius]`.

**Why it matters.** The model's patch (`pc_range`) is symmetric about the
origin. In the old `offset` frame the tile was displaced from it, so on the
25 m train split **16% of GT points and 82% of divider instances fell
outside `pc_range`**, and 651/4103 tiles kept <90% of their LiDAR points.
After re-centring: 0.01% of points outside (all sitting exactly on the
boundary), and range coverage is 100% on every tile bar 3 — and those 3 are
a *z* problem (see open items), not xy.

**The trap.** Measured against real driving-surface returns (label == 0),
`polyline - offset` sits a median 0.038 m from the road while
`polyline - tile_center` sits at 0.388 m. That looks like an argument for
the `offset` frame, and it is — *if you re-frame the polylines alone and
leave the point cloud where it was*. Shifting both by the same vector is a
rigid translation and preserves the 0.038 m figure. Do not "fix" one half.

Pkls generated before this change carry no `annotation_frame` key;
`load_annotations` rejects them outright rather than training against a
misaligned patch. The previous ones are kept at
`data/carla/*.pkl.offset-frame.bak`.

## Tile size is one number

The exporter has produced 25 m tiles (`tile_radius` 12.5) and 60 m ones
(`tile_radius` 30.0). Nothing in the converter, dataset or loader assumes a
size.

To switch exports, set `tile_radius` in **both**
`projects/configs/carla/carlasim_map.py` and
`.../gemap_carla_r50_24ep_lidar.py` (duplicated because mmcv's per-file
config isolation means a `_base_` variable is not visible as a plain Python
name in the deriving file's scope). Everything else —
`point_cloud_range`, `lidar_point_cloud_range`, `sparse_shape`,
`post_center_range`, `bev_h_`/`bev_w_` — is derived from it. At 12.5 every
derived value reproduces the previously hardcoded numbers exactly
(`[251,251,421]`, 100×100, ±14.5), so the change was a no-op on the 25 m
export.

Get it wrong and `CustomCarlaLocalMapDataset.load_annotations` fails at
load time against the pkl's own `tile_geometry`, rather than quietly
cropping the tile for a whole run.

`bev_resolution` (0.25 m/cell) is what holds BEV resolution *constant*
across tile sizes — 100×100 at 25 m, 240×240 at 60 m. Setting `bev_h_`
directly instead would stretch the cells.

### Gotcha: what must still be measured

`lidar_bev_proj.in_channels=3200` is **not** derived, deliberately. It is
`output_channels × z_out` after SparseEncoder's internal strides — not a
closed form — and it was measured with a dummy `extract_lidar_feat()` call.
It is safe against tile-size changes because it depends only on the z
extent, and the LiDAR BEV map is bicubic-resized to `(bev_h_, bev_w_)` in
`MapTRPerceptionTransformer.get_bev_features` before this projection sees
it. **Re-measure it if the z range or `lidar_voxel_size[2]` changes.**

`sparse_shape` *is* derived, because `round(extent / voxel) + 1` reproduces
both previously-measured values exactly (25/0.1+1 = 251, 168/0.4+1 = 421).

## Gotchas

* **`data/carla/carla_map_gt.json` is a cache and `_format_gt` skips it if
  it exists.** Any change to the GT frame, the tile set, or `--classes`
  means deleting it, or eval silently scores against the old GT.
* **`lidar_path` in the pkl is relative to `--data-root`**, rejoined
  against the config's `raw_data_root`. Keep it that way — MapTR's copy of
  this converter stores absolute paths, which bakes in the conversion
  machine's mount and breaks on a cluster (and `os.path.join` with an
  absolute second argument hides the bug until then).
* **`--lane-types` never worked** — it compared `lane_type_lookup` ids
  against each polyline's `type`, which holds a geometry kind
  (`'arc'`/`'straight'`), so it silently produced an empty pkl. Use
  `--classes` (ids or names against `class_lookup`); it raises on an
  unknown name and on class-free exports. Verified:
  `--classes driving_centerline` on the grid export yields 299, matching
  the manifest's own `polyline_counts_by_class`.
* **Manifest counts are not trustworthy.** `manifest.json` in the grid
  export reports `n_tiles: 4103` while listing 30. Derive from the tile
  entries, never the counts. `manifest.json` is the *curated* view (it
  drops tiles with no driving centerline) and wins over
  `grid_manifest.json`, which only backfills.
* **Some tiles have a 150 m+ z spread** within one tile (town03 highway
  overpasses). `z_max` and `lidar_point_cloud_range`'s z bounds must stay
  in sync across `carlasim_map.py` and the model config, or those tiles
  lose every point and crash `extract_lidar_feat` on a zero-voxel input.
* **Raw point counts reach 5,000,000 per tile**, at which the legacy
  `Voxelization` CUDA kernel silently under-reports occupied voxels by
  ~36%. `GridSamplePoints` (before voxelization) is what makes this both
  correct and ~26× faster; it clamps out-of-range points rather than
  dropping them.

## Open items

* **`lidar_point_cloud_range`'s z lower bound (−72) is still too high.**
  3/4103 train tiles keep <90% of their points, worst 79.0%
  (`town03_tile_00202`, z spanning −79.6…83.4). Purely z — xy coverage is
  100.000% on that tile. Widening z costs `sparse_shape[2]` and a
  re-measure of `lidar_bev_proj.in_channels`.
* **All polyline classes are collapsed into one `divider` class**, a
  deliberate choice to maximise GT density on the small local test set.
  Before a real training run, consider `--classes driving_centerline`.
* **The 60 m export has no train/val split** — it is a single 30-tile grid,
  useful for verifying size-independence, not for training.

## 50 m rotated-crop benchmark (2026-10-13)

Ported from the MapTRv2 repo's `maptrv2` branch
(`projects/mmdet3d_plugin/datasets/carla50m_crop_dataset.py` there). The
tiles on disk are **50 m** (`tile_radius` 25, e.g. `../carla_test_V2`); the
model is the unchanged 30 m one, because every crop it sees is a 30 m tile.

- `CustomCarla50mCropDataset` + `LoadCarla50mCrop`
  (`projects/mmdet3d_plugin/datasets/carla50m_crop_dataset.py`) read the
  export's `<split>/manifest.json` directly -- **no converter pkl**, so none
  of the pkl bookkeeping (frame key, tile-geometry assert, `recenter_shift`,
  `raw_data_root`) applies. `data_root` is the export root. Per
  `__getitem__`: uniform yaw about the tile centre, centre shift up to
  3.79 m (25 - 15*sqrt2), 30 m crop, sigma 0.02 m point noise; GT gets the
  same pose and is clipped with the tiler's own `clip_polyline_to_box`.
- **Eval crops are frozen by `crop_seed`** (no yaw, no noise, aimed at GT so
  they are non-empty) because `_format_gt()` caches the GT json once. Delete
  `carla_map_gt_50m_crop_2cls.json` whenever the tile set, crop kwargs or
  seed change. On `../carla_test_V2` (2170 tiles) 0 frozen crops are
  GT-empty.
- Config: `projects/configs/carla/gemap_carla_r50_24ep_lidar_50m_crop.py`,
  overlay on the 30 m config. Classes `['driving', 'curb']` (export names,
  label order = list order); `aux_seg.seg_classes=2` on both the model and
  `data.train` side.
- `VectorizedCarlaLocalMap` (`carla_offlinemap_dataset.py`) builds
  `CLASS2LABEL` from `map_classes` -- the AV2 base's hardcoded nuScenes names
  would vectorise `driving`/`curb` to **zero instances with no error**. It is
  bit-identical for `['divider']`.
- `_evaluate_single` adds per-class precision/recall/F1 and matched chamfer
  (`carla50m_metrics.py`) to the usual `CarlaMap_chamfer/*` keys.
- Verify: `python3 tools/gemap/verify_crop_dataset.py <config> --data-root
  <export>` (GT-to-LiDAR distance, quadrant occupancy, yaw histogram, val
  determinism). Smoke-tested 2026-10-13 in `gemap:latest` on a 34-tile
  subset: median GT-to-LiDAR 0.023 m, 1 epoch train + eval end-to-end.

## HRDX San Jose 50 m benchmark (2026-10-29)

Real-LiDAR counterpart of the CARLA 50 m crop benchmark, ported from the
MapTRv2 repo (see its CLAUDE.md section of the same name for the data facts).
`CustomSanJose50mCropDataset` (`datasets/sanjose50m_crop_dataset.py`)
subclasses `CustomCarla50mCropDataset`: flat export (`split=None`, `ann_file`
= a `../san_jose_splits_v1/v1/*.json` manifest, `data_root` =
`data/san_jose_50m/`), `gt_classes=('curb',)` (never add `curb_parametric`,
it re-traces the same curbs), and a check of the manifest's `finalize` block
against `crop_seed`/`max_shift`/`gt_classes` -- the splits guarantee non-empty
>= 1000-point frozen crops only for `crop_seed=0`, `rotate=False`,
`translate=True` in manifest order. Config
`projects/configs/carla/gemap_sanjose_r50_24ep_lidar_50m_crop.py` (+ `_smoke`
on `v1_smoke`), one class, `aux_seg.seg_classes=1`, separate `map_ann_file`
per frozen split under `data/carla30/` (val != test here). Verified in
`gemap:latest` on the 4 local tiles (`tools/gemap/verify_crop_dataset.py
--ann-file-train/--ann-file-val`, then 1 epoch train+eval). Cluster script:
`~/submit_scripts/run_gemap_sanjose.sh` (batch 16, no converter step).

Empty-crop guard (2026-11-04): real LiDAR has coverage gaps, and the train
crop pose is re-drawn every epoch, so now and then a crop holds GT but zero
points in `lidar_point_cloud_range`. Unguarded, `GridSamplePoints` died on
`torch.unique(...).max()` of an empty tensor and the DataLoader worker took
the whole run down -- at a random step, not every epoch. Ported the sibling
repos' fix: `GridSamplePoints(min_points=1)` raises `EmptyLidarTileError`,
`CustomCarlaLocalMapDataset.prepare_train_data` catches it, warns with the
`sample_idx`, and returns None so `Custom3DDataset.__getitem__` resamples
(> 100 consecutive skips re-raise: that means the config's range is wrong).
Train mode only; the frozen val/test crops are guaranteed >= 1000 points by
`sj_finalize_splits.py`, so test mode is left strict on purpose.

Train-time sample filter (2026-11-05), identical in all five repos: a freshly
drawn train crop must hold >= the manifest's `filters.min_points` RAW points
(the tiler's own floor, 20000 for `san_jose_tiles_50m`; the split jsons carry
the same `filters` block) and at least one curb (`filter_empty_gt=True`;
mapdiffusion/SuperMapNet check the crop GT themselves). The floor is read from
the manifest, not copied into configs: the San Jose dataset's
`min_crop_points='manifest'` (SuperMapNet: `min_lidar_points='manifest'`) is
resolved in `load_annotations`, passed through `get_data_info` as
`crop_min_points` for FRESH crops only, and enforced by `LoadCarla50mCrop` on
the raw in-crop count (raises `EmptyLidarTileError`; `prepare_train_data`
resamples). An int overrides, 0 disables; `'manifest'` on a manifest without
`filters.min_points` is a ValueError. A failing crop is skipped with a
warning and another tile drawn; > 100 consecutive skips raise (config/range
error). Roughly 2-3% of tiles sit below 20k. The frozen val/test crops are
NEVER filtered (`crop_min_points=0`): their p1 is ~9-11k points, so a 20k
floor there would crash eval, and the eval set stays exactly the finalised
list. `GridSamplePoints.min_points` stays at its default 1 everywhere (the
zero-voxel guard). Augmentation otherwise unchanged: uniform yaw, <= 3.79 m
shift aimed at a curb vertex (32 tries), sigma 0.02 m xyz point noise; no
flip, no scaling, no actor paste, no colour.

## HRDX whole-50 m-tile benchmark, three cities (2026-11-09)

Whole-50 m-tile benchmark over three cities (San Jose, San Francisco, San Mateo), defined
by the reference repo `../MapTR_hrdx_wt` (`hrdx_ORIG_L1.py`, `datasets/hrdx50m_dataset.py`)
and ported to all six sister repos (MapTR, GeMap, PseudoMapTrainer, mapdiffusion,
SuperMapNet, maptracker). The benchmark part is the same everywhere; change it in all six or
in none:

- data `data/hrdx_tiles_50m/<city>_tiles_50m_lines/{blocks,reference_lines}`, splits
  `data/hrdx_splits_v1/{train,val,test}.json` (20,499 / 2,227 / 3,392 tiles, each with
  `name`, `city`, `center`), smoke subsets `smoke/{train200,val30}.json`
- sample = the whole tile, +-25 m, no crop, no translation
- GT: one class, `gt_source='parametric'` (`curb_parametric` design samples chained per
  `alignment_id` by `hrdx_param_gt.chain_parametric`), clipped to the box, pieces < 2 m dropped
- input `(x, y, z, strength)`: z re-zeroed on the tile's 2nd percentile, strength =
  intensity / 255; `GridSamplePoints` 0.1 x 0.1 x 0.4 m; shuffled in train
- train aug `PTV3_NUSCENES_AUG`: rotate U(-pi, pi) p 0.5, scale U(0.9, 1.1), flip x / flip y
  p 0.5 each as ONE 2x2 map on points and GT, jitter sigma 0.005 clip 0.02 on points only;
  up to 8 redraws when the augmented tile has no GT; tiles under 5000 in-range points or
  without GT are resampled. RNG order matters for cross-repo identity: rotate coin, angle,
  scale, flip x, flip y, then the loader's `randn(n, 3)` jitter
- val/test never augmented (`crop_seed=0`); eval every 2 epochs, chamfer AP
- schedule: 16 epochs, AdamW wd 0.01, cosine with 4000-iteration linear warmup, batch 2 x 8
  accumulated iterations (`GradientCumulativeFp16OptimizerHook`, loss scale 512, grad clip
  35), checkpoint every epoch keep 4
- GT caches `data/hrdx_splits_v1/{val,test}_gt_param.json` are written at the first
  evaluation (atomically) and checked against the dataset at construction
  (`RuntimeError('stale GT cache ...')` means delete the cache)

Proof of identity: `~/submit_scripts/hrdx_checks/hrdx_dump_mmdet.py CONFIG OUT.npz` dumps
augmentation matrices, GT, raw and piped points for fixed seeds; `hrdx_compare.py REF OTHER`
must print `IDENTICAL`. Verified 2026-11-09 against the reference repo on 4 local San Jose
tiles posing as an HRDX city (the three-city data is only on the cluster).

Training on the cluster no longer uses a container: one shared python environment
(`~/submit_scripts/pc_env/README.md`) and `~/submit_scripts/run_*_hrdx.sh` on top of
`hrdx_common.sh` (repo rsynced from `/scratch/$USER/PC_encoding/<repo>` to the node).

Here: `datasets/hrdx50m_dataset.py` and `hrdx_param_gt.py` are byte-identical to the MapTR
repo's. Config `projects/configs/carla/gemap_hrdx_16ep_lidar_50m_tile.py` (+ `_smoke`) is the
MapTRv2 HRDX config with `GeMapNMSFreeCoder` and `GeMapAssigner`, standalone (no `_base_`),
all ranges +-25 m (the San Jose config's encoder `pc_range` of +-12.5 is not inherited),
lr 6e-4. HRDX GT caches live in `data/hrdx_splits_v1/`, not `data/carla30/`. Submit script
`run_gemap_hrdx.sh`.
