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
