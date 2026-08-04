"""Profile a few real MapTRv2 training steps to see where GPU time actually
goes -- useful when you can't read GPU utilization directly (e.g. on a MIG
slice) but want to know if the GPU is compute-bound or sitting idle waiting
on CPU-side kernel dispatch between many small ops.

Run from /MapTR inside the container:

  python3 profile_train.py <config> [--batch-size N] [--iters N] \
      [--warmup N] [--cfg-options key=val ...] [--trace out.json]

Example matching a real run:
  python3 profile_train.py \
      projects/configs/maptrv2/maptrv2_carla_r50_24ep_lidar.py \
      --batch-size 10 --iters 20 --trace /tmp/trace.json

Prints the single most useful number first: average wall-clock time per
iteration vs average CUDA-busy time per iteration. If CUDA-busy time is a
small fraction of wall-clock, the GPU is idle most of the time waiting on
the CPU -- the model has too many small serialized ops for the CPU to keep
it fed (dispatch-bound), which matches "high VRAM use, low utilization,
more dataloader workers don't help". If CUDA-busy time is close to
wall-clock, the GPU genuinely doesn't have enough parallel work per
iteration (low arithmetic intensity) -- batch size is the lever there.

Then prints top ops by self CUDA time (what's actually expensive on-GPU)
and top ops by self CPU time (dispatch-overhead candidates when the first
number above is low).
"""
import argparse
import random
import time

import torch
from mmcv import Config
from mmcv.parallel import collate
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: registers custom modules (MapTRv2, etc.)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--batch-size', type=int, default=None,
                    help='overrides data.samples_per_gpu for this profiling run '
                         '(default: whatever the config says)')
    p.add_argument('--cfg-options', nargs='+', default=[])
    p.add_argument('--iters', type=int, default=15)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--trace', default=None,
                    help='optional path to write a chrome trace json '
                         '(open in chrome://tracing or https://ui.perfetto.dev)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def apply_cfg_options(cfg, opts):
    d = {}
    for o in opts:
        k, v = o.split('=', 1)
        try:
            v = eval(v)
        except Exception:
            pass
        d[k] = v
    if d:
        cfg.merge_from_dict(d)
    return cfg


def to_cuda(data):
    """Mirrors the exact data-prep pattern used elsewhere in this project's
    benchmarking scripts -- keep in sync if the model's expected input keys
    change."""
    out = {}
    for k, v in data.items():
        out[k] = v.data[0] if hasattr(v, 'data') else v
    for k in ('points', 'gt_bboxes_3d', 'gt_labels_3d'):
        if k in out and isinstance(out[k], list):
            out[k] = [x.cuda().contiguous() if hasattr(x, 'cuda') else x for x in out[k]]
    if 'gt_seg_mask' in out:
        gsm = out['gt_seg_mask']
        out['gt_seg_mask'] = ([x.cuda() for x in gsm] if isinstance(gsm, list)
                               else gsm.cuda())
    return out


def main():
    args = parse_args()
    random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg = apply_cfg_options(cfg, args.cfg_options)

    bs = args.batch_size or cfg.data.samples_per_gpu
    print(f"config={args.config}")
    print(f"batch_size={bs}  (config default was {cfg.data.samples_per_gpu})")

    ds = build_dataset(cfg.data.train)
    print(f"dataset size={len(ds)}")

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                         test_cfg=cfg.get('test_cfg')).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)

    idx_pool = list(range(len(ds)))
    random.shuffle(idx_pool)

    def get_batch(offset):
        idxs = [idx_pool[(offset + i) % len(idx_pool)] for i in range(bs)]
        batch = collate([ds[i] for i in idxs], samples_per_gpu=bs)
        return to_cuda(batch)

    def train_step(data):
        optimizer.zero_grad()
        losses = model(return_loss=True, **data)
        loss = sum(v for k, v in losses.items() if 'loss' in k and torch.is_tensor(v))
        loss.backward()
        optimizer.step()

    print(f"Warming up ({args.warmup} iters)...")
    for i in range(args.warmup):
        train_step(get_batch(i))
    torch.cuda.synchronize()

    print(f"Profiling ({args.iters} iters)...")
    wall_times = []
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        for i in range(args.iters):
            data = get_batch(args.warmup + i)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_step(data)
            torch.cuda.synchronize()
            wall_times.append(time.perf_counter() - t0)
            prof.step()

    avg_wall_ms = sum(wall_times) / len(wall_times) * 1000
    key_avgs = prof.key_averages()
    total_cuda_ms = sum(e.self_cuda_time_total for e in key_avgs) / 1000 / args.iters
    busy_pct = 100 * total_cuda_ms / avg_wall_ms if avg_wall_ms > 0 else 0

    print("\n" + "=" * 70)
    print(f"Avg wall-clock time/iter : {avg_wall_ms:.1f} ms")
    print(f"Avg CUDA-busy time/iter  : {total_cuda_ms:.1f} ms")
    print(f"=> implied GPU busy fraction: {busy_pct:.1f}%")
    if busy_pct < 60:
        print("   GPU idle a large fraction of the time -- likely dispatch/")
        print("   kernel-launch-overhead-bound (many small serialized ops),")
        print("   not compute-bound. Check the self-CPU-time table below for")
        print("   which ops dominate dispatch overhead.")
    else:
        print("   GPU is busy most of the wall-clock time -- utilization is")
        print("   limited by genuinely low parallel work per iteration, not")
        print("   dispatch overhead. Batch size is the main lever here.")
    print("=" * 70)

    print("\nTop 15 ops by self CUDA time (what's actually expensive on-GPU):")
    print(key_avgs.table(sort_by="self_cuda_time_total", row_limit=15))

    print("\nTop 15 ops by self CPU time (dispatch-overhead candidates):")
    print(key_avgs.table(sort_by="self_cpu_time_total", row_limit=15))

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"\nChrome trace written to {args.trace}")


if __name__ == '__main__':
    main()
