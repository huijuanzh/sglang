# SPDX-License-Identifier: Apache-2.0
"""Measure the XAttention ring backend: correctness, cost, and comm/compute overlap.

Two configurations are timed, both doing identical attention FLOPs:

  ring  XAttnRingAttentionImpl.forward -- K/V rotated around the ring group
  off   the same XAttention FMHA kernel over K/V that is already local, i.e.
        zero communication. The gap to it is the communication the ring failed
        to hide.

A Chrome/Perfetto trace is written per rank per configuration, and the overlap
between the device's communication and compute intervals is reported.

    torchrun --nproc-per-node=4 benchmark/ring_attention/profile_xattn_ring.py \
        --seq-len 32768 --heads 16 --head-dim 128 --out-dir ring_profile
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.xattn_ring_attn import (
    XAttnRingAttentionBackend,
)

DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
COMM_MARKERS = ("ccl", "nccl", "xccl", "sendrecv", "peer", "memcpy d2d", "memcpy dtod")


def _merge(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersect(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        overlap = min(a[i][1], b[j][1]) - max(a[i][0], b[j][0])
        if overlap > 0:
            total += overlap
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze_trace(path: Path) -> dict:
    """Fraction of device communication time that runs concurrently with compute."""
    events = json.loads(path.read_text())["traceEvents"]
    comm, compute = [], []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e or e.get("cat") not in DEVICE_CATEGORIES:
            continue
        span = (e["ts"], e["ts"] + e["dur"])
        lowered = e["name"].lower()
        (comm if any(m in lowered for m in COMM_MARKERS) else compute).append(span)
    comm, compute = _merge(comm), _merge(compute)
    comm_total = sum(e - s for s, e in comm)
    overlapped = _intersect(comm, compute)
    return {
        "comm_ms": comm_total / 1000,
        "compute_ms": sum(e - s for s, e in compute) / 1000,
        "overlap_ms": overlapped / 1000,
        "hidden_pct": 100.0 * overlapped / comm_total if comm_total else 0.0,
    }


def _run(impl, q, k, v, mode, k_full, v_full):
    if mode == "off":
        # Same kernel, no communication: k/v already hold the whole sequence.
        from xattention.ring_attn_interface import ring_attn_fmha

        return ring_attn_fmha(
            q, k_full, v_full, softmax_scale=impl.softmax_scale, causal=False
        )
    out = impl.forward(q, k, v)
    return out[0] if isinstance(out, tuple) else out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=32768, help="global sequence length")
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--profile-iters", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("xattn_ring_profile"))
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.xpu.set_device(local_rank)
    device = torch.device("xpu", local_rank)

    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=local_rank, backend="xccl"
    )
    initialize_model_parallel(
        sequence_parallel_degree=world_size, ulysses_degree=1, ring_degree=world_size
    )

    if args.seq_len % world_size:
        raise SystemExit(f"--seq-len must be divisible by {world_size}")
    s_local = args.seq_len // world_size
    scale = args.head_dim**-0.5

    gen = torch.Generator(device="cpu").manual_seed(1234)
    shape = (args.batch, args.seq_len, args.heads, args.head_dim)
    q_full, k_full, v_full = (
        torch.randn(shape, generator=gen, dtype=torch.float32).to(device, torch.bfloat16)
        for _ in range(3)
    )
    shard = slice(rank * s_local, (rank + 1) * s_local)
    q = q_full[:, shard].contiguous()
    k = k_full[:, shard].contiguous()
    v = v_full[:, shard].contiguous()

    impl = XAttnRingAttentionBackend.get_impl_cls()(
        num_heads=args.heads, head_size=args.head_dim, causal=False, softmax_scale=scale
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # `off` runs the same kernel over the full K/V, so it doubles as the
    # reference: any gap is the ring's rotation + online-softmax merge.
    # Correctness against true attention is covered by test_xattn_ring_attn.py.
    reference = _run(impl, q, k, v, "off", k_full, v_full).float()
    denom = reference.abs().max().clamp_min(1e-6)

    results = {}
    for mode in ("off", "ring"):
        got = _run(impl, q, k, v, mode, k_full, v_full).float()
        rel_err = ((got - reference).abs().max() / denom).item()
        print(f"[rank {rank}] {mode:4s} rel err vs no-ring result: {rel_err:.3e}", flush=True)
        if rel_err > 2e-2:
            raise SystemExit(f"rank {rank}: {mode} mismatch ({rel_err:.3e})")

        for _ in range(args.warmup):
            _run(impl, q, k, v, mode, k_full, v_full)
        torch.xpu.synchronize()
        dist.barrier()

        samples = []
        for _ in range(args.iters):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            _run(impl, q, k, v, mode, k_full, v_full)
            torch.xpu.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        results[mode] = statistics.median(samples)
        dist.barrier()

        trace_path = args.out_dir / f"rank{rank}_{mode}.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        ) as prof:
            for _ in range(args.profile_iters):
                _run(impl, q, k, v, mode, k_full, v_full)
            torch.xpu.synchronize()
        prof.export_chrome_trace(str(trace_path))
        results[f"{mode}_trace"] = analyze_trace(trace_path)
        dist.barrier()

    if rank == 0:
        print(f"\nXAttention ring backend: world_size={world_size} ring_degree={world_size}")
        print(
            f"shape per rank: B={args.batch} S_local={s_local} "
            f"H={args.heads} D={args.head_dim} dtype=bfloat16"
        )
        print(f"  off  (no ring, same kernel) median {results['off']:8.3f} ms")
        print(f"  ring (XAttn IPC ring)       median {results['ring']:8.3f} ms")
        exposed = results["ring"] - results["off"]
        print(
            f"  ring vs off: {results['ring'] / results['off']:.3f}x  "
            f"(+{exposed:.3f} ms exposed communication)"
        )
        t = results["ring_trace"]
        print(
            f"  device timeline: comm {t['comm_ms']:.2f} ms, compute {t['compute_ms']:.2f} ms, "
            f"concurrent {t['overlap_ms']:.2f} ms -> {t['hidden_pct']:.1f}% of comm hidden"
        )
        print(f"traces written to {args.out_dir.resolve()}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
