# SPDX-License-Identifier: Apache-2.0
"""Communication/computation overlap report for a Wan2.2 denoise-step trace.

Unlike the attention microbenchmark, a full pipeline trace mixes several kinds
of traffic, and lumping them together makes "% hidden" meaningless. Events are
therefore split into three classes:

  ring     K/V rotation of ring attention -- peer/IPC copies and the small
           barriers the ring uses. This is the traffic the xattn_ring backend is
           supposed to hide behind the attention kernels.
  collect  Ulysses all-to-all and other collectives (all_reduce, all_gather).
           Sequence-parallel traffic that is *not* the ring.
  offload  host-to-device streaming of layerwise-offloaded DiT weights. Present
           in every configuration and unrelated to the attention backend, so it
           would otherwise dominate and flatter the numbers.

Usage:
    python benchmark/ring_attention/analyze_wan22_overlap.py <trace-dir> [...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}

# Checked in order; first match wins.
CLASSIFIERS = (
    ("offload", ("memcpy htod", "memcpy h2d", "host -> device", "hosttodevice")),
    ("collect", ("all_to_all", "alltoall", "allreduce", "all_reduce", "allgather", "all_gather")),
    ("ring", ("ccl", "nccl", "xccl", "sendrecv", "peer", "memcpy d2d", "memcpy dtod", "device -> device")),
)


def _classify(name: str) -> str:
    lowered = name.lower()
    for label, markers in CLASSIFIERS:
        if any(m in lowered for m in markers):
            return label
    return "compute"


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


def _total(intervals) -> float:
    return sum(e - s for s, e in intervals)


def _intersect(a, b) -> float:
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        overlap = min(a[i][1], b[j][1]) - max(a[i][0], b[j][0])
        if overlap > 0:
            total += overlap
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path: Path) -> dict:
    events = json.loads(path.read_text()).get("traceEvents", [])
    buckets: dict[str, list] = {"ring": [], "collect": [], "offload": [], "compute": []}
    for e in events:
        if e.get("ph") != "X" or "dur" not in e or e.get("cat") not in DEVICE_CATEGORIES:
            continue
        buckets[_classify(e["name"])].append((e["ts"], e["ts"] + e["dur"]))

    merged = {k: _merge(v) for k, v in buckets.items()}
    compute = merged["compute"]
    out = {"trace": path.name, "compute_ms": _total(compute) / 1000}
    for label in ("ring", "collect", "offload"):
        total = _total(merged[label])
        hidden = _intersect(merged[label], compute)
        out[label] = {
            "ms": total / 1000,
            "hidden_ms": hidden / 1000,
            "hidden_pct": 100.0 * hidden / total if total else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="trace files or directories")
    args = parser.parse_args()

    traces: list[Path] = []
    for p in args.paths:
        traces.extend(sorted(p.rglob("*.json")) if p.is_dir() else [p])
    traces = [t for t in traces if "summary" not in t.name]
    if not traces:
        raise SystemExit("no traces found")

    header = (
        f"{'trace':44s} {'compute':>9s} "
        f"{'ring':>9s} {'hid%':>6s} {'collect':>9s} {'hid%':>6s} {'offload':>9s} {'hid%':>6s}"
    )
    print(header)
    print("-" * len(header))
    for trace in traces:
        try:
            r = analyze(trace)
        except (ValueError, KeyError):
            continue
        print(
            f"{r['trace'][:44]:44s} {r['compute_ms']:8.1f}m "
            f"{r['ring']['ms']:8.1f}m {r['ring']['hidden_pct']:5.1f}% "
            f"{r['collect']['ms']:8.1f}m {r['collect']['hidden_pct']:5.1f}% "
            f"{r['offload']['ms']:8.1f}m {r['offload']['hidden_pct']:5.1f}%"
        )
    print("\nms totals are device-busy time within the profiled window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
