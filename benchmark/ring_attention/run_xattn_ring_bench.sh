#!/usr/bin/env bash
# XAttention ring attention backend: correctness + comm/compute overlap.
#
# Backend under test:
#   python/sglang/multimodal_gen/runtime/layers/attention/backends/xattn_ring_attn.py
#   selected with  --attention-backend xattn_ring_attn  --ring-degree N
#
# Two steps:
#   1. test   unit tests -- backend contract (enum, impl class, CLI name
#             resolution) and, on >=2 Intel XPUs, the ring output against
#             attention over the full sequence.
#   2. bench  times the backend against a no-ring baseline that runs the *same*
#             XAttention FMHA kernel over K/V that is already local. Identical
#             FLOPs, zero communication, so the gap is exactly the communication
#             the ring failed to hide. Also dumps a Chrome/Perfetto trace per
#             rank (loadable at https://ui.perfetto.dev) and reports what
#             fraction of the device's transfer time runs concurrently with the
#             attention kernels.
#
# Requires the xattention extension on Intel XPU; bf16 only.
#
# Usage:
#   ./run_xattn_ring_bench.sh                 # both steps
#   ./run_xattn_ring_bench.sh bench
#   NPROC=2 SEQ_LEN=16384 ./run_xattn_ring_bench.sh bench
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NPROC="${NPROC:-4}"
SEQ_LEN="${SEQ_LEN:-32768}"      # global sequence, sharded over NPROC ranks
HEADS="${HEADS:-16}"
HEAD_DIM="${HEAD_DIM:-128}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
PROFILE_ITERS="${PROFILE_ITERS:-3}"
OUT_DIR="${OUT_DIR:-/workspace/outputs/xattn_ring_profile}"
MASTER_PORT="${MASTER_PORT:-29661}"

step() { printf '\n\033[1m=== %s ===\033[0m\n%s\n\n' "$1" "$2"; }

run_test() {
  step "1/2 unit tests" \
"Backend contract: get_enum() is XATTN_RING_ATTN, the impl advertises
handles_sequence_parallel_ring so USPAttention skips its own ring loop, and
--attention-backend xattn_ring_attn resolves. Then, on >=2 XPUs, a 2- and
4-rank ring is checked against attention over the full sequence."
  cd "$REPO_ROOT"
  python -m pytest python/sglang/multimodal_gen/test/unit/test_xattn_ring_attn.py \
    -q -p no:cacheprovider
}

run_bench() {
  step "2/2 ring vs no-ring + overlap" \
"S=$SEQ_LEN over $NPROC ranks, H=$HEADS, D=$HEAD_DIM, bf16.
'off' = same XAttention FMHA kernel, K/V already local, no communication.
'ring' = XAttnRingAttentionImpl.forward. The delta is the exposed communication;
the trace report says how much of the transfer was hidden behind compute."
  cd "$REPO_ROOT"
  rm -rf "$OUT_DIR"
  torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" \
    "$SCRIPT_DIR/profile_xattn_ring.py" \
    --seq-len "$SEQ_LEN" --heads "$HEADS" --head-dim "$HEAD_DIM" \
    --warmup "$WARMUP" --iters "$ITERS" --profile-iters "$PROFILE_ITERS" \
    --out-dir "$OUT_DIR"
}

main() {
  local steps=("$@")
  [[ ${#steps[@]} -eq 0 ]] && steps=(test bench)
  echo "repo    : $REPO_ROOT"
  echo "topology: nproc=$NPROC seq_len=$SEQ_LEN heads=$HEADS head_dim=$HEAD_DIM dtype=bfloat16"
  for s in "${steps[@]}"; do
    case "$s" in
      test) run_test ;;
      bench) run_bench ;;
      *) echo "unknown step: $s (want: test bench)" >&2; exit 1 ;;
    esac
  done
  printf '\n\033[1mdone\033[0m  traces in %s\n' "$OUT_DIR"
}

main "$@"
