# SPDX-License-Identifier: Apache-2.0
"""Tests for the XAttention ring attention backend."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.layers.attention.backends.xattn_ring_attn import (
    XAttnRingAttentionBackend,
    is_xattn_ring_available,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

B, S_LOCAL, H, D = 1, 1024, 8, 128
DTYPE = torch.bfloat16

requires_xattn = pytest.mark.skipif(
    not is_xattn_ring_available(),
    reason="requires the xattention extension on an Intel XPU",
)


def test_backend_contract():
    assert XAttnRingAttentionBackend.get_enum() is AttentionBackendEnum.XATTN_RING_ATTN
    impl_cls = XAttnRingAttentionBackend.get_impl_cls()
    # USPAttention keys off this to skip its own ring loop.
    assert XAttnRingAttentionBackend.handles_sequence_parallel_ring is True
    assert impl_cls.handles_sequence_parallel_ring is True
    # USPAttention rejects ring on any backend that does not declare this.
    assert XAttnRingAttentionBackend.supports_ring_rotation() is True
    assert 128 in XAttnRingAttentionBackend.get_supported_head_sizes()


def test_backend_name_resolves_from_cli():
    from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

    normalized = ServerArgs._normalize_attention_backend_name("xattn_ring_attn")
    assert normalized == "xattn_ring_attn"
    assert AttentionBackendEnum[normalized.upper()] is AttentionBackendEnum.XATTN_RING_ATTN


def test_backend_rejects_causal():
    impl_cls = XAttnRingAttentionBackend.get_impl_cls()
    with pytest.raises(NotImplementedError, match="non-causal"):
        impl_cls(num_heads=H, head_size=D, causal=True, softmax_scale=D**-0.5)


def test_platform_rejects_unsupported_head_size():
    from sglang.multimodal_gen.runtime.platforms.xpu import XpuPlatform

    with pytest.raises(ValueError, match="head_size"):
        XpuPlatform.get_attn_backend_cls_str(
            AttentionBackendEnum.XATTN_RING_ATTN, 48, torch.bfloat16
        )


def test_platform_ignores_probe_dtype():
    """DenoisingStage probes the selector with a hardcoded fp16 to size metadata."""
    from sglang.multimodal_gen.runtime.platforms.xpu import XpuPlatform

    cls_str = XpuPlatform.get_attn_backend_cls_str(
        AttentionBackendEnum.XATTN_RING_ATTN, 128, torch.float16
    )
    assert cls_str.endswith("XAttnRingAttentionBackend")


def _reference_attention(q, k, v, scale):
    """Q shard against the full K/V, in fp32."""
    logits = (q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2)) * scale
    out = torch.softmax(logits, dim=-1) @ v.transpose(1, 2).float()
    return out.transpose(1, 2)


def _reference_attention_thd(q, k, v, scale):
    """Same, for the packed ``[T, H, D]`` layout ``forward_varlen`` takes."""
    logits = (q.transpose(0, 1).float() @ k.transpose(0, 1).float().transpose(-1, -2)) * scale
    out = torch.softmax(logits, dim=-1) @ v.transpose(0, 1).float()
    return out.transpose(0, 1)


def _worker(rank, world_size, port, queue):
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        torch.xpu.set_device(rank)
        device = torch.device("xpu", rank)

        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=world_size, rank=rank, local_rank=rank, backend="xccl"
        )
        initialize_model_parallel(
            sequence_parallel_degree=world_size, ulysses_degree=1, ring_degree=world_size
        )
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_ring_parallel_world_size,
        )

        assert get_ring_parallel_world_size() == world_size

        gen = torch.Generator(device="cpu").manual_seed(0)
        shape = (B, S_LOCAL * world_size, H, D)
        q, k, v = (
            torch.randn(shape, generator=gen, dtype=torch.float32).to(device, DTYPE)
            for _ in range(3)
        )
        shard = slice(rank * S_LOCAL, (rank + 1) * S_LOCAL)
        scale = D**-0.5

        impl = XAttnRingAttentionBackend.get_impl_cls()(
            num_heads=H, head_size=D, causal=False, softmax_scale=scale
        )
        got = impl.forward(q[:, shard].contiguous(), k[:, shard].contiguous(), v[:, shard].contiguous())
        if isinstance(got, tuple):
            got = got[0]

        expected = _reference_attention(q[:, shard], k, v, scale)
        rel_err = ((got.float() - expected).abs().max() / expected.abs().max()).item()

        # A ring that drops the remote steps would still land near the local-only
        # result, so require it to be far from that and close to the full one.
        local_only = _reference_attention(q[:, shard], k[:, shard], v[:, shard], scale)
        local_gap = (
            (local_only - expected).abs().max() / expected.abs().max()
        ).item()
        queue.put((rank, rel_err, local_gap, None))
    except Exception as exc:
        queue.put((rank, None, None, repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@requires_xattn
@pytest.mark.parametrize("world_size", [2, 4])
def test_ring_matches_full_attention(world_size):
    if torch.xpu.device_count() < world_size:
        pytest.skip(f"requires {world_size} XPUs")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, world_size, port, queue))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=300) for _ in range(world_size)]
    for p in procs:
        p.join(timeout=60)

    for rank, rel_err, local_gap, err in results:
        assert err is None, f"rank {rank} failed: {err}"
        # bf16 accumulation over `world_size` merged blocks lands around 5e-3.
        assert rel_err < 1e-2, f"rank {rank} rel err {rel_err:.3e}"
        # Guards against the ring degenerating to local-only attention.
        assert local_gap > 10 * rel_err, (
            f"rank {rank}: local-only attention is within {local_gap:.3e} of the "
            f"reference, so rel err {rel_err:.3e} does not prove the ring rotated"
        )


@requires_xattn
def test_forward_varlen_rejects_multi_document():
    """A per-round KV clamp cannot mask interior packed-segment boundaries."""
    impl = XAttnRingAttentionBackend.get_impl_cls()(
        num_heads=H, head_size=D, causal=False, softmax_scale=D**-0.5
    )
    q = torch.zeros(8, H, D, dtype=DTYPE)
    with pytest.raises(NotImplementedError, match="single right-padded"):
        impl.forward_varlen(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
            max_seqlen=4,
            cu_seqlens_host=(0, 4, 8),
        )


def _worker_varlen(rank, world_size, port, queue):
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        torch.xpu.set_device(rank)
        device = torch.device("xpu", rank)

        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=world_size, rank=rank, local_rank=rank, backend="xccl"
        )
        initialize_model_parallel(
            sequence_parallel_degree=world_size, ulysses_degree=1, ring_degree=world_size
        )

        total = S_LOCAL * world_size
        # Leave the last rank's block part real, part padding -- the case the
        # per-rank valid_kv_lens clamp exists for.
        real_seq_len = total - S_LOCAL // 4
        scale = D**-0.5
        gen = torch.Generator(device="cpu").manual_seed(0)
        q, k, v = (
            torch.randn((total, H, D), generator=gen, dtype=torch.float32).to(device, DTYPE)
            for _ in range(3)
        )
        # Poison the padding rows: attending across the boundary is then obvious
        # instead of being lost in bf16 noise.
        k[real_seq_len:] = 100.0
        v[real_seq_len:] = 100.0

        shard = slice(rank * S_LOCAL, (rank + 1) * S_LOCAL)
        n_real = max(0, min(real_seq_len - rank * S_LOCAL, S_LOCAL))

        impl = XAttnRingAttentionBackend.get_impl_cls()(
            num_heads=H, head_size=D, causal=False, softmax_scale=scale
        )
        got = impl.forward_varlen(
            q[shard].contiguous(),
            k[shard].contiguous(),
            v[shard].contiguous(),
            cu_seqlens=torch.tensor([0, real_seq_len], dtype=torch.int32, device=device),
            max_seqlen=real_seq_len,
            cu_seqlens_host=(0, real_seq_len),
        )

        # Padding rows produce unused garbage; only the real rows are compared.
        expected = _reference_attention_thd(
            q[shard][:n_real], k[:real_seq_len], v[:real_seq_len], scale
        )
        denom = expected.abs().max()
        rel_err = ((got[:n_real].float() - expected).abs().max() / denom).item()

        local_only = _reference_attention_thd(
            q[shard][:n_real], k[shard][:n_real], v[shard][:n_real], scale
        )
        local_gap = ((local_only - expected).abs().max() / denom).item()
        queue.put((rank, rel_err, local_gap, None))
    except Exception as exc:
        queue.put((rank, None, None, repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_xattn
@pytest.mark.parametrize("world_size", [2, 4])
def test_ring_varlen_ignores_padding(world_size):
    """The H3 entry point: ring over a right-padded packed sequence."""
    if torch.xpu.device_count() < world_size:
        pytest.skip(f"requires {world_size} XPUs")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker_varlen, args=(r, world_size, port, queue))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=300) for _ in range(world_size)]
    for p in procs:
        p.join(timeout=60)

    for rank, rel_err, local_gap, err in results:
        assert err is None, f"rank {rank} failed: {err}"
        # Poisoned padding rows would push this far past bf16 noise.
        assert rel_err < 1e-2, f"rank {rank} rel err {rel_err:.3e}"
        assert local_gap > 10 * rel_err, (
            f"rank {rank}: local-only attention is within {local_gap:.3e} of the "
            f"reference, so rel err {rel_err:.3e} does not prove the ring rotated"
        )
