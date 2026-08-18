# SPDX-License-Identifier: Apache-2.0
"""Ring attention backend built on XAttention's SYCL-TLA IPC kernel (Intel XPU).

Ported from vLLM-Omni's XPU ring path
(``vllm_omni/diffusion/attention/parallel/ring.py``), which dispatches to
``xattention.ring_attn_ipc_forward``.

The kernel owns the whole ring pass: it rotates the K/V shards around the ring
group over Level-Zero IPC peer copies, runs the FMHA for each step and merges
the partial results with an online softmax. Because the copies run on a
dedicated copy engine rather than the compute queue, the transfer overlaps the
attention kernels instead of serializing against them.
Limits inherited from the kernel: bf16 only, non-causal only, no joint K/V.
Select it with ``--attention-backend xattn_ring`` and ``--ring-degree > 1``.
"""

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def is_xattn_ring_available() -> bool:
    try:
        import xattention  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


class XAttnRingAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = False
    # The kernel performs the ring itself, so USPAttention must hand it the
    # local shards directly instead of wrapping it in another ring loop.
    handles_sequence_parallel_ring: bool = True

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        # The fused kernel rotates KV and merges partials internally, so it
        # never has to hand a per-hop LSE back to the generic ring loop.
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.XATTN_RING_ATTN

    @staticmethod
    def get_impl_cls() -> type["XAttnRingAttentionImpl"]:
        return XAttnRingAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder


class XAttnRingAttentionImpl(AttentionImpl):

    handles_sequence_parallel_ring: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        from xattention import ring_attn_ipc_forward

        if causal:
            # The kernel applies the mask identically on every ring round, so it
            # rejects causal rather than silently masking the peer blocks too.
            raise NotImplementedError(
                "xattn_ring attention backend supports non-causal attention only."
            )

        self._ring_attn_ipc_forward = ring_attn_ipc_forward
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        return_softmax_lse: bool = False,
    ):
        """Run one ring pass over ``[B, S_local, H, D]`` sequence-parallel shards.

        Sequence-parallel padding (vLLM-Omni's ``valid_kv_lens``) is not plumbed
        through: USPAttention refuses ring parallelism on its masked and
        replicated-prefix paths, so no padded shard reaches this kernel.
        """
        if query.dtype is not torch.bfloat16:
            raise ValueError(
                f"{type(self).__name__} requires bfloat16, got {query.dtype}"
            )

        ring_group = get_sp_group().ring_group
        assert ring_group is not None, "Ring process group is not initialized."

        return self._ring_attn_ipc_forward(
            process_group=ring_group,
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            softmax_scale=self.softmax_scale,
            causal=False,
            return_softmax_lse=return_softmax_lse,
        )
