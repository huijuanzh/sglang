# SPDX-License-Identifier: Apache-2.0
"""Ring attention backend built on deepklox's SYCL-TLA IPC kernel (Intel XPU).

Ported from vLLM-Omni's XPU ring path
(``vllm_omni/diffusion/attention/parallel/ring.py``), which dispatches to
``deepklox.ring_attn_ipc_forward``.

The kernel owns the whole ring pass: it rotates the K/V shards around the ring
group over Level-Zero IPC peer copies, runs the FMHA for each step and merges
the partial results with an online softmax. Because the copies run on a
dedicated copy engine rather than the compute queue, the transfer overlaps the
attention kernels instead of serializing against them.
Limits inherited from the kernel: bf16 only, non-causal only, no joint K/V.
Select it with ``--attention-backend xattn_ring_attn`` and ``--ring-degree > 1``;
it only serves DiT components and only on a pure ring topology (no tensor
parallelism, Ulysses 1) -- see ``supports_component``/``validate_parallelism``.
"""

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_sp_group,
)
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
        import deepklox  # noqa: F401
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

    @classmethod
    def supports_component(cls, module) -> bool:
        from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT

        # Only the DiT is sequence parallel; every other component holds whole
        # sequences, so rotating their KV over the ring group is meaningless.
        return isinstance(module, BaseDiT)

    @classmethod
    def validate_parallelism(cls, server_args) -> None:
        name = cls.get_enum().name.lower()
        if server_args.ring_degree <= 1 or server_args.ulysses_degree != 1:
            raise ValueError(
                f"{name} requires ring_degree > 1 and ulysses_degree = 1; got "
                f"ring_degree={server_args.ring_degree}, "
                f"ulysses_degree={server_args.ulysses_degree}"
            )
        # The kernel rotates KV over the whole DiT sequence shard, so the DiT
        # must be pure sequence parallel across every GPU.
        if server_args.tp_size != 1 or server_args.sp_degree != server_args.num_gpus:
            raise ValueError(
                f"{name} requires the DiT to run full sequence parallelism "
                f"without tensor parallelism (tp_size = 1, sp_degree = num_gpus); "
                f"got tp_size={server_args.tp_size}, "
                f"sp_degree={server_args.sp_degree}, num_gpus={server_args.num_gpus}"
            )

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
        from deepklox import ring_attn_ipc_forward
        from deepklox.ring_attn_interface import ring_attn_fmha

        if causal:
            # The kernel applies the mask identically on every ring round, so it
            # rejects causal rather than silently masking the peer blocks too.
            raise NotImplementedError(
                "xattn_ring attention backend supports non-causal attention only."
            )

        self._ring_attn_ipc_forward = ring_attn_ipc_forward
        self._ring_attn_fmha = ring_attn_fmha
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

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
        ring_active: bool = False,
    ) -> torch.Tensor:
        """Attention over one rank's packed ``[T, H, D]`` chunk.

        H3 also drives its (unsharded, redundant-per-rank) text refiner through
        this same method with ``ring_active=False``: that call's rows are not
        split across the ring group at all, so rotating it against the global
        ring world size would attend across ranks that hold the same data
        rather than a genuine peer shard. Only ``ring_active=True`` -- the
        packed multi-segment attention's own row shard -- rotates KV.
        """
        if query.dtype is not torch.bfloat16:
            raise ValueError(
                f"{type(self).__name__} requires bfloat16, got {query.dtype}"
            )

        real_seq_len = max_seqlen
        if cu_seqlens_host is not None:
            boundaries = tuple(int(v) for v in cu_seqlens_host)
            # A per-round KV clamp can only trim a trailing padding run; it
            # cannot mask interior boundaries, so reject true multi-document
            # packing rather than silently attending across segments.
            if (
                len(boundaries) not in (2, 3)
                or boundaries[0] != 0
                or list(boundaries) != sorted(boundaries)
            ):
                raise NotImplementedError(
                    f"{type(self).__name__} supports a single right-padded "
                    f"packed document ([0, used] or [0, used, total]), got "
                    f"cu_seqlens={boundaries}."
                )
            real_seq_len = boundaries[1]

        chunk_len = query.shape[0]
        if not ring_active:
            # No cross-rank rotation: just mask off this chunk's own padding
            # tail with a single local FMHA round.
            valid_len = min(max(real_seq_len, 0), chunk_len)
            if valid_len <= 0:
                raise ValueError(
                    f"real_seq_len={real_seq_len} leaves no real rows in this "
                    f"{chunk_len}-row chunk."
                )
            out = self._ring_attn_fmha(
                query.unsqueeze(0).contiguous(),
                key.unsqueeze(0).contiguous(),
                value.unsqueeze(0).contiguous(),
                softmax_scale=self.softmax_scale,
                causal=False,
                return_softmax_lse=False,
                valid_kv_len=valid_len,
            )
            return out.squeeze(0)

        ring_ws, _ = get_ring_ctx()
        valid_kv_lens = [
            min(max(real_seq_len - rank * chunk_len, 0), chunk_len)
            for rank in range(ring_ws)
        ]
        if any(length <= 0 for length in valid_kv_lens):
            # The kernel requires every rank to hold at least one real row, and
            # the ring degree is pinned to the GPU count, so the fix is a longer
            # sequence or fewer GPUs rather than a smaller ring.
            raise NotImplementedError(
                f"ring degree {ring_ws} leaves an all-padding block for "
                f"real_seq_len={real_seq_len} over {chunk_len}-row chunks "
                f"(valid_kv_lens={valid_kv_lens}); run this request on fewer GPUs."
            )

        ring_group = get_sp_group().ring_group
        assert ring_group is not None, "Ring process group is not initialized."

        out = self._ring_attn_ipc_forward(
            process_group=ring_group,
            q=query.unsqueeze(0).contiguous(),
            k=key.unsqueeze(0).contiguous(),
            v=value.unsqueeze(0).contiguous(),
            softmax_scale=self.softmax_scale,
            causal=False,
            return_softmax_lse=False,
            valid_kv_lens=valid_kv_lens,
        )
        return out.squeeze(0)
