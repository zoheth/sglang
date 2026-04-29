# mapping on device memory, host memory and memory allocator

import os
import weakref
from typing import List, Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import get_num_new_pages

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


class HiSparseNSATokenToKVPool(NSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
        index_layer_mask: Optional[List[bool]] = None,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_buf_size=size * host_to_device_ratio,
            index_layer_mask=index_layer_mask,
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")


_HISPARSE_TRACE = os.environ.get("SGL_HISPARSE_POOL_TRACE", "0") == "1"
_HISPARSE_TRACE_OP_ID = 0


def _hisparse_trace(allocator, op: str, before, after, **info):
    if not _HISPARSE_TRACE:
        return
    global _HISPARSE_TRACE_OP_ID
    _HISPARSE_TRACE_OP_ID += 1
    caller = ""
    try:
        import traceback as _tb

        stack = _tb.extract_stack()[:-2]
        for frame in reversed(stack):
            if "hisparse_memory_pool.py" in frame.filename:
                continue
            caller = f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}"
            break
    except Exception:
        pass
    logical_before, hisparse_before = before
    logical_after, hisparse_after = after
    extras = " ".join(f"{k}={v}" for k, v in info.items())
    print(
        f"[HSP-TRACE #{_HISPARSE_TRACE_OP_ID:05d}] {op} "
        f"L:{logical_before}->{logical_after}({logical_after-logical_before:+d}) "
        f"H:{hisparse_before}->{hisparse_after}({hisparse_after-hisparse_before:+d}) "
        f"{extras} <- {caller}",
        flush=True,
    )


def _hisparse_pool_avail(allocator):
    return (
        allocator.logical_attn_allocator.available_size(),
        allocator.hisparse_attn_allocator.available_size(),
    )


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: NSATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
    ):
        self._kvcache = kvcache
        self._size_full = size * host_to_device_ratio
        self._size_hisparse = size
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.need_sort = need_sort

        self.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_full,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )

        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._size_full + self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self._kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size(),
        )

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "Page size = 1 is not supported in HiSparse allocator"
        )

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate only logical indices without hisparse device indices.

        Used in the direct-to-host transfer path where KV data is written
        directly to host memory by the prefill node, skipping GPU staging.
        """
        before = _hisparse_pool_avail(self)
        out = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        _hisparse_trace(
            self,
            "alloc_logical_only",
            before,
            _hisparse_pool_avail(self),
            extend=extend_num_tokens,
            out_len=(0 if out is None else len(out)),
        )
        return out

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.page_size == 0
        before = _hisparse_pool_avail(self)
        n_alloc_in = len(allocated_indices)
        # Read the prefill mapping but DO NOT clear it: subsequent alloc_extend
        # calls (e.g. spec-mode verify) read mapping[last_prefill_loc] as
        # hisparse_last_loc; clearing it would force the kernel to write into
        # page 0 (the sentinel page) and corrupt the pool. Stale mapping for
        # any prefill slots not retained in the device buffer is acceptable —
        # those slots will be reallocated and the mapping overwritten.
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        # Filter valid (non-zero) hisparse indices.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        n_reused = int(hisparse_indices.numel())
        n_extra = 0
        n_residual = 0
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            self.free_hisparse_indices(hisparse_indices[need_size:])
        else:
            # page alignment, claiming the residual space for an incomplete page
            page_residual_length = len(hisparse_indices) % self.page_size
            if page_residual_length != 0:
                n_residual = self.page_size - page_residual_length
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            n_extra = int(extra_indices.numel())
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        _hisparse_trace(
            self,
            "alloc_device_buffer",
            before,
            _hisparse_pool_avail(self),
            need_size=need_size,
            allocated_in=n_alloc_in,
            reused=n_reused,
            residual=n_residual,
            extra=n_extra,
            out_len=int(buffer_indices.numel()),
        )
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        before = _hisparse_pool_avail(self)
        n_in = int(buffer_indices.numel())
        # disable free group mechanism for device buffer free
        self.hisparse_attn_allocator.is_not_in_free_group = True
        kept = buffer_indices[buffer_indices > 0]
        self.hisparse_attn_allocator.free(kept)
        _hisparse_trace(
            self,
            "free_hisparse_indices",
            before,
            _hisparse_pool_avail(self),
            in_len=n_in,
            kept_len=int(kept.numel()),
        )

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        hisparse_last_locs = self._kvcache._translate_loc_to_hisparse_device(last_locs)
        return hisparse_last_locs

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1
        before = _hisparse_pool_avail(self)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if (
            num_new_pages
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            _hisparse_trace(
                self,
                "alloc_extend(FAIL_LOGICAL)",
                before,
                _hisparse_pool_avail(self),
                extend=extend_num_tokens,
                new_pages=num_new_pages,
            )
            return None
        if (
            num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.page_size
        ):
            _hisparse_trace(
                self,
                "alloc_extend(FAIL_HISPARSE)",
                before,
                _hisparse_pool_avail(self),
                extend=extend_num_tokens,
                new_pages=num_new_pages,
            )
            return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_last_loc_first = (
            int(hisparse_last_loc[0].item()) if hisparse_last_loc.numel() > 0 else -1
        )
        last_loc_first = int(last_loc[0].item()) if last_loc.numel() > 0 else -1
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
            len(logical_indices),
        )
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"
        _hisparse_trace(
            self,
            "alloc_extend",
            before,
            _hisparse_pool_avail(self),
            extend=extend_num_tokens,
            new_pages=num_new_pages,
            last_loc=last_loc_first,
            hisparse_last_loc=hisparse_last_loc_first,
            log_len=int(logical_indices.numel()),
            his_len=int(hisparse_indices.numel()),
            log_first=(
                int(logical_indices[0].item()) if logical_indices.numel() > 0 else -1
            ),
            his_first=(
                int(hisparse_indices[0].item())
                if hisparse_indices.numel() > 0
                else -1
            ),
        )

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices

        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        before = _hisparse_pool_avail(self)
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        _hisparse_trace(
            self,
            "alloc_decode",
            before,
            _hisparse_pool_avail(self),
            bs=int(seq_lens.numel()),
            out_len=(0 if logical_indices is None else int(logical_indices.numel())),
        )
        return logical_indices

    def alloc_decode_debug(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_decode(
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
        )

        if logical_indices is None or hisparse_indices is None:
            return None

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices

        return logical_indices

    def free_hisparse(self, free_indices: torch.Tensor):
        before = _hisparse_pool_avail(self)
        n_in = int(free_indices.numel())
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(free_indices)
        n_pre_filter = int(hisparse_indices.numel())
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        n_post_filter = int(hisparse_indices.numel())
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[free_indices] = 0
        _hisparse_trace(
            self,
            "free_hisparse",
            before,
            _hisparse_pool_avail(self),
            free_in=n_in,
            translated=n_pre_filter,
            kept=n_post_filter,
            log_first=int(free_indices[0].item()) if n_in > 0 else -1,
        )

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()

        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def backup_state(self):
        before = _hisparse_pool_avail(self)
        out = (
            self.logical_attn_allocator.backup_state(),
            self.hisparse_attn_allocator.backup_state(),
        )
        _hisparse_trace(self, "backup_state", before, _hisparse_pool_avail(self))
        return out

    def restore_state(self, state):
        before = _hisparse_pool_avail(self)
        logical_state, hisparse_state = state
        self.logical_attn_allocator.restore_state(logical_state)
        self.hisparse_attn_allocator.restore_state(hisparse_state)
        _hisparse_trace(self, "restore_state", before, _hisparse_pool_avail(self))

    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        before = _hisparse_pool_avail(self)
        n_in = int(free_index.numel())
        in_group = not self.is_not_in_free_group

        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
            self.free_hisparse(free_index)
        else:
            self.free_group.append(free_index)
        _hisparse_trace(
            self,
            "free",
            before,
            _hisparse_pool_avail(self),
            n=n_in,
            grouped=in_group,
            log_first=int(free_index[0].item()) if n_in > 0 else -1,
        )
        assert (
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        )
        assert (
            self.hisparse_attn_allocator.available_size()
            <= self.hisparse_attn_allocator.size
        )
