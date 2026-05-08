"""Synthetic decode batch injection for decode-only performance benchmarking.

Lets a benchmark driver hit `/inject_decode_batch` with (num_reqs, seq_len,
decode_steps) and have the server skip the long real prefill, going straight
into a steady-state decode loop. The KV slots backing each fake prefix are
allocated but never written by a real prefill -- they hold whatever GPU bytes
were there before. This is fine for performance measurement: attention kernel
timing depends on shape/dtype/hardware, not on the numerical values.

Implementation note (1-token extend, not strictly zero):
    The injector primes each request with `seq_len - 1` pre-allocated KV slots
    as a fake prefix and `extend_input_len = 1`. The scheduler's normal extend
    path then runs a 1-token extend to bootstrap state (sampling_info, spec
    decode draft input, hisparse coordinator, etc.) before transitioning to
    decode. The 1-token extend is well below profiling noise floor (~ms) and
    crucially is the path that initializes spec decode (`spec_info`) correctly,
    so we avoid having to manually fake EagleDraftInput / MTP state.

    All injected requests share the global `num_reqs`; each scheduler computes
    its own share from `attn_dp_rank` / dp_size. Ranks getting 0 reqs simply
    return success and let SGLang's existing DP-attention idle batch logic
    keep them in lockstep with busy ranks.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.managers.io_struct import BaseReq

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDecodeReqInput(BaseReq):
    """Inject N synthetic decode-ready requests into the running batch.

    `num_reqs` is the GLOBAL total across all attn DP ranks. Each scheduler
    computes its local share via `n // dp_size + (1 if rank < n % dp_size else 0)`.
    """

    num_reqs: int = 0
    seq_len: int = 0
    decode_steps: int = 0


@dataclass
class SyntheticDecodeReqOutput(BaseReq):
    success: bool = False
    message: str = ""
    n_local: int = 0


def _attn_dp_size(scheduler: "Scheduler") -> int:
    return (
        scheduler.dp_size
        if getattr(scheduler.server_args, "enable_dp_attention", False)
        else 1
    )


def _local_share(num_reqs: int, dp_size: int, dp_rank: int) -> int:
    base = num_reqs // dp_size
    remainder = num_reqs % dp_size
    return base + (1 if dp_rank < remainder else 0)


def handle_synthetic_decode_request(
    scheduler: "Scheduler",
    recv_req: SyntheticDecodeReqInput,
) -> SyntheticDecodeReqOutput:
    """Construct N decode-ready Reqs with pre-allocated fake KV prefix and
    push them onto the scheduler's waiting queue.

    Reqs carry `is_synthetic = True` so:
      - match_prefix_for_req leaves their pre-set prefix_indices alone
      - the output streamer skips them (no client to push to)
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    dp_size = max(_attn_dp_size(scheduler), 1)
    dp_rank = scheduler.attn_dp_rank
    n_local = _local_share(recv_req.num_reqs, dp_size, dp_rank)

    if n_local == 0:
        logger.info(
            "[synthetic-decode] rank=%d gets 0 reqs (idle); total=%d dp_size=%d",
            dp_rank,
            recv_req.num_reqs,
            dp_size,
        )
        return SyntheticDecodeReqOutput(success=True, n_local=0)

    seq_len = recv_req.seq_len
    decode_steps = recv_req.decode_steps
    if decode_steps < 1:
        return SyntheticDecodeReqOutput(
            success=False, message="decode_steps must be >= 1"
        )

    allocator = scheduler.token_to_kv_pool_allocator
    device = scheduler.req_to_token_pool.device

    # HiSparse's BaseTokenToKVPoolAllocator.alloc() raises NotImplementedError;
    # it requires alloc_extend semantics that we don't replicate here. Fail
    # loudly so the user knows to disable HiSparse for this benchmark mode.
    allocator_cls = type(allocator).__name__
    if "HiSparse" in allocator_cls:
        return SyntheticDecodeReqOutput(
            success=False,
            message=(
                f"{allocator_cls}.alloc() is not implemented; "
                "synthetic decode injection currently only supports the standard "
                "(non-HiSparse) token-to-kv allocator."
            ),
        )

    # Page-align the fake prefix so the paged allocator returns exactly the
    # number of slots we asked for, and so the extend kernel's last_loc/
    # prefix_len alignment invariant holds. The unaligned tail becomes the
    # bootstrap-extend portion (between 1 and page_size tokens, still cheap).
    page_size = max(getattr(allocator, "page_size", 1), 1)
    prefix_len = ((seq_len - 1) // page_size) * page_size
    extend_len = seq_len - prefix_len  # in [1, page_size]
    if prefix_len <= 0:
        return SyntheticDecodeReqOutput(
            success=False,
            message=(
                f"seq_len={seq_len} too small for page_size={page_size}; "
                f"need seq_len > page_size to leave a non-empty fake prefix."
            ),
        )

    prefix_indices_list: List[torch.Tensor] = []
    for i in range(n_local):
        slots = allocator.alloc(prefix_len)
        if slots is None or len(slots) < prefix_len:
            # Free any partial allocation we just got back, plus everything
            # previously grabbed for earlier reqs. Skipping this leaks the
            # whole partial chunk (paged allocators may return floor-aligned
            # slot counts when prefix_len is not a page multiple).
            if slots is not None and len(slots) > 0:
                allocator.free(slots)
            for prev in prefix_indices_list:
                allocator.free(prev)
            msg = (
                f"token_to_kv_pool_allocator exhausted: needed {prefix_len} "
                f"slots for synthetic prefix #{i}, got "
                f"{0 if slots is None else len(slots)}"
            )
            logger.error(msg)
            return SyntheticDecodeReqOutput(success=False, message=msg)
        prefix_indices_list.append(slots.to(device, non_blocking=True))

    fake_token = 1  # arbitrary; never read by attention compute

    for i in range(n_local):
        rid = f"synthetic-{uuid.uuid4().hex[:12]}-{i}"
        # Each Req needs its own SamplingParams instance so per-req state
        # (e.g. min_new_tokens clamping, custom_params) doesn't bleed across.
        sp = SamplingParams(
            temperature=0.0,
            max_new_tokens=decode_steps,
            ignore_eos=True,
        )
        sp.normalize(scheduler.tokenizer)

        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=[fake_token] * seq_len,
            sampling_params=sp,
            stream=False,
            return_logprob=False,
            eos_token_ids=scheduler.model_config.hf_eos_token_id,
            vocab_size=scheduler.model_config.vocab_size,
        )
        req.tokenizer = scheduler.tokenizer
        req.is_synthetic = True
        # Pre-set the prefix so match_prefix_for_req's early-return preserves it
        # and prepare_for_extend treats `prefix_len` tokens as already cached.
        req.prefix_indices = prefix_indices_list[i]
        req.fill_ids = list(req.origin_input_ids)
        # logprob_start_len = -1 + set_extend_input_len() makes
        # extend_logprob_start_len = extend_input_len, which keeps the DP
        # attention sync invariant `num_tokens_for_logprob == batch_size` for
        # non-logprob extend reqs (otherwise the all-gather assertion in
        # scheduler_dp_attn_mixin.prepare_mlp_sync_batch_raw fires).
        req.logprob_start_len = -1
        req.set_extend_input_len(extend_len)  # in [1, page_size]
        req.already_computed = prefix_len
        req.cached_tokens = prefix_len

        # Cap max_new_tokens via the same path normal requests use, so the
        # `max_req_len` bound is respected.
        scheduler.init_req_max_new_tokens(req)

        scheduler._add_request_to_queue(req)

    logger.info(
        "[synthetic-decode] rank=%d injected %d reqs (seq_len=%d, decode_steps=%d, "
        "page_size=%d, prefix_len=%d, extend_len=%d). Bootstrap extend will "
        "process %d token(s), then pure decode.",
        dp_rank,
        n_local,
        seq_len,
        decode_steps,
        page_size,
        prefix_len,
        extend_len,
        extend_len,
    )
    return SyntheticDecodeReqOutput(
        success=True,
        message=f"injected {n_local} synthetic decode reqs",
        n_local=n_local,
    )
