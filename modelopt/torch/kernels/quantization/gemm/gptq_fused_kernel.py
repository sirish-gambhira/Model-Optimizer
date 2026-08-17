# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton kernels for GPTQ blockwise weight-update.

A kernel for scalar NVFP4 quantization using production-computed effective
block scales. It fuses quantization + per-column GPTQ error propagation into
one launch per GPTQ block, avoiding the Python-level per-column loop.

Architecture:
  - One Triton program per output row.
  - ``w_full [BLOCK_SIZE]`` register tensor holds working weights.
  - Per-column: calls ``nvfp4_scalar_quant()`` with the exact production scale,
    then propagates error via ``w_full -= err * h_inv_row``.
"""

import torch
import triton
import triton.language as tl

from ..common.nvfp4_quant import nvfp4_scalar_quant

__all__ = ["gptq_fused_block_scalar"]


# ---------------------------------------------------------------------------
# Scalar kernel — NVFP4 QDQ + error propagation
# ---------------------------------------------------------------------------


@triton.jit
def _gptq_scalar_kernel(
    w_ptr,
    qw_ptr,
    err_ptr,
    block_scale_ptr,
    hinv_ptr,
    candidate_scale_ptr,
    selected_amax_ptr,
    num_rows,
    n_amax_blocks,
    quant_block_size,
    block_start,
    n_candidates: tl.constexpr,
    search_candidates: tl.constexpr,
    dynamic: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    w_base = w_ptr + row * BLOCK_SIZE
    qw_base = qw_ptr + row * BLOCK_SIZE
    err_base = err_ptr + row * BLOCK_SIZE
    block_scale_base = block_scale_ptr + row * n_amax_blocks

    j_range = tl.arange(0, BLOCK_SIZE)
    w_full = tl.load(w_base + j_range)
    active_scale = 0.0

    for col in range(0, BLOCK_SIZE, 1):
        quant_block = (block_start + col) // quant_block_size
        block_scale = tl.load(block_scale_base + quant_block)

        if dynamic and col % quant_block_size == 0:
            group_end = col + quant_block_size
            group_mask = (j_range >= col) & (j_range < group_end)
            target_scale = tl.max(tl.where(group_mask, tl.abs(w_full), 0.0)) / 6.0

            lo = 0
            hi = n_candidates
            for _ in range(7):
                mid = (lo + hi) // 2
                mid_scale = tl.load(candidate_scale_ptr + mid)
                lo = tl.where(mid_scale < target_scale, mid + 1, lo)
                hi = tl.where(mid_scale < target_scale, hi, mid)
            upper = tl.minimum(lo, n_candidates - 1)
            lower = tl.maximum(upper - 1, 0)
            upper_scale = tl.load(candidate_scale_ptr + upper)
            lower_scale = tl.load(candidate_scale_ptr + lower)
            base = tl.where(
                tl.abs(upper_scale - target_scale) < tl.abs(target_scale - lower_scale),
                upper,
                lower,
            )

            best_loss = float("inf")
            best_scale = lower_scale
            for candidate_offset in range(search_candidates):
                if search_candidates == n_candidates:
                    candidate_idx = candidate_offset
                else:
                    candidate_idx = tl.maximum(
                        0,
                        tl.minimum(
                            n_candidates - 1,
                            base - (search_candidates - 2) + candidate_offset,
                        ),
                    )
                candidate_scale = tl.load(candidate_scale_ptr + candidate_idx)
                candidate_q = nvfp4_scalar_quant(w_full, candidate_scale, BLOCK_SIZE)
                loss = tl.sum(
                    tl.where(group_mask, (candidate_q - w_full) * (candidate_q - w_full), 0.0)
                )
                choose = loss < best_loss
                best_loss = tl.where(choose, loss, best_loss)
                best_scale = tl.where(choose, candidate_scale, best_scale)
            active_scale = best_scale
            tl.store(
                selected_amax_ptr
                + row * (BLOCK_SIZE // quant_block_size)
                + col // quant_block_size,
                best_scale * 6.0,
            )

        w_scalar = tl.sum(tl.where(j_range == col, w_full, 0.0))
        if dynamic:
            q_scalar = tl.sum(
                nvfp4_scalar_quant(tl.full([1], w_scalar, dtype=tl.float32), active_scale, 1)
            )
        else:
            q_scalar = tl.sum(
                nvfp4_scalar_quant(
                    tl.full([1], w_scalar, dtype=tl.float32), block_scale, 1
                )
            )

        d_val = tl.load(hinv_ptr + col * BLOCK_SIZE + col)
        err_val = (w_scalar - q_scalar) / d_val
        tl.store(err_base + col, err_val)
        tl.store(qw_base + col, q_scalar)

        remaining = j_range > col
        hinv_row = tl.load(hinv_ptr + col * BLOCK_SIZE + j_range, mask=remaining, other=0.0)
        w_full = w_full - err_val * hinv_row


def gptq_fused_block_scalar(
    w_block: torch.Tensor,
    block_amax: torch.Tensor,
    global_scale: float,
    h_inv_cho_blk: torch.Tensor,
    quant_block_size: int,
    block_start: int,
    candidate_scales: torch.Tensor | None = None,
    dynamic_scale_candidates: int = 8,
    effective_block_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run scalar GPTQ (NVFP4) column loop for one block in a single Triton kernel launch.

    Uses production-computed effective FP8 block scales, then performs NVFP4
    fake quantization and GPTQ error propagation per column.

    Args:
        w_block:         Working weights ``[num_rows, block_size]`` (float32).
        block_amax:      Per-block amax values ``[num_rows, n_amax_blocks]`` (float32).
        global_scale:    Pre-computed ``global_amax / (6.0 * 448.0)`` (scalar).
        h_inv_cho_blk:   Block of upper-Cholesky inverse Hessian ``[block_size, block_size]``.
        quant_block_size: Number of elements sharing one scale factor.
        block_start:     Column offset of this block in the full weight matrix.

    Returns:
        ``(qw_block, err_block)`` each ``[num_rows, block_size]``.
    """
    num_rows, block_size = w_block.shape

    qw_block = torch.empty_like(w_block)
    err_block = torch.empty_like(w_block)
    dynamic = candidate_scales is not None
    if effective_block_scale is None and dynamic:
        # Dynamic mode supplies its selected scale separately; this pointer is unused.
        effective_block_scale = block_amax
    elif effective_block_scale is None:
        from .fp4_kernel import compute_fp4_scales

        global_amax = torch.as_tensor(
            global_scale * 6.0 * 448.0, device=w_block.device, dtype=torch.float32
        )
        effective_block_scale = compute_fp4_scales(block_amax, global_amax)
    if dynamic:
        selected_amax = torch.empty(
            (num_rows, block_size // quant_block_size),
            device=w_block.device,
            dtype=torch.float32,
        )
    else:
        selected_amax = torch.empty(1, device=w_block.device, dtype=torch.float32)
        candidate_scales = torch.empty(1, device=w_block.device, dtype=torch.float32)

    h_inv_cho_blk = h_inv_cho_blk.contiguous()
    candidate_scales = candidate_scales.contiguous()
    _gptq_scalar_kernel[(num_rows,)](
        w_block.contiguous(),
        qw_block,
        err_block,
        effective_block_scale.contiguous(),
        h_inv_cho_blk,
        candidate_scales,
        selected_amax,
        num_rows,
        block_amax.shape[1],
        quant_block_size,
        block_start,
        n_candidates=candidate_scales.numel(),
        search_candidates=dynamic_scale_candidates,
        dynamic=dynamic,
        BLOCK_SIZE=block_size,
    )

    return qw_block, err_block, selected_amax if dynamic else None
