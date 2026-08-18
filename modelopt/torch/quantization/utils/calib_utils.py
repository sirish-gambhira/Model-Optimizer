# Adapted from https://github.com/IST-DASLab/FP-Quant/blob/d2e3092/src/quantization/gptq.py
# with minor modifications to the original forms to accommodate minor architectural differences
# to be reused in the Model-Optimizer pipeline.
# Copyright (c) Andrei Panferov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""GPTQ helper and Hessian utilities for calibration."""

import math
import os

import torch

from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.network import bind_forward_method, unpatch_forward_method
from modelopt.torch.utils.perf import get_used_gpu_mem_fraction


def update_hessian(input, hessian, n_samples):
    """Update hessian matrix with new input samples using incremental formula.

    Args:
        input: Input tensor (batch_size, ..., features)
        hessian: Current Hessian matrix to update in-place
        n_samples: Number of samples already processed
    Returns:
        Tuple of (updated_hessian, new_sample_count)

    Note: input must be non-empty (batch_size > 0); a zero-sized input causes division by zero.
    """
    # Flatten to 2D (total_tokens, features) first, so batch_size counts tokens
    input_flat = input.reshape(-1, input.shape[-1]).t().float()
    batch_size = input_flat.shape[1]

    if batch_size == 0:  # in MOEs some experts receive no tokens
        return hessian, n_samples

    # Incremental averaging: scale down old hessian
    hessian *= n_samples / (n_samples + batch_size)
    n_samples += batch_size

    # Compute outer product: H += (2/n_samples) * X @ X^T
    scaled_input = math.sqrt(2 / n_samples) * input_flat
    hessian.add_((scaled_input @ scaled_input.t()).to(hessian.device))

    return hessian, n_samples


def compute_hessian_inverse(
    hessian, weight, perc_damp, factorization_device="current", factorization_dtype="float32"
):
    """Compute damped upper-Cholesky inverse Hessian.

    Dead input dimensions are identified by an exactly-zero Hessian
    diagonal. Their Hessian diagonal is set to one and their working-weight
    columns are zeroed before inversion, matching Quark and GPTQModel.

    Args:
        hessian: Hessian matrix ``[in_features, in_features]``.
        weight: Mutable GPTQ working-weight matrix ``[out_features, in_features]``.
        perc_damp: Percentage of average Hessian diagonal for damping.
        factorization_device: ``'current'`` uses ``hessian.device``. ``'cpu'`` moves the
            factorization to CPU and returns the result on ``weight.device``.
        factorization_dtype: Floating-point dtype used by the dense factorization. The
            resulting factor is converted back to the working-weight dtype.

    Returns:
        Upper-triangular Cholesky factor of the damped inverse Hessian
        ``[in_features, in_features]``, or ``None`` when the Hessian cannot
        be factorized. Callers must treat ``None`` as an RTN fallback; an
        identity factor is not a valid substitute for Hessian-aware GPTQ.
    """
    result_device = weight.device
    h = hessian.clone()
    dead = torch.diag(h) == 0
    if dead.all():
        print_rank_0("Warning: Hessian has no observed input dimensions; retaining scale-only RTN")
        return None
    weight[:, dead.to(result_device)] = 0
    if factorization_device == "cpu":
        h = h.cpu()
        dead = dead.cpu()
    elif factorization_device != "current":
        raise ValueError(f"Unsupported Hessian factorization device: {factorization_device}")
    if factorization_dtype == "float64":
        h = h.double()
    elif factorization_dtype != "float32":
        raise ValueError(f"Unsupported Hessian factorization dtype: {factorization_dtype}")
    h[dead, dead] = 1

    damp = perc_damp * torch.mean(torch.diag(h))
    diag_indices = torch.arange(h.shape[0], device=h.device)
    h[diag_indices, diag_indices] += damp

    try:
        h = torch.cholesky_inverse(torch.linalg.cholesky(h))
        return torch.linalg.cholesky(h, upper=True).to(
            device=result_device, dtype=weight.dtype
        )
    except (RuntimeError, torch.linalg.LinAlgError):
        print_rank_0("Warning: Hessian factorization failed; retaining scale-only RTN")
        return None


class GPTQHelper:
    """Encapsulates per-module GPTQ state and operations.

    Owns the Hessian, patches the forward during collection, and contains
    the blockwise weight-update logic.

    Instance attributes set during ``__init__``:
        module, name, hessian, n_samples

    Instance attributes set during ``update_weights``:
        weight: float working copy of module weights (mutated in-place by update methods)
        h_inv: upper-triangular Cholesky factor of the damped inverse Hessian
    """

    CACHE_NAME = "_forward_no_gptq_hessian"

    def __init__(
        self,
        module,
        name,
        offload_to_cpu=False,
        fused=False,
        hessian_storage_device="auto",
    ):
        """Initialize GPTQHelper with module state and Hessian storage."""
        self.module = module
        self.name = name
        self.fused = fused
        in_features = module.weight.shape[-1]
        device = module.weight.device
        if hessian_storage_device == "cpu" or (hessian_storage_device == "auto" and (
            device.type == "meta"
            or (offload_to_cpu and get_used_gpu_mem_fraction(device) > 0.65)
        )):
            device = "cpu"
        elif hessian_storage_device not in ("auto", "current"):
            raise ValueError(
                "hessian_storage_device must be one of 'auto', 'current', or 'cpu'"
            )
        self.hessian = torch.zeros(in_features, in_features, dtype=torch.float32, device=device)
        self.n_samples = 0
        # Set by update_weights(); listed here for documentation.
        self.weight: torch.Tensor | None = None
        self.h_inv: torch.Tensor | None = None
        self.updated_amax: torch.Tensor | None = None
        self.sequential_loss: float | None = None

    def setup(self):
        """Patch the module's forward to accumulate Hessian during the collection pass."""
        gptq_helper = self

        def hessian_forward(self, input, *args, **kwargs):
            inp = input.to_local() if hasattr(input, "to_local") else input
            if self.input_quantizer is not None and self.input_quantizer.is_enabled:
                hessian_input = self.input_quantizer(inp)
            else:
                hessian_input = inp
            gptq_helper.hessian, gptq_helper.n_samples = update_hessian(
                hessian_input, gptq_helper.hessian, gptq_helper.n_samples
            )

            out = self._forward_no_gptq_hessian(input, *args, **kwargs)

            return out

        bind_forward_method(self.module, hessian_forward, self.CACHE_NAME)

    def cleanup(self):
        """Unpatch the module's forward method."""
        unpatch_forward_method(self.module, self.CACHE_NAME)

    def free(self):
        """Release Hessian and working tensors to reclaim memory."""
        self.hessian = None
        self.weight = None
        self.h_inv = None
        self.updated_amax = None
        self.sequential_loss = None

    def update_weights(
        self,
        block_size,
        perc_damp,
        compare_vs_rtn=False,
        block_scale_update="static",
        dynamic_scale_candidates=8,
        *,
        hessian_inverse_device="current",
        hessian_inverse_dtype="float32",
    ):
        """Run GPTQ blockwise weight update on this module.

        Populates ``self.weight`` and ``self.h_inv``, runs the blockwise update,
        and writes the result back to the module unless an optional RTN-relative
        calibration-loss guard rejects it.
        """
        hessian = self.hessian.to(self.module.weight.device)
        original_weight = self.module.weight.data.float()
        self.weight = original_weight.clone()
        if not self._prepare_hessian_inverse(
            hessian, perc_damp, hessian_inverse_device, hessian_inverse_dtype
        ):
            return {
                "accepted": False,
                "fallback_reason": "hessian_factorization",
                "gptq_reconstruction_loss": None,
                "rtn_reconstruction_loss": None,
                "gptq_elementwise_mse": None,
                "rtn_elementwise_mse": None,
                "gptq_relative_mse": None,
                "rtn_relative_mse": None,
                "gptq_mse_ratio_vs_rtn": None,
            }
        self._blockwise_update(block_size, block_scale_update, dynamic_scale_candidates)
        metrics = self._candidate_metrics(
            original_weight,
            hessian,
            self.module.weight_quantizer,
            compare_vs_rtn,
            perc_damp,
        )
        self._print_mse_error(metrics)
        if metrics["accepted"]:
            self.module.weight.data = self.weight.reshape(self.module.weight.shape).to(
                self.module.weight.data.dtype
            )
            if self.updated_amax is not None:
                self.module.weight_quantizer.amax = self.updated_amax.reshape_as(
                    self.module.weight_quantizer.amax
                )
        return metrics

    # ------------------------------------------------------------------
    # Quantize helpers — all read from self.module, self.weight, self.h_inv
    # ------------------------------------------------------------------

    def _prepare_hessian_inverse(
        self,
        hessian,
        perc_damp,
        hessian_inverse_device="current",
        hessian_inverse_dtype="float32",
    ):
        """Compute damped inverse Hessian and store as ``self.h_inv``."""
        assert self.weight is not None, "_prepare_hessian_inverse called before update_weights()"
        self.h_inv = compute_hessian_inverse(
            hessian,
            self.weight,
            perc_damp,
            hessian_inverse_device,
            hessian_inverse_dtype,
        )
        return self.h_inv is not None

    def _blockwise_update(
        self, block_size, block_scale_update="static", dynamic_scale_candidates=8
    ):
        """Column-wise GPTQ update.

        When ``self.fused`` is True and the weight quantizer is an
        ``NVFP4StaticQuantizer``, uses :func:`gptq_blockwise_update_fused_scalar`
        (a fused Triton kernel).  Otherwise falls back to
        :func:`gptq_blockwise_update` (unfused column-by-column loop).
        """
        assert self.weight is not None and self.h_inv is not None, (
            "_blockwise_update called before _prepare_hessian_inverse()"
        )
        quantizer = self.module.weight_quantizer

        if self.fused and getattr(quantizer, "_is_nvfp4_static_quantizer", False):
            block_sizes = quantizer.block_sizes
            quant_block_size = block_sizes.get(-1) or block_sizes.get(1)
            if quant_block_size is not None and block_size % quant_block_size != 0:
                raise ValueError(
                    f"GPTQ block_size ({block_size}) must be divisible by the quantizer"
                    f" group_size ({quant_block_size})"
                )
            out_features, num_cols = self.weight.shape
            n_blocks = num_cols // quant_block_size
            block_amax = quantizer.amax.reshape(out_features, n_blocks).float()
            global_scale = quantizer.global_amax.float().item() / (6.0 * 448.0)
            if block_scale_update == "static":
                from modelopt.torch.quantization.utils.numeric_utils import (
                    fp8_max_for_normalization,
                )

                self.updated_amax = None
                self.sequential_loss = gptq_blockwise_update_static_nvfp4_groups(
                    self.weight,
                    block_amax,
                    global_scale,
                    self.h_inv,
                    block_size,
                    quant_block_size,
                    return_sequential_loss=True,
                    global_amax=quantizer.global_amax,
                    fp8_max_for_normalization=fp8_max_for_normalization(quantizer),
                    pass_through_bwd=quantizer._pass_through_bwd,
                )
            else:
                self.updated_amax, self.sequential_loss = gptq_blockwise_update_fused_scalar(
                    self.weight,
                    block_amax,
                    global_scale,
                    self.h_inv,
                    block_size,
                    quant_block_size,
                    dynamic=True,
                    dynamic_scale_candidates=dynamic_scale_candidates,
                    return_sequential_loss=True,
                )
        elif block_scale_update == "dynamic_mse":
            if not getattr(quantizer, "_is_nvfp4_static_quantizer", False):
                raise ValueError("dynamic_mse block scales require a static NVFP4 quantizer")
            self.updated_amax = gptq_blockwise_update_unfused_dynamic(
                self.weight,
                self.h_inv,
                block_size,
                quantizer,
                dynamic_scale_candidates,
            )
        else:
            gptq_blockwise_update(self.weight, self.h_inv, block_size, quantizer)

    def _candidate_metrics(self, original_weight, hessian, quantizer, compare_vs_rtn, perc_damp):
        """Compare the GPTQ candidate with the original-weight RTN candidate."""
        signal = original_weight.mm(hessian).mul(original_weight).mean().abs() + 1e-6

        def weighted_mse(candidate):
            delta = candidate.float() - original_weight
            return delta.mm(hessian).mul(delta).mean().clamp_min(0)

        gptq_mse = weighted_mse(self.weight)
        gptq_elementwise_mse = (self.weight.float() - original_weight).square().mean()
        damped_hessian = hessian.clone()
        dead = torch.diag(damped_hessian) == 0
        damped_hessian[dead, dead] = 1
        damp = perc_damp * torch.mean(torch.diag(damped_hessian))
        indices = torch.arange(damped_hessian.shape[0], device=damped_hessian.device)
        damped_hessian[indices, indices] += damp

        def damped_weighted_mse(candidate):
            delta = candidate.float() - original_weight
            return delta.mm(damped_hessian).mul(delta).mean().clamp_min(0)

        gptq_damped_mse = damped_weighted_mse(self.weight)
        metrics = {
            "accepted": True,
            "gptq_reconstruction_loss": gptq_mse.item(),
            "rtn_reconstruction_loss": None,
            "gptq_elementwise_mse": gptq_elementwise_mse.item(),
            "rtn_elementwise_mse": None,
            "gptq_relative_mse": (gptq_mse / signal).item(),
            "rtn_relative_mse": None,
            "gptq_mse_ratio_vs_rtn": None,
            "gptq_damped_reconstruction_loss": gptq_damped_mse.item(),
            "rtn_damped_reconstruction_loss": None,
            "gptq_damped_mse_ratio_vs_rtn": None,
            "gptq_sequential_objective": (
                2.0 * self.sequential_loss / self.weight.numel()
                if self.sequential_loss is not None
                else None
            ),
            "dead_hessian_dimensions": dead.sum().item(),
        }
        if compare_vs_rtn:
            rtn_weight = quantizer(original_weight).float()
            rtn_mse = weighted_mse(rtn_weight)
            rtn_elementwise_mse = (rtn_weight - original_weight).square().mean()
            rtn_damped_mse = damped_weighted_mse(rtn_weight)
            ratio = gptq_mse / (rtn_mse + 1e-12)
            damped_ratio = gptq_damped_mse / (rtn_damped_mse + 1e-12)
            metrics.update(
                {
                    "rtn_reconstruction_loss": rtn_mse.item(),
                    "rtn_elementwise_mse": rtn_elementwise_mse.item(),
                    "rtn_relative_mse": (rtn_mse / signal).item(),
                    "gptq_mse_ratio_vs_rtn": ratio.item(),
                    "rtn_damped_reconstruction_loss": rtn_damped_mse.item(),
                    "gptq_damped_mse_ratio_vs_rtn": damped_ratio.item(),
                }
            )
        return metrics

    def _print_mse_error(self, metrics):
        """Log Hessian-weighted relative MSE and optional comparison with RTN."""
        suffix = f", n_hessian_samples: {self.n_samples}" if self.n_samples else ""
        if metrics["gptq_mse_ratio_vs_rtn"] is not None:
            sequential = metrics["gptq_sequential_objective"]
            sequential_text = "n/a" if sequential is None else f"{sequential:.6e}"
            suffix += (
                f", rtn_loss: {metrics['rtn_reconstruction_loss']:.6e}, "
                f"rtn_elementwise_mse: {metrics['rtn_elementwise_mse']:.6e}, "
                f"vs_rtn: {metrics['gptq_mse_ratio_vs_rtn']:.2e}, "
                f"damped_vs_rtn: {metrics['gptq_damped_mse_ratio_vs_rtn']:.2e}, "
                f"sequential: {sequential_text}"
            )
        print_rank_0(
            f"[{self.name}] GPTQ reconstruction loss: "
            f"{metrics['gptq_reconstruction_loss']:.6e}, "
            f"elementwise_mse: {metrics['gptq_elementwise_mse']:.6e}, "
            f"relative: {metrics['gptq_relative_mse']:.2e}{suffix}"
        )

class FusedExpertGPTQHelper(GPTQHelper):
    """GPTQ state for one expert projection stored in a fused 3-D parameter."""

    def __init__(
        self,
        module,
        name,
        weight_name,
        expert_idx,
        quantizer,
        input_quantizer,
        offload_to_cpu=False,
        fused=False,
        hessian_storage_device="auto",
    ):
        """Initialize one helper for a fused parameter's expert slice."""
        self.module = module
        self.name = name
        self.weight_name = weight_name
        self.expert_idx = expert_idx
        self.quantizer = quantizer
        self.input_quantizer = input_quantizer
        self.fused = fused
        weight = self._weight_slice()
        device = weight.device
        if hessian_storage_device == "cpu" or (hessian_storage_device == "auto" and (
            device.type == "meta"
            or (offload_to_cpu and get_used_gpu_mem_fraction(device) > 0.65)
        )):
            device = "cpu"
        elif hessian_storage_device not in ("auto", "current"):
            raise ValueError(
                "hessian_storage_device must be one of 'auto', 'current', or 'cpu'"
            )
        self.hessian = torch.zeros(
            weight.shape[-1], weight.shape[-1], dtype=torch.float32, device=device
        )
        self.n_samples = 0
        self.weight = None
        self.h_inv = None
        self.updated_amax = None
        self.sequential_loss = None
        self._hook = None

    def _weight_slice(self):
        return getattr(self.module, self.weight_name)[self.expert_idx]

    def setup(self):
        """Collect quantized inputs when the fused wrapper evaluates this expert."""

        def collect(_quantizer, _args, output):
            if self.module._current_expert_idx == self.expert_idx:
                self.hessian, self.n_samples = update_hessian(output, self.hessian, self.n_samples)

        self._hook = self.input_quantizer.register_forward_hook(collect)

    def cleanup(self):
        """Remove the shared-input-quantizer collection hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def update_weights(
        self,
        block_size,
        perc_damp,
        compare_vs_rtn=False,
        block_scale_update="static",
        dynamic_scale_candidates=8,
        *,
        hessian_inverse_device="current",
        hessian_inverse_dtype="float32",
    ):
        """Run GPTQ on this expert slice and copy it into the fused parameter."""
        hessian = self.hessian.to(self._weight_slice().device)
        original_weight = self._weight_slice().float()
        self.weight = original_weight.clone()
        if not self._prepare_hessian_inverse(
            hessian, perc_damp, hessian_inverse_device, hessian_inverse_dtype
        ):
            return {
                "accepted": False,
                "fallback_reason": "hessian_factorization",
                "gptq_reconstruction_loss": None,
                "rtn_reconstruction_loss": None,
                "gptq_elementwise_mse": None,
                "rtn_elementwise_mse": None,
                "gptq_relative_mse": None,
                "rtn_relative_mse": None,
                "gptq_mse_ratio_vs_rtn": None,
            }
        self._blockwise_update(block_size, block_scale_update, dynamic_scale_candidates)
        metrics = self._candidate_metrics(
            original_weight,
            hessian,
            self.quantizer,
            compare_vs_rtn,
            perc_damp,
        )
        self._print_mse_error(metrics)
        if metrics["accepted"]:
            self._weight_slice().copy_(self.weight.to(self._weight_slice().dtype))
            if self.updated_amax is not None:
                self.quantizer.amax = self.updated_amax.reshape_as(self.quantizer.amax)
        return metrics

    def _blockwise_update(
        self, block_size, block_scale_update="static", dynamic_scale_candidates=8
    ):
        assert self.weight is not None and self.h_inv is not None
        if self.fused and getattr(self.quantizer, "_is_nvfp4_static_quantizer", False):
            quant_block_size = self.quantizer.block_sizes.get(-1) or self.quantizer.block_sizes.get(
                1
            )
            if block_size % quant_block_size != 0:
                raise ValueError(
                    f"GPTQ block_size ({block_size}) must be divisible by the quantizer "
                    f"group_size ({quant_block_size})"
                )
            verify_fused = os.environ.get("MODELOPT_GPTQ_VERIFY_FUSED") == "1"
            reference_weight = self.weight.clone() if verify_fused else None
            block_amax = self.quantizer.amax.reshape(self.weight.shape[0], -1).float()
            global_scale = self.quantizer.global_amax.float().item() / (6.0 * 448.0)
            dump_path = os.environ.get("MODELOPT_GPTQ_VERIFY_DUMP")
            if dump_path:
                torch.save(
                    {
                        "name": self.name,
                        "weight": self.weight.cpu(),
                        "h_inv": self.h_inv.cpu(),
                        "block_amax": block_amax.cpu(),
                        "global_scale": global_scale,
                        "block_size": block_size,
                        "quant_block_size": quant_block_size,
                    },
                    dump_path,
                )
            if block_scale_update == "static":
                from modelopt.torch.quantization.utils.numeric_utils import (
                    fp8_max_for_normalization,
                )

                self.updated_amax = None
                self.sequential_loss = gptq_blockwise_update_static_nvfp4_groups(
                    self.weight,
                    block_amax,
                    global_scale,
                    self.h_inv,
                    block_size,
                    quant_block_size,
                    return_sequential_loss=True,
                    global_amax=self.quantizer.global_amax,
                    fp8_max_for_normalization=fp8_max_for_normalization(self.quantizer),
                    pass_through_bwd=self.quantizer._pass_through_bwd,
                )
            else:
                self.updated_amax, self.sequential_loss = gptq_blockwise_update_fused_scalar(
                    self.weight,
                    block_amax,
                    global_scale,
                    self.h_inv,
                    block_size,
                    quant_block_size,
                    dynamic=True,
                    dynamic_scale_candidates=dynamic_scale_candidates,
                    return_sequential_loss=True,
                )
            if reference_weight is not None:
                gptq_blockwise_update(
                    reference_weight, self.h_inv, block_size, self.quantizer
                )

                mismatch = self.weight != reference_weight
                indices = mismatch.nonzero(as_tuple=False)
                first = indices[0].tolist() if indices.numel() else None
                print_rank_0(
                    f"[{self.name}] fused shadow parity: mismatches={indices.shape[0]}/"
                    f"{self.weight.numel()}, max_abs="
                    f"{(self.weight - reference_weight).abs().max().item():.9g}, "
                    f"first={first}, weight_device={self.weight.device}, "
                    f"amax_device={self.quantizer.amax.device}, "
                    f"amax_shape={tuple(self.quantizer.amax.shape)}, "
                    f"global_amax_device={self.quantizer.global_amax.device}"
                )
        elif block_scale_update == "dynamic_mse":
            if not getattr(self.quantizer, "_is_nvfp4_static_quantizer", False):
                raise ValueError("dynamic_mse block scales require a static NVFP4 quantizer")
            self.updated_amax = gptq_blockwise_update_unfused_dynamic(
                self.weight,
                self.h_inv,
                block_size,
                self.quantizer,
                dynamic_scale_candidates,
            )
        else:
            gptq_blockwise_update(self.weight, self.h_inv, block_size, self.quantizer)


def gptq_blockwise_update(weight, h_inv, block_size, quantize_fn):
    """Column-wise GPTQ update using full-matrix fake quantization.

    For each column, quantizes the full weight matrix via ``quantize_fn`` and
    extracts the quantized column.  Error is propagated to remaining columns
    within the block and then to all subsequent columns via the inverse Hessian.

    Args:
        weight: Weight tensor ``[out_features, in_features]``, modified **in-place**
            with fake-quantized values.
        h_inv: Upper-triangular Cholesky factor of the damped inverse Hessian
            ``[in_features, in_features]``.
        block_size: Number of columns to process per GPTQ block.
        quantize_fn: Callable ``(weight) -> qdq_weight`` that fake-quantizes
            the full weight matrix.
    """
    num_cols = weight.shape[1]

    for block_start in range(0, num_cols, block_size):
        block_end = min(block_start + block_size, num_cols)
        n_cols_blk = block_end - block_start
        h_inv_cho_blk = h_inv[block_start:block_end, block_start:block_end]

        wblk = weight.clone()
        errs = torch.zeros_like(weight[:, block_start:block_end])

        for i in range(n_cols_blk):
            w_ci = wblk[:, block_start + i]
            d = h_inv_cho_blk[i, i]
            qdq = quantize_fn(wblk)
            weight[:, block_start + i] = qdq[:, block_start + i]
            err = (w_ci - qdq[:, block_start + i]) / d
            wblk[:, block_start + i : block_end].addr_(err, h_inv_cho_blk[i, i:], alpha=-1)
            errs[:, i] = err

        weight[:, block_end:].addmm_(errs, h_inv[block_start:block_end, block_end:], alpha=-1)


def gptq_blockwise_update_static_nvfp4_groups(
    weight,
    block_amax,
    global_scale,
    h_inv,
    block_size,
    quant_block_size,
    return_sequential_loss=False,
    *,
    global_amax=None,
    fp8_max_for_normalization=448.0,
    pass_through_bwd=False,
):
    """Exact PyTorch GPTQ recurrence with group-local static NVFP4 QDQ.

    Static blockwise quantization is independent between quantization groups.
    The generic unfused reference nevertheless requantizes the full weight matrix
    for every selected column. Quantizing only that column's 16-value group gives
    the identical selected value while retaining PyTorch's stable ``addr_`` update.
    """
    from modelopt.torch.quantization.tensor_quant import static_blockwise_fp4_fake_quant

    block_amax = block_amax.to(device=weight.device, dtype=torch.float32)
    if global_amax is None:
        global_amax = global_scale * 6.0 * 448.0
    global_amax = torch.as_tensor(global_amax, device=weight.device, dtype=torch.float32)
    num_cols = weight.shape[1]
    sequential_loss = 0.0
    for block_start in range(0, num_cols, block_size):
        block_end = min(block_start + block_size, num_cols)
        h_inv_cho_blk = h_inv[block_start:block_end, block_start:block_end]
        wblk = weight.clone()
        errs = torch.zeros_like(weight[:, block_start:block_end])

        for i in range(block_end - block_start):
            column = block_start + i
            group_idx = column // quant_block_size
            group_start = group_idx * quant_block_size
            group_end = group_start + quant_block_size
            qdq_group = static_blockwise_fp4_fake_quant(
                wblk[:, group_start:group_end].contiguous(),
                block_amax[:, group_idx : group_idx + 1].contiguous(),
                global_amax,
                True,
                fp8_max_for_normalization,
                wblk.dtype,
                pass_through_bwd,
            )
            qdq_column = qdq_group[:, column - group_start]
            d = h_inv_cho_blk[i, i]
            err = (wblk[:, column] - qdq_column) / d
            weight[:, column] = qdq_column
            wblk[:, column:block_end].addr_(err, h_inv_cho_blk[i, i:], alpha=-1)
            errs[:, i] = err
            sequential_loss += err.float().square().sum().item() / 2.0

        weight[:, block_end:].addmm_(
            errs, h_inv[block_start:block_end, block_end:], alpha=-1
        )

    if return_sequential_loss:
        return sequential_loss


def gptq_blockwise_update_unfused_dynamic(
    weight,
    h_inv,
    block_size,
    quantizer,
    dynamic_scale_candidates=8,
):
    """Reference GPTQ loop with per-group dynamic NVFP4 scale selection.

    At the start of each quantization group, choose the scale that minimizes
    elementwise error on the current compensated working weights. Quantization
    itself uses the production block-16 NVFP4 QDQ path rather than the fused
    kernel's scalar reimplementation.
    """
    from modelopt.torch.kernels.quantization.gemm._fp8_scale_candidates import fp8_scale_candidates
    from modelopt.torch.kernels.quantization.gemm.fp4_kernel import compute_fp4_scales
    from modelopt.torch.quantization.tensor_quant import static_blockwise_fp4_fake_quant
    from modelopt.torch.quantization.utils.numeric_utils import fp8_max_for_normalization

    quant_block_size = quantizer.block_sizes.get(-1) or quantizer.block_sizes.get(1)
    if quant_block_size is None or block_size % quant_block_size != 0:
        raise ValueError("dynamic block scales require aligned GPTQ and quantization blocks")

    num_rows, num_cols = weight.shape
    n_quant_blocks = num_cols // quant_block_size
    updated_amax = quantizer.amax.reshape(num_rows, n_quant_blocks).float().clone()
    global_amax = quantizer.global_amax.detach().to(weight.device, torch.float32).reshape(())
    fp8_max = fp8_max_for_normalization(quantizer)
    all_candidate_amaxes = fp8_scale_candidates(weight.device).float() * global_amax
    all_candidate_scales = compute_fp4_scales(
        all_candidate_amaxes,
        global_amax,
        quantize_block_scales=True,
        fp8_max_for_normalization=fp8_max,
    )

    def quantize(values, amax):
        return static_blockwise_fp4_fake_quant(
            values,
            amax,
            global_amax,
            True,
            fp8_max,
            values.dtype,
            quantizer._pass_through_bwd,
        )

    candidate_offsets = torch.arange(dynamic_scale_candidates, device=weight.device)
    candidate_offsets -= dynamic_scale_candidates - 2

    for block_start in range(0, num_cols, block_size):
        block_end = min(block_start + block_size, num_cols)
        h_inv_cho_blk = h_inv[block_start:block_end, block_start:block_end]
        wblk = weight.clone()
        errs = torch.zeros_like(weight[:, block_start:block_end])

        for i in range(block_end - block_start):
            column = block_start + i
            if column % quant_block_size == 0:
                group = wblk[:, column : column + quant_block_size]
                if dynamic_scale_candidates == all_candidate_amaxes.numel():
                    candidate_indices = torch.arange(
                        all_candidate_amaxes.numel(), device=weight.device
                    ).expand(num_rows, -1)
                else:
                    target_scale = group.abs().amax(dim=1) / 6.0
                    base = (target_scale[:, None] - all_candidate_scales[None, :]).abs().argmin(
                        dim=1
                    )
                    candidate_indices = (
                        base[:, None] + candidate_offsets[None, :]
                    ).clamp_(0, all_candidate_amaxes.numel() - 1)

                candidate_amaxes = all_candidate_amaxes[candidate_indices]
                expanded_group = group[:, None, :].expand(
                    -1, dynamic_scale_candidates, -1
                )
                candidate_qdq = quantize(expanded_group, candidate_amaxes)
                candidate_loss = (candidate_qdq - expanded_group).float().square().sum(dim=2)
                best = candidate_loss.argmin(dim=1, keepdim=True)
                selected_amax = candidate_amaxes.gather(1, best).squeeze(1)
                updated_amax[:, column // quant_block_size] = selected_amax

            w_ci = wblk[:, column]
            d = h_inv_cho_blk[i, i]
            qdq = quantize(wblk, updated_amax)
            weight[:, column] = qdq[:, column]
            err = (w_ci - qdq[:, column]) / d
            wblk[:, column:block_end].addr_(err, h_inv_cho_blk[i, i:], alpha=-1)
            errs[:, i] = err

        weight[:, block_end:].addmm_(
            errs, h_inv[block_start:block_end, block_end:], alpha=-1
        )

    return updated_amax


def gptq_blockwise_update_fused_scalar(
    weight,
    block_amax,
    global_scale,
    h_inv,
    block_size,
    quant_block_size,
    dynamic=False,
    dynamic_scale_candidates=8,
    return_sequential_loss=False,
):
    """Fused GPTQ blockwise update for NVFP4 scalar quantization.

    Uses a fused Triton kernel that combines scale computation, quantization,
    and per-column error propagation into one launch per GPTQ block, avoiding
    the Python-level per-column loop in :func:`gptq_blockwise_update`.

    Args:
        weight: Weight tensor ``[out_features, in_features]``, modified **in-place**
            with fake-quantized values.
        block_amax: Per-block amax values ``[out_features, n_amax_blocks]``.
        global_scale: Pre-computed ``global_amax / (6.0 * 448.0)`` (scalar).
        h_inv: Upper-triangular Cholesky factor of the damped inverse Hessian
            ``[in_features, in_features]``.
        block_size: Number of columns to process per GPTQ block.
        quant_block_size: Number of elements sharing one quantization scale factor.
    """
    from modelopt.torch.kernels.quantization.gemm.gptq_fused_kernel import gptq_fused_block_scalar

    # Layerwise calibration with Accelerate can offload quantizer buffers while
    # keeping the active expert weight on CUDA. Triton accepts the tensor-shaped
    # argument as a raw pointer, so normalize metadata to the working device here
    # instead of letting a host pointer reach the kernel.
    block_amax = block_amax.to(device=weight.device, dtype=torch.float32)

    if dynamic and block_size % quant_block_size != 0:
        raise ValueError("dynamic block scales require GPTQ blocks aligned to quant blocks")
    candidate_scales = None
    updated_amax = torch.empty_like(block_amax) if dynamic else None
    effective_block_scale = None
    if dynamic:
        from modelopt.torch.kernels.quantization.gemm._fp8_scale_candidates import (
            fp8_scale_candidates,
        )

        candidate_scales = fp8_scale_candidates(weight.device) * (global_scale * 448.0)
    else:
        from modelopt.torch.kernels.quantization.gemm.fp4_kernel import compute_fp4_scales

        global_amax = torch.as_tensor(
            global_scale * 6.0 * 448.0, device=weight.device, dtype=torch.float32
        )
        effective_block_scale = compute_fp4_scales(block_amax, global_amax)

    num_cols = weight.shape[1]
    sequential_loss = 0.0
    for bs in range(0, num_cols, block_size):
        be = min(bs + block_size, num_cols)
        qw, err, selected_amax = gptq_fused_block_scalar(
            weight[:, bs:be].clone().contiguous(),
            block_amax,
            global_scale,
            h_inv[bs:be, bs:be].contiguous(),
            quant_block_size,
            bs,
            candidate_scales=candidate_scales,
            dynamic_scale_candidates=dynamic_scale_candidates,
            effective_block_scale=effective_block_scale,
        )
        weight[:, bs:be] = qw
        sequential_loss += err.float().square().sum().item() / 2.0
        if be < num_cols:
            weight[:, be:].addmm_(err, h_inv[bs:be, be:], alpha=-1)
        if updated_amax is not None:
            first = bs // quant_block_size
            updated_amax[:, first : first + selected_amax.shape[1]] = selected_amax

    if return_sequential_loss:
        return updated_amax, sequential_loss
    return updated_amax


_GPTQ_HELPER_REGISTRY: dict[str, type[GPTQHelper]] = {}


def register_gptq_helper(backend: str, factory: type[GPTQHelper]) -> None:
    """Register a :class:`GPTQHelper` subclass for a quantizer backend.

    When :func:`modelopt.torch.quantization.model_calib.gptq` encounters a
    module whose ``weight_quantizer.backend`` matches ``backend``, it will
    construct ``factory`` instead of the default ``GPTQHelper``.
    """
    _GPTQ_HELPER_REGISTRY[backend] = factory
