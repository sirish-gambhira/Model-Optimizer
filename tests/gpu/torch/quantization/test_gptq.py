# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import time

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from conftest import requires_triton

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.model_calib import gptq
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.utils.calib_utils import (
    FusedExpertGPTQHelper,
    compute_hessian_inverse,
    gptq_blockwise_update,
    gptq_blockwise_update_fused_scalar,
    gptq_blockwise_update_static_nvfp4_groups,
    gptq_blockwise_update_unfused_dynamic,
    update_hessian,
)
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

RAND_SEED = 42
torch.manual_seed(RAND_SEED)


def test_fused_expert_hessian_collects_quantizer_output():
    """Fused W4A4 GPTQ must form its Hessian from Q(X), not raw X."""

    class RoundingQuantizer(torch.nn.Module):
        def forward(self, inputs):
            return inputs.round()

    class FusedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(1, 3, 2))
            self._current_expert_idx = 0

    module = FusedExperts()
    input_quantizer = RoundingQuantizer()
    helper = FusedExpertGPTQHelper(
        module,
        "mlp.experts.gate_up_proj.0",
        "gate_up_proj",
        0,
        quantizer=None,
        input_quantizer=input_quantizer,
    )
    inputs = torch.tensor([[0.2, 1.6], [1.4, -0.3]])
    quantized_inputs = inputs.round()

    helper.setup()
    output = input_quantizer(inputs)
    helper.cleanup()

    expected = torch.zeros_like(helper.hessian)
    expected, expected_samples = update_hessian(quantized_inputs, expected, 0)
    raw = torch.zeros_like(helper.hessian)
    raw, _ = update_hessian(inputs, raw, 0)

    assert torch.equal(output, quantized_inputs)
    assert helper.n_samples == expected_samples
    torch.testing.assert_close(helper.hessian, expected)
    assert not torch.allclose(helper.hessian, raw)


@pytest.mark.parametrize(
    ("block_size", "dim", "model_weight", "expect_weight_change"),
    [
        (16, 128, torch.randn(128, 128).to("cuda"), True),  # random weight
        (
            16,
            128,
            torch.ones(128, 128).to("cuda"),
            False,
        ),  # all same weight -> no quantization error -> no GPTQ update
    ],
)
def test_gptq_updates(block_size, dim, model_weight, expect_weight_change):
    model = torch.nn.Linear(dim, dim).to("cuda")
    model.weight.data = model_weight
    original_weight = model_weight.clone()
    input_tensor = torch.randn(2, 16, dim).to("cuda")
    quant_cfg = mtq.NVFP4_DEFAULT_CFG

    mtq.quantize(model, quant_cfg, forward_loop=lambda model: model(input_tensor))

    # Get qdq weight
    q_dq_weight = model.weight_quantizer(model.weight.data)

    # Restore original weight before GPTQ
    model.weight.data = original_weight.clone()

    # Run GPTQ through the public API
    gptq(model, forward_loop=lambda m: m(input_tensor), perc_damp=0.1, block_size=block_size)
    if expect_weight_change:
        # Weight must change as GPTQ updates weights to adjust for quantization error
        assert not torch.allclose(model.weight.data, q_dq_weight), "Weight should not be equal"
    else:
        assert torch.allclose(model.weight.data, q_dq_weight), "Weight should be equal"


def test_gptq_rtn_fallback_preserves_original_weight():
    """An under-sampled module keeps its original weight for scale-only RTN export."""
    dim = 128
    model = torch.nn.Linear(dim, dim, bias=False, device="cuda")
    original_weight = model.weight.detach().clone()
    input_tensor = torch.randn(2, 16, dim, device="cuda")

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=lambda m: m(input_tensor))
    model.weight.data.copy_(original_weight)
    gptq(
        model,
        forward_loop=lambda m: m(input_tensor),
        perc_damp=0.1,
        block_size=16,
        rtn_fallback={"min_samples_per_input_dim": 1.0, "module_patterns": ["*"]},
    )
    assert model._gptq_calibration_stats["action"] == "rtn_fallback"
    assert model._gptq_calibration_stats["n_samples"] == 32
    assert model._gptq_calibration_stats["required_samples"] == dim
    assert torch.equal(model.weight.data, original_weight)


def test_gptq_reports_rtn_comparison_without_rejecting_candidate():
    """An eligible module reports RTN-relative loss without using it as a fallback gate."""
    dim = 128
    model = torch.nn.Linear(dim, dim, bias=False, device="cuda")
    original_weight = model.weight.detach().clone()
    input_tensor = torch.randn(8, 16, dim, device="cuda")

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=lambda m: m(input_tensor))
    model.weight.data.copy_(original_weight)
    gptq(
        model,
        forward_loop=lambda m: m(input_tensor),
        perc_damp=0.1,
        block_size=16,
        rtn_fallback={"min_samples_per_input_dim": 1.0},
    )

    stats = model._gptq_calibration_stats
    assert stats["action"] == "gptq"
    assert stats["accepted"]
    assert stats["gptq_elementwise_mse"] >= 0
    assert stats["rtn_elementwise_mse"] >= 0
    assert stats["gptq_mse_ratio_vs_rtn"] >= 0


def test_gptq_failed_hessian_factorization_uses_rtn_not_identity():
    """A sampled module with an unusable Hessian skips GPTQ and retains RTN."""
    dim = 128
    model = torch.nn.Linear(dim, dim, bias=False, device="cuda")
    original_weight = model.weight.detach().clone()
    input_tensor = torch.zeros(1, dim, dim, device="cuda")

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=lambda m: m(input_tensor))
    model.weight.data.copy_(original_weight)
    gptq(
        model,
        forward_loop=lambda m: m(input_tensor),
        perc_damp=0.01,
        block_size=16,
        rtn_fallback={"min_samples_per_input_dim": 1.0},
    )

    stats = model._gptq_calibration_stats
    assert stats["n_samples"] == dim
    assert stats["action"] == "rtn_hessian_fallback"
    assert stats["fallback_reason"] == "hessian_factorization"
    assert not stats["accepted"]
    assert torch.equal(model.weight.data, original_weight)


def test_gptq_module_patterns_leave_nonmatching_weights_untouched():
    dim = 128
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, dim, bias=False),
        torch.nn.Linear(dim, dim, bias=False),
    ).to("cuda")
    original_weights = [layer.weight.detach().clone() for layer in model]
    input_tensor = torch.randn(2, 16, dim, device="cuda")

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=lambda m: m(input_tensor))
    for layer, weight in zip(model, original_weights):
        layer.weight.data.copy_(weight)
    gptq(
        model,
        forward_loop=lambda m: m(input_tensor),
        perc_damp=0.1,
        block_size=16,
        module_patterns=["0"],
        rtn_fallback={"min_samples_per_input_dim": 1.0},
    )

    assert model[0]._gptq_calibration_stats["action"] == "rtn_fallback"
    assert not hasattr(model[1], "_gptq_calibration_stats")
    assert torch.equal(model[0].weight.data, original_weights[0])
    assert torch.equal(model[1].weight.data, original_weights[1])


def test_gptq_export_roundtrip():
    """Test that GPTQ export + dequantize produces weights matching in-memory QDQ."""
    torch.manual_seed(RAND_SEED)
    dim = 128
    block_size = 16

    # Step 1: Create a simple linear model and quantize to install NVFP4 quantizers
    model = torch.nn.Linear(dim, dim, dtype=torch.bfloat16).to("cuda")
    original_weight = model.weight.data.clone()
    input_tensor = torch.randn(2, 16, dim, dtype=torch.bfloat16).to("cuda")
    quant_cfg = mtq.NVFP4_DEFAULT_CFG

    mtq.quantize(model, quant_cfg, forward_loop=lambda m: m(input_tensor))

    # Restore original weight before GPTQ
    model.weight.data = original_weight.clone()

    # Step 2: Perform GPTQ — compute Hessian and update weights
    gptq(model, forward_loop=lambda m: m(input_tensor), perc_damp=0.1, block_size=block_size)

    # Save the QDQ reference from the quantizer applied to GPTQ'd weights
    gptq_weight_shape = model.weight.data.shape
    gptq_weight_dtype = model.weight.data.dtype
    qdq_ref = model.weight.data.clone()

    # Step 3: Export — converts weight to packed NVFP4 and registers scale buffers
    _export_quantized_weight(model, torch.bfloat16)

    # Verify export produced the expected buffers
    assert hasattr(model, "weight_scale"), "Export should register weight_scale buffer"
    assert hasattr(model, "weight_scale_2"), "Export should register weight_scale_2 buffer"

    # Step 4: Dequantize the exported packed weight and compare with QDQ reference
    packed_weight = model.weight.data
    weight_scale = model.weight_scale
    weight_scale_2 = model.weight_scale_2

    nvfp4_qtensor = NVFP4QTensor(gptq_weight_shape, gptq_weight_dtype, packed_weight)
    deq_weight = nvfp4_qtensor.dequantize(
        dtype=torch.bfloat16,
        scale=weight_scale,
        double_scale=weight_scale_2,
        block_sizes={-1: 16},
    )

    assert deq_weight.shape == qdq_ref.shape, (
        f"Shape mismatch: dequantized {deq_weight.shape} vs QDQ ref {qdq_ref.shape}"
    )
    assert torch.allclose(deq_weight, qdq_ref, atol=1e-2), (
        f"Dequantized weight does not match QDQ reference. "
        f"Max diff: {(deq_weight - qdq_ref).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "quant_cfg", [mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG]
)
def test_gptq_e2e_flow(quant_cfg, tiny_tokenizer):
    model = get_tiny_llama(vocab_size=tiny_tokenizer.vocab_size).to("cuda")
    model.eval()

    quant_cfg = copy.deepcopy(quant_cfg)
    quant_cfg["algorithm"] = {"method": "gptq", "layerwise": True}
    calib_dataloader = get_dataset_dataloader(
        dataset_name="cnn_dailymail",
        tokenizer=tiny_tokenizer,
        batch_size=2,
        num_samples=8,
        device="cuda",
        include_labels=False,
    )

    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)


# ---------------------------------------------------------------------------
# Fused Triton GPTQ kernel tests for NVFP4 scalar quantization
# ---------------------------------------------------------------------------


def _make_nvfp4_test_data(quant_block_size, out_features, dim, algorithm="max"):
    """Create weight, weight_quantizer, block_amax, global_scale, and h_inv for NVFP4 GPTQ tests."""
    # Build a quantized Linear with NVFP4 static config at the desired block size
    model = torch.nn.Linear(dim, out_features, bias=False, device="cuda")
    weight = model.weight.data.clone()

    nvfp4_static_cfg = {
        "num_bits": (2, 1),
        "block_sizes": {-1: quant_block_size, "type": "static", "scale_bits": (4, 3)},
    }
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": nvfp4_static_cfg},
        ],
        "algorithm": algorithm,
    }
    inp = torch.randn(4, 32, dim, device="cuda")
    mtq.quantize(model, quant_cfg, forward_loop=lambda m: m(inp))

    # Restore original weight (GPTQ operates on original weights)
    model.weight.data = weight.clone()

    weight_quantizer = model.weight_quantizer
    block_amax = weight_quantizer.amax.reshape(out_features, -1).float()
    global_scale = weight_quantizer.global_amax.float().item() / (6.0 * 448.0)

    # Compute Hessian
    hessian = torch.zeros(dim, dim, dtype=torch.float32)
    hessian, _ = update_hessian(inp, hessian, 0)
    hessian = hessian.to("cuda")
    h_inv = compute_hessian_inverse(hessian, weight, perc_damp=0.01)

    return weight, weight_quantizer, block_amax, global_scale, h_inv


def _run_unfused_gptq_nvfp4(weight, weight_quantizer, h_inv, gptq_block_size):
    """Unfused NVFP4 GPTQ using the production blockwise update with weight_quantizer."""
    w = weight.float().clone()
    gptq_blockwise_update(w, h_inv, gptq_block_size, weight_quantizer)
    return w


def _run_fused_gptq_nvfp4(
    weight, block_amax, global_scale, h_inv, gptq_block_size, quant_block_size
):
    """Fused Triton GPTQ for NVFP4 using the production fused update."""
    w = weight.float().clone()
    gptq_blockwise_update_fused_scalar(
        w, block_amax, global_scale, h_inv, gptq_block_size, quant_block_size
    )
    return w


_NVFP4_QUANT_BLOCK_SIZES = [16, 128]
_NVFP4_GPTQ_BLOCK_SIZES = [16, 128]


@requires_triton
@pytest.mark.parametrize("quant_block_size", _NVFP4_QUANT_BLOCK_SIZES)
@pytest.mark.parametrize("gptq_block_size", _NVFP4_GPTQ_BLOCK_SIZES)
def test_fused_vs_unfused_nvfp4(quant_block_size, gptq_block_size):
    """Fused Triton NVFP4 GPTQ must match unfused production reference."""
    torch.manual_seed(42)
    dim = max(256, quant_block_size * 4)
    out_features = 64

    weight, weight_quantizer, block_amax, global_scale, h_inv = _make_nvfp4_test_data(
        quant_block_size,
        out_features,
        dim,
    )

    weight_fused = _run_fused_gptq_nvfp4(
        weight,
        block_amax,
        global_scale,
        h_inv,
        gptq_block_size,
        quant_block_size,
    )
    weight_unfused = _run_unfused_gptq_nvfp4(
        weight,
        weight_quantizer,
        h_inv,
        gptq_block_size,
    )

    assert not torch.equal(weight_fused, weight.float()), "Fused did not update weights"
    assert not torch.equal(weight_unfused, weight.float()), "Unfused did not update weights"

    diff = (weight_fused - weight_unfused).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = weight_unfused.abs().max().item()
    rel_max = max_abs / denom if denom > 0 else 0.0

    print(
        f"\n[nvfp4] gptq_bs={gptq_block_size} quant_bs={quant_block_size}: "
        f"max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  rel_max={rel_max:.2e}"
    )

    torch.testing.assert_close(weight_fused, weight_unfused, atol=1e-4, rtol=1e-4)


@requires_triton
def test_fused_vs_unfused_nvfp4_mse_grid():
    """The fused kernel must also match RTN calls using exhaustive-MSE scales."""
    torch.manual_seed(42)
    quant_block_size = 16
    gptq_block_size = 128
    weight, weight_quantizer, block_amax, global_scale, h_inv = _make_nvfp4_test_data(
        quant_block_size,
        out_features=64,
        dim=256,
        algorithm={"method": "mse", "fp8_scale_sweep": True},
    )

    weight_fused = _run_fused_gptq_nvfp4(
        weight,
        block_amax,
        global_scale,
        h_inv,
        gptq_block_size,
        quant_block_size,
    )
    weight_unfused = _run_unfused_gptq_nvfp4(
        weight,
        weight_quantizer,
        h_inv,
        gptq_block_size,
    )

    torch.testing.assert_close(weight_fused, weight_unfused, atol=1e-4, rtol=1e-4)


@requires_triton
def test_fused_vs_unfused_nvfp4_correlated_production_width_exact():
    """Fused static GPTQ must be bit-exact over many blocks under correlated updates."""
    torch.manual_seed(42)
    dim = 2048
    weight, weight_quantizer, block_amax, global_scale, _ = _make_nvfp4_test_data(
        quant_block_size=16,
        out_features=8,
        dim=dim,
        algorithm={"method": "mse", "fp8_scale_sweep": True},
    )

    # A long-range upper-triangular recurrence makes sub-ULP QDQ differences cross
    # later FP4 boundaries. This is the failure mode missed by the old 256-wide,
    # two-block, tolerance-only test.
    offset = torch.arange(dim, device="cuda")
    distance = offset[None, :] - offset[:, None]
    h_inv = torch.where(
        distance > 0,
        0.015 * torch.pow(0.98, distance.float()),
        torch.zeros((), device="cuda"),
    )
    h_inv.diagonal().fill_(1.0)

    weight_fused = _run_fused_gptq_nvfp4(
        weight,
        block_amax.cpu(),  # Accelerate may offload this metadata during layerwise calibration.
        global_scale,
        h_inv,
        gptq_block_size=128,
        quant_block_size=16,
    )
    weight_unfused = _run_unfused_gptq_nvfp4(
        weight, weight_quantizer, h_inv, gptq_block_size=128
    )

    assert torch.equal(weight_fused, weight_unfused)

    weight_group_local = weight.float().clone()
    gptq_blockwise_update_static_nvfp4_groups(
        weight_group_local,
        block_amax.cpu(),
        global_scale,
        h_inv,
        block_size=128,
        quant_block_size=16,
    )
    assert torch.equal(weight_group_local, weight_unfused)


@requires_triton
def test_dynamic_nvfp4_scales_roundtrip_and_improve_rtn():
    """Dynamic GPTQ weights remain on their exported grid and beat RTN locally."""
    torch.manual_seed(42)
    weight, weight_quantizer, block_amax, global_scale, h_inv = _make_nvfp4_test_data(
        quant_block_size=16,
        out_features=64,
        dim=256,
        algorithm={"method": "mse", "fp8_scale_sweep": True},
    )
    inp = torch.randn(4, 32, 256, device="cuda")
    hessian = torch.zeros(256, 256, dtype=torch.float32)
    hessian, _ = update_hessian(inp, hessian, 0)
    hessian = hessian.cuda()
    h_inv = compute_hessian_inverse(hessian, weight, perc_damp=0.01)

    candidate = weight.float().clone()
    updated_amax, sequential_loss = gptq_blockwise_update_fused_scalar(
        candidate,
        block_amax,
        global_scale,
        h_inv,
        block_size=128,
        quant_block_size=16,
        dynamic=True,
        dynamic_scale_candidates=8,
        return_sequential_loss=True,
    )

    original_amax = weight_quantizer.amax.clone()
    rtn = weight_quantizer(weight.float()).float()
    weight_quantizer.amax = updated_amax.reshape_as(weight_quantizer.amax)
    exported_grid_candidate = weight_quantizer(candidate).float()

    def weighted_loss(value):
        delta = value - weight.float()
        return delta.mm(hessian).mul(delta).mean()

    torch.testing.assert_close(exported_grid_candidate, candidate, atol=1e-7, rtol=1e-7)
    assert weighted_loss(candidate) < weighted_loss(rtn)
    damped_hessian = hessian.clone()
    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    damped_hessian[diagonal, diagonal] += 0.01 * torch.diag(hessian).mean()
    delta = candidate - weight.float()
    exact_damped_objective = delta.mm(damped_hessian).mul(delta).mean()
    sequential_objective = 2.0 * sequential_loss / candidate.numel()
    assert sequential_objective == pytest.approx(exact_damped_objective.item(), rel=2e-3)
    weight_quantizer.amax = original_amax


@requires_triton
def test_unfused_dynamic_nvfp4_scales_roundtrip_and_improve_rtn():
    """Reference dynamic GPTQ uses exported grids and improves the local objective."""
    torch.manual_seed(42)
    weight, weight_quantizer, _, _, h_inv = _make_nvfp4_test_data(
        quant_block_size=16,
        out_features=16,
        dim=256,
        algorithm={"method": "mse", "fp8_scale_sweep": True},
    )
    inp = torch.randn(4, 32, 256, device="cuda")
    hessian = torch.zeros(256, 256, dtype=torch.float32)
    hessian, _ = update_hessian(inp, hessian, 0)
    hessian = hessian.cuda()
    h_inv = compute_hessian_inverse(hessian, weight, perc_damp=0.01)

    original_amax = weight_quantizer.amax.clone()
    candidate = weight.float().clone()
    updated_amax = gptq_blockwise_update_unfused_dynamic(
        candidate,
        h_inv,
        block_size=128,
        quantizer=weight_quantizer,
        dynamic_scale_candidates=8,
    )
    torch.testing.assert_close(weight_quantizer.amax, original_amax)

    rtn = weight_quantizer(weight.float()).float()
    weight_quantizer.amax = updated_amax.reshape_as(weight_quantizer.amax)
    exported_grid_candidate = weight_quantizer(candidate).float()

    def weighted_loss(value):
        delta = value - weight.float()
        return delta.mm(hessian).mul(delta).mean()

    torch.testing.assert_close(exported_grid_candidate, candidate, atol=1e-7, rtol=1e-7)
    assert weighted_loss(candidate) < weighted_loss(rtn)
    weight_quantizer.amax = original_amax


_NVFP4_BENCH_CONFIGS = [
    (16, 128, 256, 512),
    (16, 128, 256, 2048),
    (16, 128, 256, 4096),
    (128, 128, 256, 512),
    (128, 128, 256, 2048),
    (128, 128, 256, 4096),
]


def bench_fused_nvfp4():
    """Benchmark fused Triton NVFP4 GPTQ vs unfused production loop (informational-only).

    Not collected by pytest. Run directly: ``python tests/gpu/torch/quantization/test_gptq.py``
    """

    def _bench(fn, n_warmup=2, n_iters=5):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        total = 0.0
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            total += time.perf_counter() - t0
        return total / n_iters

    for quant_block_size, gptq_block_size, out_features, dim in _NVFP4_BENCH_CONFIGS:
        torch.manual_seed(42)
        weight, weight_quantizer, block_amax, global_scale, h_inv = _make_nvfp4_test_data(
            quant_block_size, out_features, dim
        )

        def run_fused():
            return _run_fused_gptq_nvfp4(
                weight, block_amax, global_scale, h_inv, gptq_block_size, quant_block_size
            )

        def run_unfused():
            return _run_unfused_gptq_nvfp4(weight, weight_quantizer, h_inv, gptq_block_size)

        t_fused = _bench(run_fused)
        t_unfused = _bench(run_unfused)
        speedup = t_unfused / t_fused if t_fused > 0 else float("inf")

        tag = f"qbs{quant_block_size}_gbs{gptq_block_size}_{out_features}x{dim}"
        print(
            f"[{tag}] Fused: {t_fused * 1e3:8.2f} ms | "
            f"Unfused: {t_unfused * 1e3:8.2f} ms | Speedup: {speedup:.1f}x"
        )


if __name__ == "__main__":
    bench_fused_nvfp4()
