"""Whole-model float boundaries preserve products and expose format loss."""

from types import SimpleNamespace

import numpy as np

from merlin.runtime.dispatch_runtime import boundary_scale, float_boundary_operands


def test_float_boundary_scaling_and_representability_remain_bound_to_selected_formats():
    binding = SimpleNamespace(
        operand_dtype="fp8_e4m3", accum_dtype="fp16", subnormal_operand_flush=True
    )
    a = np.full((2, 64), 100.0)
    b = np.full((64, 2), 100.0)
    qa, qb, product_scale, report = float_boundary_operands(a, b, binding)

    assert boundary_scale(np.zeros((1,)), binding.operand_dtype) == 1.0
    assert report["operand_dtype"] == binding.operand_dtype
    assert report["scale_a"] > 1.0 and report["scale_b"] > 1.0
    assert product_scale == report["scale_a"] * report["scale_b"]
    assert report["a"]["n_values"] == a.size and report["b"]["n_values"] == b.size
    assert report["a"]["n_flushed_to_zero"] == report["a"]["n_subnormal"]
    np.testing.assert_allclose(np.asarray(qa) @ np.asarray(qb) * product_scale, a @ b)
