"""Directed and randomized checks of the SmolVLA reference primitives.

Run with the selected interpreter; MODEL_ROOT (npu_model checkout),
ATLAS_REF_ROOT (atlas_ref package root), and MERLIN_CONSTANT_MATH_HELPER
select the pinned instruction model, MXU reference, and MPFR helper.
"""
from fractions import Fraction
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROTARY = HERE.parents[1] / "rotary-host" / "reference"
sys.path[:0] = [os.environ["MODEL_ROOT"], str(ROTARY), str(HERE)]
import layer_reference  # noqa: E402  rotary-host qualified exact BF16 add
sys.path.insert(0, os.environ["ATLAS_REF_ROOT"])
import smolvla_primitives as sp  # noqa: E402

RNG = np.random.default_rng(20261111)


def test_round_fraction_matches_ieee_binary32_and_binary64():
    values = np.concatenate([RNG.standard_normal(4000) * 10.0 ** RNG.integers(-45, 38, 4000),
                             np.ldexp(RNG.random(500), RNG.integers(-160, -120, 500)), [0.0, -0.0, 1.0, -2.5]])
    for value in values:
        assert sp.round_fraction(Fraction(float(value)), "f32", negative_zero=math.copysign(1, value) < 0) == \
            int(np.float32(value).view(np.uint32))
        assert sp.round_fraction(Fraction(float(value)), "f64", negative_zero=math.copysign(1, value) < 0) == \
            int(np.float64(value).view(np.uint64))
    assert sp.scalar_bits(-3.4028234663852886e38, "f32") == 0xff7fffff
    assert sp.scalar_bits(-3.3895313892515355e38, "bf16") == 0xff7f
    with pytest.raises(ValueError, match="overflows"):
        sp.round_fraction(Fraction(2) ** 128, "f32")


def test_exact_binary_matches_hardware_ieee():
    a = (RNG.standard_normal(3000) * 10.0 ** RNG.integers(-18, 18, 3000)).astype(np.float32)
    b = (RNG.standard_normal(3000) * 10.0 ** RNG.integers(-18, 18, 3000)).astype(np.float32)
    for op, ufunc in (("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide)):
        got = sp.exact_elementwise(op, "f32", a.view(np.uint32), b.view(np.uint32))
        assert np.array_equal(got, ufunc(a, b).view(np.uint32)), op
    c, d = a.astype(np.float64) * 1e100, b.astype(np.float64) * np.pi
    got = sp.exact_elementwise("mul", "f64", c.view(np.uint64), d.view(np.uint64))
    assert np.array_equal(got, (c * d).view(np.uint64))


def test_conversions_match_ieee():
    ints = RNG.integers(-(1 << 40), 1 << 40, 2000)
    assert np.array_equal(sp.convert_exact(ints, "i64", "f32"), ints.astype(np.float32).view(np.uint32))
    doubles = RNG.standard_normal(2000) * 10.0 ** RNG.integers(-40, 37, 2000)
    assert np.array_equal(sp.convert_exact(doubles.view(np.uint64), "f64", "f32"), doubles.astype(np.float32).view(np.uint32))
    singles = doubles.astype(np.float32)
    assert np.array_equal(sp.convert_exact(singles.view(np.uint32), "f32", "f64"), singles.astype(np.float64).view(np.uint64))
    with pytest.raises(ValueError, match="NaN"):
        sp.f32_to_bf16_rne(np.array([0x7fc00001], np.uint32))


def _structured_bf16(count):
    exponents = RNG.integers(0, 256, count)
    exponents[RNG.random(count) < 0.1] = 0          # subnormals and zeros
    exponents[RNG.random(count) < 0.02] = 255      # infinities (mantissa cleared below)
    mantissa = RNG.integers(0, 128, count)
    mantissa[exponents == 255] = 0
    sign = RNG.integers(0, 2, count)
    return ((sign << 15) | (exponents << 7) | mantissa).astype(np.uint16)


def test_exact_bf16_add_matches_qualified_integer_add():
    a, b = _structured_bf16(60000), _structured_bf16(60000)
    near = (a & 0xff80) | (RNG.integers(0, 128, a.size)).astype(np.uint16)   # close magnitudes, cancellation
    swap = (RNG.random(a.size) < 0.3) & ((a & 0x7f80) != 0x7f80)
    b = np.where(swap, near ^ np.uint16(0x8000), b).astype(np.uint16)
    both_inf = ((a & 0x7fff) == 0x7f80) & ((b & 0x7fff) == 0x7f80)
    a, b = a[~both_inf], b[~both_inf]
    expected = layer_reference.host_bf16_add(a, b)
    assert np.array_equal(sp.exact_bf16_add(a, b), expected)
    assert np.array_equal(sp.exact_bf16_add(a, b, subtract=True), layer_reference.host_bf16_add(a, b, subtract=True))
    zeros = np.array([0x0000, 0x8000, 0x0000, 0x8000, 0x3f80], np.uint16)
    others = np.array([0x8000, 0x8000, 0x0000, 0x0000, 0xbf80], np.uint16)
    assert np.array_equal(sp.exact_bf16_add(zeros, others), [0x0000, 0x8000, 0x0000, 0x0000, 0x0000])
    assert sp.exact_bf16_add(np.uint16(0x7f7f), np.uint16(0x7b00)) == 0x7f80  # RNE overflow to infinity
    with pytest.raises(ValueError, match="cancellation"):
        sp.exact_bf16_add(np.uint16(0x7f80), np.uint16(0xff80))
    with pytest.raises(ValueError, match="NaN"):
        sp.exact_bf16_add(np.uint16(0x7fc0), np.uint16(0))


def test_f64_to_bf16_matches_exact_rounding():
    values = np.concatenate([RNG.standard_normal(5000) * 10.0 ** RNG.integers(-45, 39, 5000),
                             np.ldexp(RNG.integers(1, 1 << 20, 2000).astype(np.float64), RNG.integers(-160, -125, 2000))])
    got = sp._f64_to_bf16_rne(values)
    for value, word in zip(values, got):
        expected = sp.round_fraction(Fraction(float(value)), "bf16", negative_zero=value < 0, allow_overflow=True)
        assert int(word) == expected


def test_restore_exponents_equals_power_of_two_readout_in_domain():
    from atlas_ref.arithmetic import readout_bf16
    words = _structured_bf16(20000)
    words = words[((words & 0x7f80) != 0) & ((words & 0x7f80) != 0x7f80)]
    shifts = RNG.integers(-20, 21, words.size)
    exponent = (words >> 7) & 255
    ok = (exponent + shifts > 0) & (exponent + shifts < 255)
    words, shifts = words[ok], shifts[ok]
    assert np.array_equal(sp.restore_exponents(words, shifts), readout_bf16(words, shifts))
    assert sp.restore_exponents(np.uint16(0x8000), np.int64(-50)) == 0x8000
    with pytest.raises(ValueError, match="finite normal"):
        sp.restore_exponents(np.uint16(0x0080), np.int64(-1))
    with pytest.raises(ValueError, match=r"\[-133,127\]"):
        sp.restore_exponents(np.uint16(0x3f80), np.int64(128))


def test_cap448_code_domain():
    sp.valid_cap448(np.array([0, 8, 126, 136, 254], np.uint8), "test")
    for code in (1, 7, 127, 128, 255):
        with pytest.raises(ValueError, match="cap448"):
            sp.valid_cap448(np.array([code], np.uint8), "test")


def test_contraction_numpy_small():
    from atlas_ref.contractions import contraction_nt
    a = RNG.choice(np.r_[0, 8:127, 136:255].astype(np.uint8), (2, 5, 40))
    b = RNG.choice(np.r_[0, 8:127, 136:255].astype(np.uint8), (2, 3, 40))
    ae, be = np.array([[[[-3]], [[2]]]], np.int32)[0], np.array(-5, np.int32)
    accumulator, restored = sp.contraction(a, ae, b, be)
    reference = contraction_nt(a, ae, b, be, unit="mxu1")
    assert np.array_equal(accumulator, reference.accumulator_bf16)
    assert np.array_equal(restored, reference.readout_bf16)


def test_gelu_constants_and_domain():
    literals, words = sp.gelu_constants()
    assert literals == [0x3f000000, 0x3f800000, 0x3d372713, 0x3f4c422a]
    assert words == [0x3f00, 0x3f80, 0x3d37, 0x3f4c]
    x = np.linspace(-6, 6, 241, dtype=np.float32)
    got = sp.tanh_gelu(x.view(np.uint32)).view(np.float32)
    exact = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    assert np.max(np.abs(got - exact)) < 0.05  # sanity only; exactness is the selected composition
    assert sp.tanh_gelu(np.array([0, 0x80000000], np.uint32)).tolist() == [0, 0x80000000]
    with pytest.raises(ValueError, match="tanh GELU input leaves"):
        sp.tanh_gelu(np.array([np.float32(2.0 ** 33)]).view(np.uint32))
    with pytest.raises(ValueError, match="tanh GELU input leaves"):
        sp.tanh_gelu(np.array([np.float32(2.0 ** -40)]).view(np.uint32))


def test_silu_composition_and_domain():
    x = np.linspace(-31, 31, 125, dtype=np.float32)
    got = sp.silu(x.view(np.uint32)).view(np.float32)
    assert np.max(np.abs(got - x / (1 + np.exp(-x.astype(np.float64)))) / np.maximum(1, np.abs(x))) < 0.02
    with pytest.raises(ValueError, match="SiLU sigmoid input leaves"):
        sp.silu(np.array([np.float32(33.0)]).view(np.uint32))


def test_row_softmax_refuses_float32_minimum_mask():
    scores = np.zeros((2, 40), np.float32)
    scores[1, 3:] = np.float32(-3.4028234663852886e38)
    probabilities, _ = sp.row_softmax_rowspan80(scores[:1].view(np.uint32))
    assert np.allclose(probabilities.view(np.float32).sum(), 1, atol=0.05)
    with pytest.raises(ValueError, match="row softmax score input leaves its selected BF16 code domain: 37 of 80"):
        sp.row_softmax_rowspan80(scores.view(np.uint32))


def test_mpfr_power_and_binary64_trig():
    path = os.environ.get("MERLIN_CONSTANT_MATH_HELPER")
    if not path:
        pytest.skip("MPFR helper not selected")
    sp.math_helper(Path(path))
    exponents = (np.arange(32, dtype=np.float32) * np.float32(0.03125)).view(np.uint32)
    base = np.full(32, np.float32(10000).view(np.uint32))
    got = sp.constant_power(32, base, exponents).view(np.float32)
    assert np.max(np.abs(got / np.power(np.float64(10000), np.arange(32) / 32) - 1)) < 2 ** -23
    angles = np.array([0.0, 1e-3, 1.0, 3.0, 6.283185307179586, 123.456])
    for operation, function in (("sin", math.sin), ("cos", math.cos)):
        got = sp.trig_f64(angles.view(np.uint64), operation).view(np.float64)
        assert np.all(np.abs(got - [function(v) for v in angles]) <= np.spacing(np.abs(got)))
