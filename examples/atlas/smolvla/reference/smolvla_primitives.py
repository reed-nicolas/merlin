"""SmolVLA-specific independent primitives for the flow-loss reference.

Every function implements one selected compiled arithmetic policy from its
definition. None reads or calls a compiler evaluator, emitted program, or
semantic JSON. BF16 VPU steps use the pinned instruction model's RtlNumerics
(as the rotary-host layer and attention references do); exact IEEE steps use
integer or rational arithmetic, or binary64 sums proven exact below.
Raw words: bf16 uint16, f32 uint32, f64 uint64.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

FORMATS = {"bf16": (16, 7, 127), "f32": (32, 23, 127), "f64": (64, 52, 1023)}
RAW = {"bf16": np.uint16, "f32": np.uint32, "f64": np.uint64}

GELU_POLICY = "f32-to-bf16-bounded-tanh-gelu-cube-chop-v1"
SIGMOID_POLICY = "f32-to-bf16-bounded-sigmoid-exp-recip-v1"
ROW_SOFTMAX_POLICY = "bf16-tiled-row-softmax-ltr-fp32-rne-chop-rcp-lut-v1"
ROWSPAN80_POLICY = "bf16-minfinite-zero-rowspan80-nonempty-v1"
ALREADY_QUANTIZED_POLICY = "atlas-already-quantized-e4m3-signed-exponents-mxu1-model47-v1"
CONSTANT_POWER_POLICY = "positive-power-correct-rne-mpfr-v1"

# Admitted BF16 code ranges (inclusive) of each selected input encoding.
GELU_DOMAIN = ((0, 0), (0x2F80, 0x4F80), (0x8000, 0x8000), (0xAF80, 0xCF80))
SIGMOID_DOMAIN = ((0, 0), (0x2000, 0x4200), (0x8000, 0x8000), (0xA000, 0xC200))
ROWSPAN80_SCORE_DOMAIN = ((0, 0), (0x0400, 0x4300), (0x8000, 0x8000), (0x8400, 0xC300))


# ---------------------------------------------------------------- exact rounding

def bits_to_fraction(bits, fmt):
    """Exact value of one finite raw word."""
    width, fraction, bias = FORMATS[fmt]
    bits = int(bits)
    sign = bits >> (width - 1)
    exponent = (bits >> fraction) & ((1 << (width - fraction - 1)) - 1)
    mantissa = bits & ((1 << fraction) - 1)
    if exponent == (1 << (width - fraction - 1)) - 1:
        raise ValueError(f"nonfinite {fmt} constant operand is unqualified")
    value = Fraction(mantissa if not exponent else mantissa | (1 << fraction)) * Fraction(2) ** ((exponent or 1) - bias - fraction)
    return -value if sign else value


def round_fraction(value, fmt, *, negative_zero=False, allow_overflow=False):
    """Round an exact rational to fmt with ties-to-even and gradual underflow."""
    width, fraction, bias = FORMATS[fmt]
    value = Fraction(value)
    sign = 1 if value < 0 or (value == 0 and negative_zero) else 0
    magnitude = abs(value)
    if magnitude == 0:
        return sign << (width - 1)
    n, d = magnitude.numerator, magnitude.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n < d << exponent) if exponent >= 0 else (n << -exponent < d):
        exponent -= 1
    quantum = max(exponent - fraction, 1 - bias - fraction)
    scaled_n, scaled_d = (n, d << quantum) if quantum >= 0 else (n << -quantum, d)
    q, r = divmod(scaled_n, scaled_d)
    if 2 * r > scaled_d or (2 * r == scaled_d and q & 1):
        q += 1
    if q >= 1 << (fraction + 1):
        q >>= 1
        quantum += 1
    biased = quantum + fraction + bias if q >> fraction else 0
    if biased >= (1 << (width - fraction - 1)) - 1:
        if not allow_overflow:
            raise ValueError(f"{fmt} rounding overflows")
        return (sign << (width - 1)) | (((1 << (width - fraction - 1)) - 1) << fraction)
    return (sign << (width - 1)) | (biased << fraction) | (q & ((1 << fraction) - 1))


def scalar_bits(value, fmt):
    """Exact Python number -> raw fmt word, ties-to-even."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("scalar operand is not a declared number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite scalar operand is unqualified")
    negative_zero = isinstance(value, float) and value == 0 and math.copysign(1.0, value) < 0
    return round_fraction(Fraction(value), fmt, negative_zero=negative_zero)


def exact_binary(left, right, fmt, operation):
    """IEEE RNE add/sub/mul/div of raw words, through exact rationals."""
    a, b = bits_to_fraction(left, fmt), bits_to_fraction(right, fmt)
    width = FORMATS[fmt][0]
    sa, sb = int(left) >> (width - 1), int(right) >> (width - 1)
    if operation == "sub":
        b, sb, operation = -b, sb ^ 1, "add"
    if operation == "add":
        total = a + b
        return round_fraction(total, fmt, negative_zero=(total == 0 and sa == sb == 1))
    if operation == "mul":
        return round_fraction(a * b, fmt, negative_zero=(a * b == 0 and sa != sb))
    if operation == "div":
        if b == 0:
            raise ValueError("constant division by zero is unqualified")
        return round_fraction(a / b, fmt, negative_zero=(a == 0 and sa != sb))
    raise ValueError("unknown constant arithmetic")


def exact_elementwise(operation, fmt, *operands):
    """Constant folding: exact IEEE RNE elementwise on raw words with broadcasting."""
    arrays = np.broadcast_arrays(*(np.asarray(x, dtype=RAW[fmt]) for x in operands))
    out = np.empty(arrays[0].shape, dtype=RAW[fmt])
    for index in np.ndindex(out.shape):
        out[index] = exact_binary(*(int(a[index]) for a in arrays), fmt, operation)
    return out


def convert_exact(raw, source, target):
    """Float/int conversions: exact widening, RNE narrowing, RNE int->float."""
    raw = np.asarray(raw)
    out = np.empty(raw.shape, dtype=RAW[target])
    for index in np.ndindex(raw.shape):
        word = int(raw[index])
        if source in ("i64", "i32", "bool", "u8"):
            out[index] = round_fraction(Fraction(word), target)
        else:
            width = FORMATS[source][0]
            out[index] = round_fraction(bits_to_fraction(word, source), target,
                                        negative_zero=bool(word >> (width - 1)))
    return out


def f32_to_bf16_rne(raw):
    """f32 -> bf16 RNE (exact integer formula), refusing NaN payloads."""
    bits = np.asarray(raw, dtype=np.uint32)
    if np.any((bits & 0x7fffffff) > 0x7f800000):
        raise ValueError("NaN in f32 -> bf16 conversion is unqualified")
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def widen(bits):
    return np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)


# ---------------------------------------------------------------- exact BF16 add

def _bf16_to_f64(bits):
    return (widen(bits)).view(np.float32).astype(np.float64)


def _f64_to_bf16_rne(values):
    """Round binary64 values to BF16 RNE, with subnormals and overflow to infinity.

    Each value is scaled by an exact power of two to its 8-bit BF16 quantum,
    rounded half-to-even by rint, and scaled back; every step is exact in
    binary64 except the single intended rounding.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    _, exponent = np.frexp(safe)
    quantum = np.maximum(exponent - 8, -133)
    rounded = np.ldexp(np.rint(np.ldexp(safe, -quantum)), quantum)
    rounded = np.where(finite, rounded, values)
    with np.errstate(over="ignore"):
        single = rounded.astype(np.float32)
    return (single.view(np.uint32) >> 16).astype(np.uint16)


def exact_bf16_add(left, right, *, subtract=False):
    """IEEE BF16 add/sub with one RNE rounding (ieee754-bf16-add-rne-v1).

    The binary64 sum of two BF16 values is exact when their exponents differ
    by at most 44. Otherwise the smaller magnitude is below 2^-36 BF16 ulps of
    the larger, so the binary64 sum and the exact sum round to the same BF16
    word (the larger operand). NaN operands and opposite-infinity sums refuse.
    Signed zeros follow binary64 RNE, which equals BF16 RNE.
    """
    a, b = np.broadcast_arrays(np.asarray(left, np.uint16), np.asarray(right, np.uint16))
    if subtract:
        b = b ^ np.uint16(0x8000)
    if np.any((a & 0x7fff) > 0x7f80) or np.any((b & 0x7fff) > 0x7f80):
        raise ValueError("BF16 add/subtract NaN operand is unqualified")
    if np.any(((a & 0x7fff) == 0x7f80) & ((b & 0x7fff) == 0x7f80) & ((a ^ b) >> 15 == 1)):
        raise ValueError("BF16 infinity cancellation is unqualified")
    with np.errstate(invalid="raise"):
        total = _bf16_to_f64(a) + _bf16_to_f64(b)
    return _f64_to_bf16_rne(total)


# ---------------------------------------------------------------- VPU compositions

def _rtl():
    from npu_model.configs.numerics import RtlNumerics
    return RtlNumerics


def _tb(bits):
    import torch
    return torch.from_numpy(np.ascontiguousarray(np.asarray(bits, np.uint16)).copy()).view(torch.bfloat16)


def _nb(value):
    import torch
    return value.contiguous().view(torch.uint16).numpy().copy()


def check_domain(bits, ranges, label):
    bits = np.asarray(bits, np.uint16)
    good = np.zeros(bits.shape, bool)
    for low, high in ranges:
        good |= (bits >= low) & (bits <= high)
    if not np.all(good):
        bad = np.flatnonzero(~good.reshape(-1))
        raise ValueError(f"{label} input leaves its selected BF16 code domain: {bad.size} of {bits.size} elements, "
                         f"first flat index {int(bad[0])} word 0x{int(bits.reshape(-1)[bad[0]]):04x}")
    return bits


def _finite_result(bits, label):
    if np.any((np.asarray(bits) & 0x7f80) == 0x7f80):
        raise ValueError(f"{label} produced a nonfinite BF16 result")
    return bits


def gelu_constants():
    """f32 literals of torch tanh-GELU, then the policy's BF16 RNE narrowing."""
    literals = [scalar_bits(x, "f32") for x in (0.5, 1.0, 0.044715, math.sqrt(2.0 / math.pi))]
    return literals, [int(f32_to_bf16_rne(np.uint32(x))) for x in literals]


def tanh_gelu(x_f32):
    """f32-to-bf16-bounded-tanh-gelu-cube-chop-v1 on f32 words; returns f32 words."""
    rtl = _rtl()
    bits = check_domain(f32_to_bf16_rne(x_f32), GELU_DOMAIN, "tanh GELU")
    _, (half, one, cubic, root) = gelu_constants()
    x = _tb(bits)
    c = lambda word: _tb(np.full(bits.shape, word, np.uint16))
    cube = rtl.unary("cube", x)
    scaled = rtl.mul(cube, c(cubic))
    inner = rtl.mul(rtl.add(x, scaled), c(root))
    outer = rtl.add(c(one), rtl.unary("tanh", inner))
    out = _finite_result(_nb(rtl.mul(rtl.mul(x, c(half)), outer)), "tanh GELU")
    return widen(out)


def silu(x_f32):
    """SiLU: f32-to-bf16-bounded-sigmoid-exp-recip-v1, then the BF16 product."""
    rtl = _rtl()
    bits = check_domain(f32_to_bf16_rne(x_f32), SIGMOID_DOMAIN, "SiLU sigmoid")
    x = _tb(bits)
    c = lambda word: _tb(np.full(bits.shape, word, np.uint16))
    exponential = rtl.unary("exp", rtl.sub(c(0), x))
    sigmoid = rtl.unary("rcp", rtl.add(c(0x3f80), exponential))
    out = _finite_result(_nb(rtl.mul(x, sigmoid)), "SiLU")
    return widen(out)


def row_softmax_rowspan80(scores_f32):
    """Unmasked whole-row softmax under the selected row-relative span80 domain.

    The f32 scores are RNE-encoded to BF16 and must lie in the admitted codes;
    the row-relative proof domain is then checked and the qualified pinned
    model composition (adjacent 32-lane trees, one divisor) is applied.
    """
    import rowspan_softmax

    bits = f32_to_bf16_rne(scores_f32)
    check_domain(bits, ROWSPAN80_SCORE_DOMAIN, "row softmax score")
    result = rowspan_softmax.normalized_softmax(bits, np.ones(bits.shape, np.bool_))
    return widen(result["probabilities_bf16"]), result


# ---------------------------------------------------------------- already-quantized contraction

def valid_cap448(codes, label):
    codes = np.asarray(codes)
    if codes.dtype != np.uint8:
        raise ValueError(f"{label} codes must be raw uint8")
    good = (codes == 0) | ((codes >= 8) & (codes <= 126)) | ((codes >= 136) & (codes <= 254))
    if not np.all(good):
        raise ValueError(f"{label} requires finite normal cap448 E4M3 codes or positive zero")
    return codes


def restore_exponents(words, shifts):
    """Add the signed operand-exponent sum to the BF16 exponent field exactly.

    Zero words stay unchanged. A nonzero word must be normal and its scaled
    exponent must stay in [1, 254]; otherwise the compiled readout refuses.
    """
    words = np.asarray(words, np.uint16)
    shifts = np.broadcast_to(np.asarray(shifts, np.int64), words.shape)
    if np.any((shifts < -133) | (shifts > 127)):
        raise ValueError("already-quantized readout exponent sum leaves [-133,127]")
    nonzero = (words & 0x7fff) != 0
    exponent = ((words >> 7) & 255).astype(np.int64)
    scaled = exponent + shifts
    if np.any(nonzero & ((exponent == 0) | (exponent == 255) | (scaled <= 0) | (scaled >= 255))):
        raise ValueError("already-quantized scaled readout leaves the finite normal BF16 domain")
    return np.where(nonzero, (words & 0x807f) | (np.clip(scaled, 0, 255) << 7), words).astype(np.uint16)


def contraction(a_codes, a_exp, b_codes, b_exp, *, engine="numpy", threads=1, label="contraction"):
    """[...,M,K] x [...,N,K] on MXU1 (ascending K32), then exact exponent restoration."""
    from atlas_ref.contractions import contraction_nt

    valid_cap448(a_codes, label)
    valid_cap448(b_codes, label)
    for value in (a_exp, b_exp):
        value = np.asarray(value)
        if value.dtype.kind != "i" or np.any((value < -127) | (value > 127)):
            raise ValueError(f"{label} signed exponents leave [-127,127]")
    a_codes, b_codes = np.asarray(a_codes), np.asarray(b_codes)
    ax = np.broadcast_to(np.asarray(a_exp), a_codes.shape[:-1] + (1,)).astype(np.int64)
    bx = np.broadcast_to(np.asarray(b_exp), b_codes.shape[:-1] + (1,)).astype(np.int64)
    result = contraction_nt(a_codes, a_exp, b_codes, b_exp, unit="mxu1", engine=engine, threads=threads)
    restored = restore_exponents(result.accumulator_bf16, ax + np.swapaxes(bx, -1, -2))
    return result.accumulator_bf16, restored


# ---------------------------------------------------------------- binary64 host math

_HELPER = None


def math_helper(path=None):
    global _HELPER
    if _HELPER is None:
        from merlin.semantic_compiler.constant_math_backend import MathHelper
        if path is None:
            raise ValueError("binary64 trig and constant power need the selected MPFR helper")
        _HELPER = MathHelper(path)
    return _HELPER


def trig_f64(raw, operation):
    """Correctly rounded binary64 sin/cos (selected-libm-rne-finite-v1) via MPFR bounds."""
    helper = math_helper()
    raw = np.asarray(raw, np.uint64)
    if np.any((raw & np.uint64(0x7ff0000000000000)) == np.uint64(0x7ff0000000000000)):
        raise ValueError("binary64 trig input must be finite")
    out = np.empty(raw.shape, np.uint64)
    for index in np.ndindex(raw.shape):
        out[index] = helper.unary_reference(64, int(raw[index]), operation)[0]
    return out


def constant_power(width, base, exponent, maximum=1024):
    """positive-power-correct-rne-mpfr-v1 for constant operands (raw words)."""
    helper = math_helper()
    base, exponent = np.broadcast_arrays(np.asarray(base), np.asarray(exponent))
    out = np.empty(base.shape, RAW["f32" if width == 32 else "f64"])
    for index in np.ndindex(base.shape):
        out[index] = helper.power(width, int(base[index]), int(exponent[index]), maximum)[0]
    return out


def row_softmax_unchecked(scores_f32, keep):
    """DIAGNOSTIC ONLY: the same tiled composition with its proof-domain checks waived.

    Masked lanes (keep False) and the N tail use the -32 pad and a +0 multiplier
    after exponentiation, exactly like the selected masked row-softmax tiling.
    Kept scores must still encode as finite BF16. This is not a selected
    compiled policy; results computed through it are never expected words.
    """
    rtl = _rtl()
    keep = np.asarray(keep, np.bool_)
    bits = f32_to_bf16_rne(np.where(keep, np.asarray(scores_f32, np.uint32), 0).astype(np.uint32))
    if np.any(keep & ((bits & 0x7f80) == 0x7f80)) or not np.all(np.any(keep, axis=-1)):
        raise ValueError("diagnostic softmax needs finite kept scores and nonempty rows")
    tiles, masks, maxima = [], [], []
    for start in range(0, bits.shape[-1], 32):
        count = min(32, bits.shape[-1] - start)
        tile = np.full((*bits.shape[:-1], 32), 0xc200, np.uint16)
        valid = np.zeros(tile.shape, np.bool_)
        tile[..., :count] = np.where(keep[..., start:start + count], bits[..., start:start + count], 0xc200)
        valid[..., :count] = keep[..., start:start + count]
        tiles.append(_tb(tile))
        masks.append(_tb(np.where(valid, 0x3f80, 0).astype(np.uint16)))
        maxima.append(rtl.amax(tiles[-1], -1))
    maximum = maxima[0]
    for partial in maxima[1:]:
        maximum = rtl.maximum(maximum, partial)
    exponentials, sums = [], []
    for tile, mask in zip(tiles, masks, strict=True):
        exponentials.append(rtl.mul(rtl.unary("exp", rtl.sub(tile, maximum)), mask))
        sums.append(rtl.row_sum(exponentials[-1]))
    denominator = sums[0]
    for partial in sums[1:]:
        denominator = rtl.add(denominator, partial)
    reciprocal = rtl.unary("rcp", denominator)
    probabilities = np.concatenate([_nb(rtl.mul(e, reciprocal)) for e in exponentials], axis=-1)[..., :bits.shape[-1]]
    return widen(probabilities), {"row_max_bf16": _nb(maximum), "denominator_bf16": _nb(denominator)}
