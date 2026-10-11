"""Opt-in FP32 math translations retaining fused BF16 operator boundaries.

Original operands and results remain BF16. FP32 internal arithmetic avoids adding
BF16 rounding points inside a mathematical fused operation. Native linear can
choose an unfused layout-dependent kernel; native attention can use blocked BF16
exponential storage. These are FP32-math candidates, not reproductions of every
native CPU kernel. Both operations remain strictly source-reference gated.
"""
import math
from typing import Optional


def bf16_linear_error_diagnostic(input, weight, bias, torch):
    """A conditional worst-case CPU arithmetic diagnostic, never an active gate.

    Covers BF16 operands, FP32 multiply/add/reduction, optional FP32 bias add,
    and ONE final BF16 rounding. It does not infer that a native source kernel
    follows this operational contract. FP64 evaluation is a labeled diagnostic
    and its own roundoff is included; it is not an exact real-valued oracle.
    """
    operands = (input, weight) + (() if bias is None else (bias,))
    if any(value.device.type != "cpu" or value.dtype != torch.bfloat16
           or not torch.isfinite(value).all() for value in operands):
        raise ValueError("Linear diagnostic requires finite original BF16 CPU operands")
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1] or \
            bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("Linear diagnostic shape mismatch")
    width = weight.shape[1]
    if width <= 0:
        raise ValueError("Linear diagnostic requires a nonempty reduction")
    operations = 2 * width + (1 if bias is not None else 0)

    def gamma(dtype):
        unit = torch.finfo(dtype).eps / 2
        if operations * unit >= 1:
            raise ValueError("Reduction length exceeds this floating-point error model")
        return operations * unit / (1 - operations * unit)

    with torch.no_grad():
        left, right = input.double(), weight.double().T
        ideal = left @ right
        magnitude = left.abs() @ right.abs()
        if bias is not None:
            ideal = ideal + bias.double()
            magnitude = magnitude + bias.double().abs()
        g32, g64 = gamma(torch.float32), gamma(torch.float64)
        # An upper bound on sum(abs(products))+abs(bias), accounting for FP64
        # roundoff in the diagnostic's magnitude computation.
        magnitude_upper = magnitude / (1 - g64)
        if not torch.isfinite(magnitude_upper).all() or \
                torch.any(magnitude_upper * (1 + g32) >= torch.finfo(torch.float32).max):
            raise ValueError("Diagnostic cannot exclude FP32 intermediate overflow")
        minimum_subnormal32 = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
        underflow = operations * minimum_subnormal32 / (2 * (1 - operations * torch.finfo(torch.float32).eps / 2))
        arithmetic = (g32 + g64) * magnitude_upper + underflow
        # Relative rounding plus a subnormal absolute term. This avoids an
        # arbitrary near-zero floor or a fixed number of BF16 representable steps.
        bf16 = torch.finfo(torch.bfloat16)
        rounding = (bf16.eps / 2) * (ideal.abs() + arithmetic) + bf16.tiny * bf16.eps / 2
        bound = arithmetic + rounding
        rounded = ideal.bfloat16()
        if not torch.isfinite(rounded).all():
            raise ValueError("Diagnostic reference overflows BF16 at the declared output cast")
    return {
        "higher_precision_value": ideal, "bf16_rounded_diagnostic": rounded,
        "absolute_error_bound_to_fp64_diagnostic": bound,
        "fp32_arithmetic_error_bound": arithmetic, "sum_absolute_products_and_bias_upper": magnitude_upper,
        "diagnostic_only": True, "acceptance_contract_selected": False,
        "assumptions": "BF16 exact operands; FP32 products/reduction/bias; one BF16 output cast; IEEE rounding; no overflow; gradual underflow",
        "roundoff_model": "gamma_(2K+bias) sum(abs(products))+abs(bias), plus FP64 diagnostic roundoff and subnormal terms",
        "reduction_width": width, "operation_bound": operations,
        "placement": "CPU; regular Gemmini INT8/INT32 is outside this diagnostic",
    }


def translations(torch):
    # Optional exporter dependencies are loaded only when this recipe is selected.
    from onnxscript import ir, opset18 as op
    from onnxscript.function_libs.torch_lib.ops import nn
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.onnx_types import TensorType

    def linear(input: TFloat, weight: TFloat, bias: Optional[TFloat] = None) -> TFloat:
        if input.dtype != ir.DataType.BFLOAT16:
            return nn.aten_linear(input, weight, bias)
        if weight.dtype != input.dtype or bias is not None and bias.dtype != input.dtype:
            raise ValueError("Fused BF16 linear requires matching operand dtypes")
        if len(weight.shape) != 2 or len(input.shape) < 1:
            raise ValueError("Fused BF16 linear requires a matrix weight and a non-scalar input")
        product = op.MatMul(op.Cast(input, to=1), op.Transpose(op.Cast(weight, to=1), perm=[1, 0]))
        if bias is not None:
            product = op.Add(product, op.Cast(bias, to=1))
        return op.Cast(product, to=16)

    def attention(query: TFloat, key: TFloat, value: TFloat, attn_mask: Optional[TensorType] = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False) -> TFloat:
        if query.dtype != ir.DataType.BFLOAT16:
            return nn.aten_scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("Fused BF16 attention requires matching Q/K/V dtypes")
        if any(len(tensor.shape) != 4 for tensor in (query, key, value)):
            raise ValueError("Fused BF16 attention requires rank-four Q/K/V")
        if dropout_p != 0.0 or enable_gqa:
            raise ValueError("Fused BF16 attention supports inference without dropout or grouped-query expansion")
        if is_causal and attn_mask is not None:
            raise ValueError("Fused BF16 attention cannot combine a causal flag and an explicit mask")
        width = query.shape[-1]
        if scale is None and (not isinstance(width, int) or width <= 0):
            raise ValueError("Fused BF16 attention requires a static positive head width for its default scale")
        factor = 1.0 / math.sqrt(width) if scale is None else scale
        logits = op.Mul(op.MatMul(op.Cast(query, to=1), op.Transpose(op.Cast(key, to=1), perm=[0, 1, 3, 2])), factor)
        if is_causal:
            rows = op.Gather(op.Shape(query), 2, axis=0)
            columns = op.Gather(op.Shape(key), 2, axis=0)
            row = op.Unsqueeze(op.Range(0, rows, 1), [1])
            column = op.Unsqueeze(op.Range(0, columns, 1), [0])
            logits = op.Where(op.LessOrEqual(column, row), logits, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == ir.DataType.BOOL:
                logits = op.Where(attn_mask, logits, float("-inf"))
            elif attn_mask.dtype in (ir.DataType.BFLOAT16, ir.DataType.FLOAT):
                logits = op.Add(logits, op.Cast(attn_mask, to=1))
            else:
                raise ValueError("Fused BF16 attention requires a boolean, BF16, or FP32 mask")
        if is_causal or attn_mask is not None:
            # Native SDPA returns zero for fully masked rows rather than NaN.
            valid = op.Cast(op.Not(op.Equal(logits, float("-inf"))), to=7)
            blocked = op.Equal(op.ReduceSum(valid, [-1], keepdims=1), 0)
            logits = op.Where(blocked, 0.0, logits)
        probabilities = op.Softmax(logits, axis=-1)
        if is_causal or attn_mask is not None:
            probabilities = op.Where(blocked, 0.0, probabilities)
        return op.Cast(op.MatMul(probabilities, op.Cast(value, to=1)), to=16)

    return {torch.ops.aten.linear.default: linear, torch.ops.aten.scaled_dot_product_attention.default: attention}
