"""Source-bound scalar/reduction exports and opt-in fused BF16 math.

The independent scalar hook preserves original Python values in typed DOUBLE
constants; other ordinary scalar dtypes use their built-in translations.
The Smol mean hook preserves a separately pinned native reduction order.

Original operands and results remain BF16. FP32 internal arithmetic avoids adding
BF16 rounding points inside a mathematical fused operation. Native linear can
choose an unfused layout-dependent kernel; native attention can use blocked BF16
exponential storage. LayerNorm retains its single BF16 output boundary, without
claiming native reduction order or auxiliary statistics. These are FP32-math
candidates, not reproductions of every native CPU kernel, and remain source-reference gated.
"""
import math
from typing import Optional, Sequence


SOURCE_MEAN_ARTIFACTS = {
    "lib/libtorch_cpu.so": "c960407c516ec84d7994fd2c46cf2adc282d0fe6d3d0e928e1d64cc66598e5e4",
    "include/ATen/native/DispatchStub.h": "45cde3f5f24a89acc536a064a17f007189255d6998321a7ac534c7b6f14ddb27",
    "include/ATen/AccumulateType.h": "93381e925f39f8f7f689df6a07f528db2e50c97c4c18d71a9971becafb9e8746",
    "include/ATen/native/cpu/Reduce.h": "f7286c1fa97e96597b8728968a9f639d1d17f1913c64926f8e485ed570df854f",
    "include/ATen/cpu/vec/vec256/vec256_float.h": "62cd6915726701b7b62dea00c16406eb1792c140c197ae1a12e4d35b9c6d81dd",
}

SOURCE_SOFTMAX_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/DispatchStub.h": SOURCE_MEAN_ARTIFACTS["include/ATen/native/DispatchStub.h"],
    "include/ATen/cpu/vec/functional_base.h": "f60163d76edecb5bc3871b46af52f6148be9cb12d2d378bde86ccef6b528853a",
    "include/ATen/cpu/vec/vec512/vec512_float.h": "0150e7eebb07a5235f19ce5c759368ac3c3bf93f65ff459632a30bcf99d5896c",
    "include/ATen/native/cpu/SoftmaxKernel.h": "2e5d98a57e767e29ace4db183eb6d3f3281f669e7093b0b00bdfbb946118a536",
    "include/sleef.h": "2955f7429d04a05c5534a634f1aa77519510c90c603b4cbb4afa29d01e9a5923",
}

SOURCE_LAYERNORM_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/DispatchStub.h": SOURCE_MEAN_ARTIFACTS["include/ATen/native/DispatchStub.h"],
    "include/ATen/native/cpu/moments_utils.h": "0225459908f720e74b4d6814e4b61734042b3d7db22fc8c9d98a889ed1126dff",
    "include/ATen/cpu/vec/vec256/vec256_float.h": SOURCE_MEAN_ARTIFACTS["include/ATen/cpu/vec/vec256/vec256_float.h"],
    "include/ATen/cpu/vec/vec256/vec256_bfloat16.h": "a337093b5269dd1349f15a3dec8c6113f1af9e11714a669b6273139a67f72242",
    "include/ATen/cpu/vec/vec256/vec256_16bit_float.h": "aa2fd19c04c46964061cea8395d292613827e77572c6e8f95d105d5cd2bc3e63",
    "include/ATen/cpu/vec/functional.h": "01672298ad7826eef730bec0d2b1172ed6380bd907daa0de3748cb09d5699258",
    "include/ATen/cpu/vec/functional_base.h": SOURCE_SOFTMAX_ARTIFACTS["include/ATen/cpu/vec/functional_base.h"],
    "include/ATen/cpu/vec/functional_bfloat16.h": "6d40c9bd4b21750b0140bdeaf93cd8f455c4f0466b4501c91e6218b420747ef9",
}


def _source_cpu_bindings(torch, artifacts, operation):
    if torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0" or torch.backends.cpu.get_cpu_capability() != "AVX512" or torch.get_num_threads() != 1:
        raise ValueError(f"{operation} requires the pinned PyTorch CPU build, AVX512 capability and one native thread")
    from pathlib import Path
    from verify_resnet import file_identity, safe_path
    root = safe_path(Path(torch.__file__).parent)
    bindings = [file_identity(root / relative, safe_path) for relative in artifacts]
    if any(record["sha256"] != expected for record, expected in zip(bindings, artifacts.values())):
        raise ValueError(f"{operation} requires the independently proved native library and headers")
    return bindings


def source_mean_contract(torch, policy="native"):
    """Bind the native CPU mean used by the scoped Smol expert export."""
    if policy == "native":
        return None
    if policy != "torch_cpu_mean720_v1":
        raise ValueError("Unknown source mean export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_MEAN_ARTIFACTS, "Source mean")
    return {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchMean720Fp32",
            "native": "unchanged_Tensor_mean_float32_last_axis_keepdim_Smol_expert_1x50x720_contiguous",
            "native_dispatch": "pinned_AVX2_eight_lane_sum;AVX512_sum_registration_null", "native_threads": 1,
            "export_shape": [1, 50, 720], "workspace": "65_FP32_private_output_words_260_bytes_plus_mean_output_and_reshape",
            "algorithm": "source_cascade_FP32_sum_then_direct_FP32_division_by720;no_reassociation_or_reciprocal_rewrite",
            "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/SumKernel.cpp",
            "unchanged": ["square", "epsilon", "rsqrt", "affine", "BF16_casts", "native_reference", "acceptance_gates"],
            "bindings": bindings}


def source_mean_translations(torch, policy="native"):
    """Export only the proved Smol mean geometry; preserve other translations."""
    if source_mean_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.onnx_types import FLOAT
    from onnxscript.values import Opset, Op
    source = Opset("merlin.source_preserving", 1)
    schema = defs.OpSchema("TorchMean720Fp32", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter("a", "T")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(float)"], "Bound source FP32 mean")])
    mean_op = Op(source, "TorchMean720Fp32", schema)

    def mean_dim(self: FLOAT, dim: Optional[Sequence[int]] = None, keepdim: bool = False, dtype: int = -1) -> FLOAT:
        shape = tuple(self.shape) if self.shape is not None else ()
        if shape == (1, 50, 720) and isinstance(dim, (tuple, list)) and tuple(dim) in ((-1,), (2,)) and keepdim and dtype in (None, -1, ir.DataType.FLOAT):
            return mean_op(self)
        return core.aten_mean_dim(self, dim, keepdim, dtype)

    return {torch.ops.aten.mean.dim: mean_dim}


def source_softmax_contract(torch, policy="native"):
    """Bind the scoped native CPU softmax; leave QK/PV and masks unchanged."""
    if policy == "native":
        return None
    if policy != "torch_cpu_softmax177_v1":
        raise ValueError("Unknown source softmax export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_SOFTMAX_ARTIFACTS, "Source softmax")
    return {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchSoftmaxAvx512F32LastDim177",
            "native": "unchanged_Tensor_softmax_float32_last_axis_Smol_expert_1x15x50x177_contiguous",
            "native_dispatch": "pinned_AVX512_sixteen_lane_softmax", "native_threads": 1,
            "export_shape": [1, 15, 50, 177],
            "algorithm": "source_lane_reduction_SLEEF_exp_explicit_FMA_then_FP32_reciprocal_multiply",
            "qualification": "bounded_saved_actual_and_controls;complete_native_attention_and_policy_unqualified",
            "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/SoftMaxKernel.cpp",
            "unchanged": ["QK", "PV", "masks", "BF16_casts", "native_reference", "acceptance_gates"],
            "bindings": bindings}


def source_softmax_translations(torch, policy="native"):
    """Export only the proved FP32 Smol shape and last axis."""
    if source_softmax_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.onnx_types import FLOAT
    from onnxscript.values import Opset, Op
    schema = defs.OpSchema("TorchSoftmaxAvx512F32LastDim177", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter("a", "T")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(float)"], "Bound source FP32 softmax")])
    softmax_op = Op(Opset("merlin.source_preserving", 1), "TorchSoftmaxAvx512F32LastDim177", schema)

    def softmax(self: FLOAT, dim: int, dtype: int = -1) -> FLOAT:
        if tuple(self.shape or ()) == (1, 15, 50, 177) and dim in (-1, 3) and dtype in (None, -1, ir.DataType.FLOAT):
            return softmax_op(self)
        return core.aten_softmax(self, dim, dtype)

    def internal_softmax(self: FLOAT, dim: int, half_to_float: bool) -> FLOAT:
        if tuple(self.shape or ()) == (1, 15, 50, 177) and dim in (-1, 3) and not half_to_float:
            return softmax_op(self)
        return core.aten__softmax(self, dim, half_to_float)

    return {torch.ops.aten.softmax.int: softmax, torch.ops.aten._softmax.default: internal_softmax}


def source_layernorm_contract(torch, policy="native"):
    """Bind the single-output BF16 vision LayerNorm's native CPU arithmetic."""
    if policy == "native":
        return None
    if policy != "torch_cpu_layernorm768_v1":
        raise ValueError("Unknown source LayerNorm export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_LAYERNORM_ARTIFACTS, "Source LayerNorm")
    return {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchLayerNormAvx2Bf16Width768",
            "native": "unchanged_single_output_BF16_Smol_vision_LayerNorm_1x1024x768_original_affines_epsilon1e-6",
            "native_dispatch": "pinned_AVX2_BF16_sixteen_lane_FP32_eight_lane_moments;AVX512_registration_null",
            "native_threads": 1, "export_shape": [1, 1024, 768], "epsilon_fp32_word": "0x358637bd",
            "algorithm": "source_Welford_cascade_scalar_merge_sqrt_reciprocal_and_two_affine_FMAs",
            "qualification": "bounded_saved_actual_and_controls;complete_vision_prefix_and_policy_unqualified",
            "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp",
            "unchanged": ["parameters", "epsilon", "native_reference", "acceptance_gates", "ordinary_ONNX_LayerNormalization"],
            "bindings": bindings}


def source_layernorm_translations(torch, policy="native"):
    """Preserve only the proved native BF16 vision shape and original epsilon."""
    if source_layernorm_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir, opset18 as op
    from onnxscript.onnx_types import BFLOAT16
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.values import Opset, Op
    schema = defs.OpSchema("TorchLayerNormAvx2Bf16Width768", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter(name, "T" if name != "epsilon" else "F") for name in ("input", "weight", "bias", "epsilon")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(bfloat16)"], "Original BF16 operands"), ("F", ["tensor(float)"], "Original FP32 kernel epsilon")])
    layernorm_op = Op(Opset("merlin.source_preserving", 1), "TorchLayerNormAvx2Bf16Width768", schema)
    fallback = translations(torch)[torch.ops.aten.layer_norm.default]

    def layer_norm(input: BFLOAT16, normalized_shape: Sequence[int], weight: Optional[TFloat] = None, bias: Optional[TFloat] = None, eps: float = 1e-5, cudnn_enable: bool = True) -> BFLOAT16:
        if tuple(input.shape or ()) == (1, 1024, 768) and tuple(normalized_shape) == (768,) and eps == 1e-6 and all(value is not None and value.dtype == ir.DataType.BFLOAT16 and tuple(value.shape or ()) == (768,) for value in (weight, bias)):
            epsilon = op.Constant(value=ir.tensor(eps, dtype=ir.DataType.FLOAT))
            return layernorm_op(input, weight, bias, epsilon)
        return fallback(input, normalized_shape, weight, bias, eps, cudnn_enable)

    return {torch.ops.aten.layer_norm.default: layer_norm}


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


def scalar_translations(torch):
    """Preserve original Python scalars when materializing DOUBLE constants.

    The scalar attribute annotation avoids the built-in TensorType argument's
    implicit FLOAT constant before its DOUBLE cast. Other ordinary dtypes keep
    the built-in translation. This does not select floating arithmetic order.
    """
    from onnxscript import ir, opset18 as op
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.onnx_types import TensorType

    def scalar_tensor(s: float, dtype: int = -1, layout: str = "", device: str = "", pin_memory: bool = False) -> TensorType:
        # This annotation marks an attribute; the exporter retains Python int
        # and bool arguments too. Never numerically convert those through float.
        if type(s) not in (int, float, bool):
            raise ValueError("Scalar translation requires an original static Python number")
        if layout not in (None, "", 0, "strided", "torch.strided"):
            raise ValueError("Scalar translation requires default strided layout")
        selected_device = str(torch.get_default_device()) if device in (None, "") else str(device)
        if selected_device not in ("cpu", "cpu:0"):
            raise ValueError("Scalar translation requires default or explicit CPU device")
        if pin_memory:
            raise ValueError("Scalar translation cannot preserve pinned allocation")
        if dtype in (None, -1):
            defaults = {torch.float16: ir.DataType.FLOAT16, torch.bfloat16: ir.DataType.BFLOAT16, torch.float32: ir.DataType.FLOAT, torch.float64: ir.DataType.DOUBLE}
            dtype = defaults[torch.get_default_dtype()]
        selected = ir.DataType(dtype)
        if selected not in (ir.DataType.FLOAT16, ir.DataType.FLOAT, ir.DataType.DOUBLE, ir.DataType.BFLOAT16, ir.DataType.INT8, ir.DataType.INT16, ir.DataType.INT32, ir.DataType.INT64, ir.DataType.UINT8, ir.DataType.BOOL):
            raise ValueError("Scalar translation requires a supported ordinary real dtype")
        if selected == ir.DataType.DOUBLE:
            return op.Constant(value=ir.tensor(s, dtype=selected))
        return core.aten_scalar_tensor(s, dtype=selected, layout=layout, device=device, pin_memory=False)

    return {torch.ops.aten.scalar_tensor.default: scalar_tensor}


def translations(torch):
    # Optional exporter dependencies are loaded only when this recipe is selected.
    from onnxscript import ir, opset18 as op
    from onnxscript.function_libs.torch_lib.ops import nn
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.onnx_types import BFLOAT16, TensorType

    def silu(self: BFLOAT16) -> BFLOAT16:
        # Fused BF16 SiLU has one logical output cast, not a BF16 sigmoid store.
        if self.dtype != ir.DataType.BFLOAT16:
            raise ValueError("Fused SiLU translation requires BF16 dispatcher input")
        widened = op.Cast(self, to=1)
        return op.Cast(op.Mul(widened, op.Sigmoid(widened)), to=16)

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

    def layer_norm(input: BFLOAT16, normalized_shape: Sequence[int], weight: Optional[TFloat] = None, bias: Optional[TFloat] = None, eps: float = 1e-5, cudnn_enable: bool = True) -> BFLOAT16:
        # A BF16-only signature lets the exporter retain its built-in FP32 path.
        # Register the observed single-output ATen op; native_layer_norm's CPU
        # auxiliary statistics can be BF16 and are outside this translation.
        if input.dtype != ir.DataType.BFLOAT16:
            raise ValueError("LayerNorm translation requires BF16 dispatcher input")
        if not normalized_shape or any(type(value) is not int or value <= 0 for value in normalized_shape) or input.shape is None or len(input.shape) < len(normalized_shape):
            raise ValueError("BF16 LayerNorm requires a known rank and positive static normalized shape")
        trailing_shape = tuple(input.shape)[-len(normalized_shape):]
        if any(type(value) is not int for value in trailing_shape) or trailing_shape != tuple(normalized_shape):
            raise ValueError("BF16 LayerNorm normalized shape must equal the static trailing input dimensions")
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("BF16 LayerNorm requires a finite nonnegative epsilon")
        if any(value is not None and value.dtype != ir.DataType.BFLOAT16 for value in (weight, bias)):
            raise ValueError("BF16 LayerNorm requires original BF16 scale/bias")
        if any(value is not None and (value.shape is None or tuple(value.shape) != tuple(normalized_shape)) for value in (weight, bias)):
            raise ValueError("BF16 LayerNorm affine shapes must equal the normalized shape")
        # Explicit FLOAT standardization and affine have no intervening BF16
        # normalization cast. Original BF16 parameters widen exactly. The
        # decomposition also supports omitted affine and all trailing axes.
        axes = list(range(-len(normalized_shape), 0))
        widened = op.Cast(input, to=1)
        centered = op.Sub(widened, op.ReduceMean(widened, axes, keepdims=1))
        variance = op.ReduceMean(op.Mul(centered, centered), axes, keepdims=1)
        output = op.Div(centered, op.Sqrt(op.Add(variance, eps)))
        if weight is not None:
            output = op.Mul(output, op.Cast(weight, to=1))
        if bias is not None:
            output = op.Add(output, op.Cast(bias, to=1))
        return op.Cast(output, to=16)

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

    return {torch.ops.aten.linear.default: linear, torch.ops.aten.scaled_dot_product_attention.default: attention, torch.ops.aten.layer_norm.default: layer_norm, torch.ops.aten.silu.default: silu}
