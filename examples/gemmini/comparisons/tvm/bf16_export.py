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

SOURCE_CONV_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/dnnl_version.h": "f51d7a673319a0519ace0a799ee718687a6b0571bb42a9b627aec41d850367af",
    "include/oneapi/dnnl/dnnl_version.h": "99128af2408719334bee22737425b5b4fad774ebba550906f93a6f259a55a6e8",
}

SOURCE_SGEMM_ARTIFACTS = {
    "lib/libtorch_cpu.so": SOURCE_MEAN_ARTIFACTS["lib/libtorch_cpu.so"],
    "include/ATen/native/CPUBlas.h": "a7f1b8b3bcd6669902170c730917ceefa0da1aa18489dbea6d5e086f706987b1",
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
    if policy not in ("torch_cpu_softmax177_v1", "torch_cpu_softmax177_227_v1"):
        raise ValueError("Unknown source softmax export policy")
    bindings = _source_cpu_bindings(torch, SOURCE_SOFTMAX_ARTIFACTS, "Source softmax")
    record = {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchSoftmaxAvx512F32LastDim177",
            "native": "unchanged_Tensor_softmax_float32_last_axis_Smol_expert_1x15x50x177_contiguous",
            "native_dispatch": "pinned_AVX512_sixteen_lane_softmax", "native_threads": 1,
            "export_shape": [1, 15, 50, 177],
            "algorithm": "source_lane_reduction_SLEEF_exp_explicit_FMA_then_FP32_reciprocal_multiply",
            "qualification": "bounded_saved_actual_and_controls;complete_native_attention_and_policy_unqualified",
            "source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/SoftMaxKernel.cpp",
            "unchanged": ["QK", "PV", "masks", "BF16_casts", "native_reference", "acceptance_gates"],
            "bindings": bindings}
    if policy == "torch_cpu_softmax177_227_v1":
        record.pop("operator")
        record.pop("export_shape")
        record.update({"operators": ["TorchSoftmaxAvx512F32LastDim177", "TorchSoftmaxAvx512F32LastDim227"], "export_shapes": [[1, 15, 50, 177], [1, 15, 50, 227]], "native": "unchanged_FP32_Smol_manual_attention_softmax_width177_or227"})
    return record


def source_softmax_translations(torch, policy="native"):
    """Export only the proved FP32 Smol shape and last axis."""
    if source_softmax_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.onnx_types import FLOAT
    from onnxscript.values import Opset, Op
    widths = (177,) if policy == "torch_cpu_softmax177_v1" else (177, 227)
    ops = {}
    for width in widths:
        name = f"TorchSoftmaxAvx512F32LastDim{width}"
        schema = defs.OpSchema(name, "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter("a", "T")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(float)"], "Bound source FP32 softmax")])
        ops[(1, 15, 50, width)] = Op(Opset("merlin.source_preserving", 1), name, schema)

    def softmax(self: FLOAT, dim: int, dtype: int = -1) -> FLOAT:
        selected = ops.get(tuple(self.shape or ()))
        if selected is not None and dim in (-1, 3) and dtype in (None, -1, ir.DataType.FLOAT):
            return selected(self)
        return core.aten_softmax(self, dim, dtype)

    def internal_softmax(self: FLOAT, dim: int, half_to_float: bool) -> FLOAT:
        selected = ops.get(tuple(self.shape or ()))
        if selected is not None and dim in (-1, 3) and not half_to_float:
            return selected(self)
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


def source_conv_contract(torch, policy="native"):
    """Bind the proved software-BF16 native patch convolution dispatch."""
    if policy == "native":
        return None
    if policy != "torch_cpu_conv3x16_v1":
        raise ValueError("Unknown source convolution export policy")
    import os
    overrides = ("ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "ONEDNN_DEFAULT_FPMATH_MODE", "DNNL_DEFAULT_FPMATH_MODE", "ATEN_CPU_CAPABILITY")
    if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled or torch.cpu._is_avx512_bf16_supported() or torch.cpu._is_amx_tile_supported() or any(name in os.environ for name in overrides):
        raise ValueError("Source convolution requires the proved software-BF16 oneDNN dispatch without ISA/fpmath overrides")
    bindings = _source_cpu_bindings(torch, SOURCE_CONV_ARTIFACTS, "Source convolution")
    return {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchConv2dAvx512Bf16IC3Kernel16Stride16",
            "native": "unchanged_BF16_Smol_patch_convolution_1x3x512x512_original_768x3x16x16_weight_and_768_bias",
            "native_dispatch": "oneDNN3.7.1_jit_bf16_avx512_core_software_BF16", "native_threads": 1,
            "export_shape": [1, 3, 512, 512], "workspace": "32_FP32_private_words_128_bytes_plus_output",
            "algorithm": "KH_KW_channel1_channel0_zero_partner_channel2_explicit_FP32_FMA_then_original_bias_add_and_one_BF16_RNE_cast",
            "qualification": "bounded_actual_and_controls;finite_normal_product_scope;complete_vision_prefix_and_policy_unqualified",
            "source": "https://github.com/oneapi-src/oneDNN/blob/8d263e693366ef8db40acc569cc7d8edf644556d/src/cpu/x64/jit_avx512_core_bf16_conv_kernel.cpp",
            "unchanged": ["parameters", "stride", "padding", "dilation", "groups", "native_reference", "acceptance_gates", "ordinary_ONNX_Conv"],
            "bindings": bindings}


def source_conv_translations(torch, policy="native"):
    """Export the proved patch geometry; keep other convolutions ordinary."""
    if source_conv_contract(torch, policy) is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.onnx_types import BFLOAT16
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.values import Opset, Op
    schema = defs.OpSchema("TorchConv2dAvx512Bf16IC3Kernel16Stride16", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter(name, "T") for name in ("input", "weight", "bias")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(bfloat16)"], "Original BF16 patch operands")])
    conv_op = Op(Opset("merlin.source_preserving", 1), schema.name, schema)

    def pair(value):
        if isinstance(value, int):
            return (value, value)
        values = tuple(value) if isinstance(value, (tuple, list)) else ()
        return values * 2 if len(values) == 1 else values

    def admitted(input, weight, bias, stride, padding, dilation, groups):
        return all(value is not None and value.dtype == ir.DataType.BFLOAT16 and tuple(value.shape or ()) == shape for value, shape in ((input, (1, 3, 512, 512)), (weight, (768, 3, 16, 16)), (bias, (768,)))) and pair(stride) == (16, 16) and pair(padding) == (0, 0) and pair(dilation) == (1, 1) and groups == 1

    def conv2d(input: BFLOAT16, weight: TFloat, bias: Optional[TFloat] = None, stride: Sequence[int] = (1, 1), padding: Sequence[int] = (0, 0), dilation: Sequence[int] = (1, 1), groups: int = 1) -> BFLOAT16:
        if admitted(input, weight, bias, stride, padding, dilation, groups):
            return conv_op(input, weight, bias)
        return core.aten_conv2d(input, weight, bias, stride, padding, dilation, groups)

    def convolution(input: BFLOAT16, weight: TFloat, bias: Optional[TFloat] = None, stride: Sequence[int] = (1,), padding: Sequence[int] = (0,), dilation: Sequence[int] = (1,), transposed: bool = False, output_padding: Sequence[int] = (0,), groups: int = 1) -> BFLOAT16:
        if not transposed and pair(output_padding) == (0, 0) and admitted(input, weight, bias, stride, padding, dilation, groups):
            return conv_op(input, weight, bias)
        return core.aten_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

    return {torch.ops.aten.conv2d.default: conv2d, torch.ops.aten.convolution.default: convolution}


def source_sgemm_contract(torch, policy="native"):
    """Bind the scoped native attention kernels, leaving scaling intact."""
    if policy == "native":
        return None
    if policy not in ("torch_cpu_sgemm177_v1", "torch_cpu_attention177_227_v1"):
        raise ValueError("Unknown source SGEMM export policy")
    import os
    if not torch.backends.mkl.is_available() or torch.get_float32_matmul_precision() != "highest" or any(name in os.environ for name in ("MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS", "MKL_DEBUG_CPU_TYPE", "ATEN_CPU_CAPABILITY")):
        raise ValueError("Source SGEMM requires the proved highest-precision MKL dispatch without ISA/consistency overrides")
    artifacts = SOURCE_SGEMM_ARTIFACTS
    if policy == "torch_cpu_attention177_227_v1":
        overrides = ("ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "ONEDNN_DEFAULT_FPMATH_MODE", "DNNL_DEFAULT_FPMATH_MODE")
        if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled or torch.cpu._is_avx512_bf16_supported() or torch.cpu._is_amx_tile_supported() or any(name in os.environ for name in overrides):
            raise ValueError("Source attention requires the proved software-BF16 oneDNN dispatch without ISA/fpmath overrides")
        artifacts = {**artifacts, **SOURCE_CONV_ARTIFACTS}
    bindings = _source_cpu_bindings(torch, artifacts, "Source SGEMM")
    record = {"name": policy, "status": "opt_in_source_preserving_export_not_policy_qualified",
            "domain": "merlin.source_preserving", "version": 1,
            "operators": ["TorchMklSgemmF32QK177", "TorchMklSgemmF32PV177"],
            "native": "unchanged_FP32_manual_Smol_denoiser_Q50_K177_QK_and_PV_matmul",
            "native_dispatch": "observed_AVX512_copied_QK_kernel_0_b0_and_nocopy_NN_PV", "native_threads": 1,
            "qk_native_strides": [[48000, 64, 960, 1], [169920, 64, 1, 960]],
            "pv_native_strides": [[132750, 8850, 177, 1], [169920, 64, 960, 1]],
            "algorithm": "positive_zero_then_increasing_k_explicit_FP32_FMA;native_alpha1_final_multiply_is_identity_in_admitted_scope",
            "qualification": "bounded_actual_and_controls;finite_RNE_gradual_underflow;native_physical_layouts_bound_by_capture;complete_attention_and_policy_unqualified",
            "unchanged": ["parameters", "QK_scale", "masks", "softmax", "BF16_casts", "native_reference", "acceptance_gates", "ordinary_ONNX_MatMul"],
            "bindings": bindings}
    if policy == "torch_cpu_attention177_227_v1":
        record["operators"] += ["TorchMklSgemmF32QK227", "TorchOnednnGemmBf16PV227"]
        record.update({"native": "unchanged_manual_Smol_Q50_K177_or227_attention_contractions", "native_dispatch": "MKL_AVX512_FP32_QK177_QK227_PV177;oneDNN3.7.1_gemm_jit_software_BF16_PV227", "qk227_native_strides": [[48000, 64, 960, 1], [217920, 64, 1, 960]], "pv227_native_strides": [[170250, 11350, 227, 1], [217920, 64, 960, 1]], "pv227_algorithm": "positive_zero_then_pair_high_low_explicit_FP32_FMA;odd_tail_FMA_positive_zero_times_last_probability_then_last_product;one_BF16_RNE_store", "pv227_evidence": "all_four_installed_JIT_variants_and_isolated_copy_packing;original_selected_variant_and_internal_buffers_not_observed;driver_alpha1_beta0_inferred_from_ordinary_matmul_semantics"})
    return record


def source_sgemm_translations(torch, policy="native"):
    """Preserve only the proved manual-attention contraction geometries."""
    record = source_sgemm_contract(torch, policy)
    if record is None:
        return {}
    from onnx import defs
    from onnxscript import ir
    from onnxscript.function_libs.torch_lib.ops import core
    from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
    from onnxscript.values import Opset, Op
    source = Opset("merlin.source_preserving", 1)
    ops = {}
    for name in record["operators"]:
        dtype = "tensor(bfloat16)" if name == "TorchOnednnGemmBf16PV227" else "tensor(float)"
        schema = defs.OpSchema(name, source.domain, 1, inputs=[defs.OpSchema.FormalParameter(value, "T") for value in ("left", "right")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", [dtype], "Original contraction operands")])
        ops[name] = Op(source, name, schema)
    geometries = {
        (ir.DataType.FLOAT, (1, 15, 50, 64), (1, 15, 64, 177)): "TorchMklSgemmF32QK177",
        (ir.DataType.FLOAT, (1, 15, 50, 177), (1, 15, 177, 64)): "TorchMklSgemmF32PV177",
        (ir.DataType.FLOAT, (1, 15, 50, 64), (1, 15, 64, 227)): "TorchMklSgemmF32QK227",
        (ir.DataType.BFLOAT16, (1, 15, 50, 227), (1, 15, 227, 64)): "TorchOnednnGemmBf16PV227",
    }

    def matmul(self: TFloat, other: TFloat) -> TFloat:
        key = (self.dtype, tuple(self.shape or ()), tuple(other.shape or ()))
        selected = ops.get(geometries.get(key)) if self.dtype == other.dtype else None
        return selected(self, other) if selected is not None else core.aten_matmul(self, other)

    return {torch.ops.aten.matmul.default: matmul}


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
