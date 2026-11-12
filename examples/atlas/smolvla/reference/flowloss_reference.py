"""Independent Atlas-arithmetic reference for the SmolVLA flow-loss boundary.

Interprets the authored SmolVLA bundle FX (model.pt2, loaded after
atlas_fx_ops) node by node with explicitly selected independent arithmetic.
Dispatch uses only original ATen/custom-op targets, scalar arguments, and
dtype/shape metadata. No compiler evaluator, emitted program, model2MLIR
graph, Merlin semantic JSON, or ordinary framework numerical kernel computes
any value. Policies follow the selected precision policy; the compiled
application is the same forward re-exported live with the documented
semantics-preserving rewrites, which this interpreter applies as bodies:
biased atlas.qlinear = bias-free qlinear then `+ bias`; atlas.qconv2d = the
same contraction on im2col codes, reshape/permute/reshape, then `+ bias`;
stride-equals-kernel unpadded aten.im2col = reshape/permute/reshape.

Constant-only float computation follows exact IEEE constant folding (powers:
the selected MPFR power helper). A value outside a selected policy domain
refuses with an exact message; no fallback semantics are introduced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
import operator
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROTARY_REFERENCE = HERE.parents[1] / "rotary-host" / "reference"
SCHEMA = "atlas.independent_smolvla_flowloss.v1"
FLOATS = {"bf16", "f32", "f64"}
RAW = {"bf16": np.uint16, "f32": np.uint32, "f64": np.uint64, "u8": np.uint8, "i32": np.int32,
       "i64": np.int64, "bool": np.bool_}
EXPECTED_POLICY = {
    "pointwise": "f32-to-bf16-bounded-pointwise-rne-chop-v1",
    "bf16_pointwise": "bf16-bounded-pointwise-rne-v1",
    "reduction": "bf16-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1",
    "pointwise_math": "f32-to-bf16-positive-denominator-lut-div-rsqrt-v1",
    "source_exp": "f32-to-bf16-bounded32-exp-lut-v1",
    "already_quantized_mask": "bf16-minfinite-zero-rowspan80-nonempty-v1",
    "already_quantized": "atlas-already-quantized-e4m3-signed-exponents-mxu1-model47-v1",
    "row_softmax": "bf16-tiled-row-softmax-ltr-fp32-rne-chop-rcp-lut-v1",
    "gelu": "f32-to-bf16-bounded-tanh-gelu-cube-chop-v1",
    "sigmoid": "f32-to-bf16-bounded-sigmoid-exp-recip-v1",
}
REQUIRED_LISTS = {
    "host_float_arithmetic": {"bf16_add", "bf16_sub", "f64_mul"},
    "host_float_casts": {"f32_to_bf16", "bf16_to_f32", "f32_to_f64", "f64_to_f32"},
    "host_integer_ops": {"i64_add", "i64_sub", "i64_cumsum", "signed_i64_to_f32", "i64_min_value", "i64_first_min_index"},
    "host_predicates": {"mask", "select"},
}
UNARY_OPERATIONS = {"source_f32_cos_unqualified", "source_f32_sin_unqualified",
                    "source_f64_cos_unqualified", "source_f64_sin_unqualified"}
RETAIN_TARGETS = {"aten.layer_norm.default", "aten.embedding.default", "aten.cat.default", "atlas.qconv2d.default",
                  "aten.mse_loss.default", "aten.softmax.int"}
RETAIN_LIMIT = 4 << 20  # elements


def sha_file(path):
    digest = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def info(raw):
    raw = np.asarray(raw)
    return {"dtype": raw.dtype.str, "shape": list(raw.shape), "sha256": sha256(raw.tobytes()).hexdigest()}


@dataclass
class Tensor:
    raw: np.ndarray
    dtype: str
    const: bool = False


def dtype_name(dtype):
    import torch
    return {torch.bfloat16: "bf16", torch.float32: "f32", torch.float64: "f64", torch.uint8: "u8",
            torch.int32: "i32", torch.int64: "i64", torch.bool: "bool"}.get(dtype)


def raw_tensor(value, const=False):
    import torch
    name = dtype_name(value.dtype)
    if name is None:
        raise ValueError("unsupported original FX tensor dtype " + str(value.dtype))
    value = value.detach().contiguous()
    view = {"bf16": torch.uint16, "f32": torch.uint32, "f64": torch.uint64}.get(name)
    data = (value.view(view) if view is not None else value).numpy().copy()
    return Tensor(data, name, const)


def read_safetensors(path, keys):
    """Read declared tensors as raw words straight from the archive bytes."""
    names = {"BF16": ("bf16", "<u2"), "F32": ("f32", "<u4"), "F64": ("f64", "<u8"), "I64": ("i64", "<i8"),
             "I32": ("i32", "<i4"), "U8": ("u8", "u1"), "BOOL": ("bool", "u1")}
    values = {}
    with open(path, "rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 0 < size < 1 << 24:
            raise ValueError("fixture archive header is invalid")
        header = json.loads(stream.read(size))
        for key in keys:
            row = header[key]
            dtype, encoding = names[row["dtype"]]
            begin, end = row["data_offsets"]
            stream.seek(8 + size + begin)
            data = np.frombuffer(stream.read(end - begin), dtype=encoding).copy().reshape(row["shape"])
            if dtype == "bool":
                if np.any(data > 1):
                    raise ValueError("fixture Boolean bytes are not 0/1")
                data = data.astype(np.bool_)
            values[key] = Tensor(data.astype(RAW[dtype]), dtype)
    return values


def as_index(value):
    if type(value) is not int:
        raise ValueError("source layout argument is not an integer")
    return value


class Refusal(ValueError):
    pass


class FlowLossInterpreter:
    def __init__(self, program, policy, *, engine="numpy", threads=1, retain=(), progress=None, waivers=()):
        import layer_reference
        self.lr = layer_reference
        self.program, self.policy = program, policy
        self.engine, self.threads = engine, threads
        self.retain, self.progress = set(retain), progress
        self.specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
        self.env, self.saved, self.boundaries, self.constants = {}, {}, {}, {}
        self.dispatch = Counter()
        self.position_ids_check = None
        self.waivers, self.waived = set(waivers), []
        self.started = time.monotonic()

    # ------------------------------------------------------------ helpers
    def dtype(self, node):
        return dtype_name(getattr(node.meta.get("val"), "dtype", None))

    def resolve(self, value):
        import torch
        if isinstance(value, torch.fx.Node):
            return self.env[value]
        if isinstance(value, (list, tuple)):
            return type(value)(self.resolve(item) for item in value)
        if isinstance(value, dict):
            return {key: self.resolve(item) for key, item in value.items()}
        return value

    def count(self, policy):
        self.dispatch[policy] += 1

    def f32_words(self, value):
        """Operand as f32 words: exact BF16 widening, RNE scalar literal."""
        import smolvla_primitives as sp
        if isinstance(value, Tensor):
            if value.dtype == "f32":
                return value.raw
            if value.dtype == "bf16":
                return sp.widen(value.raw)
            raise Refusal("floating arithmetic operand is not BF16/f32 data: " + value.dtype)
        return np.asarray(sp.scalar_bits(value, "f32"), np.uint32)

    def const_words(self, value, fmt):
        """Exact conversion of a constant operand to fmt words."""
        import smolvla_primitives as sp
        if isinstance(value, Tensor):
            if value.dtype == fmt:
                return value.raw
            source = value.dtype
            raw = value.raw.astype(np.int64) if source == "bool" else value.raw
            return sp.convert_exact(raw, source, fmt)
        return np.asarray(sp.scalar_bits(value, fmt), sp.RAW[fmt])

    @staticmethod
    def is_const(values):
        return all(value.const for value in values if isinstance(value, Tensor))

    # ------------------------------------------------------------ arithmetic
    def arith(self, op, left, right, out):
        """add/sub/mul/div dispatch by result dtype and constant-ness."""
        import smolvla_primitives as sp
        lr = self.lr
        tensors = [v for v in (left, right) if isinstance(v, Tensor)]
        const = self.is_const(tensors)
        if out in {"i64", "i32", "bool"}:
            a = left.raw if isinstance(left, Tensor) else left
            b = right.raw if isinstance(right, Tensor) else right
            if out == "bool":
                if op != "mul":
                    raise Refusal("Boolean arithmetic other than product is unqualified")
                self.count("host_predicates:mask")
                return Tensor(np.logical_and(a, b), "bool", const)
            a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
            if np.any(np.abs(a) > 1 << 40) or np.any(np.abs(b) > 1 << 40):
                raise Refusal("integer source arithmetic leaves the exact small-integer domain")
            value = {"add": np.add, "sub": np.subtract, "mul": np.multiply}[op](a, b)
            self.count("host_integer_ops:i64_" + op if not const else "constant:i64_" + op)
            return Tensor(value.astype(RAW[out]), out, const)
        if const:
            words = [self.const_words(v, out) for v in (left, right)]
            self.count(f"constant_fold:ieee754-{out}-{op}-rne-v1")
            return Tensor(sp.exact_elementwise(op, out, *words), out, True)
        if out == "f64":
            if op != "mul":
                raise Refusal("runtime binary64 arithmetic other than multiply is unqualified")
            words = [self.const_words(v, "f64") if not (isinstance(v, Tensor) and v.dtype == "f32")
                     else sp.convert_exact(v.raw, "f32", "f64") for v in (left, right)]
            self.count("host_float_arithmetic:f64_mul")
            return Tensor(sp.exact_elementwise("mul", "f64", *words), "f64")
        both_bf16 = all(isinstance(v, Tensor) and v.dtype == "bf16" for v in (left, right))
        if out == "bf16" and both_bf16 and op in {"add", "sub"}:
            self.count("exact_float:ieee754-bf16-" + op + "-rne-v1")
            return Tensor(sp.exact_bf16_add(left.raw, right.raw, subtract=op == "sub"), "bf16")
        if out == "bf16" and both_bf16 and op == "mul":
            self.count("bf16_pointwise:bf16_mul")
            return Tensor(lr.selected_pointwise("bf16_mul", [left.raw, right.raw], source_bf16=True), "bf16")
        if out not in {"bf16", "f32"}:
            raise Refusal("floating arithmetic result dtype is unqualified: " + out)
        words = [self.f32_words(v) for v in (left, right)]
        self.count(("pointwise_math:f32_div" if op == "div" else "pointwise:f32_" + op)
                   + (" (bf16 result)" if out == "bf16" else ""))
        value = lr.selected_pointwise("f32_" + op, words)
        return Tensor(sp.f32_to_bf16_rne(value) if out == "bf16" else value, out)

    def convert(self, value, target):
        import smolvla_primitives as sp
        source = value.dtype
        if source == target:
            return Tensor(value.raw.copy(), target, value.const)
        if (source, target) == ("bf16", "f32"):
            self.count("cast:bf16_to_f32")
            return Tensor(sp.widen(value.raw), target, value.const)
        if (source, target) == ("f32", "bf16"):
            self.count("cast:f32_to_bf16")
            return Tensor(sp.f32_to_bf16_rne(value.raw), target, value.const)
        if (source, target) == ("f32", "bool") and value.const:
            self.count("constant:f32_to_bool")
            return Tensor((value.raw & np.uint32(0x7fffffff)) != 0, "bool", True)
        if (source, target) in {("i64", "f32"), ("f64", "f32"), ("f32", "f64"), ("bool", "f32")}:
            self.count(f"cast:{source}_to_{target}")
            raw = value.raw.astype(np.int64) if source == "bool" else value.raw
            return Tensor(sp.convert_exact(raw, source, target), target, value.const)
        raise Refusal(f"source dtype conversion {source}->{target} has no selected policy")

    # ------------------------------------------------------------ composite ops
    def layer_norm(self, x, shape, weight, bias, eps):
        import smolvla_primitives as sp
        lr = self.lr
        if x.dtype != "bf16" or len(shape) != 1 or shape[0] != x.raw.shape[-1] or weight is None or bias is None:
            raise Refusal("layer_norm requires final-axis BF16 data with weight and bias")
        n = shape[0]
        xf, wf, bf = sp.widen(x.raw), sp.widen(weight.raw), sp.widen(bias.raw)
        seed = np.zeros(xf.shape[:-1], np.uint32)
        count = np.asarray(sp.scalar_bits(float(n), "f32"), np.uint32)
        policy = self.policy["reduction"]
        mean = lr.selected_pointwise("f32_div", [lr.selected_sum(xf, seed, axis=xf.ndim - 1, reduction_policy=policy), count])
        diff = lr.selected_pointwise("f32_sub", [xf, mean[..., None]])
        square = lr.selected_pointwise("f32_mul", [diff, diff])
        var = lr.selected_pointwise("f32_div", [lr.selected_sum(square, seed, axis=xf.ndim - 1, reduction_policy=policy), count])
        shifted = lr.selected_pointwise("f32_add", [var[..., None], np.asarray(sp.scalar_bits(eps, "f32"), np.uint32)])
        rstd = lr.selected_pointwise("source_f32_rsqrt_unqualified", [shifted])
        y = lr.selected_pointwise("f32_mul", [diff, rstd])
        y = lr.selected_pointwise("f32_mul", [y, wf])
        y = lr.selected_pointwise("f32_add", [y, bf])
        self.count("composite:layer_norm(reduction,pointwise,pointwise_math)")
        return Tensor(sp.f32_to_bf16_rne(y), "bf16")

    def contraction(self, a_codes, a_exp, b_codes, b_exp, label):
        import smolvla_primitives as sp
        if [v.dtype for v in (a_codes, a_exp, b_codes, b_exp)] != ["u8", "i32", "u8", "i32"]:
            raise Refusal(label + " code/exponent dtypes differ")
        self.count("already_quantized:" + label)
        return sp.contraction(a_codes.raw, a_exp.raw, b_codes.raw, b_exp.raw, engine=self.engine,
                              threads=self.threads, label=label)

    def qlinear(self, node, x, xe, w, we, bias, out_dtype, weight_format):
        if weight_format != "atlas_e4m3" or out_dtype not in {"bfloat16", "float32"} or w.raw.ndim != 2:
            raise Refusal("qlinear weight format or output dtype differs")
        k = x.raw.shape[-1]
        rows = x.raw.reshape(-1, k)
        exponent = xe.raw if xe.raw.ndim == 0 else xe.raw.reshape(-1, 1)
        accumulator, restored = self.contraction(Tensor(rows, "u8"), Tensor(exponent, "i32"), w, we, "qlinear")
        restored = restored.reshape(x.raw.shape[:-1] + (w.raw.shape[0],))
        y = Tensor(restored, "bf16") if out_dtype == "bfloat16" else Tensor(restored.astype(np.uint32) << np.uint32(16), "f32")
        if bias is None:
            return y
        self.count("rewrite:qlinear_bias_add")
        return self.arith("add", y, bias, self.dtype(node))

    # ------------------------------------------------------------ dispatch
    def call(self, node):
        import torch
        import smolvla_primitives as sp
        lr = self.lr
        target = str(node.target)
        args, kwargs = self.resolve(node.args), self.resolve(node.kwargs)
        out = self.dtype(node)
        if node.target is operator.getitem:
            return args[0][args[1]]
        if node.target in (operator.ge, operator.le):
            return node.target(*args)
        if target == "aten.sym_size.int":
            return int(args[0].raw.shape[args[1]])
        if target == "aten._assert_scalar.default":
            if args[0] is not True:
                raise Refusal("source runtime assertion failed: " + str(args[1]))
            return None
        if target == "aten._assert_tensor_metadata.default":
            if "dtype" in kwargs and dtype_name(kwargs["dtype"]) != args[0].dtype:
                raise Refusal("source dtype assertion differs")
            return None
        const = self.is_const([v for v in args if isinstance(v, Tensor)])

        # ---------------- custom Atlas operations
        if target.startswith("atlas."):
            schema = node.target._schema
            ordered = [args[i] if i < len(args) else kwargs.get(spec.name, spec.default_value)
                       for i, spec in enumerate(schema.arguments)]
            if target == "atlas.quantize_act.default":
                from atlas_ref.activation import quantize_act
                x, mode, reduce, rule, static_exp, code_dtype, tag, rows = ordered
                if code_dtype != "uint8" or x.dtype not in {"bf16", "f32"}:
                    raise Refusal("activation quantizer code or input dtype differs")
                self.count(f"activation_quantize:{mode}/{reduce}/{rule}")
                values = x.raw if x.dtype == "bf16" else x.raw.view(np.float32)
                codes, exponent = quantize_act(values, mode, reduce, rule, static_exp, rows)
                return Tensor(codes, "u8"), Tensor(np.asarray(exponent, np.int32), "i32")
            if target == "atlas.qlinear.default":
                return self.qlinear(node, *ordered)
            if target == "atlas.qconv2d.default":
                x, xe, w, we, bias, out_dtype, weight_format, batch, oh, ow = ordered
                y = self.qlinear(node, x, xe, w, we, None, out_dtype, weight_format)
                channels = w.raw.shape[0]
                if x.raw.shape[0] != batch * oh * ow:
                    raise Refusal("qconv2d im2col rows differ from batch*out_h*out_w")
                raw = y.raw.reshape(batch, oh * ow, channels).transpose(0, 2, 1).reshape(batch, channels, oh, ow).copy()
                self.count("rewrite:qconv2d_layout_bias")
                if bias is None:
                    return Tensor(raw, y.dtype)
                if bias.dtype != y.dtype:
                    bias = self.convert(bias, y.dtype)
                return self.arith("add", Tensor(raw, y.dtype), Tensor(bias.raw.reshape(1, -1, 1, 1), y.dtype), out)
            if target == "atlas.qk_matmul.default":
                _, restored = self.contraction(*ordered[:4], "qk_matmul")
                return Tensor(sp.widen(restored), "f32")
            if target == "atlas.pv_matmul.default":
                if len(ordered) > 4 and ordered[4] != 0:
                    raise Refusal("folded PV pop_shift is unsupported")
                _, restored = self.contraction(*ordered[:4], "pv_matmul")
                return Tensor(sp.widen(restored), "f32")
            raise Refusal("unsupported custom operation " + target)

        # ---------------- arithmetic
        if target in {"aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor"}:
            if kwargs.get("alpha", 1) != 1 or set(kwargs) - {"alpha"}:
                raise Refusal("binary arithmetic scalar modifier is unqualified")
            return self.arith(target.split(".")[1], args[0], args[1], out)
        if target == "aten.rsub.Scalar":
            if kwargs.get("alpha", 1) != 1 or len(args) > 2:
                raise Refusal("rsub modifier is unqualified")
            return self.arith("sub", args[1], args[0], out)
        if target == "aten.add_.Tensor":
            destination = args[0]
            value = self.arith("add", destination, args[1], destination.dtype)
            if not destination.raw.flags.writeable:
                raise Refusal("in-place add destination is not writable storage")
            np.copyto(destination.raw, value.raw)
            destination.const = destination.const and value.const
            return destination
        if target == "aten.pow.Tensor_Scalar":
            x, exponent = args
            if exponent != 2:
                raise Refusal("tensor power other than square is unqualified")
            return self.arith("mul", x, x, out)
        if target == "aten.pow.Scalar":
            base, exponent = args
            if not exponent.const or out not in {"f32", "f64"}:
                raise Refusal("scalar-base power needs a constant f32/f64 exponent")
            words = self.const_words(exponent, out)
            base_word = np.full(words.shape, sp.scalar_bits(base, out), sp.RAW[out])
            self.count("source_constant_math:" + sp.CONSTANT_POWER_POLICY)
            return Tensor(sp.constant_power(32 if out == "f32" else 64, base_word, words), out, True)
        if target == "aten.reciprocal.default":
            if not const or out not in {"f32", "f64"}:
                raise Refusal("runtime reciprocal has no selected policy")
            return self.arith("div", 1.0, args[0], out)
        if target == "aten.mean.dim":
            value, axes, keepdim = args
            if value.dtype != "f32" or tuple(axes) not in {(-1,), (value.raw.ndim - 1,)} or value.const:
                raise Refusal("mean requires runtime final-axis f32 data")
            summed = lr.selected_sum(value.raw, np.zeros(value.raw.shape[:-1], np.uint32), axis=value.raw.ndim - 1,
                                     reduction_policy=self.policy["reduction"])
            result = lr.selected_pointwise("f32_div", [summed, np.asarray(sp.scalar_bits(float(value.raw.shape[-1]), "f32"), np.uint32)])
            self.count("composite:mean(reduction,pointwise_math)")
            return Tensor(result[..., None] if keepdim else result, "f32")
        if target == "aten.rsqrt.default":
            if args[0].dtype != "f32" or args[0].const:
                raise Refusal("rsqrt requires runtime f32 data")
            self.count("pointwise_math:source_f32_rsqrt_unqualified")
            return Tensor(lr.selected_pointwise("source_f32_rsqrt_unqualified", [args[0].raw]), "f32")
        if target == "aten.silu.default":
            x = args[0]
            self.count("sigmoid:silu " + x.dtype)
            value = sp.silu(self.f32_words(x))
            return Tensor(sp.f32_to_bf16_rne(value) if out == "bf16" else value, out)
        if target == "aten.gelu.default":
            if kwargs.get("approximate") != "tanh" or args[0].dtype != "bf16":
                raise Refusal("only BF16 tanh-approximate GELU has a selected policy")
            self.count("gelu:tanh")
            return Tensor(sp.f32_to_bf16_rne(sp.tanh_gelu(sp.widen(args[0].raw))), "bf16")
        if target == "aten.layer_norm.default":
            x, shape, weight, bias, eps = args[:5]
            return self.layer_norm(x, shape, weight, bias, eps)
        if target == "aten.softmax.int":
            value, axis = args[:2]
            if value.dtype != "f32" or axis not in {-1, value.raw.ndim - 1} or (len(args) > 2 and args[2]):
                raise Refusal("softmax requires final-axis f32 scores")
            try:
                probabilities, extra = sp.row_softmax_rowspan80(value.raw)
                self.count("row_softmax:rowspan80")
            except ValueError as error:
                masked = value.raw == np.uint32(0xff7fffff)
                waiver = "softmax-where-mask" if masked.any() else "softmax-span"
                if waiver not in self.waivers:
                    raise
                probabilities, extra = sp.row_softmax_unchecked(value.raw, ~masked)
                self.count("DIAGNOSTIC:row_softmax_unchecked:" + waiver)
                self.waived.append({"node": node.name, "waiver": waiver, "message": str(error),
                                    "masked_lanes": int(masked.sum())})
            if node.name in self.retain or "*" in self.retain:
                for key in ("row_max_bf16", "denominator_bf16"):
                    self.saved[node.name + "." + key] = extra[key]
            return Tensor(probabilities, "f32")
        if target == "aten.mse_loss.default":
            a, b, reduction = args
            if reduction != 0 or a.dtype != "f32" or b.dtype != "f32":
                raise Refusal("mse_loss requires reduction none on f32 operands")
            diff = self.arith("sub", a, b, "f32")
            return self.arith("mul", diff, diff, "f32")
        if target in {"aten.sin.default", "aten.cos.default"}:
            x, operation = args[0], target.split(".")[1]
            if x.const:
                raise Refusal("constant trigonometry has no selected folding policy")
            if x.dtype == "f32":
                import rotary_exact
                self.count("unary_math:source_f32_" + operation + "_unqualified")
                return Tensor(rotary_exact.trig_f32(x.raw, operation), "f32")
            if x.dtype == "f64":
                self.count("unary_math:source_f64_" + operation + "_unqualified")
                return Tensor(sp.trig_f64(x.raw, operation), "f64")
            raise Refusal("trigonometry dtype is unqualified")
        if target == "aten.where.ScalarOther":
            condition, selected, other = args
            if condition.dtype != "bool" or not isinstance(selected, Tensor) or selected.dtype != out:
                raise Refusal("where requires a Boolean condition and a same-dtype tensor")
            fill = sp.scalar_bits(other, out) if out in FLOATS else other
            self.count("host_predicates:select")
            return Tensor(np.where(condition.raw, selected.raw, np.asarray(fill, RAW[out])).astype(RAW[out]), out,
                          const and condition.const)
        if target in {"aten.le.Tensor", "aten.ge.Scalar"}:
            a, b = args
            values = lambda v: (v.raw.view(np.float32) if v.dtype == "f32" else v.raw) if isinstance(v, Tensor) else v
            if any(isinstance(v, Tensor) and v.dtype in {"bf16", "f64"} for v in (a, b)):
                raise Refusal("comparison dtype is unqualified")
            result = values(a) <= values(b) if target == "aten.le.Tensor" else values(a) >= values(b)
            self.count("host_predicates:mask")
            return Tensor(np.asarray(result, np.bool_), "bool", const)
        if target == "aten.__and__.Tensor":
            self.count("host_predicates:mask")
            return Tensor(np.logical_and(args[0].raw, args[1].raw), "bool", const)
        if target == "aten.cumsum.default":
            value, axis = args
            if value.dtype in {"bool", "i64"} and out == "i64":
                self.count("host_integer_ops:i64_cumsum" if not const else "constant:i64_cumsum")
                return Tensor(np.cumsum(value.raw.astype(np.int64), axis=axis), "i64", const)
            if value.dtype == "f32" and out == "f32" and const:
                raw = np.moveaxis(value.raw, axis, -1).copy()
                total = np.zeros(raw.shape[:-1], np.uint32)
                result = np.empty_like(raw)
                for k in range(raw.shape[-1]):
                    total = sp.exact_elementwise("add", "f32", total, raw[..., k])
                    result[..., k] = total
                self.count("constant_fold:ieee754-f32-add-rne-v1 (scan)")
                return Tensor(np.moveaxis(result, -1, axis).copy(), "f32", True)
            raise Refusal("cumsum dtype or runtime float scan is unqualified")
        if target == "aten.min.dim":
            value, axis, keepdim = args
            if value.dtype != "i64":
                raise Refusal("min.dim requires int64 data")
            self.count("host_integer_ops:i64_min_value")
            values = np.min(value.raw, axis=axis, keepdims=keepdim)
            indices = np.argmin(value.raw, axis=axis)
            indices = np.expand_dims(indices, axis) if keepdim else indices
            return Tensor(values, "i64", const), Tensor(indices.astype(np.int64), "i64", const)
        if target == "aten.sum.dim_IntList":
            value, axes = args[:2]
            if value.dtype not in {"bool", "i64"} or out != "i64":
                raise Refusal("sum requires integer or Boolean data")
            keep = len(args) > 2 and args[2]
            self.count("constant:i64_sum" if const else "host_integer_ops:i64_sum")
            return Tensor(np.sum(value.raw.astype(np.int64), axis=tuple(axes), keepdims=bool(keep)), "i64", const)
        if target == "aten.clamp.default":
            value, low, high = (list(args) + [None, None])[:3]
            if not const or value.dtype != "f32":
                raise Refusal("runtime clamp has no selected policy")
            data = value.raw.view(np.float32)
            result = value.raw.copy()
            for bound, take in ((low, np.less), (high, np.greater)):
                if bound is not None:
                    word = np.uint32(sp.scalar_bits(bound, "f32"))
                    result = np.where(take(result.view(np.float32), word.view(np.float32)), word, result).astype(np.uint32)
            self.count("constant_fold:ieee754-f32-clamp")
            return Tensor(result, "f32", True)
        if target == "aten.bucketize.Tensor":
            value, boundaries = args
            if not (value.const and boundaries.const) or out != "i64" or kwargs.get("out_int32", False):
                raise Refusal("runtime bucketize has no selected policy")
            right = kwargs.get("right", False)
            exact = lambda t: [sp.bits_to_fraction(int(w), t.dtype) for w in t.raw.reshape(-1)]
            edges = exact(boundaries)
            if any(b <= a for a, b in zip(edges, edges[1:])):
                raise Refusal("bucketize boundaries are not strictly increasing")
            values = exact(value)
            ids = [sum(1 for e in edges if (e <= v if right else e < v)) for v in values]
            self.count("constant:bucketize(exact)")
            return Tensor(np.asarray(ids, np.int64).reshape(value.raw.shape), "i64", True)

        # ---------------- conversion
        if target == "aten.to.dtype":
            return self.convert(args[0], dtype_name(args[1]))
        if target in {"aten.to.dtype_layout", "aten.to.device"}:
            dtype = kwargs.get("dtype") if target == "aten.to.dtype_layout" else (args[2] if len(args) > 2 else None)
            target_dtype = dtype_name(dtype) if dtype is not None else args[0].dtype
            return self.convert(args[0], target_dtype)

        # ---------------- factories
        if target in {"aten.ones.default", "aten.full.default", "aten.new_ones.default"}:
            size = args[1] if target == "aten.new_ones.default" else args[0]
            fill = args[1] if target == "aten.full.default" else 1
            word = sp.scalar_bits(fill, out) if out in FLOATS else fill
            self.count("constant:" + target.split(".")[1])
            return Tensor(np.full(tuple(size), word, RAW[out]), out, True)
        if target in {"aten.arange.default", "aten.arange.start_step"}:
            start, end, step = (0, args[0], 1) if len(args) == 1 else (list(args) + [1])[:3]
            count = math.ceil((Fraction(end) - Fraction(start)) / Fraction(step)) if out in FLOATS else len(range(start, end, step))
            if out == "i64":
                return Tensor(np.arange(start, end, step, dtype=np.int64), "i64", True)
            words = []
            for i in range(count):
                exact = Fraction(start) + i * Fraction(step)
                word = sp.round_fraction(exact, out)
                if sp.bits_to_fraction(word, out) != exact:
                    raise Refusal("floating arange element is not exactly representable")
                words.append(word)
            self.count("constant:arange")
            return Tensor(np.asarray(words, RAW[out]), out, True)
        if target == "aten.linspace.default":
            start, end, steps = args
            if out != "f64" or steps < 2:
                raise Refusal("linspace requires binary64 with at least two steps")
            # Compiled iota: delta=(end-start)/(steps-1) in binary64, then start + f64(i)*delta.
            delta = sp.round_fraction(Fraction(end) - Fraction(start), "f64")
            delta = sp.exact_binary(delta, sp.round_fraction(Fraction(steps - 1), "f64"), "f64", "div")
            start_word = sp.scalar_bits(float(start), "f64")
            words = [sp.exact_binary(start_word, sp.exact_binary(sp.round_fraction(Fraction(i), "f64"), delta, "f64", "mul"),
                                     "f64", "add") for i in range(steps)]
            self.count("constant:linspace(start+i*delta)")
            return Tensor(np.asarray(words, np.uint64), "f64", True)
        if target == "aten.empty_like.default":
            value = args[0]
            sentinel = {"f32": 0x7fc00000, "bf16": 0x7fc0, "f64": 0x7ff8000000000000}.get(value.dtype)
            if sentinel is None:
                raise Refusal("empty_like dtype is unqualified")
            return Tensor(np.full(value.raw.shape, sentinel, RAW[value.dtype]), value.dtype)

        # ---------------- layout / identity
        if target in {"aten.alias.default", "aten.clone.default", "aten.contiguous.default",
                      "aten.lift_fresh_copy.default", "aten.detach_.default"}:
            value = args[0]
            if target == "aten.detach_.default":
                return value
            return Tensor(np.array(value.raw, order="C", copy=True), value.dtype, value.const)
        if target in {"aten.view.default", "aten.reshape.default"}:
            return Tensor(args[0].raw.reshape(tuple(args[1])).copy(), args[0].dtype, args[0].const)
        if target == "aten.flatten.using_ints":
            value, begin = args[0], args[1]
            end = args[2] if len(args) > 2 else -1
            shape = value.raw.shape
            end = end % len(shape)
            new = shape[:begin] + (math.prod(shape[begin:end + 1]),) + shape[end + 1:]
            return Tensor(value.raw.reshape(new).copy(), value.dtype, value.const)
        if target == "aten.transpose.int":
            return Tensor(np.swapaxes(args[0].raw, args[1], args[2]).copy(), args[0].dtype, args[0].const)
        if target == "aten.permute.default":
            return Tensor(np.transpose(args[0].raw, tuple(args[1])).copy(), args[0].dtype, args[0].const)
        if target == "aten.unsqueeze.default":
            return Tensor(np.expand_dims(args[0].raw, args[1]).copy(), args[0].dtype, args[0].const)
        if target in {"aten.expand.default", "aten.expand_as.default"}:
            value = args[0]
            shape = tuple(args[1].raw.shape) if target == "aten.expand_as.default" else tuple(args[1])
            offset = len(shape) - value.raw.ndim
            shape = tuple(value.raw.shape[i - offset] if extent == -1 else extent for i, extent in enumerate(shape))
            self.count("layout:broadcast")
            return Tensor(np.broadcast_to(value.raw, shape).copy(), value.dtype, value.const)
        if target == "aten.broadcast_tensors.default":
            values = args[0]
            shape = np.broadcast_shapes(*(v.raw.shape for v in values))
            return [Tensor(np.broadcast_to(v.raw, shape).copy(), v.dtype, v.const) for v in values]
        if target == "aten.slice.Tensor":
            value, axis = args[0], args[1]
            begin = args[2] if len(args) > 2 else None
            end = args[3] if len(args) > 3 else None
            step = args[4] if len(args) > 4 else 1
            indices = [slice(None)] * value.raw.ndim
            indices[axis] = slice(begin, end, step)
            view = value.raw[tuple(indices)]
            # A slice consumed only as an in-place copy destination stays a view of its base.
            if node.users and all(str(user.target) == "aten.copy_.default" and user.args[0] is node for user in node.users):
                return Tensor(view, value.dtype, value.const)
            return Tensor(view.copy(), value.dtype, value.const)
        if target == "aten.select.int":
            return Tensor(np.take(args[0].raw, args[2], axis=args[1]).copy(), args[0].dtype, args[0].const)
        if target == "aten.split.Tensor":
            value, size = args[0], args[1]
            axis = args[2] if len(args) > 2 else 0
            extent = value.raw.shape[axis]
            pieces = []
            for begin in range(0, extent, size):
                indices = [slice(None)] * value.raw.ndim
                indices[axis] = slice(begin, begin + size)
                pieces.append(Tensor(value.raw[tuple(indices)].copy(), value.dtype, value.const))
            return pieces
        if target == "aten.cat.default":
            values, axis = args[0], (args[1] if len(args) > 1 else 0)
            dtypes = {v.dtype for v in values}
            if dtypes != {out}:
                if not all(v.const for v in values):
                    raise Refusal("runtime concat with mixed dtypes is unqualified")
                values = [self.convert(v, out) for v in values]
            self.count("layout:concat")
            return Tensor(np.concatenate([v.raw for v in values], axis=axis), out, all(v.const for v in values))
        if target == "aten.copy_.default":
            destination, source = args[:2]
            if destination.dtype != source.dtype or destination.raw.shape != np.broadcast_shapes(destination.raw.shape, source.raw.shape):
                raise Refusal("in-place copy requires same-dtype broadcastable data")
            if not destination.raw.flags.writeable:
                raise Refusal("in-place copy destination is not writable storage")
            np.copyto(destination.raw, source.raw)
            self.count("layout:copy_into_view")
            return destination
        if target == "aten.embedding.default":
            weight, indices = args[:2]
            if indices.dtype != "i64" or np.any(indices.raw < 0) or np.any(indices.raw >= weight.raw.shape[0]):
                raise Refusal("embedding indices leave the table")
            self.count("host_support:embedding_lookup")
            if indices.const:
                # Input-independent SigLIP position ids: the compiled application binds arange(extent).
                expected = np.arange(indices.raw.shape[-1], dtype=np.int64)
                equal = bool(np.array_equal(indices.raw.reshape(-1), expected))
                self.position_ids_check = {"node": node.name, "shape": list(indices.raw.shape), "equals_arange": equal,
                                           "sha256": info(indices.raw)["sha256"]}
                if not equal:
                    raise Refusal("computed constant position ids differ from the compiled arange binding")
            return Tensor(weight.raw[indices.raw].copy(), weight.dtype, weight.const and indices.const)
        if target == "aten.index.Tensor":
            value, indices = args
            if len(indices) == 1 and isinstance(indices[0], Tensor) and indices[0].dtype == "bool":
                result = value.raw[indices[0].raw]
            else:
                key = tuple(slice(None) if v is None else v.raw for v in indices)
                if any(isinstance(v, Tensor) and v.dtype != "i64" for v in indices):
                    raise Refusal("advanced index dtype is unqualified")
                result = value.raw[key]
            return Tensor(np.asarray(result).copy(), value.dtype, const)
        if target == "aten.index_put_.default":
            destination, indices, values = args[:3]
            if (len(args) > 3 and args[3]) or len(indices) != 1 or indices[0].dtype != "bool" or values.dtype != destination.dtype:
                raise Refusal("index_put_ requires one Boolean mask without accumulation")
            destination.raw[indices[0].raw] = values.raw
            destination.const = destination.const and const
            return destination
        if target == "aten.im2col.default":
            value, kernel, dilation, padding, stride = args
            if list(dilation) != [1, 1] or list(padding) != [0, 0] or list(stride) != list(kernel):
                raise Refusal("im2col is outside the stride-equals-kernel unpadded layout rewrite")
            b, c, h, w = value.raw.shape
            kh, kw = kernel
            oh, ow = h // kh, w // kw
            if oh * kh != h or ow * kw != w:
                raise Refusal("im2col kernel does not tile the image")
            raw = value.raw.reshape(b, c, oh, kh, ow, kw).transpose(0, 1, 3, 5, 2, 4).reshape(b, c * kh * kw, oh * ow).copy()
            self.count("rewrite:im2col_layout")
            return Tensor(raw, value.dtype, value.const)
        raise Refusal("original FX operation has no explicit selected independent arithmetic: " + target)

    # ------------------------------------------------------------ execution
    def live_nodes(self, nodes):
        """Output dependency closure, including in-place mutations of live bases."""
        mutators = {"aten.copy_.default", "aten.add_.Tensor", "aten.index_put_.default", "aten.detach_.default"}
        def base(node):
            while str(getattr(node, "target", "")) == "aten.slice.Tensor":
                node = node.args[0]
            return node
        live = {nodes[-1]}
        changed = True
        while changed:
            changed = False
            for node in reversed(nodes):
                if node in live:
                    for dependency in node.all_input_nodes:
                        if dependency not in live:
                            live.add(dependency)
                            changed = True
                elif node.op == "call_function" and str(node.target) in mutators and base(node.args[0]) in live:
                    live.add(node)
                    changed = True
                elif (node.op == "call_function" and str(node.target).startswith("aten._assert")
                      and all(d in live for d in node.all_input_nodes) and node.all_input_nodes):
                    live.add(node)
                    changed = True
        return live

    def save(self, node, value):
        if isinstance(value, Tensor):
            self.boundaries[node.name] = {**info(value.raw), "logical_dtype": value.dtype}
            target = str(node.target)
            residual = (target in {"aten.add.Tensor", "aten.add_.Tensor"} and value.dtype == "bf16"
                        and all(arg.op != "placeholder" for arg in node.all_input_nodes))
            wanted = (node.name in self.retain or "*" in self.retain
                      or ((target in RETAIN_TARGETS or residual) and value.raw.size <= RETAIN_LIMIT))
            if wanted:
                self.saved[node.name] = np.array(value.raw, order="C", copy=True)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, Tensor):
                    self.boundaries[f"{node.name}.{i}"] = {**info(item.raw), "logical_dtype": item.dtype}

    def run(self, inputs):
        nodes = list(self.program.graph.nodes)
        live = self.live_nodes(nodes)
        last = {}
        for index, node in enumerate(nodes):
            for dependency in node.all_input_nodes:
                last[dependency] = index
        user_inputs = iter(inputs)
        executed = Counter()
        self.skipped = sum(1 for node in nodes if node.op == "call_function" and node not in live)
        for index, node in enumerate(nodes):
            if node.op == "placeholder":
                spec = self.specs[node.target]
                kind = spec.kind.name
                if kind == "USER_INPUT":
                    value = next(user_inputs)
                elif kind in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR"}:
                    source = self.program.state_dict if spec.target in self.program.state_dict else self.program.constants
                    value = raw_tensor(source[spec.target], const=kind == "CONSTANT_TENSOR")
                    self.constants[spec.target] = {**info(value.raw), "logical_dtype": value.dtype, "kind": kind}
                else:
                    raise Refusal("original FX graph has an unsupported input role")
            elif node.op == "call_function":
                if node not in live:
                    continue
                try:
                    value = self.call(node)
                except (ValueError, ArithmeticError) as error:
                    stack = node.meta.get("nn_module_stack") or {}
                    raise Refusal(json.dumps({"node": node.name, "target": str(node.target),
                                              "module": list(stack.values())[-1][0] if stack else None,
                                              "message": str(error)})) from error
                executed[str(node.target)] += 1
            elif node.op == "output":
                result = self.resolve(node.args[0])
                self.executed = executed
                return result
            else:
                raise Refusal("original FX graph has an unsupported node kind")
            if isinstance(value, Tensor):
                expected = node.meta.get("val")
                dims = [d if type(d) is int else None for d in expected.shape]
                if (len(dims) != value.raw.ndim or any(d is not None and d != s for d, s in zip(dims, value.raw.shape))
                        or value.dtype != self.dtype(node)):
                    raise Refusal(f"result shape/dtype differs at {node.name}: {value.raw.shape} {value.dtype}")
                if value.raw.dtype != np.dtype(RAW[value.dtype]):
                    raise Refusal("result raw encoding differs at " + node.name)
            self.env[node] = value
            self.save(node, value)
            for dependency in node.all_input_nodes:
                if last.get(dependency) == index:
                    self.env.pop(dependency, None)
            if self.progress and node.op == "call_function" and (str(node.target).startswith("atlas.q")
                                                                 or str(node.target) in RETAIN_TARGETS):
                self.progress(node.name, str(node.target), round(time.monotonic() - self.started, 1))
        raise Refusal("original FX graph has no output")


def check_policy(policy):
    for field, expected in EXPECTED_POLICY.items():
        if policy.get(field) != expected:
            raise ValueError("selected precision policy differs for " + field)
    for field, members in REQUIRED_LISTS.items():
        if not members <= set(policy.get(field, ())):
            raise ValueError("selected host policy list is incomplete: " + field)
    unary = policy.get("host_unary_math", {})
    if unary.get("policy") != "selected-libm-rne-finite-v1" or set(unary.get("operations", ())) != UNARY_OPERATIONS:
        raise ValueError("selected host unary math policy differs")
    constant = policy.get("source_constant_math", {})
    if constant.get("policy") != "positive-power-correct-rne-mpfr-v1":
        raise ValueError("selected constant power policy differs")
    return constant


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fx-bundle", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True, help="atlas_ref package root (with native.c)")
    parser.add_argument("--validation", type=Path, required=True, help="ordinary-arithmetic layer validation receipt")
    parser.add_argument("--mask-validation", type=Path, required=True, help="rowspan80 softmax qualification receipt")
    parser.add_argument("--math-helper", type=Path, required=True)
    parser.add_argument("--engine", choices=("numpy", "native"), default="numpy")
    parser.add_argument("--native-qualification", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--retain", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostic-waive", action="append", default=[], choices=("softmax-span", "softmax-where-mask"),
                        help="DIAGNOSTIC ONLY: continue past a row-softmax domain refusal with the same tiled "
                             "composition (where-mask lanes as masked lanes); results are never expected words")
    args = parser.parse_args()
    started = time.monotonic()

    sys.path[:0] = [str(args.model_root.resolve()), str(ROTARY_REFERENCE), str(HERE)]
    import layer_reference, attention_reference, rowspan_softmax, rotary_exact  # noqa: E401  rotary-host modules
    sys.path.insert(0, str(args.reference_root.resolve()))
    import atlas_ref.activation, atlas_ref.arithmetic, atlas_ref.contractions  # noqa: E401
    import smolvla_primitives as sp
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for module, root in ((layer_reference, ROTARY_REFERENCE), (rowspan_softmax, ROTARY_REFERENCE),
                         (atlas_ref.arithmetic, args.reference_root), (sp, HERE)):
        if not Path(module.__file__).resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("reference module resolved outside its selected root: " + module.__name__)
    model_sources = layer_reference.authenticate_model(args.model_root)
    import npu_model.configs.numerics as numerics
    if Path(numerics.__file__).resolve() != args.model_root.resolve() / "npu_model/configs/numerics.py":
        raise ValueError("loaded instruction model differs from the selected root")
    validation = json.loads(args.validation.read_text())
    if (validation.get("status") != "PASS" or validation.get("mismatches") != 0
            or validation.get("adapter_sha256") != sha_file(layer_reference.__file__)
            or validation.get("attention_adapter_sha256") != sha_file(attention_reference.__file__)
            or validation.get("model_arithmetic_sources") != model_sources
            or validation.get("reference_sources") != attention_reference.reference_sources()):
        raise ValueError("ordinary arithmetic validation differs from the reused primitives")
    mask = json.loads(args.mask_validation.read_text())
    if (mask.get("status") != "PASS" or mask.get("mismatches") != 0 or mask.get("selected_policy") != sp.ROWSPAN80_POLICY
            or mask.get("oracle_sha256") != rowspan_softmax.source_hash() or mask.get("model_arithmetic_sources") != model_sources):
        raise ValueError("rowspan80 qualification differs from the reused softmax oracle")
    policy = json.loads(args.policy.read_text())
    constant_policy = check_policy(policy)
    helper = sp.math_helper(args.math_helper)
    if helper.identity != {key: constant_policy[key] for key in ("helper_sha256", "dependencies")}:
        raise ValueError("MPFR helper identity differs from the selected constant math policy")
    engine = layer_reference.configure_engine(args.engine, args.threads, args.native_qualification)

    fx = args.fx_bundle
    parity = json.loads((fx / "parity.json").read_text())
    case = parity["cases"][args.case]
    fixtures = read_safetensors(fx / "fixtures.safetensors", case["inputs"])
    inputs = [fixtures[key] for key in case["inputs"]]
    import atlas_fx_ops  # noqa: F401  registers the authored custom-op schemas
    program = torch.export.load(str(fx / "model.pt2"))
    interpreter = FlowLossInterpreter(program, policy, engine=args.engine, threads=args.threads, retain=args.retain,
                                      waivers=args.diagnostic_waive,
                                      progress=lambda name, target, elapsed: print(json.dumps(
                                          {"node": name, "target": target, "elapsed_s": elapsed}), flush=True))
    code = {name: sha_file(path) for name, path in {
        "flowloss_reference.py": __file__, "smolvla_primitives.py": sp.__file__,
        "rotary-host/layer_reference.py": layer_reference.__file__,
        "rotary-host/attention_reference.py": attention_reference.__file__,
        "rotary-host/rowspan_softmax.py": rowspan_softmax.__file__,
        "rotary-host/rotary_exact.py": rotary_exact.__file__,
        "atlas_ref/activation.py": atlas_ref.activation.__file__,
        "atlas_ref/arithmetic.py": atlas_ref.arithmetic.__file__,
        "atlas_ref/contractions.py": atlas_ref.contractions.__file__,
        "atlas_ref/native.py": Path(atlas_ref.arithmetic.__file__).with_name("native.py"),
        "atlas_ref/native.c": Path(atlas_ref.arithmetic.__file__).with_name("native.c"),
    }.items()}
    record = {"schema": SCHEMA, "status": "RUNNING", "case": args.case,
              "inputs": {key: {**info(fixtures[key].raw), "logical_dtype": fixtures[key].dtype} for key in case["inputs"]},
              "graph": {"model_pt2_sha256": sha_file(fx / "model.pt2"), "graph_txt_sha256": sha_file(fx / "graph.txt"),
                        "fixtures_sha256": sha_file(fx / "fixtures.safetensors"), "parity_sha256": sha_file(fx / "parity.json")},
              "policy": {"path": str(args.policy), "sha256": sha_file(args.policy),
                         "fields": {key: policy[key] for key in EXPECTED_POLICY}},
              "code_sha256": code, "model_arithmetic_sources": model_sources, "engine": engine,
              "receipts": {"validation_sha256": sha_file(args.validation), "mask_validation_sha256": sha_file(args.mask_validation),
                           "native_qualification_sha256": sha_file(args.native_qualification) if args.native_qualification else None},
              "math_helper": {"path": str(args.math_helper), **helper.identity},
              "independence": {"compiler_evaluator_or_emitted_program_inputs": False, "semantic_json_inputs": False,
                               "framework_numerical_kernels_for_compiled_arithmetic": False,
                               "dispatch": "authored FX ATen/custom-op targets, scalar arguments, dtype/shape metadata"}}
    args.out.mkdir(parents=True, exist_ok=False)
    status, refusal, outputs = "EXPECTED", None, None
    try:
        outputs = interpreter.run(inputs)
    except Refusal as error:
        status = "REFUSED"
        try:
            refusal = json.loads(str(error))
        except json.JSONDecodeError:
            refusal = {"message": str(error)}
    if interpreter.waived:
        status = "DIAGNOSTIC_WAIVED" if status == "EXPECTED" else "DIAGNOSTIC_REFUSED"
    arrays = dict(interpreter.saved)
    arrays.update({f"input_{i}": value.raw for i, value in enumerate(inputs)})
    if outputs is not None:
        outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
        arrays.update({f"output_{i}": value.raw for i, value in enumerate(outputs)})
        record["outputs"] = {f"output_{i}": {**info(value.raw), "logical_dtype": value.dtype} for i, value in enumerate(outputs)}
    np.savez(args.out / "expected.npz", **arrays)
    for name, raw in program.state_dict.items():
        if name in interpreter.constants and info(raw_tensor(raw).raw)["sha256"] != interpreter.constants[name]["sha256"]:
            raise ValueError("authored lifted state changed during interpretation: " + name)
    if layer_reference.authenticate_model(args.model_root) != model_sources:
        raise ValueError("instruction-model arithmetic changed during interpretation")
    record.update(status=status, refusal=refusal, wall_seconds=round(time.monotonic() - started, 3),
                  counts={"graph_call_nodes": sum(1 for n in program.graph.nodes if n.op == "call_function"),
                          "executed_by_target": dict(sorted(getattr(interpreter, "executed", Counter()).items())),
                          "dead_call_nodes_skipped": getattr(interpreter, "skipped", None),
                          "policy_dispatch": dict(sorted(interpreter.dispatch.items())),
                          "retained_arrays": len(arrays), "hashed_boundaries": len(interpreter.boundaries)},
                  expected_npz_sha256=sha_file(args.out / "expected.npz"),
                  lifted_constants=interpreter.constants, boundaries=interpreter.boundaries,
                  position_ids_check=interpreter.position_ids_check,
                  diagnostic_waivers={"selected": sorted(interpreter.waivers), "applied": interpreter.waived,
                                      "note": "waived results are not expected words of any selected compiled policy"},
                  device_exactness="PENDING separate compiled execution comparison")
    (args.out / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": status, "refusal": refusal, "outputs": record.get("outputs"), "wall_seconds": record["wall_seconds"]}))
    return 0 if status == "EXPECTED" else 3


if __name__ == "__main__":
    sys.exit(main())
