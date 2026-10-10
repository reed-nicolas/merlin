"""State routing and real LLVM frontend checks over a diagnostic two-stage session."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest

from merlin.common.paths import repo_root


def get_model_and_inputs(*, model2mlir_root):
    """Explicit local CLI fixture; this diagnostic contains no model checkpoint."""
    import torch
    from verify_session import load_protocol
    fixture = SessionTests()
    fixture.torch, fixture.protocol = torch, load_protocol(Path(model2mlir_root))
    session = fixture.session()
    return SimpleNamespace(external_runtime_session=lambda: session), ()


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        root = os.environ.get("TVM_SESSION_MODEL2MLIR")
        if not root:
            raise unittest.SkipTest("Set TVM_SESSION_MODEL2MLIR to the existing model2MLIR protocol checkout")
        sys.path.insert(0, root)
        path = repo_root() / "examples/gemmini/comparisons/tvm/verify_session.py"
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("tvm_session_verifier", path)
        cls.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verifier)
        cls.protocol = cls.verifier.load_protocol(Path(root))

    def session(self):
        torch, protocol = self.torch, self.protocol

        class Prefix(torch.nn.Module):
            def forward(self, image):
                return image + 1

        class Decode(torch.nn.Module):
            def forward(self, token, state):
                return state + token * 2, state + token

        tokens = torch.tensor([[1., -2.], [-3., 4.], [2., 1.]], dtype=torch.float32)
        spec = {"steps": 3, "streams": [{"name": "token", "input_index": 0, "values": tokens}],
                "states": [{"name": "state", "input_index": 1, "output_index": 1}], "quality": {"output_index": 0}}
        return protocol.make_external_runtime_session(
            version=2,
            programs=(protocol.ExternalRuntimeProgram("prefill", Prefix().eval(), (torch.tensor([0., 1.]),), 1),
                      protocol.ExternalRuntimeProgram("decode", Decode().eval(), (tokens[0], torch.zeros(2)), 3, spec)),
            metadata={"kind": "diagnostic_recurrence", "paper_ready": False, "stages": ["prefill", "decode"],
                      "stage_schedule": [{"name": "prefill", "steps": 1}, {"name": "decode", "steps": 3}],
                      "quality_program": "decode", "bindings": [{"name": "initial_state", "from": {"program": "prefill", "output_index": 0},
                                                                 "to": {"program": "decode", "input_index": 1}}],
                      "provenance": {"checkpoint": "randomless_toy", "full_checkpoint": False, "synthetic_inputs": True}})

    def reference_compiler(self, program, inputs, output_count):
        def run(values):
            return program.module(*(self.verifier.torch_array(value, self.np, self.torch) for value in values))
        return run

    def test_exporter_and_schema_selection_are_explicit(self):
        import onnx
        compiler = self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None)
        self.assertEqual(compiler.opset, 17)
        compiler = self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, exporter="dynamo")
        self.assertEqual(compiler.opset, 18)
        compiler = self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, opset=22)
        self.assertEqual(compiler.opset, 22)
        for value in (0, True, "22", onnx.defs.onnx_opset_version() + 1):
            with self.subTest(opset=value), self.assertRaisesRegex(ValueError, "opset"):
                self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, opset=value)
        with self.assertRaisesRegex(ValueError, "exporter"):
            self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, exporter="unknown")
        for options in ({}, {"exporter": "dynamo", "opset": 17}):
            with self.assertRaisesRegex(ValueError, "Fused BF16 export"):
                self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, bf16_fused_export=True, **options)
        self.assertEqual(compiler.export_options, {})
        self.assertIsNone(compiler.translation_source)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_bf16_export_preserves_bias_boundary_and_attention_dtypes(self):
        import onnx
        import tvm
        torch, np = self.torch, self.np

        class Fused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                weight = torch.zeros((3, 17), dtype=torch.bfloat16)
                weight[:, :2] = 1
                self.register_buffer("weight", weight)
                self.register_buffer("bias", torch.full((3,), 1 / 256, dtype=torch.bfloat16))

            def forward(self, image, q, k, v):
                return torch.nn.functional.linear(image, self.weight, self.bias), torch.nn.functional.scaled_dot_product_attention(q, k, v)

        image = torch.zeros((1, 2, 17), dtype=torch.bfloat16)
        image[:, :, 0], image[:, :, 1] = 1, 1 / 256
        q, k = torch.zeros((1, 2, 17, 8), dtype=torch.bfloat16), torch.zeros((1, 2, 17, 8), dtype=torch.bfloat16)
        v = torch.arange(17, dtype=torch.bfloat16).reshape(1, 1, 17, 1).expand(1, 2, 17, 8).contiguous()
        values, module = (image, q, k, v), Fused().eval()
        with torch.no_grad():
            expected = module(*values)
            split = (image @ module.weight.t()) + module.bias
        self.assertFalse(torch.equal(split, expected[0]))
        inputs = tuple(self.verifier.array(value, np, torch) for value in values)
        program = SimpleNamespace(name="fused_bf16", module=module)
        original_weights = self.verifier.parameter_digest(module, torch)
        with tempfile.TemporaryDirectory(prefix="tvm-session-fused-bf16-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "baseline", np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
            actual = self.verifier.outputs(compiler(program, inputs, 2)(inputs), np, torch)
            for value, reference in zip(actual, expected):
                self.verifier.compare(value, self.verifier.array(reference, np, torch), np, 0, 0)
            graph = onnx.shape_inference.infer_shapes(onnx.load(str(Path(temp) / "stage_0/model.onnx")))
            types = {value.name: value.type.tensor_type.elem_type for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
            for node in graph.graph.node:
                if node.op_type in ("MatMul", "Add", "Mul", "Softmax"):
                    self.assertTrue(all(types[name] == onnx.TensorProto.FLOAT for name in node.output))
            self.assertTrue(all(value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16 for value in graph.graph.input))
            self.assertTrue(all(value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16 for value in graph.graph.output))
            parameters = {value.name: value for value in graph.graph.initializer if value.name in ("weight", "bias")}
            self.assertEqual(set(parameters), {"weight", "bias"})
            self.assertTrue(all(value.data_type == onnx.TensorProto.BFLOAT16 for value in parameters.values()))
            self.assertEqual(compiler.records[0]["translation_source"], compiler.translation_source)
            self.assertEqual(compiler.records[0]["translated_aten_operations"], {"aten.linear.default": 1, "aten.scaled_dot_product_attention.default": 1})
            self.assertEqual(self.verifier.parameter_digest(module, torch), original_weights)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_bf16_attention_masks_causal_and_scale(self):
        import math
        import onnx
        import tvm
        torch, np = self.torch, self.np

        class MaskedAttention(torch.nn.Module):
            def __init__(self, mask=None, causal=False, scale=None):
                super().__init__()
                self.register_buffer("mask", mask)
                self.causal, self.scale = causal, scale

            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.mask, is_causal=self.causal, scale=self.scale)

        mask = torch.zeros((3, 5), dtype=torch.bool)
        mask[1:, -1] = True
        additive = torch.where(mask, 0., -torch.inf)
        q, k = torch.zeros((1, 1, 3, 1), dtype=torch.bfloat16), torch.zeros((1, 1, 5, 1), dtype=torch.bfloat16)
        v = torch.arange(0, 10, 2, dtype=torch.bfloat16).reshape(1, 1, 5, 1)
        masked_output = torch.tensor([0., 8., 8.], dtype=torch.bfloat16).reshape(1, 1, 3, 1)
        causal_output = torch.tensor([0., 1., 2.], dtype=torch.bfloat16).reshape(1, 1, 3, 1)
        scaled_values = (torch.ones((1, 1, 3, 1), dtype=torch.bfloat16), torch.tensor([-1., 1.], dtype=torch.bfloat16).reshape(1, 1, 2, 1), torch.tensor([0., 4.], dtype=torch.bfloat16).reshape(1, 1, 2, 1))
        cases = [("bool", MaskedAttention(mask), (q, k, v), masked_output), ("bf16-mask", MaskedAttention(additive.bfloat16()), (q, k, v), masked_output), ("fp32-mask", MaskedAttention(additive), (q, k, v), masked_output), ("rectangular-causal", MaskedAttention(causal=True), (q, k, v), causal_output), ("explicit-scale", MaskedAttention(scale=math.log(3) / 2), scaled_values, torch.full_like(q, 3))]
        for name, module, values, known_output in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix="tvm-session-bf16-mask-") as temp:
                with torch.no_grad():
                    reference = module(*values)
                self.assertTrue(torch.equal(reference, known_output))
                inputs = tuple(self.verifier.array(value, np, torch) for value in values)
                compiler = self.verifier.StageCompiler(Path(temp), "baseline", np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
                actual = self.verifier.array(compiler(SimpleNamespace(name=name, module=module), inputs, 1)(inputs), np, torch)
                self.verifier.compare(actual, self.verifier.array(reference, np, torch), np, 0, 0)
        malformed = MaskedAttention(torch.full((3, 5), torch.nan))
        with tempfile.TemporaryDirectory(prefix="tvm-session-bf16-nan-mask-") as temp:
            inputs = tuple(self.verifier.array(value, np, torch) for value in (q, k, v))
            compiler = self.verifier.StageCompiler(Path(temp), "baseline", np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
            result = compiler(SimpleNamespace(name="nan-mask", module=malformed), inputs, 1)(inputs)
            actual = self.verifier.array(result, np, torch, require_finite=False)
            self.assertTrue(np.isnan(actual).all())
            with self.assertRaisesRegex(ValueError, "finite"):
                self.verifier.array(result, np, torch)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_importer_literal_and_cast_like_refusals(self):
        import tvm
        from tvm.relax.frontend.onnx.onnx_frontend import CastLike, Constant
        for attributes in ({}, {"value_int": 1, "value_float": 1.}, {"value_string": b"unsupported"}):
            with self.subTest(attributes=attributes), self.assertRaisesRegex(ValueError, "exactly one supported"):
                Constant._impl_v13(None, [], attributes, None)
        target = tvm.relax.Var("target", tvm.relax.TensorStructInfo([1], dtype=""))
        with self.assertRaisesRegex(ValueError, "known target dtype"):
            CastLike._impl_v15(None, [None, target], {}, None)
        with self.assertRaisesRegex(ValueError, "attributes"):
            CastLike._impl_v15(None, [None, target], {"round_mode": b"unknown"}, None)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_export_keeps_fp32_builtin_semantics(self):
        import onnx
        import tvm
        torch, np = self.torch, self.np

        class FloatOperators(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, q, k, v):
                return self.linear(q), torch.nn.functional.scaled_dot_product_attention(q, k, v)

        with torch.random.fork_rng():
            torch.manual_seed(151)
            module = FloatOperators().eval()
            values = tuple(torch.randn((1, 2, 17, 8)) for _ in range(3))
        with torch.no_grad():
            expected = module(*values)
        inputs = tuple(self.verifier.array(value, np, torch) for value in values)
        with tempfile.TemporaryDirectory(prefix="tvm-session-fused-fp32-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "baseline", np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
            actual = self.verifier.outputs(compiler(SimpleNamespace(name="fp32", module=module), inputs, 2)(inputs), np, torch)
            for value, reference in zip(actual, expected):
                self.verifier.compare(value, self.verifier.array(reference, np, torch), np, 1e-4, 1e-4)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_bf16_export_refuses_unverified_contracts(self):
        from onnxscript import ir
        helper = self.verifier.load_module("fused_bf16_test_helper", repo_root() / "examples/gemmini/comparisons/tvm/bf16_export.py")
        translations = helper.translations(self.torch)
        bf16 = SimpleNamespace(dtype=ir.DataType.BFLOAT16, shape=(1, 2, 17, 8))
        fp32 = SimpleNamespace(dtype=ir.DataType.FLOAT, shape=(1, 2, 17, 8))
        linear = translations[self.torch.ops.aten.linear.default]
        attention = translations[self.torch.ops.aten.scaled_dot_product_attention.default]
        with self.assertRaisesRegex(ValueError, "matching operand"):
            linear(bf16, fp32)
        for options in ({"dropout_p": 0.1}, {"enable_gqa": True}, {"is_causal": True, "attn_mask": bf16}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                attention(bf16, bf16, bf16, **options)
        with self.assertRaisesRegex(ValueError, "matching Q/K/V"):
            attention(bf16, fp32, bf16)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_modern_export_preserves_source_cache_across_repeats(self):
        import onnx
        import tvm
        torch = self.torch

        class CachedPrefix(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cache = SimpleNamespace(length=torch.zeros((), dtype=torch.int64))

            def forward(self, value):
                self.cache.length = torch.zeros_like(self.cache.length)
                self.cache.length += value.numel()
                return value + self.cache.length.to(value.dtype)

        session = self.session()
        prefix = CachedPrefix().eval()
        session = replace(session, programs=(replace(session.programs[0], module=prefix), session.programs[1]))
        with tempfile.TemporaryDirectory(prefix="tvm-session-modern-cache-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, torch, onnx, tvm, exporter="dynamo", opset=22)
            result = self.verifier.verify_session(session, compiler, self.np, torch)
            self.assertTrue(result["reset_verified"])
            self.assertEqual(len(result["checks"]), 8)
            self.assertEqual(type(prefix.cache.length), torch.Tensor)
            self.assertEqual(prefix.cache.length.numpy().item(), 2)
            self.assertTrue(all(record["opset"] == 22 and record["exporter"] == "dynamo" for record in compiler.records))

    def growing_session(self):
        torch, protocol = self.torch, self.protocol

        class Grow(torch.nn.Module):
            def forward(self, token, cache):
                updated = torch.cat((cache, token))
                return updated.sum().reshape(1), updated

        tokens = torch.tensor([[1.], [-2.], [3.]], dtype=torch.float32)
        spec = {"steps": 3, "streams": [{"name": "token", "input_index": 0, "values": tokens}],
                "states": [{"name": "cache", "input_index": 1, "output_index": 1}], "quality": {"output_index": 0}}
        return protocol.make_external_runtime_session(
            version=1, programs=(protocol.ExternalRuntimeProgram("grow", Grow().eval(), (tokens[0], torch.tensor([10.])), 3, spec),),
            metadata={"kind": "diagnostic_growing_state", "paper_ready": False})

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_changing_state_extents_have_distinct_static_compilations(self):
        import onnx
        import tvm
        with tempfile.TemporaryDirectory(prefix="tvm-session-grow-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, self.torch, onnx, tvm)
            result = self.verifier.verify_session(self.growing_session(), compiler, self.np, self.torch)
            self.assertEqual(result["compilation_signatures"], 3)
            self.assertEqual([record["inputs"][1]["shape"] for record in compiler.records], [[1], [2], [3]])
            self.assertEqual(len(result["checks"]), 6)
            self.assertTrue(result["reset_verified"])
            for row, wanted in zip(result["checks"][:3], (11., 9., 12.)):
                expected = self.np.array([wanted], dtype="float32")
                self.assertEqual(row["outputs"][0]["actual"], self.verifier.tensor_identity(expected))

    def test_streams_routes_reset_and_repeated_outputs(self):
        result = self.verifier.verify_session(self.session(), self.reference_compiler, self.np, self.torch)
        self.assertTrue(result["reset_verified"])
        self.assertEqual(result["compilation_signatures"], 2)
        self.assertEqual(len(result["checks"]), 8)
        rows = [row for row in result["checks"] if row["program"] == "decode" and row["repeat"] == 0]
        wanted = ([3., -2.], [-4., 8.], [3., 6.])
        for row, values in zip(rows, wanted):
            expected = self.np.array(values, dtype="float32")
            self.assertEqual(row["outputs"][0]["actual"], self.verifier.tensor_identity(expected))
            self.assertEqual(len(row["routed_states"]), 1)
        self.assertFalse(result["session"]["metadata"]["paper_ready"])

    def test_wrong_output_and_wrong_route_fail(self):
        def wrong(program, inputs, output_count):
            run = self.reference_compiler(program, inputs, output_count)
            return lambda values: tuple(value + 1 for value in run(values)) if program.name == "decode" else run(values)
        with self.assertRaisesRegex(AssertionError, "numerical comparison"):
            self.verifier.verify_session(self.session(), wrong, self.np, self.torch)
        session = self.session()
        route = session.routes[-1]
        corrupted = replace(route, source=replace(route.source, output_index=99))
        session = replace(session, routes=(*session.routes[:-1], corrupted))
        with self.assertRaisesRegex(ValueError, "absent output"):
            self.verifier.verify_session(session, self.reference_compiler, self.np, self.torch)

    def test_biased_stage_has_structured_failure_and_failed_receipt(self):
        from unittest.mock import patch
        def biased(program, inputs, output_count):
            run = self.reference_compiler(program, inputs, output_count)
            return lambda values: tuple(value + 0.5 for value in run(values)) if program.name == "decode" else run(values)
        with self.assertRaisesRegex(AssertionError, "numerical comparison") as caught:
            self.verifier.verify_session(self.session(), biased, self.np, self.torch)
        failure = caught.exception.diagnostics
        self.assertEqual({key: failure[key] for key in ("stage", "program", "repeat", "observation", "tensor", "output_index")},
                         {"stage": "per_observation", "program": "decode", "repeat": 0, "observation": 0, "tensor": "output", "output_index": 0})
        self.assertEqual(failure["max_absolute_error"], 0.5)
        self.assertEqual(failure["mismatch_count"], 2)
        self.assertEqual((failure["rtol"], failure["atol"]), (1e-4, 1e-4))
        self.assertEqual(failure["actual"]["shape"], [2])
        self.assertEqual(failure["actual"]["dtype"], "float32")
        self.assertIn("sha256", failure["actual"]["identity"])
        json.dumps(failure, allow_nan=False)
        with tempfile.TemporaryDirectory(prefix="tvm-session-failure-") as temp:
            folder = Path(temp) / "receipt"
            argv = ["verify_session.py", "--output-dir", str(folder)]
            for name in ("model2mlir-root", "tvm-source", "tvm-build", "factory"):
                argv.extend(("--" + name, str(Path(temp) / name)))
            with patch.object(sys, "argv", argv), patch.object(self.verifier, "file_identity", side_effect=caught.exception):
                self.assertEqual(self.verifier.main(), 1)
            receipt = json.loads((folder / "results.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(receipt["host_session_verified"])
            self.assertEqual(receipt["failure_diagnostics"], failure)
            self.assertIn("SessionComparisonError", receipt["traceback"])
            self.assertEqual([path.name for path in folder.iterdir()], ["results.json"])

    def test_invalid_tensor_diagnostics_do_not_claim_numerical_metrics(self):
        np = self.np
        reference = np.ones(2, dtype="float32")
        for actual in (np.ones(3, dtype="float32"), np.ones(2, dtype="float64"), np.array([np.nan, np.inf], dtype="float32")):
            with self.subTest(dtype=actual.dtype, shape=actual.shape), self.assertRaisesRegex(AssertionError, "shape/dtype/finite") as caught:
                self.verifier.compare(actual, reference, np, 1e-4, 1e-4)
            failure = caught.exception.diagnostics
            self.assertIsNone(failure["max_absolute_error"])
            self.assertIsNone(failure["mismatch_count"])
            json.dumps(failure, allow_nan=False)
        self.assertIsNone(failure["actual"]["identity"])
        def nonfinite(program, inputs, output_count):
            return lambda values: self.torch.full((2,), float("nan"))
        with self.assertRaisesRegex(AssertionError, "finite") as caught:
            self.verifier.verify_session(self.session(), nonfinite, np, self.torch)
        self.assertEqual(caught.exception.diagnostics["program"], "prefill")
        self.assertEqual(caught.exception.diagnostics["output_index"], 0)
        self.assertFalse(caught.exception.diagnostics["actual"]["finite"])
        json.dumps(caught.exception.diagnostics, allow_nan=False)

    def test_difference_metrics_preserve_large_integers_and_handle_overflow(self):
        np = self.np
        for actual, expected, delta in ((np.array([2 ** 63 - 1], dtype="int64"), np.array([-2 ** 63], dtype="int64"), 2 ** 64 - 1),
                                        (np.array([np.finfo("float64").max]), np.array([-np.finfo("float64").max]), None)):
            with self.assertRaisesRegex(AssertionError, "numerical comparison") as caught:
                self.verifier.compare(actual, expected, np, 0, 0)
            failure = caught.exception.diagnostics
            self.assertEqual(failure["mismatch_count"], 1)
            self.assertEqual(failure["max_absolute_error"], delta)
            json.dumps(failure, allow_nan=False)

    def test_unknown_cadence_route_update_and_dtype_are_rejected(self):
        session = self.session()
        for invocation in (replace(session.execution_schedule[0], cadence="unknown"),):
            malformed = replace(session, execution_schedule=(invocation, session.execution_schedule[1]))
            with self.assertRaisesRegex(ValueError, "cadence"):
                self.verifier.verify_session(malformed, self.reference_compiler, self.np, self.torch)
        malformed = replace(session, routes=(replace(session.routes[0], update="after_session"), *session.routes[1:]))
        with self.assertRaisesRegex(ValueError, "state route"):
            self.verifier.verify_session(malformed, self.reference_compiler, self.np, self.torch)
        malformed = replace(session, execution_schedule=(replace(session.execution_schedule[0], cadence="per_observation", repeats=3),
                            replace(session.execution_schedule[1], cadence="once_after_observations")))
        with self.assertRaisesRegex(ValueError, "per-observation stage"):
            self.verifier.verify_session(malformed, self.reference_compiler, self.np, self.torch)
        with self.assertRaisesRegex(AssertionError, "dtype"):
            self.verifier.compare(self.np.ones(2, dtype="int32"), self.np.ones(2, dtype="int64"), self.np, 1e-4, 1e-4)
        with self.assertRaisesRegex(ValueError, "Unsupported session dtype"):
            self.verifier.array(self.torch.ones(2, dtype=self.torch.complex64), self.np, self.torch)

    @unittest.skipUnless(importlib.util.find_spec("ml_dtypes"), "Original BF16 verification needs ml_dtypes")
    def test_bfloat16_torch_numpy_tvm_round_trip_preserves_bits(self):
        words = self.np.array([0, 0x8000, 0x3F81, 0xBFC1, 0x0080, 0x7F7F], dtype="uint16")
        tensor = self.torch.from_numpy(words.copy()).view(self.torch.bfloat16)
        value = self.verifier.array(tensor, self.np, self.torch)
        self.assertEqual(value.dtype.name, "bfloat16")
        self.np.testing.assert_array_equal(value.view(self.np.uint16), words)
        restored = self.verifier.torch_array(value, self.np, self.torch)
        self.assertTrue(self.torch.equal(restored.view(self.torch.uint16), tensor.view(self.torch.uint16)))
        self.assertEqual(self.verifier.tensor_identity(value)["bytes"], 12)
        if os.environ.get("TVM_LIBRARY_PATH"):
            import tvm
            candidate = self.verifier.outputs(tvm.nd.array(value), self.np, self.torch)[0]
            self.np.testing.assert_array_equal(candidate.view(self.np.uint16), words)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.verifier.array(self.torch.tensor([float("nan")], dtype=self.torch.bfloat16), self.np, self.torch)

    @unittest.skipUnless(importlib.util.find_spec("ml_dtypes") and os.environ.get("TVM_LIBRARY_PATH"), "Needs BF16 dependency and TVM")
    def test_original_bfloat16_session_compiles_and_routes_without_casts(self):
        import onnx
        import tvm
        session = self.session()
        bf16 = self.torch.bfloat16
        session = replace(session,
            programs=tuple(replace(program, inputs=tuple(value.to(bf16) for value in program.inputs)) for program in session.programs),
            input_bindings=tuple(replace(binding, initial=binding.initial.to(bf16)) for binding in session.input_bindings),
            streams=tuple(replace(stream, values=stream.values.to(bf16)) for stream in session.streams))
        with tempfile.TemporaryDirectory(prefix="tvm-session-bfloat16-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, self.torch, onnx, tvm)
            result = self.verifier.verify_session(session, compiler, self.np, self.torch, rtol=0, atol=0)
        self.assertEqual(len(result["checks"]), 8)
        self.assertTrue(result["reset_verified"])
        self.assertTrue(all(row["outputs"][0]["actual"]["dtype"] == "bfloat16" for row in result["checks"]))

    def test_network_and_signature_budget_are_enforced(self):
        with self.verifier.offline(), self.assertRaisesRegex(RuntimeError, "network"):
            socket.create_connection(("127.0.0.1", 9))
        with self.assertRaisesRegex(ValueError, "signature budget"):
            self.verifier.verify_session(self.session(), self.reference_compiler, self.np, self.torch, max_signatures=1)

    def external_graph(self, folder):
        import onnx
        value = self.np.array([[1., -2.], [3., 4.]], dtype="float32")
        weight = onnx.numpy_helper.from_array(value, name="weight")
        graph = onnx.helper.make_graph([onnx.helper.make_node("Add", ["input", "weight"], ["output"])], "external",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])], [weight])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        path = folder / "model.onnx"
        onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
        return path, value

    def test_external_weights_are_loaded_and_content_bound(self):
        import onnx
        with tempfile.TemporaryDirectory(prefix="tvm-session-external-") as temp:
            path, value = self.external_graph(Path(temp))
            graph, identities = self.verifier.load_onnx_graph(path, onnx)
            self.assertEqual(len(identities), 2)
            self.assertEqual({Path(item["path"]).name for item in identities}, {"model.onnx", "weights.bin"})
            self.np.testing.assert_array_equal(onnx.numpy_helper.to_array(graph.graph.initializer[0]), value)
            self.assertEqual(graph.graph.initializer[0].data_location, onnx.TensorProto.DEFAULT)

    def test_external_paths_metadata_and_ranges_are_checked_before_loading(self):
        import onnx
        with tempfile.TemporaryDirectory(prefix="tvm-session-external-") as temp:
            folder = Path(temp)
            path, _ = self.external_graph(folder)
            original = onnx.load(str(path), load_external_data=False)
            for location in ("../outside.bin", "/outside.bin"):
                malformed = onnx.ModelProto()
                malformed.CopyFrom(original)
                malformed.graph.initializer[0].external_data[0].value = location
                path.write_bytes(malformed.SerializeToString())
                with self.assertRaisesRegex(ValueError, "stage directory"):
                    self.verifier.load_onnx_graph(path, onnx)
            for key, value, expected in (("offset", "-1", "nonnegative"), ("length", "17", "range exceeds"), ("basepath", "elsewhere", "metadata")):
                malformed = onnx.ModelProto()
                malformed.CopyFrom(original)
                entries = malformed.graph.initializer[0].external_data
                matches = [entry for entry in entries if entry.key == key]
                entry = matches[0] if matches else entries.add()
                entry.key, entry.value = key, value
                path.write_bytes(malformed.SerializeToString())
                with self.assertRaisesRegex(ValueError, expected):
                    self.verifier.load_onnx_graph(path, onnx)
            malformed = onnx.ModelProto()
            malformed.CopyFrom(original)
            entry = malformed.graph.initializer[0].external_data.add()
            entry.key, entry.value = "location", "weights.bin"
            path.write_bytes(malformed.SerializeToString())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.verifier.load_onnx_graph(path, onnx)

    def test_external_weight_changes_during_check_are_rejected(self):
        import onnx
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix="tvm-session-external-") as temp:
            folder = Path(temp)
            path, _ = self.external_graph(folder)
            check = onnx.checker.check_model
            def mutate(*args, **kwargs):
                check(*args, **kwargs)
                weights = folder / "weights.bin"
                weights.write_bytes(bytes([weights.read_bytes()[0] ^ 1]) + weights.read_bytes()[1:])
            with patch.object(onnx.checker, "check_model", mutate), self.assertRaisesRegex(AssertionError, "changed"):
                self.verifier.load_onnx_graph(path, onnx)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_real_onnx_relax_pipeline_verifies_every_stage(self):
        import onnx
        import tvm
        with tempfile.TemporaryDirectory(prefix="tvm-session-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, self.torch, onnx, tvm)
            result = self.verifier.verify_session(self.session(), compiler, self.np, self.torch)
            self.assertTrue(result["reset_verified"])
            self.assertEqual(len(compiler.records), 2)
            self.assertEqual({record["program"] for record in compiler.records}, {"prefill", "decode"})
            self.assertEqual(len(result["checks"]), 8)
            for record in compiler.records:
                self.assertEqual(record["pipelines"], ["zero", "default_build"])
                self.assertIn("tir", record["vm"]["inventory"]["function_counts"])


if __name__ == "__main__":
    unittest.main()
