"""State routing and real LLVM frontend checks over a diagnostic two-stage session."""
from dataclasses import replace
import importlib.util
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
