"""Fail-closed artifact and workload gates; no checkpoint substitute is executed."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from merlin.common.paths import repo_root


SOURCE = repo_root() / "examples/gemmini/comparisons/tvm/tinyllama_session.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("tinyllama_session_tested", SOURCE)
FACTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FACTORY)


class TinyLlamaArtifactTests(unittest.TestCase):
    def write_artifacts(self, root, config=None):
        for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json"):
            (root / name).write_text("fixture only; never a model")
        (root / "config.json").write_text(json.dumps(
            config or {**FACTORY.MODEL_SEMANTICS, "torch_dtype": "bfloat16"}))

    def pinned_identity(self, path, validator):
        path = validator(path)
        return {"path": str(path), "sha256": FACTORY.WEIGHT_SHA256}

    def test_absent_checkpoint_fails_before_framework_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                FACTORY.checkpoint_artifacts(directory)

    def test_nonofficial_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root)
            with self.assertRaisesRegex(ValueError, "pinned official full checkpoint"):
                FACTORY.checkpoint_artifacts(root)

    def test_truncated_geometry_is_rejected_after_identity_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root, {**FACTORY.MODEL_SEMANTICS, "num_hidden_layers": 2,
                                        "torch_dtype": "bfloat16"})
            with mock.patch.object(FACTORY, "file_identity", self.pinned_identity):
                with self.assertRaisesRegex(ValueError, "complete pinned model geometry"):
                    FACTORY.checkpoint_artifacts(root)

    def test_precision_and_recurrence_require_explicit_valid_choices(self):
        base = {"checkpoint_dir": "/missing", "model2mlir_root": "/missing",
                "corpus_path": "/missing"}
        for changes in ({"precision": "float16"}, {"prefill_tokens": 0},
                        {"decode_tokens": 1}, {"cpu_threads": 32},
                        {"prefill_tokens": 2048, "decode_tokens": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FACTORY.get_model_and_inputs(**{**base, **changes})

    def test_quantization_and_language_workload_fail_closed(self):
        base = {"checkpoint_dir": "/missing", "model2mlir_root": "/missing",
                "corpus_path": "/missing"}
        for changes in ({"quantized": 1}, {"quantized": True, "precision": "bfloat16"},
                        {"quality_corpus_path": "/missing"}, {"workload": "other"},
                        {"workload": "development_language"}, {"workload": "synthetic_feasibility"},
                        {"activation_scale": "other"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FACTORY.get_model_and_inputs(**{**base, **changes})

    def test_restricted_artifact_path_is_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "Restricted path component"):
            FACTORY.checkpoint_artifacts("/tmp/" + "ham" + "mer" + "-fixture")


try:
    import torch
    import numpy as np
    import tinyllama_integer as INTEGER
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch environment is absent")
class TinyLlamaIntegerTests(unittest.TestCase):
    def test_actual_integer_projection_matches_independent_numpy(self):
        generator = np.random.default_rng(194)
        weight = generator.uniform(-2, 2, (19, 33)).astype(np.float32)
        value = generator.uniform(-4, 4, (2, 3, 33)).astype(np.float32)
        projection = INTEGER.IntegerProjection(torch.from_numpy(weight))
        actual = projection(torch.from_numpy(value)).numpy()
        q_weight, scale_weight = INTEGER.independent_quantize_rows(weight)
        q_value, scale_value = INTEGER.independent_quantize_rows(value)
        accumulator = q_value.astype(np.int64) @ q_weight.T.astype(np.int64)
        expected = accumulator.astype(np.float32) * scale_value * scale_weight.T
        np.testing.assert_array_equal(projection.weight_kn.numpy(), q_weight.T)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(INTEGER.check_projection(projection, torch.from_numpy(value))["passed"])
        self.assertEqual(projection.weight_kn.dtype, torch.int8)
        self.assertEqual(torch._int_mm(torch.from_numpy(q_value.reshape(-1, 33)),
                                      projection.weight_kn).dtype, torch.int32)

    def test_zero_subnormal_and_ties_have_explicit_semantics(self):
        value = torch.tensor([[0., 0., 0., 0.], [127., .5, 1.5, -2.5],
                              [1e-44, -1e-44, 0., 0.]])
        quantized, scale = INTEGER.quantize_rows(value)
        np_quantized, np_scale = INTEGER.independent_quantize_rows(value.numpy())
        np.testing.assert_array_equal(quantized.numpy(), np_quantized)
        np.testing.assert_array_equal(scale.numpy(), np_scale)
        self.assertEqual(scale[0].item(), 1.)
        self.assertEqual(quantized[1].tolist(), [127, 0, 2, -2])
        self.assertTrue(bool(torch.isfinite(scale).all()))
        self.assertEqual(quantized[2].tolist(), [0, 0, 0, 0])

    def test_power_of_two_scales_match_exact_frexp_oracle_at_dyadic_boundaries(self):
        values = []
        for exponent in (-100, -20, 0, 20, 100):
            exact = np.float32(2. ** exponent)
            for raw in (np.nextafter(exact, np.float32(0)), exact,
                        np.nextafter(exact, np.float32(np.inf))):
                maximum = np.float32(raw * np.float32(127))
                values.append([maximum, maximum * np.float32(.5), -maximum * np.float32(.5)])
        values.extend([[0., 0., 0.], [1e-44, -1e-44, 0.], [254., 1., 3.]])
        value = torch.from_numpy(np.asarray(values, dtype=np.float32))
        quantized, scale = INTEGER.quantize_rows(value, "power_of_two")
        expected_q, expected_scale = INTEGER.independent_quantize_rows(value.numpy(), "power_of_two")
        np.testing.assert_array_equal(scale.numpy(), expected_scale)
        np.testing.assert_array_equal(quantized.numpy(), expected_q)
        self.assertEqual(quantized[-1].tolist(), [127, 0, 2])
        self.assertEqual(scale[-1].item(), 2.)
        self.assertEqual(scale[-3].item(), 1.)
        self.assertTrue(bool(torch.isfinite(scale).all()))
        self.assertEqual(INTEGER.recipe()["name"], "tinyllama_w8a8_dynamic_token_v1")
        self.assertNotEqual(INTEGER.recipe("power_of_two")["name"], INTEGER.recipe()["name"])

    def test_power_of_two_projection_matches_complete_integer_and_float_oracles(self):
        generator = np.random.default_rng(771)
        weight = generator.uniform(-2, 2, (19, 33)).astype(np.float32)
        value = generator.uniform(-4, 4, (2, 3, 33)).astype(np.float32)
        projection = INTEGER.IntegerProjection(torch.from_numpy(weight), activation_scale="power_of_two")
        q_weight, scale_weight = INTEGER.independent_quantize_rows(weight)
        q_value, scale_value = INTEGER.independent_quantize_rows(value, "power_of_two")
        expected = (q_value.astype(np.int64) @ q_weight.T.astype(np.int64)).astype(np.float32) * scale_value * scale_weight.T
        np.testing.assert_array_equal(projection(torch.from_numpy(value)).numpy(), expected)
        self.assertTrue(INTEGER.check_projection(projection, torch.from_numpy(value))["passed"])

    def test_vector_projection_is_preserved(self):
        projection = INTEGER.IntegerProjection(torch.tensor([[1., -2., 0.], [-1., 0., 3.]]))
        output = projection(torch.tensor([1., 2., -3.]))
        self.assertEqual(tuple(output.shape), (2,))

    def test_unsupported_and_nonfinite_inputs_are_rejected(self):
        for weight in (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(2, 131072),
                       torch.full((2, 3), float("nan"))):
            with self.subTest(shape=weight.shape, dtype=weight.dtype), self.assertRaises(ValueError):
                INTEGER.IntegerProjection(weight)
        projection = INTEGER.IntegerProjection(torch.ones(2, 3))
        for value in (torch.ones(1, 3, dtype=torch.bfloat16), torch.ones(1, 4),
                      torch.full((1, 3), float("inf"))):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                projection(value)
        with self.assertRaisesRegex(ValueError, "exactly 155"):
            INTEGER.install_projections(torch.nn.Linear(3, 2, bias=False))

    def test_operation_and_memory_plan_exposes_pending_target_terms(self):
        projection = INTEGER.IntegerProjection(torch.ones(2, 3))
        record = {"packed_weight": INTEGER.identity(projection.weight_kn),
                  "weight_scale": INTEGER.identity(projection.weight_scale)}
        plan = INTEGER.memory_plan(projection, [record], 128, 32)
        self.assertEqual(plan["one_kv_state_bytes"], 2 * 22 * 4 * 160 * 64 * 4)
        self.assertEqual(plan["io_bytes"]["simultaneous_kv_input_output"], 2 * plan["one_kv_state_bytes"])
        self.assertEqual(plan["constants_duplicated_two_stage_bytes"], 2 * plan["constants_shared_once_bytes"])
        self.assertFalse(plan["stage_constant_sharing_implemented"])
        self.assertIn("graph_workspace", plan["missing_peak_terms"])
        placement = INTEGER.operation_placement()
        self.assertEqual(placement["learned_projections"]["count"], 155)
        for name in ("qk_contractions", "pv_contractions", "rope", "softmax"):
            self.assertEqual(placement[name]["placement"], "cpu")

    def test_language_corpus_minimum_unique_tokens_and_frozen_contract(self):
        class Tokenizer:
            def __call__(self, text, **_kwargs):
                number = int(text)
                return {"input_ids": torch.arange(128).reshape(1, -1) + number}
        documents = [{"id": str(index), "text": str(index), "source": "unit-test-fixture"}
                     for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            payload = {"source": "unit-test-fixture-never-quality-evidence", "selection": "all",
                       "documents": documents}
            path.write_text(json.dumps(payload))
            tokens, contract = INTEGER.language_inputs(path, Tokenizer(), FACTORY.safe_path, FACTORY.file_identity)
            self.assertEqual(len(tokens), 8)
            self.assertEqual(contract["minimum_documents"], 8)
            self.assertEqual(len(contract["sha256"]), 64)
            contract["documents"][0]["id"] = "modified"
            with self.assertRaisesRegex(ValueError, "contract changed"):
                INTEGER.language_quality(None, tokens, [{}] * 8, contract)
            documents[-1]["text"] = documents[0]["text"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "distinct selected token"):
                INTEGER.language_inputs(path, Tokenizer(), FACTORY.safe_path, FACTORY.file_identity)

    def test_wikitext_preparation_rejects_nonpinned_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "source.parquet"
            artifact.write_text("fixture; never a dataset")
            with self.assertRaisesRegex(ValueError, "pinned source bytes"):
                INTEGER.prepare_wikitext_corpus(artifact, "/missing", Path(directory) / "prepared")
            self.assertFalse((Path(directory) / "prepared").exists())

    def test_frozen_language_thresholds_reject_likelihood_or_token_loss(self):
        documents = [{"id": str(index)} for index in range(8)]
        contract = {**INTEGER.QUALITY_CONTRACT, "documents": documents}
        import hashlib
        contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        tokens = [torch.zeros((1, 128), dtype=torch.int64) for _ in range(8)]
        reference = [{"nll": torch.ones(127), "top1": torch.zeros((1, 127), dtype=torch.int64)}
                     for _ in range(8)]
        for increase, disagree, passed in ((.1, 0, True), (.3, 0, False), (.1, 40, False)):
            candidate = []
            for _ in range(8):
                top1 = torch.zeros((1, 127), dtype=torch.int64)
                top1[:, :disagree] = 1
                candidate.append({"nll": torch.full((127,), 1. + increase), "top1": top1})
            with self.subTest(increase=increase, disagree=disagree), mock.patch.object(
                    INTEGER, "language_reference", return_value=candidate):
                result = INTEGER.language_quality(None, tokens, reference, contract)
                self.assertEqual(result["passed"], passed)
                self.assertEqual(result["predicted_tokens"], 8 * 127)
                self.assertFalse(result["paper_language_quality"])

    @unittest.skipUnless(os.environ.get("TVM_TINYLLAMA_INTEGER_ONNX") == "1", "Optional ONNX/TVM integration disabled")
    def test_real_integer_export_import_and_llvm_execution(self):
        import onnx
        import tvm
        import verify_session
        from types import SimpleNamespace
        generator = np.random.default_rng(21)
        weight = generator.uniform(-2, 2, (19, 33)).astype(np.float32)
        value = generator.uniform(-4, 4, (2, 3, 33)).astype(np.float32)
        value[0, 0] = 0
        value[0, 1, :3] = [254., 1., 3.]
        for mode in ("maxabs", "power_of_two"):
            projection = INTEGER.IntegerProjection(torch.from_numpy(weight), activation_scale=mode).eval()
            with self.subTest(activation_scale=mode), tempfile.TemporaryDirectory() as directory:
                compiler = verify_session.StageCompiler(Path(directory), "optimized", np, torch, onnx, tvm,
                                                         exporter="dynamo", export_options=INTEGER.export_options(torch))
                executable = compiler(SimpleNamespace(name="projection", module=projection), [value], 1)
                actual = verify_session.outputs(executable([value]), np, torch)[0]
                np.testing.assert_allclose(actual, projection(torch.from_numpy(value)).numpy(), rtol=1e-4, atol=1e-4)
                graph = onnx.load(str(Path(directory) / "stage_0/model.onnx"))
                contractions = [node for node in graph.graph.node if node.op_type == "MatMulInteger"]
                self.assertEqual(len(contractions), 1)
                weights = {tensor.name: tensor for tensor in graph.graph.initializer}
                self.assertEqual(weights[contractions[0].input[1]].data_type, onnx.TensorProto.INT8)


if __name__ == "__main__":
    unittest.main()
