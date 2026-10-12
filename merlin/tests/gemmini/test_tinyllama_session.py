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
                        {"workload": "development_language"},
                        {"workload": "synthetic_feasibility", "prefill_tokens": 9},
                        {"activation_scale": "other"}, {"arithmetic_policy": "other"},
                        {"arithmetic_policy": "fixed_tree_fp32_v1"},
                        {"arithmetic_policy": "fixed_tree_fp32_v1", "quantized": True, "precision": "bfloat16"}):
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
    @staticmethod
    def scalar_tree_dot(left, right):
        output = np.empty((*left.shape[:2], left.shape[-2], right.shape[-2]), dtype=np.float32)
        for batch, head, row, column in np.ndindex(output.shape):
            terms = [np.float32(np.float32(a) * np.float32(b)) for a, b in zip(left[batch, head, row], right[batch, head, column])]
            while len(terms) > 1:
                terms = [np.float32(terms[index] + terms[index + 1]) for index in range(0, len(terms), 2)]
            output[batch, head, row, column] = terms[0]
        return output

    def test_fixed_tree_native_has_independent_order_and_no_fma_oracle(self):
        generator = np.random.default_rng(194)
        left = generator.uniform(-4, 4, (1, 2, 7, 64)).astype(np.float32)
        right = generator.uniform(-4, 4, (1, 2, 9, 64)).astype(np.float32)
        expected = self.scalar_tree_dot(left, right)
        actual = INTEGER.fixed_tree_dot(torch.from_numpy(left), torch.from_numpy(right)).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(INTEGER.fixed_tree_dot(torch.from_numpy(left[..., -1:, :]), torch.from_numpy(right)).numpy(), actual[..., -1:, :])
        padded = np.pad(right, ((0, 0), (0, 0), (0, 3), (0, 0)))
        np.testing.assert_array_equal(INTEGER.fixed_tree_dot(torch.from_numpy(left), torch.from_numpy(padded)).numpy()[..., :9], actual)
        ordered_left = torch.tensor([1e8, 1., -1e8, 1.]).reshape(1, 1, 1, 4)
        ordered_right = torch.ones_like(ordered_left)
        self.assertEqual(float(INTEGER.fixed_tree_dot(ordered_left, ordered_right)), 0.)
        ascending = np.float32(0)
        for value in ordered_left.numpy().flat:
            ascending = np.float32(ascending + value)
        self.assertEqual(float(ascending), 1.)
        epsilon = np.float32(2**-23)
        no_fma_left = np.array([np.float32(1 + epsilon), -1], dtype=np.float32).reshape(1, 1, 1, 2)
        no_fma_right = np.array([np.float32(1 - epsilon), 1], dtype=np.float32).reshape(1, 1, 1, 2)
        self.assertEqual(float(INTEGER.fixed_tree_dot(torch.from_numpy(no_fma_left), torch.from_numpy(no_fma_right))), 0.)
        self.assertNotEqual(float(np.float32(np.sum(no_fma_left.astype(np.float64) * no_fma_right.astype(np.float64)))), 0.)
        for shape in ((2048,), (1, 3, 2048)):
            value = generator.uniform(-1, 1, shape).astype(np.float32)
            expected_mean = self.scalar_tree_dot(value.reshape(1, 1, -1, 2048), value.reshape(1, 1, -1, 2048))
            expected_mean = np.diagonal(expected_mean[0, 0]).reshape(*shape[:-1], 1) * np.float32(1 / 2048)
            actual_mean = INTEGER.fixed_tree_mean_squares(torch.from_numpy(value)).numpy()
            np.testing.assert_array_equal(actual_mean.view(np.uint32), expected_mean.view(np.uint32))

    def test_fixed_tree_rejects_invalid_operands_without_mutation(self):
        value = torch.ones((1, 1, 2, 8))
        before = value.clone()
        for bad in (value.double(), value[..., :3], torch.empty((1, 1, 0, 8)), value + float("inf"), torch.full_like(value, torch.finfo(torch.float32).max)):
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype), self.assertRaises(ValueError):
                INTEGER.fixed_tree_mean_squares(bad)
        with self.assertRaisesRegex(ValueError, "matching"):
            INTEGER.fixed_tree_dot(value, torch.ones((1, 2, 2, 8)))
        self.assertTrue(torch.equal(value, before))
        self.assertEqual(INTEGER.arithmetic_contract()["name"], "native")
        self.assertNotIn("arithmetic_policy", INTEGER.recipe("power_of_two"))
        self.assertEqual(INTEGER.recipe("power_of_two", "fixed_tree_fp32_v1")["arithmetic_policy"]["status"], "opt_in_candidate_not_adopted")

    def test_fixed_tree_installer_preserves_norm_parameters_and_source_interface(self):
        from types import SimpleNamespace
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaAttention, eager_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        lm = torch.nn.Module()
        lm.model = torch.nn.Module()
        lm.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(22)])
        config = LlamaConfig(hidden_size=2048, intermediate_size=5632, num_hidden_layers=22, num_attention_heads=32, num_key_value_heads=4)
        config._attn_implementation = "eager"
        for index, layer in enumerate(lm.model.layers):
            layer.input_layernorm = LlamaRMSNorm(2048, 1e-5)
            layer.post_attention_layernorm = LlamaRMSNorm(2048, 1e-5)
            with torch.device("meta"):
                layer.self_attn = LlamaAttention(config, index)
        lm.model.norm = LlamaRMSNorm(2048, 1e-5)
        lm.config = config
        lm.eval()
        parameters = {name: value for name, value in lm.named_parameters()}
        metadata = INTEGER.install_fixed_tree_arithmetic(lm)
        self.assertEqual(metadata["rms_norms"], 45)
        self.assertEqual(lm.config._attn_implementation, "eager")
        self.assertTrue(all(isinstance(layer.self_attn, INTEGER.FixedTreeAttention) for layer in lm.model.layers))
        self.assertTrue(all(not module.training for module in lm.modules()))
        self.assertIs(ALL_ATTENTION_FUNCTIONS.get_interface("eager", eager_attention_forward), eager_attention_forward)
        for name, value in lm.named_parameters():
            self.assertIs(value, parameters[name])

    def test_fixed_tree_missing_mask_fails_before_caller_state_changes(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention
        config = LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
        original = LlamaAttention(config, 0).eval()
        candidate = INTEGER.FixedTreeAttention(original).eval()
        value = torch.ones((1, 1, 64))
        before = value.clone()
        cache = mock.Mock()
        parameters = {name: parameter.detach().clone() for name, parameter in candidate.named_parameters()}
        with self.assertRaisesRegex(ValueError, "explicit causal attention mask"):
            candidate(value, (torch.ones((1, 1, 16)), torch.zeros((1, 1, 16))), past_key_values=cache)
        cache.update.assert_not_called()
        self.assertTrue(torch.equal(value, before))
        for name, parameter in candidate.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters[name]))

    def test_fixed_tree_import_is_lazy_and_alias_registration_is_idempotent(self):
        import subprocess
        script = """import importlib.util, sys, torch
from pathlib import Path
path=Path(sys.argv[1]);modules=[]
assert not hasattr(torch.ops.merlin_fixed_order,'dot')
for name in ('fixed_tree_import_one','fixed_tree_import_alias'):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
assert not hasattr(torch.ops.merlin_fixed_order,'dot')
assert not hasattr(torch.ops.merlin_fixed_order,'mean_squares')
assert modules[0].recipe()['name']=='tinyllama_w8a8_dynamic_token_v1'
value=torch.ones((1,1,1,8));expected=torch.tensor([[[[8.]]]])
for module in modules:
 assert torch.equal(module.fixed_tree_dot(value,value),expected)
 assert torch.equal(module.fixed_tree_mean_squares(value),torch.ones_like(value[...,:1]))
assert modules[0]._FIXED_TREE_OPS==modules[1]._FIXED_TREE_OPS
"""
        result = subprocess.run([sys.executable, "-c", script, str(Path(INTEGER.__file__))], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_encoder_oracle_rejects_wrong_codes_scales_dtype_and_nonfinite(self):
        value = np.asarray([[127., .5, 1.5, -2.5], [0., 0., 0., 0.]], dtype=np.float32)
        codes, scales = INTEGER.independent_quantize_rows(value)
        result = INTEGER.check_operand_encoding(value, codes, scales)
        self.assertEqual(result["checked_codes"], 8)
        self.assertEqual(result["checked_scales"], 2)
        wrong_codes = codes.copy()
        wrong_codes[0, 1] = 1
        for wrong in (wrong_codes, np.full_like(codes, -128)):
            with self.assertRaisesRegex(AssertionError, "rounding/clipping"):
                INTEGER.check_operand_encoding(value, wrong, scales)
        with self.assertRaisesRegex(AssertionError, "scale encoding"):
            INTEGER.check_operand_encoding(value, codes, scales * np.float32(2))
        for wrong_codes, wrong_scales in ((codes.astype(np.int32), scales), (codes, scales.astype(np.float64)), (codes[:, :2], scales), (codes, scales[:, 0])):
            with self.assertRaises(ValueError):
                INTEGER.check_operand_encoding(value, wrong_codes, wrong_scales)
        for wrong in (value.astype(np.float64), np.asarray([[np.inf]], dtype=np.float32), np.zeros((0, 3), dtype=np.float32)):
            with self.assertRaises(ValueError):
                INTEGER.independent_quantize_rows(wrong)

    def test_oracle_detects_shared_emitted_code_bug_before_integer_dot(self):
        projection = INTEGER.IntegerProjection(torch.ones(2, 4))
        value = torch.tensor([[127., .5, 1.5, -2.5]])
        for rows, columns in ((0, 7), (3, 0), (True, 7)):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                INTEGER.check_projection(projection, value, rows=rows, columns=columns)
        codes, scales = INTEGER.quantize_rows(value)
        codes[0, 1] = 1
        with mock.patch.object(INTEGER, "quantize_rows", return_value=(codes, scales)):
            with self.assertRaisesRegex(AssertionError, "rounding/clipping"):
                INTEGER.check_projection(projection, value)

    def test_bounded_pair_isolates_boundary_crossing_from_local_encoding(self):
        source = np.asarray([[.9921875, .0664062425494194]], dtype=np.float32)
        actual = np.asarray([[.9921875, .0664062574505806]], dtype=np.float32)
        codes, scales = INTEGER.independent_quantize_rows(actual, "power_of_two")
        result = INTEGER.diagnose_activation_pair(source, actual, actual_codes=codes, actual_scales=scales)
        first = result["first_code_difference"]
        self.assertTrue(result["local_actual_encoding"]["passed"])
        self.assertEqual(first["index"], [0, 1])
        self.assertEqual((first["source_code"], first["actual_code"]), (8, 9))
        self.assertTrue(first["same_scale_crosses_rounding_boundary"])
        self.assertLess(first["source_ratio"], 8.5)
        self.assertGreater(first["actual_ratio"], 8.5)
        with self.assertRaisesRegex(ValueError, "element budget"):
            INTEGER.diagnose_activation_pair(source, actual, maximum_elements=1)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            INTEGER.diagnose_activation_pair(source, actual, actual_codes=codes)
        codes[0, 1] = 8
        with self.assertRaisesRegex(AssertionError, "rounding/clipping"):
            INTEGER.diagnose_activation_pair(source, actual, actual_codes=codes, actual_scales=scales)

    def test_trace_reports_first_float_and_later_first_code_difference(self):
        lm = torch.nn.Module()
        for name in ("first", "second"):
            lm.add_module(name, INTEGER.IntegerProjection(torch.ones(2, 2), "power_of_two"))
        source = {"first": torch.tensor([[.9921875, .0625]]), "second": torch.tensor([[.9921875, .0664062425494194]])}
        actual = {"first": torch.tensor([[.9921875, np.nextafter(np.float32(.0625), np.float32(1))]]), "second": torch.tensor([[.9921875, .0664062574505806]])}
        result = FACTORY.compare_projection_traces(actual, source, lm)
        self.assertEqual(result["first_float_input_difference"], "first")
        self.assertEqual(result["first_quantized_input_difference"]["projection"], "second")
        self.assertEqual(result["checks"][0]["code_mismatches"], 0)
        with self.assertRaisesRegex(AssertionError, "invocation order"):
            FACTORY.compare_projection_traces(dict(reversed(list(actual.items()))), source, lm)

    def test_capture_hooks_remove_on_failure_and_attention_checks_refuse_wrong_state(self):
        projection = INTEGER.IntegerProjection(torch.ones(2, 3), "power_of_two")
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with FACTORY.capture_last_token_inputs(projection):
                projection(torch.ones(1, 3))
                raise RuntimeError("intentional")
        self.assertEqual(len(projection._forward_pre_hooks), 0)
        source = {"mask": torch.zeros(1, 1, 1, 3), "rope": (torch.ones(1, 1, 2), torch.zeros(1, 1, 2))}
        actual = {"mask": torch.tensor([[[[0., 0., 0., -torch.inf]]]]), "rope": source["rope"]}
        self.assertTrue(FACTORY.compare_cache_attention(actual, source, 3, torch)["active_mask_exact"])
        actual["mask"][..., -1] = 0
        with self.assertRaisesRegex(AssertionError, "unused cache slots"):
            FACTORY.compare_cache_attention(actual, source, 3, torch)
        actual["mask"][..., -1] = -.1
        with self.assertRaisesRegex(AssertionError, "unused cache slots"):
            FACTORY.compare_cache_attention(actual, source, 3, torch)
        actual["mask"][..., -1] = -torch.inf
        actual["rope"] = source["rope"][:1]
        with self.assertRaisesRegex(AssertionError, "RoPE components"):
            FACTORY.compare_cache_attention(actual, source, 3, torch)
        actual["rope"] = (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))
        with self.assertRaises(AssertionError):
            FACTORY.compare_cache_attention(actual, source, 3, torch)

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
    def test_compiled_emitted_activation_codes_and_scales_match_independent_oracle(self):
        import onnx
        import tvm
        import verify_session
        from types import SimpleNamespace
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = INTEGER.IntegerProjection(torch.tensor([[1., -2., 3., 4.], [-1., 0., 2., 3.]]), "power_of_two")
            def forward(self, value):
                codes, scales = self.projection.integer_inputs(value)
                return self.projection(value), codes, scales
        values = np.asarray([[254., 1., 3., -5.], [0., 0., 0., 0.], [.9921875, .0664062425494194, .0664062574505806, -.0664062574505806]], dtype=np.float32)
        probe = Probe().eval()
        with tempfile.TemporaryDirectory() as directory:
            compiler = verify_session.StageCompiler(Path(directory), "optimized", np, torch, onnx, tvm,
                                                     exporter="dynamo", export_options=INTEGER.export_options(torch))
            executable = compiler(SimpleNamespace(name="emitted_operands", module=probe), [values], 3)
            _result, codes, scales = verify_session.outputs(executable([values]), np, torch)
            checked = INTEGER.check_operand_encoding(values, codes, scales, "power_of_two")
            self.assertEqual(checked["checked_codes"], values.size)
            self.assertEqual(codes[2].tolist(), [127, 8, 9, -9])

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODEL2MLIR"), "Optional pinned cache-wrapper integration disabled")
    def test_real_hf_fixed_tree_cache_dispatch_and_unchanged_gate(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaAttention
        from types import SimpleNamespace
        path = FACTORY.safe_path(Path(os.environ["TVM_SESSION_MODEL2MLIR"]) / "m2m/capture/causal_session.py")
        spec = importlib.util.spec_from_file_location("tinyllama_candidate_test_causal_session", path)
        causal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(causal)
        torch.manual_seed(194)
        config = LlamaConfig(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
        config._attn_implementation = "eager"
        lm = LlamaForCausalLM(config).eval().requires_grad_(False)
        for name, module in list(lm.named_modules()):
            if isinstance(module, LlamaRMSNorm):
                parent, _, leaf = name.rpartition(".")
                setattr(lm.get_submodule(parent), leaf, INTEGER.FixedTreeRMSNorm(module))
            elif isinstance(module, LlamaAttention):
                parent, _, leaf = name.rpartition(".")
                setattr(lm.get_submodule(parent), leaf, INTEGER.FixedTreeAttention(module))
        capture = SimpleNamespace(lm=lm, tokens=torch.arange(12).reshape(1, -1), prefill_tokens=4, decode_tokens=8, provenance={"arithmetic_policy": INTEGER.arithmetic_contract("fixed_tree_fp32_v1")})
        with mock.patch.object(INTEGER, "fixed_tree_dot", wraps=INTEGER.fixed_tree_dot) as observed:
            result = FACTORY.validate_reference_stages(capture, causal, torch)
        self.assertEqual(observed.call_count, 36)
        self.assertEqual(result["reference"], "candidate_full_sequence_hf_use_cache_false")
        self.assertEqual((result["rtol"], result["atol"]), (1e-4, 1e-4))
        self.assertEqual(len(result["checks"]), 9)
        self.assertTrue(all(check["attention"]["active_mask_exact"] and check["attention"]["last_token_rope_exact"] for check in result["checks"]))

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODEL2MLIR"), "Optional pinned cache-wrapper integration disabled")
    def test_real_hf_fixed_cache_matches_positions_masks_and_full_sequence(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from types import SimpleNamespace
        path = FACTORY.safe_path(Path(os.environ["TVM_SESSION_MODEL2MLIR"]) / "m2m/capture/causal_session.py")
        spec = importlib.util.spec_from_file_location("tinyllama_test_causal_session", path)
        causal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(causal)
        torch.manual_seed(194)
        config = LlamaConfig(hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
        config._attn_implementation = "eager"
        lm = LlamaForCausalLM(config).eval().requires_grad_(False)
        capture = SimpleNamespace(lm=lm, tokens=torch.arange(12).reshape(1, -1), prefill_tokens=4, decode_tokens=8, provenance={})
        result = FACTORY.validate_reference_stages(capture, causal, torch)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 9)
        self.assertTrue(all(check["attention"]["active_mask_exact"] and check["attention"]["last_token_rope_exact"] for check in result["checks"]))
        self.assertEqual(len(lm.model.layers[0].self_attn._forward_pre_hooks), 0)
        original_decode = causal.FixedCacheDecodeStep.forward
        def wrong_decode(stage, *args):
            logits, cache, position = original_decode(stage, *args)
            return logits + .01, cache, position
        with mock.patch.object(causal.FixedCacheDecodeStep, "forward", wrong_decode):
            with self.assertRaisesRegex(AssertionError, "cache gate failed at 5"):
                FACTORY.validate_reference_stages(capture, causal, torch)
        diagnostic = capture.provenance["functional_cache_reference"]
        self.assertFalse(diagnostic["passed"])
        self.assertEqual((diagnostic["rtol"], diagnostic["atol"]), (1e-4, 1e-4))
        self.assertTrue(diagnostic["attention"]["active_mask_exact"])
        self.assertEqual(len(lm.model.layers[0].self_attn._forward_pre_hooks), 0)

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

    @unittest.skipUnless(os.environ.get("TVM_TINYLLAMA_FIXED_TREE_ONNX") == "1", "Optional ordered FP32 ONNX/TVM integration disabled")
    def test_real_fixed_tree_export_import_and_llvm_execution(self):
        import onnx
        import tvm
        import verify_session
        from types import SimpleNamespace
        class Ordered(torch.nn.Module):
            def forward(self, left, right, value):
                return INTEGER.fixed_tree_dot(left, right), INTEGER.fixed_tree_mean_squares(value)
        generator = np.random.default_rng(981)
        left = generator.uniform(-2, 2, (1, 2, 5, 64)).astype(np.float32)
        right = generator.uniform(-2, 2, (1, 2, 7, 64)).astype(np.float32)
        value = generator.uniform(-1, 1, (1, 3, 2048)).astype(np.float32)
        candidate = SimpleNamespace(provenance={"arithmetic_policy": INTEGER.arithmetic_contract("fixed_tree_fp32_v1")})
        options = verify_session.factory_export_options(FACTORY, torch, candidate)
        self.assertEqual(len(options["custom_translation_table"]), 3)
        # A prior candidate must not opt a later native model into ordered export.
        native = SimpleNamespace(provenance={})
        native_options = verify_session.factory_export_options(FACTORY, torch, native)
        self.assertEqual(set(native_options["custom_translation_table"]), {torch.ops.aten._int_mm.default})
        self.assertEqual(set(INTEGER.export_options(torch)["custom_translation_table"]), {torch.ops.aten._int_mm.default})
        legacy = SimpleNamespace(get_onnx_export_options=lambda selected: {"custom_translation_table": {}})
        self.assertEqual(verify_session.factory_export_options(legacy, torch, candidate), {"custom_translation_table": {}})
        self.assertIsNone(verify_session.factory_export_options(SimpleNamespace(), torch, candidate))
        expected_dot = self.scalar_tree_dot(left, right)
        expected_mean = self.scalar_tree_dot(value.reshape(1, 1, 3, 2048), value.reshape(1, 1, 3, 2048))[0, 0].diagonal().reshape(1, 3, 1) * np.float32(1 / 2048)
        for mode in ("baseline", "optimized"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                compiler = verify_session.StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", export_options=options)
                executable = compiler(SimpleNamespace(name="ordered", module=Ordered().eval()), [left, right, value], 1)
                actual_dot, actual_mean = verify_session.outputs(executable([left, right, value]), np, torch)
                np.testing.assert_array_equal(actual_dot.view(np.uint32), expected_dot.view(np.uint32))
                np.testing.assert_array_equal(actual_mean.view(np.uint32), expected_mean.view(np.uint32))
                graph = onnx.load(str(Path(directory) / "stage_0/model.onnx"))
                self.assertEqual([(node.domain, node.op_type) for node in graph.graph.node], [("merlin.fixed_order", "FixedTreeDot"), ("merlin.fixed_order", "FixedTreeMeanSquares")])
                self.assertEqual(next(item.version for item in graph.opset_import if item.domain == "merlin.fixed_order"), 1)


if __name__ == "__main__":
    unittest.main()
