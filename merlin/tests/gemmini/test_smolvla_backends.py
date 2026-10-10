"""Source-backend diagnostic controls without pretrained-model qualification."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from merlin.common.paths import repo_root


SOURCE = repo_root() / "examples/gemmini/comparisons/tvm/diagnose_smolvla_backends.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("smolvla_backends_tested", SOURCE)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


class SmolVLABackendArgumentTests(unittest.TestCase):
    def test_invalid_seeds_attribution_and_existing_outputs_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ["--checkpoint-dir", str(root / "policy"), "--backbone-dir", str(root / "backbone"), "--output-dir", str(root / "new")]
            for seeds in (("0", "0"), ("-1",), (str(2**63),)):
                with self.subTest(seeds=seeds), self.assertRaises(SystemExit):
                    DIAGNOSTIC.parse_args(base + ["--seeds", *seeds])
            with self.assertRaises(SystemExit):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0", "--fixture-kind", "dataset"])
            with self.assertRaises(SystemExit):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0", "1", "--input-npz", str(root / "fixture.npz")])
            for attributes in ([], ["--fixture-kind", "synthetic"], ["--fixture-kind", "synthetic", "--input-source", " "]):
                with self.subTest(attributes=attributes), self.assertRaises(SystemExit):
                    DIAGNOSTIC.parse_args(base + ["--seeds", "0", "--input-npz", str(root / "fixture.npz"), *attributes])
            (root / "new").mkdir()
            with mock.patch("merlin.common.paths.artifacts_dir", return_value=root), self.assertRaises(FileExistsError):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0"])
            alias = root / "alias"
            alias.symlink_to(root / "new", target_is_directory=True)
            with mock.patch("merlin.common.paths.artifacts_dir", return_value=root), self.assertRaises(FileExistsError):
                DIAGNOSTIC.parse_args(base[:4] + ["--output-dir", str(alias), "--seeds", "0"])

    def test_supplied_fixtures_require_truthful_explicit_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ["--checkpoint-dir", str(root / "policy"), "--backbone-dir", str(root / "backbone"), "--output-dir", str(root / "new"), "--seeds", "0", "--input-npz", str(root / "fixture.npz")]
            for kind in ("dataset", "synthetic"):
                with self.subTest(kind=kind), mock.patch("merlin.common.paths.artifacts_dir", return_value=root):
                    args = DIAGNOSTIC.parse_args(base + ["--fixture-kind", kind, "--input-source", "Attributed supplied development fixture"])
                self.assertEqual(args.fixture_kind, kind)
                self.assertEqual(args.input_source, "Attributed supplied development fixture")


class SmolVLABackendCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.original_threads)

    def fixture(self, *, mutate_cache=False):
        torch = self.torch

        class Backbone(torch.nn.Module):
            def forward(self, **kwargs):
                embeddings = kwargs["inputs_embeds"][0]
                cache = {index: {"key_states": embeddings.clone(), "value_states": embeddings.clone()} for index in range(2)}
                return (embeddings, None), cache

        class Source(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(num_steps=10, num_vlm_layers=2, chunk_size=50, action_feature=SimpleNamespace(shape=[2]))
                self.vlm_with_expert = Backbone()
                self.register_buffer("constant", torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16)

            def embed_prefix(self, images, masks, tokens, language_mask, state):
                outputs = [torch.nn.functional.scaled_dot_product_attention(self.constant + image.mean(), self.constant, self.constant) for image in images]
                embedding = sum(outputs).reshape(1, 4, 4).bfloat16()
                return embedding, torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool)

            def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
                if mutate_cache:
                    past_key_values[0]["key_states"].add_(1)
                return x_t * 0.2 + timestep[:, None, None] + past_key_values[0]["value_states"].float().mean()

            def sample_actions(self, images, masks, tokens, language_mask, state, noise):
                embedding, padding, _ = self.embed_prefix(images, masks, tokens, language_mask, state)
                _, cache = self.vlm_with_expert.forward(inputs_embeds=[embedding, None], fill_kv_cache=True)
                value = noise
                for index in range(10):
                    timestep = torch.tensor(1.0 + index * (-1.0 / 10)).expand(noise.shape[0])
                    velocity = self.denoise_step(x_t=value, prefix_pad_masks=padding, past_key_values=cache, timestep=timestep)
                    value = value + (-1.0 / 10) * velocity
                return value

        config = SimpleNamespace(resize_imgs_with_padding=(2, 2), tokenizer_max_length=3, max_state_dim=4, max_action_dim=4, chunk_size=50, robot_state_feature=SimpleNamespace(shape=[2]))
        return Source().eval(), DIAGNOSTIC.factory.fixture_inputs(config, 3, seed=7)

    def test_complete_capture_dispatch_noise_cache_and_repeatability(self):
        torch, np = self.torch, self.np
        model, inputs = self.fixture()
        state_before = DIAGNOSTIC.state_identity(model, torch)
        flags_before = DIAGNOSTIC.backend_flags(torch)
        rng_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            for backend in ("default", "math"):
                first, metadata = DIAGNOSTIC.run_source_policy(model, inputs, backend, 7, 0, directory, np, torch, profile=True)
                second, _ = DIAGNOSTIC.run_source_policy(model, inputs, backend, 7, 1, directory, np, torch)
                self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))
                self.assertEqual(first["final_actions"].shape, (1, 50, 2))
                self.assertEqual(len(metadata["prefix_cache_checks"]), 10)
                self.assertTrue(all(item["before"] == item["after"] for item in metadata["prefix_cache_checks"]))
                self.assertEqual(metadata["captures"]["step_00_input_state"], metadata["input_bindings"]["noise"])
                self.assertEqual(metadata["attention_profile"]["native_sdpa_calls_by_stage"], dict(zip(DIAGNOSTIC.SCOPES, (3, 0, 0))))
                self.assertTrue(metadata["temporary_wrappers_restored"])
                with np.load(metadata["output_file"]["path"], allow_pickle=False) as saved:
                    self.assertEqual(saved["prefix_key_cache"].dtype, np.uint16)
                    for index in range(10):
                        np.testing.assert_array_equal(saved[f"step_{index:02d}_output_state"], saved[f"step_{index:02d}_input_state"] + (-1.0 / 10) * saved[f"step_{index:02d}_velocity"])
                runs[backend] = metadata
            self.assertIn("aten::_scaled_dot_product_flash_attention_for_cpu", runs["default"]["attention_profile"]["operators"])
            self.assertIn("aten::_scaled_dot_product_attention_math", runs["math"]["attention_profile"]["operators"])
        self.assertEqual(DIAGNOSTIC.state_identity(model, torch), state_before)
        self.assertEqual(DIAGNOSTIC.backend_flags(torch), flags_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_cache_failure_restores_context_hooks_and_rng(self):
        torch = self.torch
        model, inputs = self.fixture(mutate_cache=True)
        flags_before = DIAGNOSTIC.backend_flags(torch)
        rng_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(AssertionError, "mutated the prefix cache"):
            DIAGNOSTIC.run_source_policy(model, inputs, "math", 7, 0, directory, self.np, torch)
        self.assertEqual(DIAGNOSTIC.backend_flags(torch), flags_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertNotIn("embed_prefix", model.__dict__)
        self.assertNotIn("denoise_step", model.__dict__)
        self.assertNotIn("forward", model.vlm_with_expert.__dict__)

    def test_stride_zero_hash_and_signed_bf16_distances(self):
        torch = self.torch
        singleton = torch.tensor(0.5).expand(1)
        self.assertEqual(DIAGNOSTIC.tensor_identity(singleton, torch), DIAGNOSTIC.tensor_identity(torch.tensor([0.5]), torch))
        default = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
        alternate = torch.tensor([-0.0, 1.0078125], dtype=torch.bfloat16)
        result = DIAGNOSTIC.statistics(default, alternate, torch)
        self.assertEqual(result["bf16_representable_step_distance"]["max"], 1)
        self.assertEqual(result["exactly_unequal_count"], 1)
        expanded_default = torch.tensor(1.0, dtype=torch.bfloat16).expand(1)
        expanded_alternate = torch.tensor(1.0078125, dtype=torch.bfloat16).expand(1)
        expanded_result = DIAGNOSTIC.statistics(expanded_default, expanded_alternate, torch)
        self.assertEqual(expanded_result["bf16_representable_step_distance"]["max"], 1)
        self.assertEqual(DIAGNOSTIC.dense_flat_copy(expanded_default, torch).stride(), (1,))
        self.assertEqual((DIAGNOSTIC.RTOL, DIAGNOSTIC.ATOL), (1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
