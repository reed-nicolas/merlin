"""SmolVLA artifact refusals and full neural-session routing; no pretrained substitute."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from merlin.common.paths import repo_root


SOURCE = repo_root() / "examples/gemmini/comparisons/tvm/smolvla_session.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("smolvla_session_tested", SOURCE)
FACTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FACTORY)


class SmolVLAArtifactTests(unittest.TestCase):
    def config(self):
        return {"type": "smolvla", **FACTORY.GEOMETRY,
                "input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                    **{f"observation.images.camera{index}": {"type": "VISUAL", "shape": [3, 256, 256]}
                       for index in range(1, 4)}},
                "output_features": {"action": {"type": "ACTION", "shape": [6]}}}

    def test_absent_checkpoint_fails_before_framework_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                FACTORY.checkpoint_artifacts(directory)

    def test_nonofficial_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(self.config()))
            (root / "model.safetensors").write_text("fixture only; never a model")
            with self.assertRaisesRegex(ValueError, "pinned official full"):
                FACTORY.checkpoint_artifacts(root)

    def test_geometry_truncation_and_precision_changes_are_rejected(self):
        for change in ({"num_steps": 1}, {"num_vlm_layers": 2}, {"load_vlm_weights": False}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps({**self.config(), **change}))
                (root / "model.safetensors").write_text("fixture only; never a model")
                identity = {"path": str(root / "model.safetensors"), "sha256": FACTORY.WEIGHT_SHA256}
                with mock.patch.object(FACTORY, "file_identity", return_value=identity):
                    with self.assertRaisesRegex(ValueError, "complete pinned"):
                        FACTORY.checkpoint_artifacts(root)

    def test_unsupported_precision_and_unattributed_dataset_fail_early(self):
        base = {"checkpoint_dir": "/missing", "backbone_dir": "/missing", "model2mlir_root": "/missing"}
        for change in ({"precision": "float32"}, {"cpu_threads": 32}, {"fixture_kind": "dataset"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                FACTORY.get_model_and_inputs(**base, **change)

    def test_restricted_component_is_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "Restricted path component"):
            FACTORY.directory_artifacts("/tmp/" + "vl" + "si" + "-fixture")


class SmolVLATensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        cls.config = SimpleNamespace(resize_imgs_with_padding=(2, 2), tokenizer_max_length=3,
                                    max_state_dim=4, max_action_dim=4, chunk_size=5,
                                    robot_state_feature=SimpleNamespace(shape=[2]))

    def test_synthetic_fixture_is_repeatable_and_pads_only_physical_state(self):
        first = FACTORY.fixture_inputs(self.config, 3, seed=42)
        second = FACTORY.fixture_inputs(self.config, 3, seed=42)
        self.assertEqual(first[0].shape, (1, 3, 3, 2, 2))
        self.assertTrue(all(self.torch.equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(self.torch.count_nonzero(first[4][:, 2:]).item(), 0)

    def test_npz_precision_is_never_coerced(self):
        import numpy as np
        values = FACTORY.fixture_inputs(self.config, 3, seed=42)
        names = ("images", "image_masks", "language_tokens", "language_mask", "state", "noise")
        arrays = {name: value.numpy() for name, value in zip(names, values)}
        arrays["images"] = arrays["images"].astype(np.float16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "dtype"):
                FACTORY.fixture_inputs(self.config, 3, input_npz=path)

    def test_complete_schedule_exact_time_stream_and_physical_decode(self):
        root = os.environ.get("MODEL2MLIR_ROOT")
        if not root:
            self.skipTest("MODEL2MLIR_ROOT selects the actual loader/protocol for this contract test")
        from verify_session import load_module, load_protocol, verify_session
        import numpy as np
        torch = self.torch
        load_protocol(Path(root))
        loader = load_module("smolvla_contract_loader", Path(root) / "workloads/smolvla/loader.py")

        class Backbone(torch.nn.Module):
            def forward(self, **kwargs):
                embeds = kwargs["inputs_embeds"][0]
                cache = {index: {"key_states": embeds, "value_states": embeds}
                         for index in range(16)}
                return (embeds, None), cache

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm_with_expert = Backbone()
                self.config = SimpleNamespace(num_vlm_layers=16, num_steps=10, chunk_size=5,
                                              action_feature=SimpleNamespace(shape=[2]))

            def embed_prefix(self, images, masks, tokens, language_mask, state):
                embeds = state[:, None, :] + sum(image.mean() for image in images)
                return embeds, torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1)

            def denoise_step(self, prefix_masks, cache, flow, timestep):
                return flow * 0.2 + timestep[:, None, None] + cache[0]["key_states"].mean()

            def sample_actions(self, images, masks, tokens, language_mask, state, noise):
                embeds, _, _ = self.embed_prefix(images, masks, tokens, language_mask, state)
                value = noise
                for index in range(self.config.num_steps):
                    time = torch.tensor(1.0 + index * (-1.0 / self.config.num_steps)).expand(1)
                    value = value - (1.0 / self.config.num_steps) * (
                        value * 0.2 + time[:, None, None] + embeds.mean())
                return value

        model = Model().eval()
        inputs = FACTORY.fixture_inputs(self.config, 3, seed=42)
        provenance = {}
        session = FACTORY.make_policy_session(model, inputs, loader, provenance)
        self.assertTrue(provenance["upstream_sample_actions_equivalence"]["exact"])
        self.assertEqual([program.steps for program in session.programs], [1, 10, 1])
        self.assertEqual(session.streams[0].values.shape, (10, 1))
        self.assertEqual(session.programs[-1].module.action_dim, 2)
        self.assertFalse(any(route.name == "timestep" for route in session.routes))

        def compile_eager(program, _inputs, _outputs):
            def run(values):
                with torch.no_grad():
                    return program.module(*(torch.from_numpy(value) for value in values))
            return run

        result = verify_session(session, compile_eager, np, torch)
        counts = {name: sum(row["program"] == name for row in result["checks"])
                  for name in ("prefix_encode", "flow_denoise", "action_decode")}
        self.assertEqual(counts, {"prefix_encode": 2, "flow_denoise": 20, "action_decode": 2})
        self.assertTrue(result["reset_verified"])


if __name__ == "__main__":
    unittest.main()
