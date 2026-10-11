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
PREP_SPEC = importlib.util.spec_from_file_location("smolvla_observation_tested", SOURCE.with_name("prepare_smolvla_observation.py"))
PREPARATION = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(PREPARATION)
NUMERIC_SPEC = importlib.util.spec_from_file_location("smolvla_bf16_diagnostic_tested", SOURCE.with_name("bf16_export.py"))
NUMERICS = importlib.util.module_from_spec(NUMERIC_SPEC)
NUMERIC_SPEC.loader.exec_module(NUMERICS)


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


class SmolVLAObservationTests(unittest.TestCase):
    def schema(self):
        return {"codebase_version": "v3.0", "robot_type": "so100", "fps": 30,
                "total_episodes": 50, "total_frames": 19631, "total_tasks": 1,
                "features": {**{name: {"dtype": "video", "shape": [480, 640, 3],
                                       "names": ["height", "width", "channels"]}
                                for name in PREPARATION.CAMERA_MAP},
                             **{name: {"dtype": "float32", "shape": [6],
                                       "names": PREPARATION.STATE_NAMES}
                                for name in ("observation.state", "action")}}}

    def test_official_camera_coordinate_schema_and_frozen_episode_split(self):
        PREPARATION.validate_schema(self.schema())
        counts = {split: sum(PREPARATION.episode_split(index) == split for index in range(50))
                  for split in ("train", "development", "heldout")}
        self.assertEqual(counts, {"train": 30, "development": 10, "heldout": 10})
        for index in (-1, 50, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                PREPARATION.episode_split(index)
        changed = self.schema()
        changed["features"]["observation.state"]["names"] = list(reversed(PREPARATION.STATE_NAMES))
        with self.assertRaisesRegex(ValueError, "coordinate schema"):
            PREPARATION.validate_schema(changed)
        changed = self.schema()
        changed["features"]["observation.images.fake"] = changed["features"]["observation.images.top"]
        with self.assertRaisesRegex(ValueError, "exactly its recorded"):
            PREPARATION.validate_schema(changed)

    def test_recorded_preparation_uses_original_processor_without_inventing_camera(self):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        config = SimpleNamespace(image_features={name: {} for name in (
            "observation.images.camera1", "observation.images.camera2", "observation.images.camera3")},
            empty_cameras=0, resize_imgs_with_padding=(2, 2), max_state_dim=32,
            max_action_dim=32, chunk_size=50)
        policy = SimpleNamespace(config=config)
        policy.prepare_images = lambda batch: SmolVLAPolicy.prepare_images(policy, batch)
        policy.prepare_state = lambda batch: SmolVLAPolicy.prepare_state(policy, batch)
        observation = {"task": "Pick up the cube", "observation.state": torch.arange(6, dtype=torch.float32),
                       "action": torch.full((6,), 99.0),
                       "observation.images.top": torch.ones(3, 480, 640) * 0.25,
                       "observation.images.wrist": torch.ones(3, 480, 640) * 0.75}
        passed = []

        def processor(batch):
            passed.append(batch)
            return {**{key: value.unsqueeze(0) for key, value in batch.items() if torch.is_tensor(value)},
                    "observation.state": (batch["observation.state"] - 2.0).unsqueeze(0),
                    "observation.language.tokens": torch.zeros(1, 48, dtype=torch.int64),
                    "observation.language.attention_mask": torch.ones(1, 48, dtype=torch.bool)}

        rng = torch.random.get_rng_state().clone()
        first = PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)
        second = PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)
        self.assertEqual(first[0].shape, (1, 2, 3, 2, 2))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertTrue(torch.equal(first[4][0, :6], observation["observation.state"] - 2))
        self.assertEqual(first[4][0, 6:].count_nonzero().item(), 0)
        self.assertTrue(first[1].all())
        self.assertNotIn("action", passed[0])
        self.assertNotIn("observation.images.camera3", passed[0])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(observation["observation.state"], torch.arange(6, dtype=torch.float32)))
        observation["observation.images.top"] = observation["observation.images.top"].to(torch.float16)
        with self.assertRaisesRegex(ValueError, "FP32 CHW"):
            PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)

    def test_observation_manifest_checks_npz_and_camera_route_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "observation.npz"
            fixture.write_bytes(b"identity control only")
            manifest = {
                "version": 1, "fixture_kind": "dataset", "dataset_id": PREPARATION.DATASET_ID,
                "dataset_revision": PREPARATION.DATASET_REVISION, "checkpoint_revision": FACTORY.MODEL_REVISION,
                "precision": "original_lerobot", "schema_verified": True,
                "selected_cameras": list(PREPARATION.CAMERA_MAP.values()), "camera_mapping": PREPARATION.CAMERA_MAP,
                "state_names": PREPARATION.STATE_NAMES, "active_gate_changed": False,
                "model_quality_qualified": False, "episode_index": 0, "split": "train",
                "input_fixture": FACTORY.file_identity(fixture, FACTORY.safe_path),
                "bindings": [{"path": str(root / relative), "sha256": digest}
                             for relative, digest in PREPARATION.PINNED_FILES.items()]}
            path = root / "observation.json"
            path.write_text(json.dumps(manifest))
            identities = {item["path"]: item for item in manifest["bindings"] + [manifest["input_fixture"]]}
            with mock.patch.object(FACTORY, "file_identity", side_effect=lambda name, _: identities[str(name)]):
                self.assertEqual(FACTORY.observation_manifest(fixture, path)["selected_cameras"],
                                 ["observation.images.camera1", "observation.images.camera2"])
                manifest["selected_cameras"].reverse()
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "pinned preparation"):
                    FACTORY.observation_manifest(fixture, path)
            manifest["selected_cameras"].reverse()
            path.write_text(json.dumps(manifest))
            fixture.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "NPZ identity"):
                FACTORY.observation_manifest(fixture, path)


class SmolVLAArithmeticDiagnosticTests(unittest.TestCase):
    def test_linear_bound_handles_cancellation_and_one_declared_cast(self):
        import torch
        left = torch.tensor([[1.0, -1.0, 0.5], [0.25, 0.5, -0.125]], dtype=torch.bfloat16)
        weight = torch.tensor([[1.0, 1.0, 0.0], [0.5, -0.75, 0.25]], dtype=torch.bfloat16)
        bias = torch.tensor([0.0, 0.125], dtype=torch.bfloat16)
        result = NUMERICS.bf16_linear_error_diagnostic(left, weight, bias, torch)
        actual = (left.float() @ weight.float().T + bias.float()).bfloat16().double()
        self.assertTrue(torch.all((actual - result["higher_precision_value"]).abs()
                                  <= result["absolute_error_bound_to_fp64_diagnostic"]))
        self.assertEqual(result["higher_precision_value"][0, 0].item(), 0.0)
        self.assertGreater(result["fp32_arithmetic_error_bound"][0, 0].item(), 0.0)
        self.assertEqual(result["operation_bound"], 7)
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["acceptance_contract_selected"])

    def test_linear_diagnostic_rejects_dtype_changes_and_unbounded_overflow(self):
        import torch
        for left, weight in ((torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bfloat16)),
                             (torch.full((1, 2), 1e30, dtype=torch.bfloat16),
                              torch.full((1, 2), 1e30, dtype=torch.bfloat16))):
            with self.subTest(left=left.dtype), self.assertRaises(ValueError):
                NUMERICS.bf16_linear_error_diagnostic(left, weight, None, torch)


if __name__ == "__main__":
    unittest.main()
