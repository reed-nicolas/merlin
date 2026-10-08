"""Artifact validation only; these tests do not qualify a backend or build ResNet."""

import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from merlin.common.paths import repo_root


class ResNetArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch

        cls.np, cls.torch = np, torch
        path = repo_root() / "examples/gemmini/comparisons/tvm/verify_resnet.py"
        spec = importlib.util.spec_from_file_location("tvm_resnet_artifact_verifier", path)
        cls.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verifier)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="resnet-artifact-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def assert_identity(self, identity, path):
        self.assertEqual(identity["path"], str(path.resolve()))
        self.assertEqual(identity["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_images_preserve_all_values_order_and_duplicates(self):
        np = self.np
        images = np.zeros((3, 3, 224, 224), dtype=np.float32)
        images[0].fill(0.75)
        images[1].fill(-0.25)
        images[2] = images[0]
        for source in (images, images[:, None]):
            with self.subTest(shape=source.shape):
                path = self.root / "images.npz"
                np.savez(path, images=source)
                actual, identity = self.verifier.load_images(path, np)
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_array_equal(actual, images[:, None])
                self.assert_identity(identity, path)

    def test_invalid_images_are_rejected(self):
        np = self.np
        valid = np.zeros((1, 3, 224, 224), dtype=np.float32)
        cases = {
            "float64": valid.astype(np.float64),
            "integer": valid.astype(np.int32),
            "object": valid.astype(object),
            "empty": valid[:0],
            "wrong_rank": valid[0],
            "wrong_channels": valid[:, :2],
            "wrong_extent": valid[:, :, :223],
            "wrong_batch_axis": np.zeros((1, 2, 3, 224, 224), dtype=np.float32),
            "nan": np.full_like(valid, np.nan),
            "infinity": np.full_like(valid, np.inf),
        }
        path = self.root / "images.npz"
        for name, images in cases.items():
            with self.subTest(case=name):
                np.savez(path, images=images)
                with self.assertRaises((ValueError, TypeError)):
                    self.verifier.load_images(path, np)
        np.savez(path, unrelated=valid)
        with self.assertRaises((ValueError, KeyError)):
            self.verifier.load_images(path, np)

    def model(self):
        torch = self.torch
        model = torch.nn.Linear(2, 2)
        model.register_buffer("counter", torch.tensor(11, dtype=torch.int64))
        with torch.no_grad():
            model.weight.fill_(3)
            model.bias.fill_(4)
        return model

    def test_checkpoint_preserves_zeros_and_integer_buffers(self):
        torch = self.torch
        model = self.model()
        state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
        state["counter"] = torch.tensor(19, dtype=torch.int64)
        path = self.root / "weights.pt"
        torch.save(state, path)
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            identity = self.verifier.load_checkpoint(model, path, torch)
        self.assertIs(load.call_args.kwargs["weights_only"], True)
        self.assertEqual(str(load.call_args.kwargs["map_location"]), "cpu")
        for key, expected in state.items():
            actual = model.state_dict()[key]
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected), key)
        self.assert_identity(identity, path)

    def test_invalid_checkpoints_are_rejected_before_model_mutation(self):
        torch = self.torch
        model = self.model()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        valid = {key: torch.zeros_like(value) for key, value in before.items()}
        cases = {
            "missing_key": {key: value for key, value in valid.items() if key != "bias"},
            "extra_key": {**valid, "extra": torch.zeros(1)},
            "nested_state": {"state_dict": valid},
            "float64": {**valid, "weight": valid["weight"].double()},
            "buffer_cast": {**valid, "counter": valid["counter"].float()},
            "wrong_shape": {**valid, "bias": torch.zeros(3)},
            "non_tensor": {**valid, "counter": 0},
            "sparse": {**valid, "weight": valid["weight"].to_sparse()},
            "nan": {**valid, "bias": torch.full_like(valid["bias"], float("nan"))},
            "infinity": {**valid, "bias": torch.full_like(valid["bias"], float("inf"))},
        }
        path = self.root / "weights.pt"
        for name, state in cases.items():
            with self.subTest(case=name):
                torch.save(state, path)
                with self.assertRaises((ValueError, TypeError)):
                    self.verifier.load_checkpoint(model, path, torch)
                for key, expected in before.items():
                    self.assertTrue(torch.equal(model.state_dict()[key], expected), key)

    def test_loader_environment_is_isolated_and_restored_on_failure(self):
        inherited = {"M2M_RESNET_INPUT_NPZ": "/unused/images.npz", "M2M_RESNET_CALIBRATION_NPZ": "/unused/calibration.npz", "M2M_RESNET_PRETRAINED": "1", "M2M_RESNET_PAPER_READY": "1", "M2M_SESSION_STEPS": "700"}

        def fail():
            self.assertNotIn("M2M_RESNET_INPUT_NPZ", os.environ)
            self.assertNotIn("M2M_RESNET_CALIBRATION_NPZ", os.environ)
            self.assertEqual(os.environ["M2M_RESNET_PRETRAINED"], "0")
            self.assertEqual(os.environ["M2M_RESNET_PAPER_READY"], "0")
            self.assertEqual(os.environ["M2M_SESSION_STEPS"], "2")
            raise RuntimeError("loader failed")

        with mock.patch.dict(os.environ, inherited):
            before = dict(os.environ)
            with self.assertRaisesRegex(RuntimeError, "loader failed"):
                self.verifier.load_model(SimpleNamespace(get_model_and_inputs=fail), SimpleNamespace(inputs=None), self.torch, self.np)
            self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
