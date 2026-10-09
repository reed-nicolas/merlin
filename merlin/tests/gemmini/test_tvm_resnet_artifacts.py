"""Artifact and opt-in host graph checks; these tests do not build ResNet or qualify a device."""

import hashlib
import importlib.util
import json
import os
import sys
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

    def state_hash(self, model):
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def batchnorm_block(self, bias):
        torch = self.torch

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 3, 3, padding=1, bias=bias)
                self.bn1 = torch.nn.BatchNorm2d(3, eps=0.125)

            def forward(self, value):
                return self.bn1(self.conv1(value))

        model = Block().eval()
        with torch.no_grad():
            model.conv1.weight.copy_(torch.arange(54).reshape(3, 2, 3, 3) / 64 - 0.5)
            model.conv1.weight[1].zero_()
            if bias:
                model.conv1.bias.copy_(torch.tensor([0.5, 0.0, -0.25]))
            model.bn1.weight.copy_(torch.tensor([-2.0, 0.0, 3.0]))
            model.bn1.bias.copy_(torch.tensor([0.75, 0.0, -1.0]))
            model.bn1.running_mean.copy_(torch.tensor([0.25, 0.0, -0.5]))
            model.bn1.running_var.copy_(torch.tensor([0.0, 0.25, 16.0]))
        return model

    def test_batchnorm_folding_preserves_source_and_matches_independent_formula(self):
        torch, np = self.torch, self.np
        for has_bias in (False, True):
            with self.subTest(bias=has_bias):
                model = self.batchnorm_block(has_bias)
                before = {name: value.clone() for name, value in model.state_dict().items()}
                folded, report = self.verifier.fold_resnet_batchnorm(model, torch, self.state_hash)
                self.assertEqual(report["source_state_dict_sha256"], self.state_hash(model))
                self.assertEqual(report["derived_state_dict_sha256"], self.state_hash(folded))
                self.assertEqual(report["sites"][0]["epsilon"], 0.125)
                self.assertEqual(report["sites"][0]["conv"], "conv1")
                self.assertEqual(report["sites"][0]["batchnorm"], "bn1")
                self.assertIsInstance(folded.bn1, torch.nn.Identity)
                self.assertIsNot(folded.conv1, model.conv1)
                self.assertNotEqual(folded.conv1.weight.data_ptr(), model.conv1.weight.data_ptr())
                # Float64 scalar channel arithmetic is independent of the FP32 folding helper.
                bn = model.bn1
                scale = bn.weight.detach().numpy().astype(np.float64) / np.sqrt(bn.running_var.numpy().astype(np.float64) + bn.eps)
                expected_weight = model.conv1.weight.detach().numpy().astype(np.float64) * scale[:, None, None, None]
                bias = model.conv1.bias.detach().numpy() if has_bias else np.zeros(3)
                expected_bias = bn.bias.detach().numpy() + scale * (bias - bn.running_mean.numpy())
                np.testing.assert_allclose(folded.conv1.weight.detach().numpy(), expected_weight, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(folded.conv1.bias.detach().numpy(), expected_bias, rtol=1e-6, atol=1e-7)
                for key in ("weight", "bias"):
                    value = getattr(folded.conv1, key).detach().numpy()
                    self.assertEqual(report["sites"][0][key + "_sha256"], hashlib.sha256(value.tobytes()).hexdigest())
                for magnitude in (0.0, 1.0, 1024.0):
                    data = (torch.arange(40).reshape(1, 2, 4, 5) - 20).float() * magnitude
                    with torch.no_grad():
                        actual, expected = folded(data).numpy(), model(data).numpy()
                    self.assertTrue(self.verifier.compare_output(actual, expected, np)["passed"])
                for name, value in before.items():
                    self.assertTrue(torch.equal(model.state_dict()[name], value), name)
                self.assertTrue(torch.equal(folded.conv1.weight[1], torch.zeros_like(folded.conv1.weight[1])))
                folded.conv1.weight.data.zero_()
                self.assertTrue(torch.equal(model.conv1.weight, before["conv1.weight"]))

    def test_batchnorm_folding_rejects_incompatible_sites_without_source_mutation(self):
        torch = self.torch
        mutations = {
            "training": lambda model: model.train(),
            "child_training": lambda model: model.bn1.train(),
            "missing_mean": lambda model: setattr(model.bn1, "running_mean", None),
            "missing_variance": lambda model: setattr(model.bn1, "running_var", None),
            "no_running_statistics": lambda model: setattr(model.bn1, "track_running_stats", False),
            "no_affine": lambda model: setattr(model.bn1, "affine", False),
            "different_channels": lambda model: setattr(model.bn1, "num_features", 4),
            "zero_epsilon": lambda model: setattr(model.bn1, "eps", 0.0),
            "nonfinite_epsilon": lambda model: setattr(model.bn1, "eps", float("nan")),
            "negative_variance": lambda model: model.bn1.running_var.fill_(-1),
            "nonfinite_variance": lambda model: model.bn1.running_var.fill_(float("inf")),
            "float64": lambda model: model.double(),
            "missing_conv": lambda model: setattr(model, "conv1", torch.nn.Identity()),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                model = self.batchnorm_block(True)
                mutate(model)
                before = self.state_hash(model)
                with self.assertRaises(ValueError):
                    self.verifier.fold_resnet_batchnorm(model, torch, self.state_hash)
                self.assertEqual(self.state_hash(model), before)
        for model in (torch.nn.Identity().eval(), torch.nn.Sequential(torch.nn.BatchNorm2d(3)).eval()):
            with self.assertRaises(ValueError):
                self.verifier.fold_resnet_batchnorm(model, torch, self.state_hash)

    def test_batchnorm_folding_supports_resnet_downsample_pair(self):
        torch = self.torch
        model = torch.nn.Module()
        model.downsample = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1, bias=False), torch.nn.BatchNorm2d(3))
        model.eval()
        folded, report = self.verifier.fold_resnet_batchnorm(model, torch, self.state_hash)
        self.assertEqual(report["sites"][0]["conv"], "downsample.0")
        self.assertEqual(report["sites"][0]["batchnorm"], "downsample.1")
        self.assertIsInstance(folded.downsample[1], torch.nn.Identity)

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

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires an explicitly selected LLVM-enabled TVM host build")
    def test_graph_fusion_preserves_two_independent_inputs(self):
        import tvm
        from tvm import relax

        np = self.np
        x = relax.Var("x", relax.TensorStructInfo((2, 6), "float32"))
        bias = np.arange(6, dtype=np.float32) - 2
        builder = relax.BlockBuilder()
        with builder.function("main", [x]):
            with builder.dataflow():
                value = builder.emit(relax.op.add(x, relax.const(bias)))
                value = builder.emit(relax.op.nn.relu(value))
                value = builder.emit_output(relax.op.multiply(value, relax.const(np.float32(0.5))))
            builder.emit_func_output(value)
        original = builder.get()
        before = original.script()
        inventories = {}
        for mode in ("baseline", "optimized"):
            graph, lowered = self.verifier.prepare_graph(original, mode, tvm)
            self.assertEqual(original.script(), before, "Pipeline changed its input module")
            inventories[mode] = self.verifier.relax_inventory(lowered, tvm)
            vm = relax.VirtualMachine(relax.build(lowered, "llvm", pipeline=None), tvm.cpu())
            for offset in (0.0, -3.0):
                data = np.arange(12, dtype=np.float32).reshape(2, 6) - 4 + offset
                expected = np.maximum(data + bias, 0) * np.float32(0.5)
                actual = vm["main"](tvm.nd.array(data)).numpy()
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(self.verifier.compare_output(actual, expected, np)["passed"])
                self.assertFalse(self.verifier.compare_output(actual + 1, expected, np)["passed"])
        self.assertLess(inventories["optimized"]["function_counts"]["tir"], inventories["baseline"]["function_counts"]["tir"])
        with self.assertRaises(ValueError):
            self.verifier.prepare_graph(original, "both", tvm)

    def labels_manifest(self):
        return {"input_sha256": "a" * 64, "class_ids": [f"class-{i}" for i in range(1000)], "samples": [{"id": "duplicate", "label": 7}, {"id": "duplicate", "label": 2}]}

    def test_labels_preserve_declared_class_order_and_duplicate_sample_ids(self):
        manifest = self.labels_manifest()
        manifest["class_ids"][0], manifest["class_ids"][1] = manifest["class_ids"][1], manifest["class_ids"][0]
        path = self.root / "labels.json"
        path.write_text(json.dumps(manifest))
        actual, identity = self.verifier.load_labels(path, {"sha256": "a" * 64}, 2)
        self.assertEqual(actual, manifest)
        self.assert_identity(identity, path)

    def test_invalid_labels_are_rejected(self):
        valid = self.labels_manifest()
        cases = {
            "root_list": [],
            "missing_mapping": {key: value for key, value in valid.items() if key != "class_ids"},
            "extra_field": {**valid, "extra": None},
            "hash_mismatch": {**valid, "input_sha256": "b" * 64},
            "hash_type": {**valid, "input_sha256": 1},
            "short_mapping": {**valid, "class_ids": valid["class_ids"][:-1]},
            "mapping_type": {**valid, "class_ids": {}},
            "duplicate_class": {**valid, "class_ids": ["same"] * 1000},
            "empty_class": {**valid, "class_ids": [" "] + valid["class_ids"][1:]},
            "class_type": {**valid, "class_ids": [None] + valid["class_ids"][1:]},
            "sample_count": {**valid, "samples": valid["samples"][:1]},
            "samples_type": {**valid, "samples": {}},
        }
        for name, sample in {"bool": {"id": "x", "label": True}, "negative": {"id": "x", "label": -1}, "too_large": {"id": "x", "label": 1000}, "float": {"id": "x", "label": 2.0}, "string": {"id": "x", "label": "2"}, "empty_id": {"id": " ", "label": 2}, "id_type": {"id": 3, "label": 2}, "missing_id": {"label": 2}, "extra_sample_field": {"id": "x", "label": 2, "extra": 0}, "sample_type": []}.items():
            cases[name] = {**valid, "samples": [sample, valid["samples"][1]]}
        path = self.root / "labels.json"
        for name, manifest in cases.items():
            with self.subTest(case=name):
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    self.verifier.load_labels(path, {"sha256": "a" * 64}, 2)
        for text in ('{"input_sha256": 1, "input_sha256": 2}', '{invalid', '{"samples": [{"id": "a", "id": "b"}]}'):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    self.verifier.load_labels(path, {"sha256": "a" * 64}, 2)

    def test_ranking_ties_and_complete_stream_counts(self):
        np = self.np
        logits = np.zeros((1, 1000), dtype=np.float32)
        logits[0, [9, 7, 2]] = 1
        first = self.verifier.classify_logits(logits, 7, np)
        self.assertEqual(first, {"top5": [2, 7, 9, 0, 1], "top1_hit": False, "top5_hit": True})
        changed = logits.copy()
        changed[0, 7] = 2
        second = self.verifier.classify_logits(changed, 7, np)
        miss = self.verifier.classify_logits(logits, 999, np)
        rows = [{"torch": first, "relax": second}, {"torch": second, "relax": second}, {"torch": miss, "relax": miss}]
        counts = self.verifier.accuracy_counts(rows)
        self.assertEqual(counts["samples"], 3)
        self.assertEqual(counts["top1_disagreements"], 1)
        self.assertEqual(counts["torch"]["top1"], {"correct": 1, "rate": 1 / 3})
        self.assertEqual(counts["relax"]["top1"], {"correct": 2, "rate": 2 / 3})
        self.assertEqual(counts["torch"]["top5"], {"correct": 2, "rate": 2 / 3})
        with self.assertRaises(ValueError):
            self.verifier.accuracy_counts([])
        for invalid in (logits[0], np.full_like(logits, np.nan)):
            with self.assertRaises(ValueError):
                self.verifier.classify_logits(invalid, 0, np)

    def test_default_loader_has_no_label_artifact(self):
        model = SimpleNamespace(cpu=lambda: model, eval=lambda: model, paper_ready=False)
        loaded_model, loaded = self.verifier.load_model(SimpleNamespace(get_model_and_inputs=lambda: (model, ())), SimpleNamespace(inputs=None, labels=None), self.torch, self.np)
        self.assertIs(loaded_model, model)
        self.assertIsNone(loaded["labels"])
        self.assertEqual(loaded["artifacts"], {})

    def test_labels_reject_changes_during_loading(self):
        path = self.root / "labels.json"
        path.write_text(json.dumps(self.labels_manifest()))
        identity = {"path": str(path), "sha256": "before"}
        with mock.patch.object(self.verifier, "file_identity", side_effect=[identity, {**identity, "sha256": "after"}]):
            with self.assertRaisesRegex(ValueError, "changed while loading"):
                self.verifier.load_labels(path, {"sha256": "a" * 64}, 2)

    def test_labels_require_supplied_mode_before_output_creation(self):
        output = self.root / "output"
        argv = ["verify_resnet.py", "--model2mlir-root", str(self.root), "--tvm-source", str(self.root), "--output-dir", str(output), "--labels", str(self.root / "labels.json")]
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                self.verifier.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
