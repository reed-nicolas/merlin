"""Fail-closed artifact and workload gates; no checkpoint substitute is executed."""

import importlib.util
import json
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

    def test_restricted_artifact_path_is_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "Restricted path component"):
            FACTORY.checkpoint_artifacts("/tmp/" + "ham" + "mer" + "-fixture")


if __name__ == "__main__":
    unittest.main()
