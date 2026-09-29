"""A saved headline probe must execute the exact source-bound generated files."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from merlin.common.paths import repo_root


def test_native_probe_rejects_stale_generation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/verification"))
    from probe_headline_kernel import _verified_generation

    projection = {"source": {"model_mlir_sha256": "source"}, "projection": {"geometry": {"M": 1}}}
    capsule = {"source_kernel_window": projection, "diagnostic_limits": {"target_admission": False}}
    capsule_path = tmp_path / "capsule.yaml"
    interface_path = tmp_path / "capsule.interface.mlir"
    generation_path = tmp_path / "generation.json"
    capsule_path.write_text(yaml.safe_dump(capsule), encoding="utf-8")
    interface_path.write_text("module {}\n", encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    generation = {
        "schema": "merlin.gemmini-headline-window-generation.v1",
        "status": "generated_not_executed",
        "source_kernel_window": projection,
        "diagnostic_limits": capsule["diagnostic_limits"],
        "capsule_sha256": digest(capsule_path),
        "interface_sha256": digest(interface_path),
    }
    generation_path.write_text(json.dumps(generation), encoding="utf-8")
    binding = _verified_generation(projection, tmp_path)
    assert binding == {
        "generation_sha256": digest(generation_path),
        "capsule_sha256": digest(capsule_path),
        "interface_sha256": digest(interface_path),
    }

    interface_path.write_text("module { func.func @different() }\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ from their generated source binding"):
        _verified_generation(projection, tmp_path)

    interface_path.write_text("module {}\n", encoding="utf-8")
    capsule["source_kernel_window"]["source"]["model_mlir_sha256"] = "other"
    capsule_path.write_text(yaml.safe_dump(capsule), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from their generated source binding"):
        _verified_generation(projection, tmp_path)


def test_scaled_readout_uses_fp32_rne_and_saturation(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/verification"))
    from probe_headline_kernel import _scaled_i8_reference

    assert _scaled_i8_reference([[-255, -3, -1, 1, 3, 255]], 0.5) == [
        [-128, -2, 0, 0, 2, 127]
    ]
    with pytest.raises(ValueError, match="exactly representable FP32"):
        _scaled_i8_reference([[1]], 0.1)
