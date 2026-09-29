"""A raw frontend inventory must never masquerade as a materialized bundle."""

import hashlib
import json

import pytest

from merlin.targetgen._m2m_capture_worker import _diagnostic_model_copy
from merlin.targetgen.application_inventory import verify_capture_receipt


def test_diagnostic_model_copy_binds_bytes_without_granting_admission(tmp_path):
    loader = tmp_path / "loader.py"
    loader.write_text("# tiny model\n")
    names = (
        "linalg.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
        "inputs.json", "golden.json", "frontend-trace.json", "pytorch-opset.json", "meta.json",
    )
    for name in names:
        (tmp_path / name).write_bytes(name.encode())

    missing_api = {"same_conversion_missing": ["m2m/capture/provenance.py"]}
    _diagnostic_model_copy(tmp_path, loader, capture_api=missing_api)

    receipt = json.loads((tmp_path / "diagnostic-capture.json").read_text())
    assert (tmp_path / "model.mlir").read_bytes() == (tmp_path / "linalg.mlir").read_bytes()
    assert not (tmp_path / "capture_receipt.json").exists()
    assert verify_capture_receipt(tmp_path / "model.mlir")["status"] == "unverified"
    assert receipt["phase0_admission"] == "not_granted"
    assert receipt["source_closure_verified"] is False
    assert receipt["materialized_abi"] is False
    assert receipt["capture_api"] == missing_api
    assert set(receipt["artifacts"]) == {*names, "model.mlir"}
    for name, record in receipt["artifacts"].items():
        data = (tmp_path / name).read_bytes()
        assert record == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with pytest.raises(ValueError, match="cannot reuse"):
        _diagnostic_model_copy(tmp_path, loader, capture_api=missing_api)
