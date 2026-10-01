"""A materialized capture must bind its external quantization manifest end to end."""

import hashlib
import json

import pytest

from merlin.targetgen.application_inventory import verify_capture_receipt


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_capture(root, *, change=None):
    manifest = {"schema": "m2m.quantization_manifest.v1", "sites": [{"site_id": "one", "status": "host"}]}
    digest = _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    mlir = f'module attributes {{prov.quantization_manifest_sha256 = "{digest}"}} {{}}\n'.encode()
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    pointer = {"path": "quantization-manifest.json", "sha256": _sha(manifest_bytes), "manifest_sha256": digest}
    if change == "pointer":
        pointer["manifest_sha256"] = "0" * 64
    if change == "mlir":
        mlir = mlir.replace(digest.encode(), b"0" * 64)
    if change == "manifest":
        manifest_bytes = b'{"schema":"m2m.quantization_manifest.v1","sites":[]}\n'
        pointer["sha256"] = _sha(manifest_bytes)
    metadata = {"quantization_manifest": pointer}
    contents = {
        "model.mlir": mlir,
        "weights.safetensors": b"weights",
        "weights.safetensors.manifest.json": b"{}",
        "meta.json": json.dumps(metadata).encode(),
        "quantization-manifest.json": manifest_bytes,
    }
    if change == "missing":
        contents.pop("quantization-manifest.json")
        metadata.pop("quantization_manifest")
        contents["meta.json"] = json.dumps(metadata).encode()
    for name, raw in contents.items():
        (root / name).write_bytes(raw)
    (root / "capture_receipt.json").write_text(json.dumps({
        "schema": "m2m.capture-receipt.v1",
        "materialized_abi": {"complete": True},
        "artifacts": {name: {"bytes": len(raw), "sha256": _sha(raw)} for name, raw in contents.items()},
    }))


@pytest.mark.parametrize("change", [None, "pointer", "mlir", "manifest", "missing"])
def test_external_manifest_binding(change, tmp_path):
    _write_capture(tmp_path, change=change)
    observed = verify_capture_receipt(tmp_path / "model.mlir")
    assert observed["status"] == ("verified_materialized" if change is None else "unverified")
    assert observed["source_closure_verified"] is False
