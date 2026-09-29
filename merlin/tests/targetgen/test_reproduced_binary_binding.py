"""A reproduced build may bind a later execution only through exact input bytes."""

import hashlib
import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.rtl.build_provenance import bind_reproduced_binary


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _fixture(tmp_path):
    source = tmp_path / "source-selection.json"
    source.write_text('{"source":"selected RTL"}')
    binary = tmp_path / "simulator"
    binary.write_bytes(b"reproduced executable")
    fields = {
        "selected_source": {"selection": {"sha256": _sha(source)}},
        "toolchain": {"simulator": {"sha256": _sha(binary)}},
    }
    prior = _write(tmp_path / "prior.json", fields)
    current = _write(tmp_path / "current.json", fields)
    attestation = _write(
        tmp_path / "attestation.json",
        {
            "status": "reproduced_exact_binary",
            "source_selection_sha256": _sha(source),
            "kernel_receipt_sha256": _sha(prior),
            "kernel_tested_binary_sha256": _sha(binary),
            "rebuilt_binary_sha256": _sha(binary),
        },
    )
    return source, binary, prior, current, attestation


def _bind(paths):
    source, binary, prior, current, attestation = paths
    return bind_reproduced_binary(
        attestation_path=attestation,
        attested_execution_path=prior,
        current_execution_path=current,
        source_selection_path=source,
        executable_path=binary,
        attestation_prior_receipt_field=("kernel_receipt_sha256",),
        attestation_source_field=("source_selection_sha256",),
        attestation_tested_binary_field=("kernel_tested_binary_sha256",),
        attestation_rebuilt_binary_field=("rebuilt_binary_sha256",),
        execution_source_field=("selected_source", "selection", "sha256"),
        execution_binary_field=("toolchain", "simulator", "sha256"),
    )


def test_new_execution_may_bind_prior_exact_binary_attestation(tmp_path):
    paths = _fixture(tmp_path)
    result = _bind(paths)
    assert result["status"] == "bound_exact_binary"
    assert result["attested_execution_receipt_sha256"] == _sha(paths[2])
    assert result["current_execution_receipt_sha256"] == _sha(paths[3])
    assert result["source_selection_sha256"] == _sha(paths[0])
    assert result["executable_sha256"] == _sha(paths[1])


@pytest.mark.parametrize("mutation", ["source", "binary", "prior", "current", "attestation"])
def test_any_broken_pin_refuses_binding(tmp_path, mutation):
    paths = _fixture(tmp_path)
    source, binary, prior, current, attestation = paths
    if mutation == "source":
        source.write_text("different RTL")
    elif mutation == "binary":
        binary.write_bytes(b"different simulator")
    elif mutation == "prior":
        prior.write_text("{}")
    elif mutation == "current":
        doc = json.loads(current.read_text())
        doc["toolchain"]["simulator"]["sha256"] = "0" * 64
        _write(current, doc)
    else:
        doc = json.loads(attestation.read_text())
        doc["status"] = "unverified"
        _write(attestation, doc)
    with pytest.raises(ValueError):
        _bind(paths)


def test_target_receipt_members_cannot_escape_their_evidence_root(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/target"))
    from bind_native_simulator import _safe_member

    assert _safe_member("kernel_matmul")
    assert all(not _safe_member(name) for name in ("", ".", "..", "../other", "a/b", "a\\b"))
