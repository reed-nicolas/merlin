"""A current source inventory cannot upgrade a historical capture."""

import hashlib
import json

import pytest
from merlin_experiments.phase0.capture_execution_attestation import (
    AttestationNotVerified,
    assess_sealed_m2m_capture,
    diagnose_capture,
    require_verified_execution,
    write_diagnostic,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialized_capture(root):
    root.mkdir()
    files = {
        "model.mlir": b"module {}\n",
        "weights.safetensors": b"weights",
        "weights.safetensors.manifest.json": b"{}\n",
    }
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    receipt = {
        "schema": "m2m.capture-receipt.v1",
        "materialized_abi": {"complete": True},
        # Even a claimed true value in an M2M receipt cannot mint a Merlin execution attestation.
        "source_closure_verified": True,
        "artifacts": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()
        },
    }
    (root / "capture_receipt.json").write_text(json.dumps(receipt))


def test_diagnostic_binds_current_bytes_but_never_admits_old_capture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "loader.py").write_bytes(b"print('first')\n")
    capture = tmp_path / "capture"
    _materialized_capture(capture)

    first = diagnose_capture(capture, source, ["loader.py"])
    assert first["capture"]["materialized_receipt"]["status"] == "verified_materialized"
    assert first["source_closure_verified"] is False
    assert first["fresh_execution"] is False
    assert first["selected_source"]["members"]["loader.py"]["sha256"] == hashlib.sha256(b"print('first')\n").hexdigest()
    output = tmp_path / "diagnostic.json"
    write_diagnostic(output, first)
    with pytest.raises(FileExistsError):
        write_diagnostic(output, first)
    with pytest.raises(AttestationNotVerified, match="no verified fresh sealed execution"):
        require_verified_execution(json.loads(output.read_text()))

    (source / "loader.py").write_bytes(b"print('second')\n")
    second = diagnose_capture(capture, source, ["loader.py"])
    assert first["selected_source"]["inventory_sha256"] != second["selected_source"]["inventory_sha256"]
    forged = dict(
        first,
        status="verified_sealed_execution",
        source_closure_verified=True,
        fresh_execution=True,
        issuer="self-reported",
    )
    with pytest.raises(AttestationNotVerified, match="no supported Merlin sealed execution issuer"):
        require_verified_execution(forged)


def test_diagnostic_rejects_ambiguous_source_selection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "loader.py").write_text("pass\n")
    (source / "linked.py").symlink_to(source / "loader.py")
    capture = tmp_path / "capture"
    _materialized_capture(capture)
    with pytest.raises(ValueError, match="symlink"):
        diagnose_capture(capture, source, ["linked.py"])
    with pytest.raises(ValueError, match="unsafe selected source path"):
        diagnose_capture(capture, source, ["../loader.py"])
    with pytest.raises(ValueError, match="overlap"):
        diagnose_capture(capture, source, ["loader.py", "loader.py/child"])
    document = diagnose_capture(capture, source, ["loader.py"])
    with pytest.raises(ValueError, match="outside"):
        write_diagnostic(capture / "attestation.json", document)


def test_sealed_m2m_assessment_binds_selected_model_and_receipt_without_admission(tmp_path, monkeypatch):
    from merlin_experiments.capture_execution import sealed_m2m

    run = tmp_path / "run"
    run.mkdir()
    capture = run / "capture"
    _materialized_capture(capture)
    pending = run / "sealed_m2m_pending.json"
    pending.write_bytes(b"pending replay\n")
    model = capture / "model.mlir"
    selected = {
        "model_sha256": _digest(model),
        "capture_receipt_sha256": _digest(capture / "capture_receipt.json"),
        "sealed_receipt_sha256": _digest(pending),
    }
    calls = []

    def replay(path, *, bwrap_binary=None):
        calls.append(path)
        return {
            "schema": sealed_m2m.SCHEMA,
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "phase0_admission": "not_granted",
            "receipt_sha256": _digest(pending),
            "capture_dtype": "fp32",
        }

    monkeypatch.setattr(sealed_m2m, "replay_verify", replay)
    assessment = assess_sealed_m2m_capture(run, model, **selected)
    assert assessment["status"] == "replay_verified_nonadmissible"
    assert assessment["phase0_admission"] == "not_granted"
    assert assessment["source_closure_verified"] is False
    assert assessment["capture"]["sealed_receipt_sha256"] == _digest(pending)
    assert assessment["replay"]["receipt_sha256"] == _digest(pending)
    assert calls == [run]
    with pytest.raises(AttestationNotVerified):
        require_verified_execution(assessment)

    mismatch = assess_sealed_m2m_capture(run, model, **{**selected, "model_sha256": "0" * 64})
    assert mismatch["status"] == "unverified"
    assert "differ" in mismatch["blockers"][0]
    assert calls == [run]
    changed_run = assess_sealed_m2m_capture(run, model, **{**selected, "sealed_receipt_sha256": "0" * 64})
    assert changed_run["status"] == "unverified"
    assert "sealed receipt bytes differ" in changed_run["blockers"][0]
    assert calls == [run]
    other = tmp_path / "other.mlir"
    other.write_bytes(model.read_bytes())
    misplaced = assess_sealed_m2m_capture(run, other, **selected)
    assert "exact sealed M2M run" in misplaced["blockers"][0]
    assert calls == [run]

    def historical(path, *, bwrap_binary=None):
        return {**replay(path, bwrap_binary=bwrap_binary), "schema": sealed_m2m.SCHEMA_V1}

    monkeypatch.setattr(sealed_m2m, "replay_verify", historical)
    unsupported = assess_sealed_m2m_capture(run, model, **selected)
    assert "no supported CPU v2 proof" in unsupported["blockers"][0]
    assert unsupported["phase0_admission"] == "not_granted"


def test_sealed_m2m_assessment_rejects_failed_replay_and_changed_bytes(tmp_path, monkeypatch):
    from merlin_experiments.capture_execution import sealed_m2m

    run = tmp_path / "run"
    run.mkdir()
    capture = run / "capture"
    _materialized_capture(capture)
    pending = run / "sealed_m2m_pending.json"
    pending.write_bytes(b"pending replay\n")
    model = capture / "model.mlir"
    selected = {
        "model_sha256": _digest(model),
        "capture_receipt_sha256": _digest(capture / "capture_receipt.json"),
        "sealed_receipt_sha256": _digest(pending),
    }

    def failed(*_args, **_kwargs):
        raise sealed_m2m.SealedM2MError("source snapshot differs")

    monkeypatch.setattr(sealed_m2m, "replay_verify", failed)
    assessment = assess_sealed_m2m_capture(run, model, **selected)
    assert "source snapshot differs" in assessment["blockers"][0]
    assert assessment["phase0_admission"] == "not_granted"

    def changed(*_args, **_kwargs):
        model.write_bytes(b"changed\n")
        return {
            "schema": sealed_m2m.SCHEMA,
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "phase0_admission": "not_granted",
            "receipt_sha256": _digest(pending),
        }

    monkeypatch.setattr(sealed_m2m, "replay_verify", changed)
    assessment = assess_sealed_m2m_capture(run, model, **selected)
    assert "changed during assessment" in assessment["blockers"][0]
    assert assessment["phase0_admission"] == "not_granted"

    model.write_bytes(b"module {}\n")

    def changed_pending(*_args, **_kwargs):
        pending.write_bytes(b"different pending replay\n")
        return {
            "schema": sealed_m2m.SCHEMA,
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "phase0_admission": "not_granted",
            "receipt_sha256": selected["sealed_receipt_sha256"],
        }

    monkeypatch.setattr(sealed_m2m, "replay_verify", changed_pending)
    assessment = assess_sealed_m2m_capture(run, model, **selected)
    assert "changed during assessment" in assessment["blockers"][0]
    assert assessment["phase0_admission"] == "not_granted"
