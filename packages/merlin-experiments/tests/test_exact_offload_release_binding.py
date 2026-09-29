"""The reviewed Phase 0 source identity is reopened before exact offload."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.corpus import release
from merlin_experiments.phase0 import evidence as phase0_evidence
from merlin_experiments.runner import fingerprint
from merlin_experiments.spec import SpecError

from merlin.common.digest import sha256_bytes
from merlin.llvmlower import exact_offload
from merlin.llvmlower.exact_offload import ExactOffloadSelection, SelectedKernel


def test_installed_verifier_entry_point_is_the_release_owner(monkeypatch) -> None:
    group = "merlin.exact_offload_release"
    provider = EntryPoint(
        name="reviewed_phase0",
        value="merlin_experiments.corpus.release:verify_exact_offload_binding",
        group=group,
    )
    monkeypatch.setattr(exact_offload, "entry_points", lambda **_kwargs: (provider,))
    assert exact_offload._release_verifier() is release.verify_exact_offload_binding
    monkeypatch.setattr(exact_offload, "entry_points", lambda **_kwargs: ())
    with pytest.raises(ValueError, match="one installed"):
        exact_offload._release_verifier()


def test_exact_offload_requires_selected_reviewed_capture_and_rechecks_it(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "release"
    private = root / "private"
    private.mkdir(parents=True, mode=0o700)
    run = tmp_path / "run"
    run.mkdir()
    (run / "resolved-plan.json").write_text("{}")
    descriptor = tmp_path / "selected-descriptor.yaml"
    descriptor.write_text("target: test_device\n")
    spec = tmp_path / "software-spec.yaml"
    spec.write_bytes(b"reviewed software")
    contract = tmp_path / "target-contract.yaml"
    contract.write_bytes(b"reviewed contract")
    inventory_path = tmp_path / "inventory.json"
    capture = tmp_path / "capture.mlir"
    capture.write_bytes(b"exact captured model")
    bundle = tmp_path / "evidence"
    bundle.mkdir()
    (bundle / "evidence-manifest.json").write_text("{}")
    lineage = {"evidence_manifest_sha256": fingerprint(bundle / "evidence-manifest.json")}
    plan = {
        "target": "test_device",
        "phase0_evidence_bundle": str(bundle),
        "phases": {
            "0": {
                "inputs": {
                    "descriptor": str(descriptor),
                    "software_spec": str(spec),
                    "capability_contract": str(contract),
                }
            }
        },
    }
    prepared = {
        "schema_version": 1,
        "source_run": str(run),
        "target": "test_device",
        "source_plan_sha256": fingerprint(run / "resolved-plan.json"),
        "source_output_sha256": "output-digest",
        "source_descriptor_sha256": fingerprint(descriptor),
        "generation_lineage": lineage,
    }
    preparation = private / "preparation.json"
    preparation.write_text(json.dumps(prepared))
    preparation.chmod(0o600)
    monkeypatch.setattr(
        release,
        "verify",
        lambda _seal, _descriptor: {
            "release": str(root),
            "review_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(release, "source_run", lambda _run: (plan, {"output_sha256": "output-digest"}, tmp_path))
    monkeypatch.setattr(release, "generation_lineage", lambda _plan, _generated: lineage)
    sources = [
        SimpleNamespace(role="software-spec", path=spec, content=spec.read_bytes()),
        SimpleNamespace(role="target-contract", path=contract, content=contract.read_bytes()),
        SimpleNamespace(role="application-inventory", path=inventory_path, content=b"{}"),
        SimpleNamespace(role="application-capture:app", path=capture, content=capture.read_bytes()),
    ]
    selected = SimpleNamespace(
        target="test_device",
        status="verified",
        source_snapshots=sources,
        application_inventory_identity={"status": "digest_bound"},
        application_inventory={
            "applications": {
                "app": {
                    "capture_source_path": capture.name,
                    "capture_sha256": sha256_bytes(capture.read_bytes()),
                }
            }
        },
    )
    monkeypatch.setattr(phase0_evidence, "load_exported_evidence", lambda _bundle: selected)
    selection = ExactOffloadSelection(
        target="test_device",
        model_sha256=sha256_bytes(capture.read_bytes()),
        package_sha256="0" * 64,
        transport="test",
        abi_sha256="1" * 64,
        kernels=(SelectedKernel("op", "interface", "2" * 64),),
        software_spec_sha256=sha256_bytes(spec.read_bytes()),
        capability_contract_sha256=sha256_bytes(contract.read_bytes()),
    )
    seal = private / "seal.json"
    bound = release.bind_exact_offload(selection, seal_path=seal, descriptor=descriptor, application="app")
    monkeypatch.setattr(exact_offload, "_release_verifier", lambda: release.verify_exact_offload_binding)
    bound.check_release()

    original_spec = sources[0]
    selected.source_snapshots[0] = SimpleNamespace(role="software-spec", path=spec, content=b"changed")
    with pytest.raises(SpecError, match="software-spec bytes"):
        bound.check_release()
    selected.source_snapshots[0] = original_spec
    with pytest.raises(SpecError, match="model is not"):
        release.bind_exact_offload(
            replace(selection, model_sha256="f" * 64), seal_path=seal, descriptor=descriptor, application="app"
        )
    selected.status = "diagnostic"
    with pytest.raises(SpecError, match="verified same-target"):
        bound.check_release()
