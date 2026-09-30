"""Scheduling imports bind bytes without granting execution or performance claims."""

import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import evidence, scheduling

from merlin.runtime.backends import base
from merlin.targetgen import readout_facet, target_registry
from merlin.targetgen.rtl import facts


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    artifact = root / "nested/profile.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"observations\n")
    document = {
        "schema": scheduling.SCHEMA,
        "target": "fixture",
        "hardware": {"config": "TestConfig", "source_ir_sha256": digest(b"HW")},
        "producer": {"name": "fixture-projector", "sha256": digest(b"producer")},
        "semantics": {"origin": "acceptance"},
        "limitations": ["partial"],
        "profiles": [{"component": "unit", "fields": {"latency": 3}, "assumptions": [], "limitations": []}],
        "artifacts": [{"role": "profile", "path": "nested/profile.bin", "sha256": digest(artifact.read_bytes())}],
        "verification": {
            "artifact_bytes_verified": True,
            "projection_evidence_agreement_verified": True,
            "compiler_identity_verified": False,
            "footprints_reproduced": False,
            "schedule_legality_verified": False,
            "rtl_replayed": False,
        },
    }
    path = root / "source.json"
    path.write_text(json.dumps(document))
    return path, document, artifact


def load(path, descriptor=None, consistency=None):
    sources = []

    def observe(member, role, required=False):
        raw = member.read_bytes()
        sources.append(evidence.EvidenceSource(member, role, raw))
        return raw

    return scheduling.load_selection(
        path, target="fixture", descriptor=descriptor or {}, source_consistency=consistency or {}, observe=observe
    ), sources


def test_hardware_source_identity_not_config_alone(tmp_path):
    path, document, _ = bundle(tmp_path)
    descriptor = {"rtl": {"elaboration": {"config": "TestConfig"}}}
    assert load(path, descriptor)[0]["hardware_comparison"]["status"] == "unknown"
    consistency = {
        "status": "verified",
        "sources": [{"role": "core_hw", "sha256": document["hardware"]["source_ir_sha256"]}],
    }
    view, _ = load(path, descriptor, consistency)
    assert view["hardware_comparison"]["status"] == "matched"
    assert view["qualified_for_use"] is False
    consistency["sources"][0]["role"] = "core_hw_generic"
    assert load(path, descriptor, consistency)[0]["hardware_comparison"]["status"] == "unknown"
    consistency["sources"][0].update(role="soc_hw", sha256=digest(b"other"))
    assert load(path, descriptor, consistency)[0]["hardware_comparison"]["status"] == "mismatch"
    descriptor["rtl"]["elaboration"]["config"] = "OtherConfig"
    assert load(path, descriptor)[0]["hardware_comparison"]["status"] == "mismatch"


def test_genericization_input_never_output(tmp_path):
    path, document, _ = bundle(tmp_path)
    descriptor = {"rtl": {"elaboration": {"config": "TestConfig"}}}
    declared = document["hardware"]["source_ir_sha256"]
    consistency = {
        "status": "verified",
        "genericization": {
            "kind": "circt_generic_serialization",
            "input": {"sha256": digest(b"other")},
            "output": {"sha256": declared},
        },
    }
    assert load(path, descriptor, consistency)[0]["hardware_comparison"]["status"] == "mismatch"
    consistency["genericization"]["input"]["sha256"] = declared
    assert load(path, descriptor, consistency)[0]["hardware_comparison"]["status"] == "matched"


@pytest.mark.parametrize(
    "mutation",
    [
        "digest",
        "escape",
        "absolute",
        "reserved",
        "duplicate_role",
        "duplicate_path",
        "type",
        "empty_profiles",
        "empty_artifacts",
        "duplicate_component",
        "extra_field",
    ],
)
def test_malformed_bundle_fails(tmp_path, mutation):
    path, document, _ = bundle(tmp_path)
    row = document["artifacts"][0]
    if mutation == "digest":
        row["sha256"] = "0" * 64
    elif mutation == "escape":
        row["path"] = "../profile.bin"
    elif mutation == "absolute":
        row["path"] = "/profile.bin"
    elif mutation == "reserved":
        row["path"] = "manifest.json"
    elif mutation == "duplicate_role":
        document["artifacts"].append({**row, "path": "other.bin"})
    elif mutation == "duplicate_path":
        document["artifacts"].append({**row, "role": "other"})
    elif mutation == "empty_profiles":
        document["profiles"] = []
    elif mutation == "empty_artifacts":
        document["artifacts"] = []
    elif mutation == "duplicate_component":
        document["profiles"].append(document["profiles"][0])
    elif mutation == "extra_field":
        document["extra"] = 1
    else:
        document["profiles"][0]["fields"] = []
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load(path)


@pytest.mark.parametrize("token", ['"target":"fixture","target":"fixture",', '"number":NaN,', '"number":1e999,'])
def test_duplicate_and_nonfinite_json_fails(tmp_path, token):
    path, _, _ = bundle(tmp_path)
    path.write_text("{" + token + path.read_text()[1:])
    with pytest.raises(ValueError):
        load(path)


def test_missing_and_symlink_member_fails(tmp_path):
    path, _, artifact = bundle(tmp_path)
    artifact.unlink()
    with pytest.raises(FileNotFoundError):
        load(path)
    external = tmp_path / "external"
    external.write_bytes(b"observations\n")
    artifact.symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        load(path)


def setup_selection(monkeypatch, tmp_path):
    provider = tmp_path / "support"
    provider.mkdir()
    contract = provider / "target_contract.yaml"
    contract.write_text("name: fixture\ncompute_units: []\n")
    raw = tmp_path / "facts.json"
    raw.write_text(json.dumps({"facts": {"arrays": [{"rows": 4, "cols": 4}], "memories": []}}))
    monkeypatch.setattr(
        target_registry, "resolve", lambda target: SimpleNamespace(base=provider, capability_contract_path=contract)
    )
    monkeypatch.setattr(facts, "find_facts", lambda target, explicit=None: raw)
    monkeypatch.setattr(facts, "ensure_facts", lambda *a, **k: pytest.fail("must never extract"))
    monkeypatch.setattr(readout_facet, "capture_inputs", lambda *a, **k: {"scalar_abi": None, "readouts": None})
    monkeypatch.setattr(base, "execution_capability_facts", lambda target: {})
    return raw


def test_export_reload_relocation_preserves_bytes_and_views(monkeypatch, tmp_path):
    raw = setup_selection(monkeypatch, tmp_path)
    before = evidence.select_evidence("fixture", facts_path=raw)
    path, _, _ = bundle(tmp_path)
    selected = evidence.select_evidence("fixture", facts_path=raw, scheduling_evidence=path)
    assert selected.target_profile == before.target_profile
    assert selected.performance_facts == before.performance_facts
    assert selected.derivation_identity == before.derivation_identity
    assert "scheduling_evidence" not in json.loads(before.views_json)
    original = tmp_path / "export"
    evidence.export_evidence(selected, original)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    shutil.rmtree(path.parent)
    raw.unlink()
    restored = evidence.load_exported_evidence(relocated)
    assert restored.scheduling_evidence == selected.scheduling_evidence
    second = tmp_path / "again"
    evidence.export_evidence(restored, second)
    assert {p.relative_to(second): p.read_bytes() for p in second.rglob("*") if p.is_file()} == {
        p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()
    }
    manifest = relocated / "hardware/scheduling/manifest.json"
    for row in json.loads(manifest.read_bytes())["artifacts"]:
        assert digest((manifest.parent / row["path"]).read_bytes()) == row["sha256"]
    (relocated / "hardware/scheduling/nested/profile.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence member changed"):
        evidence.load_exported_evidence(relocated)


def test_producer_claims_never_qualify_import(tmp_path):
    path, document, _ = bundle(tmp_path)
    document["verification"] = {key: True for key in document["verification"]}
    path.write_text(json.dumps(document))
    view, _ = load(path)
    assert view["qualified_for_use"] is False
    assert "producer assertions" in view["qualification"]


def test_snapshot_view_correspondence_is_checked(tmp_path):
    path, _, _ = bundle(tmp_path)
    view, sources = load(path)
    view["manifest"]["profiles"][0]["fields"]["latency"] = 99
    with pytest.raises(ValueError, match="view differs"):
        scheduling.snapshot_outputs(view, sources, target="fixture")
