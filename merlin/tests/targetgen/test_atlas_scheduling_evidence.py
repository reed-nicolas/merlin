"""Byte-bound Atlas timing handoff conversion stays separate from qualification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def adapter():
    path = repo_root() / "examples/atlas/target/rtlgraph_evidence.py"
    spec = importlib.util.spec_from_file_location("atlas_rtlgraph_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def bundle(root, *, dma=True):
    root.mkdir()
    hardware = "a" * 64
    members = {}

    def save(path, raw):
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        members[path] = raw
        return {"path": path, "sha256": digest(raw)}

    role = "dma" if dma else "mxu1"
    fields = {"completion": "explicit-wait"} if dma else {"first_write_age": 3, "overwrite_acc_read_hold": 0}
    evidence = {
        "config": "EE290SimConfig",
        "inputs": {"hardware_ir": {"sha256": hardware}},
        "compiler_overrides": fields,
        "assumptions": ["idle entry"],
        "limitations": ["partial projection"],
    }
    if dma:
        evidence["schema"] = "atlas.rtlgraph.dma-profile.v2"
    else:
        evidence.update({"schema_version": 2, "kind": "atlas-partial-mxu1-profile"})
    evidence_record = save(f"{role}/profile.json", json_bytes(evidence))
    native_fields = {
        "schema": f"atlas-{role}-profile-{'v2' if dma else 'v1'}",
        "config": "EE290SimConfig",
        "source_ir_sha256": hardware,
        "evidence_sha256": evidence_record["sha256"],
        **fields,
    }
    projection = save(f"{role}/atlas-{role}.profile", "".join(f"{k}={v}\n" for k, v in native_fields.items()).encode())
    completion = (
        "memory lifetime until explicit matching DMA.WAIT; age is not a bound"
        if dma
        else "access at modeled DMA completion"
    )
    footprints = {
        "schema": "atlas.footprints.v1",
        "schedule_validated": False,
        "model": {
            "source_ir_sha256": hardware,
            "rtl_dma": dma,
            "rtl_dma_ranges": dma,
            "rtl_lsu": False,
            "rtl_scalar_lsu": False,
            "rtl_xlu": False,
            "rtl_vpu": False,
            "mxu1_first_write_age": 3,
            "mxu1_overwrite_acc_read_hold": False,
        },
        "semantics": {
            "age_origin": "instruction issue",
            "hold_end": "inclusive",
            "unknown_value": None,
            "at_completion": completion,
            "dma_cycles": "cost estimate, not a completion guarantee",
        },
        "blocks": [
            {
                "entry": {"xregs": [None], "dma_base": None},
                "instructions": [
                    {"footprint": {"accesses": [{"anywhere": True, "at_completion": dma}], "dma_cycles_estimate": 123}}
                ],
            }
        ],
    }
    contract = {
        "schema": "atlas.rtlgraph.contract.v1",
        "config": "EE290SimConfig",
        "source_ir_sha256": hardware,
        "compiler": {"path": "/this/compiler/must/not/be/read", "sha256": "b" * 64},
        "source": save("before.S", b"# exact bytes\r\ndma.wait 1\r\n"),
        "profiles": [{"role": role, "projection": projection, "evidence": evidence_record}],
        "footprints": save("footprints.json", json_bytes(footprints)),
        "semantics": {
            "assembly": "atlas-opt-native",
            "age_zero": "instruction-issue",
            "hold_interval": "inclusive",
            "dma_completion": "explicit-wait" if dma else "modeled-completion",
        },
        "limitations": ["Unselected rules are inherited."],
    }
    save("contract.json", json_bytes(contract))
    return root / "contract.json", contract, members


def rewrite_contract(path, contract):
    path.write_bytes(json_bytes(contract))


def rewrite_member(path, contract, record, value):
    raw = json_bytes(value)
    (path.parent / record["path"]).write_bytes(raw)
    record["sha256"] = digest(raw)
    rewrite_contract(path, contract)


@pytest.mark.parametrize("dma", [True, False])
def test_exact_originals_unknowns_and_completion_preserved(adapter, tmp_path, monkeypatch, dma):
    path, contract, members = bundle(tmp_path / "input", dma=dma)
    original_read = Path.read_bytes

    def guarded_read(self):
        assert str(self) != contract["compiler"]["path"]
        return original_read(self)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    output = adapter.convert_contract(path, tmp_path / "output")
    envelope = json.loads(output.read_bytes())
    assert set(envelope) == {
        "schema",
        "target",
        "hardware",
        "producer",
        "semantics",
        "profiles",
        "artifacts",
        "limitations",
        "verification",
    }
    assert envelope["schema"] == "merlin.scheduling_evidence.v1"
    assert envelope["semantics"] == contract["semantics"]
    assert envelope["profiles"][0]["assumptions"] == ["idle entry"]
    assert envelope["verification"] == {
        "artifact_bytes_verified": True,
        "projection_evidence_agreement_verified": True,
        "compiler_identity_verified": False,
        "footprints_reproduced": False,
        "schedule_legality_verified": False,
        "rtl_replayed": False,
    }
    assert len({a["role"] for a in envelope["artifacts"]}) == len(envelope["artifacts"])
    for artifact in envelope["artifacts"]:
        raw = (output.parent / artifact["path"]).read_bytes()
        assert raw == members[artifact["path"].removeprefix("bundle/")]
        assert digest(raw) == artifact["sha256"]
    saved = json.loads((output.parent / "bundle/footprints.json").read_bytes())
    assert saved["blocks"][0]["entry"]["dma_base"] is None
    assert saved["blocks"][0]["instructions"][0]["footprint"]["accesses"][0]["at_completion"] is dma
    with pytest.raises(FileExistsError):
        adapter.convert_contract(path, output.parent)


@pytest.mark.parametrize("role", ["source", "footprints", "projection", "evidence"])
def test_changed_bytes_rejected_before_output(adapter, tmp_path, role):
    path, contract, _ = bundle(tmp_path / "input")
    record = contract[role] if role in {"source", "footprints"} else contract["profiles"][0][role]
    (path.parent / record["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Artifact changed"):
        adapter.convert_contract(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "change", ["hardware", "config", "overrides", "schema", "duplicate", "assumptions", "limitations"]
)
def test_projection_report_disagreement_even_with_new_hashes(adapter, tmp_path, change):
    path, contract, _ = bundle(tmp_path / "input")
    entry = contract["profiles"][0]
    evidence_path = path.parent / entry["evidence"]["path"]
    evidence = json.loads(evidence_path.read_bytes())
    if change == "hardware":
        evidence["inputs"]["hardware_ir"]["sha256"] = "c" * 64
    elif change == "config":
        evidence["config"] = "AtlasRocketConfig"
    elif change == "overrides":
        evidence["compiler_overrides"]["completion"] = "fixed-delay"
    elif change == "schema":
        evidence["schema"] = "atlas.rtlgraph.dma-profile.v1"
    elif change in {"assumptions", "limitations"}:
        evidence[change] = [{}]
    raw = json_bytes(evidence)
    if change == "duplicate":
        raw = raw.replace(b'"config": "EE290SimConfig",', b'"config": "EE290SimConfig", "config": "EE290SimConfig",')
    old_hash = entry["evidence"]["sha256"]
    evidence_path.write_bytes(raw)
    entry["evidence"]["sha256"] = digest(raw)
    projection_path = path.parent / entry["projection"]["path"]
    projection_raw = projection_path.read_bytes().replace(old_hash.encode(), digest(raw).encode())
    projection_path.write_bytes(projection_raw)
    entry["projection"]["sha256"] = digest(projection_raw)
    rewrite_contract(path, contract)
    with pytest.raises(ValueError):
        adapter.convert_contract(path, tmp_path / "output")


@pytest.mark.parametrize(
    "change",
    [
        "role",
        "semantics",
        "model",
        "range_flag",
        "bound",
        "unknown",
        "mixed",
        "absolute",
        "parent",
        "symlink",
        "collision",
        "noncanonical",
        "backslash",
        "limitations",
    ],
)
def test_invalid_contract_and_model_rejected(adapter, tmp_path, change):
    path, contract, _ = bundle(tmp_path / "input")
    if change == "role":
        contract["profiles"].append(contract["profiles"][0])
    elif change == "semantics":
        contract["semantics"]["dma_completion"] = "explicit-wait-when-dma-profile-selected"
    elif change in {"model", "range_flag", "bound", "unknown"}:
        record = contract["footprints"]
        footprints = json.loads((path.parent / record["path"]).read_bytes())
        if change in {"model", "range_flag"}:
            footprints["model"]["rtl_dma" if change == "model" else "rtl_dma_ranges"] = False
        elif change == "bound":
            footprints["semantics"]["dma_cycles"] = "completion bound"
        else:
            footprints["semantics"]["unknown_value"] = 0
        rewrite_member(path, contract, record, footprints)
    elif change == "mixed":
        contract["source_ir_sha256"] = "c" * 64
    elif change == "absolute":
        contract["source"]["path"] = str((path.parent / "before.S").resolve())
    elif change == "parent":
        contract["source"]["path"] = "../input/before.S"
    elif change == "symlink":
        outside = tmp_path / "outside.S"
        outside.write_bytes((path.parent / "before.S").read_bytes())
        (path.parent / "link.S").symlink_to(outside)
        contract["source"]["path"] = "link.S"
    elif change == "collision":
        contract["source"] = {"path": "contract.json", "sha256": digest(path.read_bytes())}
    elif change == "noncanonical":
        contract["source"]["path"] = "./before.S"
    elif change == "backslash":
        contract["source"]["path"] = "sub\\before.S"
    elif change == "limitations":
        contract["limitations"] = [3]
    rewrite_contract(path, contract)
    with pytest.raises(ValueError):
        adapter.convert_contract(path, tmp_path / "output")


def test_model_timing_disagreement(adapter, tmp_path):
    path, contract, _ = bundle(tmp_path / "input", dma=False)
    record = contract["footprints"]
    footprints = json.loads((path.parent / record["path"]).read_bytes())
    footprints["model"]["mxu1_first_write_age"] = 4
    rewrite_member(path, contract, record, footprints)
    with pytest.raises(ValueError, match="mxu1_first_write_age"):
        adapter.convert_contract(path, tmp_path / "output")


@pytest.mark.parametrize("number", ["1e999", "NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_rejected(adapter, tmp_path, number):
    path, _, _ = bundle(tmp_path / "input")
    raw = path.read_bytes().replace(b'"limitations": [', f'"extra": {number}, "limitations": ['.encode())
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="Nonfinite"):
        adapter.convert_contract(path, tmp_path / "output")
