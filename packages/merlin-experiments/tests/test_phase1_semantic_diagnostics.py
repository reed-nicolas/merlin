"""Host-only semantic receipts bind the reviewed model and frozen public inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from merlin_experiments.phase1 import semantic_diagnostics as SD

_LINALG = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<1xi32>, %b: tensor<1xi32>, %o: tensor<1xi32>) -> tensor<1xi32> {
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%a, %b : tensor<1xi32>, tensor<1xi32>) outs(%o : tensor<1xi32>) {
      ^bb0(%x: i32, %y: i32, %z: i32):
        %sum = arith.addi %x, %y : i32
        linalg.yield %sum : i32
      } -> tensor<1xi32>
    return %r : tensor<1xi32>
  }
}
"""


def _inputs(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    workspace = tmp_path / "workspace"
    public = tmp_path / "frozen-public"
    private = tmp_path / "frozen-private"
    contract = tmp_path / "contract"
    for directory in (run, workspace, public / "public-one", public / "dev-one", private, contract):
        directory.mkdir(parents=True)
    (public / "public-one/capsule.linalg.mlir").write_text(_LINALG)
    (public / "dev-one/capsule.linalg.mlir").write_text(_LINALG)
    model = private / "instruction-semantics.json"
    model.write_text(
        json.dumps(
            {
                "schema": "merlin.instruction_semantics.v1",
                "target": "test-target",
                "status": "UNKNOWN",
                "unknowns": ["selected_target_contract_has_no_instruction_semantics_resource"],
                "instructions": [],
            }
        )
    )

    def capsules(root, *, labels, contract):
        assert root == public
        assert labels == {"public"}
        return [
            {"__dir__": str(public / "public-one"), "linalg_mlir": "capsule.linalg.mlir"},
        ]

    monkeypatch.setattr(SD, "discover_capsules", capsules)
    return run, workspace, public, contract, model


def test_private_receipt_uses_only_public_linalg_and_unknown_is_not_a_verdict(tmp_path, monkeypatch):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)
    record = SD.create(run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    receipt = run / "semantic_search_diagnostic.json"
    document = json.loads(receipt.read_text())

    assert record["status"] == "recorded"
    assert receipt.stat().st_mode & 0o077 == 0
    assert document["inputs"] == 1
    assert document["rows"][0]["capsule_linalg"] == "public-one/capsule.linalg.mlir"
    assert document["rows"][0]["inventory"]["regions"][0]["receipt"]["status"] == "unknown"
    assert "verdict" not in document
    SD.verify(record, run, workspace=workspace, model_path=model, public_root=public)


def test_phase1_search_requires_a_normalized_sw_spec_and_facts_model(tmp_path, monkeypatch):
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
    from merlin.targetgen.instruction_semantics import normalize_instruction_semantics
    from merlin.targetgen.semantic_search import search_linalg_inventory

    run, workspace, public, contract, model_path = _inputs(tmp_path, monkeypatch)
    parsed = parse_linalg_mlir(_LINALG)
    operation = parsed["ops"][0]
    raw_model = {
        "schema": "merlin.instruction_semantics.v1",
        "target": "test-target",
        "memory_spaces": {},
        "instructions": [
            {
                "id": "add",
                "operands": [{"name": name, "type": "tensor<1xi32>"} for name in ("lhs", "rhs", "init")],
                "results": [{"name": "out", "type": "tensor<1xi32>"}],
                "computation": {
                    "kind": "linalg.generic",
                    "indexing_maps": operation["indexing_maps"],
                    "iterator_types": operation["iterator_types"],
                    "scalar_body": operation["scalar_body"],
                },
                "parameters": {},
                "constraints": {},
                "effects": [],
                "software_operation": "add",
            }
        ],
    }
    # The low-level matcher permits raw models for exploratory unit probes.
    # Phase 1 must not promote such a pattern into a device-candidate receipt.
    assert search_linalg_inventory(parsed, raw_model)["summary"]["device_candidates"] == 1
    model_path.write_text(json.dumps(raw_model))
    SD.create(run, workspace=workspace, model_path=model_path, public_root=public, contract_root=contract)
    [row] = json.loads((run / "semantic_search_diagnostic.json").read_text())["rows"]
    assert row["status"] == "unavailable"
    assert "normalized instruction model" in row["reason"]
    assert "inventory" not in row

    # The exact same typed pattern is selectable after Phase 0 binds a reviewed
    # software declaration and selected hardware facts into a normalized model.
    normalized = normalize_instruction_semantics(
        raw_model,
        software_spec={
            "schema": "merlin.software_spec.v1",
            "target": "test-target",
            "status": "reviewed",
            "operations": [{"id": "add", "placement": "accelerator", "signature": {"dtypes": ["i32"], "ranks": [1]}}],
        },
        rtl_facts={"facts": {}},
        target="test-target",
    )
    assert normalized["status"] == "described"
    another_run = tmp_path / "another-run"
    another_run.mkdir()
    model_path.write_text(json.dumps(normalized))
    SD.create(another_run, workspace=workspace, model_path=model_path, public_root=public, contract_root=contract)
    [row] = json.loads((another_run / "semantic_search_diagnostic.json").read_text())["rows"]
    assert row["inventory"]["summary"]["device_candidates"] == 1


def test_declared_input_without_linalg_provenance_is_diagnostic_error(tmp_path, monkeypatch):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)
    (public / "public-one/capsule.linalg.mlir").write_text("module {}\n")
    SD.create(run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    [row] = json.loads((run / "semantic_search_diagnostic.json").read_text())["rows"]
    assert row["status"] == "diagnostic_error"
    assert "linalg-on-tensors provenance" in row["reason"]
    assert "inventory" not in row


def test_payload_budget_marks_file_unavailable_without_running_search(tmp_path, monkeypatch):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        SD,
        "parse_linalg_mlir",
        lambda _text: {"ops": [{"operation": "linalg.generic"}] * 33},
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("search must not run after the host diagnostic cap")

    monkeypatch.setattr(SD, "search_linalg_inventory", forbidden)
    SD.create(run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    [row] = json.loads((run / "semantic_search_diagnostic.json").read_text())["rows"]
    assert row["status"] == "limit_reached"
    assert "inventory" not in row


def test_escaping_capsule_declaration_is_only_a_diagnostic_refusal(tmp_path, monkeypatch):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        SD,
        "discover_capsules",
        lambda *_args, **_kwargs: [
            {"__dir__": str(public / "public-one"), "linalg_mlir": "../dev-one/capsule.linalg.mlir"}
        ],
    )
    record = SD.create(run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    document = json.loads((run / "semantic_search_diagnostic.json").read_text())
    assert record["status"] == "recorded"
    assert "escaping linalg input" in document["selection_error"]
    assert document["rows"] == []


def test_missing_parser_dependency_is_only_a_diagnostic_row(tmp_path, monkeypatch):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)

    def missing(_text):
        raise ImportError("xdsl is not installed")

    monkeypatch.setattr(SD, "parse_linalg_mlir", missing)
    SD.create(run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    [row] = json.loads((run / "semantic_search_diagnostic.json").read_text())["rows"]
    assert row["status"] == "diagnostic_error"
    assert "ImportError" in row["reason"]
    assert "inventory" not in row


@pytest.mark.parametrize("changed", ["model", "public", "receipt"])
def test_resume_refuses_changed_diagnostic_inputs_or_receipt(tmp_path, monkeypatch, changed):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)
    record = SD.create(run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    path = {
        "model": model,
        "public": public / "public-one/capsule.linalg.mlir",
        "receipt": run / "semantic_search_diagnostic.json",
    }[changed]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="semantic-search"):
        SD.verify(record, run, workspace=workspace, model_path=model, public_root=public)


def test_absent_model_is_bound_without_minting_a_receipt(tmp_path):
    run = tmp_path / "run"
    workspace = tmp_path / "workspace"
    public = tmp_path / "public"
    contract = tmp_path / "contract"
    for path in (run, workspace, public, contract):
        path.mkdir()
    record = SD.create(run, workspace=workspace, model_path=None, public_root=public, contract_root=contract)
    assert record["status"] == "unavailable"
    assert not (run / "semantic_search_diagnostic.json").exists()
    SD.verify(record, run, workspace=workspace, model_path=None, public_root=public)
    model = tmp_path / "new-model.json"
    model.write_text("{}")
    with pytest.raises(RuntimeError, match="availability changed"):
        SD.verify(record, run, workspace=workspace, model_path=model, public_root=public)


@pytest.mark.parametrize("exposed_root", ["workspace", "public"])
@pytest.mark.parametrize("direction", ["inside", "ancestor", "symlink"])
def test_receipt_refuses_agent_visible_placement(tmp_path, monkeypatch, exposed_root, direction):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)
    exposed = {"workspace": workspace, "public": public}[exposed_root]
    if direction == "inside":
        unsafe_run = exposed / "run"
        unsafe_run.mkdir()
    elif direction == "ancestor":
        unsafe_run = tmp_path
    else:
        unsafe_run = tmp_path / "alias"
        unsafe_run.symlink_to(exposed, target_is_directory=True)
    with pytest.raises(RuntimeError, match="host-private"):
        SD.create(unsafe_run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    assert not (unsafe_run / "semantic_search_diagnostic.json").exists()


def test_resume_refuses_receipt_newly_exposed_to_workspace(tmp_path, monkeypatch):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)
    record = SD.create(run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    exposed_workspace = tmp_path
    with pytest.raises(RuntimeError, match="host-private"):
        SD.verify(record, run, workspace=exposed_workspace, model_path=model, public_root=public)


def test_final_mount_plan_rejects_direct_and_aliased_receipt_exposure(tmp_path, monkeypatch):
    run, workspace, public, contract, model = _inputs(tmp_path, monkeypatch)
    SD.create(run, workspace=workspace, model_path=model, public_root=public, contract_root=contract)
    hidden = ["bwrap", "--tmpfs", str(tmp_path), "--bind", str(workspace), str(workspace)]
    SD.assert_private_mounts(hidden, run)

    direct = [*hidden, "--ro-bind", str(run), str(run)]
    with pytest.raises(RuntimeError, match="host-private"):
        SD.assert_private_mounts(direct, run)

    alias = [*hidden, "--ro-bind", str(tmp_path), "/agent-visible"]
    with pytest.raises(RuntimeError, match="bind alias"):
        SD.assert_private_mounts(alias, run)

    masked_alias = [*alias, "--ro-bind", "/dev/null", "/agent-visible/run/semantic_search_diagnostic.json"]
    with pytest.raises(RuntimeError, match="bind alias"):
        SD.assert_private_mounts(masked_alias, run)

    hidden_alias = [*alias, "--tmpfs", "/agent-visible/run"]
    SD.assert_private_mounts(hidden_alias, run)

    environment = run / "environment.yaml"
    environment.write_text("semantic_search_diagnostic: recorded\n")
    environment_alias = [*hidden, "--ro-bind", str(environment), "/agent-visible/environment.yaml"]
    with pytest.raises(RuntimeError, match="bind alias"):
        SD.assert_private_mounts(environment_alias, run)
