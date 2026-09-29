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
    public = tmp_path / "frozen-public"
    private = tmp_path / "frozen-private"
    contract = tmp_path / "contract"
    for directory in (run, public / "public-one", public / "dev-one", private, contract):
        directory.mkdir(parents=True)
    (public / "public-one/capsule.linalg.mlir").write_text(_LINALG)
    (public / "dev-one/capsule.linalg.mlir").write_text(_LINALG)
    model = private / "instruction-semantics.json"
    model.write_text(json.dumps({"schema": "merlin.instruction_semantics.v1", "status": "UNKNOWN"}))

    def capsules(root, *, labels, contract):
        assert root == public
        assert labels == {"public"}
        return [
            {"__dir__": str(public / "public-one"), "linalg_mlir": "capsule.linalg.mlir"},
        ]

    monkeypatch.setattr(SD, "discover_capsules", capsules)
    return run, public, contract, model


def test_private_receipt_uses_only_public_linalg_and_unknown_is_not_a_verdict(tmp_path, monkeypatch):
    run, public, contract, model = _inputs(tmp_path, monkeypatch)
    record = SD.create(run, model_path=model, public_root=public, contract_root=contract)
    receipt = run / "semantic_search_diagnostic.json"
    document = json.loads(receipt.read_text())

    assert record["status"] == "recorded"
    assert receipt.stat().st_mode & 0o077 == 0
    assert document["inputs"] == 1
    assert document["rows"][0]["capsule_linalg"] == "public-one/capsule.linalg.mlir"
    assert document["rows"][0]["inventory"]["regions"][0]["receipt"]["status"] == "unknown"
    assert "verdict" not in document
    SD.verify(record, run, model_path=model, public_root=public)


def test_declared_input_without_linalg_provenance_is_diagnostic_error(tmp_path, monkeypatch):
    run, public, contract, model = _inputs(tmp_path, monkeypatch)
    (public / "public-one/capsule.linalg.mlir").write_text("module {}\n")
    SD.create(run, model_path=model, public_root=public, contract_root=contract)
    [row] = json.loads((run / "semantic_search_diagnostic.json").read_text())["rows"]
    assert row["status"] == "diagnostic_error"
    assert "linalg-on-tensors provenance" in row["reason"]
    assert "inventory" not in row


def test_payload_budget_marks_file_unavailable_without_running_search(tmp_path, monkeypatch):
    run, public, contract, model = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        SD,
        "parse_linalg_mlir",
        lambda _text: {"ops": [{"operation": "linalg.generic"}] * 33},
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("search must not run after the host diagnostic cap")

    monkeypatch.setattr(SD, "search_linalg_inventory", forbidden)
    SD.create(run, model_path=model, public_root=public, contract_root=contract)
    [row] = json.loads((run / "semantic_search_diagnostic.json").read_text())["rows"]
    assert row["status"] == "limit_reached"
    assert "inventory" not in row


def test_escaping_capsule_declaration_is_only_a_diagnostic_refusal(tmp_path, monkeypatch):
    run, public, contract, model = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        SD,
        "discover_capsules",
        lambda *_args, **_kwargs: [
            {"__dir__": str(public / "public-one"), "linalg_mlir": "../dev-one/capsule.linalg.mlir"}
        ],
    )
    record = SD.create(run, model_path=model, public_root=public, contract_root=contract)
    document = json.loads((run / "semantic_search_diagnostic.json").read_text())
    assert record["status"] == "recorded"
    assert "escaping linalg input" in document["selection_error"]
    assert document["rows"] == []


def test_missing_parser_dependency_is_only_a_diagnostic_row(tmp_path, monkeypatch):
    run, public, contract, model = _inputs(tmp_path, monkeypatch)

    def missing(_text):
        raise ImportError("xdsl is not installed")

    monkeypatch.setattr(SD, "parse_linalg_mlir", missing)
    SD.create(run, model_path=model, public_root=public, contract_root=contract)
    [row] = json.loads((run / "semantic_search_diagnostic.json").read_text())["rows"]
    assert row["status"] == "diagnostic_error"
    assert "ImportError" in row["reason"]
    assert "inventory" not in row


@pytest.mark.parametrize("changed", ["model", "public", "receipt"])
def test_resume_refuses_changed_diagnostic_inputs_or_receipt(tmp_path, monkeypatch, changed):
    run, public, contract, model = _inputs(tmp_path, monkeypatch)
    record = SD.create(run, model_path=model, public_root=public, contract_root=contract)
    path = {
        "model": model,
        "public": public / "public-one/capsule.linalg.mlir",
        "receipt": run / "semantic_search_diagnostic.json",
    }[changed]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="semantic-search"):
        SD.verify(record, run, model_path=model, public_root=public)


def test_absent_model_is_bound_without_minting_a_receipt(tmp_path):
    run = tmp_path / "run"
    public = tmp_path / "public"
    contract = tmp_path / "contract"
    for path in (run, public, contract):
        path.mkdir()
    record = SD.create(run, model_path=None, public_root=public, contract_root=contract)
    assert record["status"] == "unavailable"
    assert not (run / "semantic_search_diagnostic.json").exists()
    SD.verify(record, run, model_path=None, public_root=public)
    model = tmp_path / "new-model.json"
    model.write_text("{}")
    with pytest.raises(RuntimeError, match="availability changed"):
        SD.verify(record, run, model_path=model, public_root=public)
