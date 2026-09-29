"""Phase 2 semantic search is bound to frozen inputs and remains a host sidecar."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase2 import semantic_diagnostic as SD
from merlin_experiments.phase2.contracts import exact_tree_record
from merlin_experiments.phase2.stage_inputs import StageE2ESentinel

from merlin.targetgen.sandbox import bwrap as BW


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _selected(tmp_path: Path, *, status: str = "UNKNOWN") -> tuple[StageE2ESentinel, Path, Path]:
    corpus = tmp_path / "frozen-functional" / "repo" / "selected" / "capsules"
    capsule = corpus / "model" / "whole"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(
        yaml.safe_dump(
            {"name": "whole", "kind": "model", "label": "public", "interface_mlir": "capsule.interface.mlir"}
        )
    )
    (capsule / "capsule.interface.mlir").write_text('module attributes {prov.level = "linalg-on-tensors"} {}\n')
    model = {
        "schema": "merlin.instruction_semantics.v1",
        "target": "fixture",
        "status": status,
        "instructions": [],
    }
    evidence = corpus / "_evidence"
    model_path = evidence / "software" / "instruction-semantics.json"
    model_path.parent.mkdir(parents=True)
    model_raw = (json.dumps(model, sort_keys=True) + "\n").encode()
    model_path.write_bytes(model_raw)
    (evidence / "evidence-manifest.json").write_text(
        json.dumps(
            {
                "schema": "phase0_evidence_v1",
                "target": "fixture",
                "artifacts": {
                    "software/instruction-semantics.json": {"sha256": _digest(model_raw), "size_bytes": len(model_raw)}
                },
            }
        )
    )
    selected = StageE2ESentinel("whole", str(capsule), str(capsule), exact_tree_record(capsule)["sha256"], (), ("L2",))
    return selected, model_path, capsule


def test_unknown_model_is_explicit_and_binds_both_frozen_inputs(tmp_path: Path) -> None:
    selected, model_path, capsule = _selected(tmp_path)
    result = SD.write_portfolio_receipt(
        tmp_path, target="fixture", members=((selected, True),), frozen_grants=(capsule.parents[1],)
    )
    destination = tmp_path / "_host_semantic_diagnostics" / "semantic_search.json"
    assert json.loads(destination.read_bytes()) == result
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert result["visibility"] == "host_private_diagnostic"
    assert result["members"][0]["status"] == "unknown"
    assert result["members"][0]["phase0"]["instruction_model_sha256"] == _digest(model_path.read_bytes())
    assert result["members"][0]["source_mlir"]["sha256"] == _digest((capsule / "capsule.interface.mlir").read_bytes())
    assert result["score_effect"] == result["timing_effect"] == result["qualification_effect"] == "none"


def test_model_manifest_or_frozen_capsule_tamper_refuses(tmp_path: Path) -> None:
    selected, model_path, capsule = _selected(tmp_path)
    model_path.write_text("{}")
    assert (
        SD.inspect_member(selected, target="fixture", capsule_linked=True, frozen_grants=(capsule.parents[1],))[
            "status"
        ]
        == "refused"
    )
    selected, _, capsule = _selected(tmp_path / "second")
    (capsule / "capsule.interface.mlir").write_text("module { changed }\n")
    row = SD.inspect_member(selected, target="fixture", capsule_linked=True, frozen_grants=(capsule.parents[1],))
    assert row["status"] == "refused"
    assert "capsule bytes changed" in row["reason"]


def test_missing_selected_evidence_is_unavailable_and_external_is_refused(tmp_path: Path) -> None:
    selected, _, capsule = _selected(tmp_path)
    assert SD.inspect_member(selected, target="fixture", capsule_linked=False, frozen_grants=())["status"] == "refused"
    evidence = Path(selected.frozen_source_path).parents[1] / "_evidence"
    (evidence / "evidence-manifest.json").unlink()
    row = SD.inspect_member(selected, target="fixture", capsule_linked=True, frozen_grants=(capsule.parents[1],))
    assert row["status"] == "unavailable"
    assert "no Phase 0 evidence manifest" in row["reason"]


def test_described_model_reaches_search_with_selected_bytes(tmp_path: Path, monkeypatch) -> None:
    selected, model_path, capsule = _selected(tmp_path, status="described")
    calls = []

    def parse(raw):
        calls.append(("mlir", raw))
        return {"ops": [{"operation": "linalg.generic"}]}

    def search(parsed, model, *, limits):
        calls.append(("model", model["target"]))
        return {"schema": "merlin.semantic_search.inventory.v1", "summary": {"operations": 0}, "regions": []}

    monkeypatch.setattr(SD, "validate_normalized_instruction_model", lambda model, expected_target: model)
    monkeypatch.setattr(SD, "parse_linalg_mlir", parse)
    monkeypatch.setattr(SD, "search_linalg_inventory", search)
    row = SD.inspect_member(selected, target="fixture", capsule_linked=True, frozen_grants=(capsule.parents[1],))
    assert row["status"] == "diagnostic"
    assert calls == [("mlir", (capsule / "capsule.interface.mlir").read_text()), ("model", "fixture")]
    assert row["phase0"]["instruction_model_sha256"] == _digest(model_path.read_bytes())


def test_non_linalg_source_cannot_appear_as_a_searched_model(tmp_path: Path, monkeypatch) -> None:
    selected, _, capsule = _selected(tmp_path, status="described")
    (capsule / "capsule.interface.mlir").write_text("module {}\n")
    selected = StageE2ESentinel(
        selected.capsule,
        selected.capsule_path,
        selected.frozen_source_path,
        exact_tree_record(capsule)["sha256"],
        selected.required_lanes,
        selected.required_tiers,
    )
    monkeypatch.setattr(SD, "validate_normalized_instruction_model", lambda model, expected_target: model)
    row = SD.inspect_member(selected, target="fixture", capsule_linked=True, frozen_grants=(capsule.parents[1],))
    assert row["status"] == "unavailable"
    assert "not linalg-on-tensors" in row["reason"]


def test_adjacent_evidence_outside_exact_frozen_grant_refuses(tmp_path: Path) -> None:
    selected, _, capsule = _selected(tmp_path)
    row = SD.inspect_member(selected, target="fixture", capsule_linked=True, frozen_grants=(capsule,))
    assert row["status"] == "refused"
    assert "one frozen functional corpus grant" in row["reason"]


def test_receipt_rejects_nonprivate_stage_and_existing_sidecar(tmp_path: Path) -> None:
    selected, _, capsule = _selected(tmp_path)
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(ValueError, match="owner-controlled"):
            SD.write_portfolio_receipt(
                tmp_path, target="fixture", members=((selected, True),), frozen_grants=(capsule.parents[1],)
            )
    finally:
        tmp_path.chmod(0o755)
    SD.write_portfolio_receipt(
        tmp_path, target="fixture", members=((selected, True),), frozen_grants=(capsule.parents[1],)
    )
    with pytest.raises(ValueError, match="fresh private directory"):
        SD.write_portfolio_receipt(
            tmp_path, target="fixture", members=((selected, True),), frozen_grants=(capsule.parents[1],)
        )


def test_sidecar_is_outside_agent_workspace_control_and_input_mounts(tmp_path: Path) -> None:
    selected, _, capsule = _selected(tmp_path)
    SD.write_portfolio_receipt(
        tmp_path, target="fixture", members=((selected, True),), frozen_grants=(capsule.parents[1],)
    )
    sidecar = tmp_path / "_host_semantic_diagnostics" / "semantic_search.json"
    workspace = tmp_path / "agent_workspaces" / "round_00"
    control = tmp_path / "global_control" / "round_0000"
    inputs = tmp_path / "_agent_inputs"
    for path in (workspace, control, inputs):
        path.mkdir(parents=True)
    argv = BW.base_argv(workspace, {}, repo=tmp_path, _policy_test_live_inputs=True)
    argv += ["--ro-bind", str(control), "/perf-control", "--ro-bind", str(inputs), "/perf-corpus"]
    assert not BW.is_exposed(argv, sidecar)
    assert BW.is_exposed([*argv, "--ro-bind", str(tmp_path), str(tmp_path)], sidecar)
