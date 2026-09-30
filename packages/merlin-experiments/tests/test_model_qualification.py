"""Coarse offline compiler qualification preserves the real process and artifact boundaries."""

from __future__ import annotations

import contextvars
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments import model_qualification as Q

from merlin.targetgen.sandbox import preflight as PF


def _bundle(root: Path):
    root.mkdir()
    for name, raw in {
        "model.mlir": b"module {}\n",
        "weights.safetensors": b"fixture",
        "weights.safetensors.manifest.json": b"{}",
        "inputs.npz": b"fixture",
        "input_order.json": b"{}",
        "golden.npy": b"fixture",
    }.items():
        (root / name).write_bytes(raw)
    return root


def _package(root: Path):
    root.mkdir()
    tool = root / "compiler.py"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\nfrom pathlib import Path\nimport xdsl\n"
        "for arg in sys.argv:\n"
        "    if arg.startswith('--emit-command-buffer='):\n"
        "        Path(arg.partition('=')[2]).write_text(json.dumps({'commands':[], 'declined':{'reason':'fixture'}}))\n"
        "print('module {}')\n"
    )
    tool.chmod(0o755)
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "package_id": "fixture",
        "language": "python",
        "integrity_exempt": False,
        "authoring": {"mode": "agent_generated_from_rtl_facts", "author": "fixture", "generated_by_agent": True},
        "entrypoints": {"tool": "compiler.py"},
        "commands": {
            name: {"argv": ["{tool}", "{input_mlir}"]}
            for name in ("parse", "lower_interface_to_target", "lower_target_to_llvm")
        },
    }
    manifest["commands"]["emit_command_buffer"] = {
        "argv": ["{tool}", "--emit-command-buffer={output_json}", "{input_mlir}"]
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return root


def test_offline_process_boundary_preserves_interpreter_and_refuses_scope_upgrade(tmp_path, monkeypatch):
    if not all(shutil.which(tool) for tool in ("bwrap", "prlimit", "taskset")):
        pytest.skip("bounded local compiler tools are unavailable")
    monkeypatch.setenv("PYTHONPATH", ":".join(str(Path(path).absolute()) for path in __import__("sys").path if path))
    monkeypatch.setenv("MERLIN_CHIPYARD", str(tmp_path / "selected-chipyard"))
    monkeypatch.setenv("MERLIN_MESH_SIM", "spike")
    bundle, package = _bundle(tmp_path / "capture"), _package(tmp_path / "compiler")
    output = tmp_path / "qualification"
    result = Q.qualify(
        bundle=bundle,
        package=package,
        target="fixture",
        output=output,
        timeout_seconds=20,
        stage_timeout_seconds=5,
        memory_gib=2,
        lower_native=True,
    )
    assert result["status"] == "completed"
    statuses = {row["entrypoint"]: row["status"] for row in result["observations"]["compiler_observations"]}
    if not PF.probe_sandbox(network_isolation=True).usable:
        assert statuses == {name: "unavailable" for name in (
            "parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm"
        )}
        assert all(
            "sandbox inoperable" in row["reason"]
            for row in result["observations"]["compiler_observations"]
        )
        assert result["observations"]["model_routes"][0]["status"] == "unresolved"
        assert result["observations"]["runtime"]["target_executed"] is False
        assert result["whole_workload_validation_verified"] is False
        return
    assert statuses == {
        "parse": "accepted",
        "lower_interface_to_target": "unchanged",
        "emit_command_buffer": "declined",
        "lower_target_to_llvm": "unqualified",
    }
    assert all(row["canonical_ir_changed"] is False for row in result["observations"]["compiler_observations"])
    command_row = next(
        row for row in result["observations"]["compiler_observations"] if row["entrypoint"] == "emit_command_buffer"
    )
    assert command_row["command_buffer"]["status"] == "declined"
    assert command_row["command_buffer"]["commands_count"] == 0
    assert result["observations"]["model_routes"][0]["accelerator"]["status"] == "declined"
    assert result["observations"]["model_routes"][0]["status"] == "observed_decline"
    assert result["whole_workload_validation_verified"] is False
    assert result["observations"]["full_native_lowering_verified"] is False
    assert result["observations"]["native_lowerings"][0]["status"] == "failed"
    assert result["observations"]["workflow"]["application_validation_blockers"]
    assert result["observations"]["runtime"]["target_executed"] is False
    assert result["request"]["tool_environment"]["MERLIN_CHIPYARD"] == str(tmp_path / "selected-chipyard")
    assert result["request"]["tool_environment"]["MERLIN_MESH_SIM"] == "spike"
    assert result["request"]["certification_output_root"] is None
    assert (output / "worker-tmp").is_dir()
    assert json.loads((output / "qualification.json").read_bytes()) == result
    assert (output / "README.md").is_file()
    assert (output / "qualification.json").stat().st_mode & 0o222 == 0
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="fresh"):
        Q.qualify(bundle=bundle, package=package, target="fixture", output=output)


def test_compiler_preflight_requires_the_same_network_isolation_as_its_launch(tmp_path, monkeypatch):
    bundle, package = _bundle(tmp_path / "capture"), _package(tmp_path / "compiler")
    from merlin.targetgen.package_runtime import load_package

    seen = []

    def refuse(_binary=None, **kwargs):
        seen.append(kwargs)
        raise PF.SandboxUnavailable(
            PF.SandboxProbe(status=PF.SANDBOX_INOPERABLE, reason="netns_denied"),
            kwargs.get("context", ""),
        )

    monkeypatch.setattr(PF, "require_working_sandbox", refuse)
    with pytest.raises(PF.SandboxUnavailable, match="netns_denied"):
        Q._compiler_check(load_package(package), bundle / "model.mlir", "parse", tmp_path, timeout=5)
    assert seen and seen[0]["network_isolation"] is True


def test_selected_certification_guards_source_but_allows_its_execution_copy(tmp_path):
    selected = _package(tmp_path / "selected")
    other = _package(tmp_path / "other")
    active = contextvars.ContextVar("test_selected_certification", default=False)
    called = []

    def certify(source, *args, **kwargs):
        copied = tmp_path / "compiler-execution-1" / "package"
        copied.parent.mkdir()
        shutil.copytree(source, copied)
        assert active.get() is True
        assert copied != selected
        assert (copied / "manifest.yaml").read_bytes() == (selected / "manifest.yaml").read_bytes()
        called.append((source, args, kwargs))
        return {"status": "pass"}

    checked = Q._selected_source_certifier(selected, certify, active)
    with pytest.raises(ValueError, match="different compiler package"):
        checked(other)
    assert not called
    assert checked(selected, "interface.mlir", target="fixture") == {"status": "pass"}
    assert called == [(selected, ("interface.mlir",), {"target": "fixture"})]
    assert active.get() is False


def test_multi_program_roster_cannot_escape_or_become_one_forward(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    _bundle(root / "prefix")
    _bundle(root / "recurrent")
    document = {
        "version": 2,
        "programs": [
            {"name": "prefix", "bundle": "prefix", "steps": 1},
            {"name": "recurrent", "bundle": "recurrent", "steps": 10},
        ],
        "provenance": {"full_checkpoint": False, "synthetic_inputs": True},
    }
    (root / "session_contract.yaml").write_text(yaml.safe_dump(document))
    report = Q.inspect_workflow(root)
    assert len(report["programs"]) == 2
    assert report["application_validation_blockers"]
    document["programs"][1]["bundle"] = "../elsewhere"
    (root / "session_contract.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="escapes"):
        Q.inspect_workflow(root)
    document["programs"][1]["bundle"] = "recurrent"
    document["version"] = 1
    (root / "session_contract.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="cannot become one forward"):
        Q.inspect_workflow(root)


def test_session_inspection_binds_typed_abi_and_producer_receipts(tmp_path, monkeypatch):
    from merlin.llvmlower import session_bundle

    root = tmp_path / "session"
    root.mkdir()
    for name in ("prefix", "decode"):
        stage = _bundle(root / name)
        (stage / "capture_receipt.json").write_text(json.dumps({"stage": name}))
    document = {
        "version": 2,
        "programs": [
            {"name": "prefix", "bundle": "prefix", "steps": 1},
            {"name": "decode", "bundle": "decode", "steps": 2},
        ],
        "provenance": {"full_checkpoint": True, "synthetic_inputs": True},
    }
    contract = root / "session_contract.yaml"
    contract.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(
        session_bundle, "load",
        lambda *_: SimpleNamespace(program_names=("prefix", "decode"), bindings=(object(),)),
    )
    receipt = {
        "schema": "merlin.model_session_capture.v1",
        "session_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "programs": [
            {
                "name": name,
                "receipt_sha256": hashlib.sha256((root / name / "capture_receipt.json").read_bytes()).hexdigest(),
                "ok": True,
                "opaque": 0,
                "materialized_abi": {"complete": True},
            }
            for name in ("prefix", "decode")
        ],
    }
    (root / "session-receipt.json").write_text(json.dumps(receipt))
    inspected = Q.inspect_workflow(root)
    assert inspected["session_abi"] == {"status": "verified_structure", "bindings": 1}
    assert inspected["session_capture"]["status"] == "producer_receipts_bound"
    assert inspected["session_capture"]["source_closure_verified"] is False
    assert "session-receipt.json" in inspected["members"]

    (root / "decode" / "capture_receipt.json").write_text("changed")
    inspected = Q.inspect_workflow(root)
    assert inspected["session_capture"]["status"] == "unverified"
    assert "session producer receipts" in " ".join(inspected["application_validation_blockers"])
    monkeypatch.setattr(session_bundle, "load", lambda *_: (_ for _ in ()).throw(ValueError("ABI differs")))
    inspected = Q.inspect_workflow(root)
    assert inspected["session_abi"]["status"] == "unverified"


def test_single_program_v1_session_remains_a_forward(tmp_path):
    root = _bundle(tmp_path / "image-session")
    (root / "session_contract.yaml").write_text(
        yaml.safe_dump({"version": 1, "steps": 1, "provenance": {"full_checkpoint": False}})
    )
    inspected = Q.inspect_workflow(root)
    assert [program["name"] for program in inspected["programs"]] == ["forward"]
    assert inspected["session_abi"]["status"] == "not_applicable"


def test_command_observation_requires_an_explicit_nonempty_route(tmp_path):
    command = tmp_path / "commands.json"
    command.write_text(json.dumps({"commands": []}))
    assert Q._command_observation(command)["status"] == "invalid"
    command.write_text(json.dumps({"commands": [{"opcode": "MATMUL"}], "declined": {"reason": "no"}}))
    assert Q._command_observation(command)["status"] == "invalid"
    command.write_text(json.dumps({"commands": [{"opcode": "MATMUL"}]}))
    assert Q._command_observation(command)["status"] == "emitted"


def test_conditional_ssa_edge_becomes_transfer_only_after_reviewed_placements(monkeypatch):
    from merlin_experiments.phase1 import model_routes

    monkeypatch.setattr(model_routes, "_graph_totality", lambda *_: ({}, []))
    monkeypatch.setattr(model_routes, "_source_complete", lambda *_: True)

    def obligation(operation_id, selected):
        row = {
            "operation_ids": [operation_id],
            "status": "resolved",
            "precision": {
                "status": "resolved",
                "numerical_contracts": {
                    "host": {"status": "resolved"},
                    "accelerator": {"status": "resolved"},
                },
            },
            "host_admission": {"status": "admitted", "reviewed": True},
            "accelerator_admission": {"status": "unsupported", "reviewed": True},
        }
        if selected == "accelerator":
            row["host_admission"]["status"] = "unsupported"
            row["accelerator_admission"]["status"] = "admitted"
        elif selected is None:
            row["host_admission"]["reviewed"] = False
        return row

    application = {
        "capture_sha256": "exact",
        "capture_receipt": {"status": "verified_materialized", "source_closure_verified": True},
        "n_mlir_operations": 2,
        "completeness": {
            "source_trace": {},
            "operation_obligations": [obligation("op:0", "host"), obligation("op:1", None)],
            "transfer_obligations": [{"producer_operation_id": "op:0", "consumer_operation_id": "op:1"}],
        },
    }
    summary, blockers = model_routes._ledger_observation(application, "exact")
    assert summary["conditional_ssa_edges"]["status"] == "pending_placement"
    assert summary["conditional_ssa_edges"]["required_crossing"] == 0
    assert not any("typed transfer lowering" in reason for reason in blockers)

    application["completeness"]["operation_obligations"][1] = obligation("op:1", "host")
    summary, blockers = model_routes._ledger_observation(application, "exact")
    assert summary["conditional_ssa_edges"]["same_lane"] == 1
    assert not any("typed transfer lowering" in reason for reason in blockers)

    application["completeness"]["operation_obligations"][1] = obligation("op:1", "accelerator")
    summary, blockers = model_routes._ledger_observation(application, "exact")
    assert summary["conditional_ssa_edges"]["required_crossing"] == 1
    assert any("typed transfer lowering" in reason for reason in blockers)


def test_support_lowering_does_not_become_a_host_lane_or_transfer_endpoint(monkeypatch):
    from merlin_experiments.phase1 import model_routes

    monkeypatch.setattr(model_routes, "_graph_totality", lambda *_: ({}, []))
    monkeypatch.setattr(model_routes, "_source_complete", lambda *_: True)
    application = {
        "capture_sha256": "exact",
        "capture_receipt": {"status": "verified_materialized", "source_closure_verified": True},
        "n_mlir_operations": 2,
        "completeness": {
            "source_trace": {},
            "operation_obligations": [
                {
                    "operation_ids": ["op:compute"],
                    "role": "compute_placement",
                    "status": "resolved",
                    "precision": {"status": "resolved", "numerical_contracts": {"accelerator": {"status": "resolved"}}},
                    "accelerator_admission": {"status": "admitted", "reviewed": True},
                    "host_admission": {"status": "unsupported", "reviewed": True},
                },
                {
                    "operation_ids": ["op:support"],
                    "role": "support_lowering",
                    "status": "resolved",
                    "precision": {"status": "resolved"},
                    "support_lowering_evidence": {"status": "not_available"},
                },
            ],
            "graph_accounting": {"edges": [{"accounting": "support_dependency"}]},
            "transfer_obligations": [],
        },
    }
    summary, blockers = model_routes._ledger_observation(application, "exact")
    assert summary["candidate_lanes"] == {"accelerator": 1}
    assert summary["support_lowering"] == {"pending": 1, "support_dependencies": 1}
    assert summary["conditional_ssa_edges"]["count"] == 0
    assert any("support-lowering/shape" in reason for reason in blockers)
    assert any("support-mediated" in reason for reason in blockers)


def test_unobserved_oot_route_is_unresolved_not_a_compiler_decline():
    from merlin_experiments.phase1.model_routes import summarize_model_routes

    routes = summarize_model_routes({"programs": [{"name": "model", "model_sha256": "exact"}]}, None, [], [])
    assert routes[0]["status"] == "unresolved"
    assert routes[0]["accelerator"]["status"] == "not_emitted"
    assert routes[0]["whole_model_compiler_verified"] is False
