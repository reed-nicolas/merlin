"""Whole-workload admission binds source obligations to the actual capsule cohort."""

from __future__ import annotations

import copy
import json

import pytest
import yaml
from merlin_experiments.phase0 import coverage_commitment as CC

from merlin.common.digest import sha256_bytes


def _numerical_contracts(lane, operation, semantics):
    from merlin.targetgen.operation_numerics import operation_numerical_contracts

    declaration = {
        "id": f"fixture.{operation}",
        "numerical_contract": {
            "status": "reviewed",
            "semantics": semantics,
            "evidence": {"source": "explicit unit-fixture arithmetic contract"},
        },
    }
    decision = {"declaration": declaration["id"], "status": "admitted", "reviewed": True, "placement": lane}
    entry = {"observed_signature": {}, "software_admissions": [], "host_admission": {"profiles": []}}
    document = {"status": "reviewed", "operations": [declaration]}
    if lane == "accelerator":
        entry["software_admissions"] = [decision]
        return operation_numerical_contracts(entry, document, {})
    entry["host_admission"]["profiles"] = [
        {"profile": "fixture.host", "status": "admitted", "reviewed": True, "decisions": [decision]}
    ]
    return operation_numerical_contracts(entry, None, {"fixture.host": {"capability_spec": document}})


def _bind_graph(inputs):
    application = inputs["accounting"]["applications"]["application"]
    graph = application["completeness"]["graph_accounting"]
    application["operation_graph_identity"] = {
        "normalized_mlir_sha256": graph["normalized_mlir_sha256"],
        "n_operations": len(graph["nodes"]),
        "n_edges": len(graph["edges"]),
        "node_roster_sha256": CC._digest(
            [
                {key: row.get(key) for key in ("operation_id", "ordinal", "mlir_operation", "parent_operation_id")}
                for row in graph["nodes"]
            ]
        ),
        "edge_roster_sha256": CC._digest(
            [
                {
                    key: row.get(key)
                    for key in ("id", "consumer_operation_id", "producer_operation_id", "value_id", "type")
                }
                for row in graph["edges"]
            ]
        ),
    }


def _fixture():
    signature = {
        "mlir_operation": "linalg.matmul",
        "semantic_family": "contraction",
        "ordered_operand_types": [{"shape": [2, 2], "dtype": "i8"}] * 2,
        "ordered_result_types": [{"shape": [2, 2], "dtype": "i32"}],
        "accumulator_dtypes": ["i32"],
        "contraction_shape": {"M": 2, "K": 2, "N": 2},
    }
    obligation = {
        "id": "application:op:0",
        "source_operation_ids": ["torch:original:0", "torch:quantized:0"],
        "mlir_ordinals": [0],
        "operation_ids": ["mlir:0"],
        "observed_signature": signature,
        "status": "resolved",
        "role": "compute_placement",
        "placement": "unselected",
        "accelerator_admission": {"status": "admitted", "reviewed": True},
        "host_admission": {"status": "unsupported", "reviewed": True},
        "precision": {
            "status": "resolved",
            "numerical_contracts": _numerical_contracts(
                "accelerator",
                "integer_dot",
                {
                    "operand_dtype": "i8",
                    "accumulator_dtype": "i32",
                    "result_dtype": "i32",
                    "overflow": "wrap_modulo_2_32",
                },
            ),
        },
    }
    trace = {
        "status": "complete",
        "original_invocation_count": 1,
        "quantized_invocation_count": 1,
        "prepared_invocation_count": 1,
        "raw_mlir_correspondence": {"status": "verified"},
        "normalization_correspondence": {"status": "identity"},
    }
    inputs = {
        "schema": CC.INPUT_SCHEMA,
        "target": "fixture",
        "software_spec": {"status": "reviewed"},
        "evidence": {"status": "diagnostic"},
        "accounting": {
            "applications": {
                "application": {
                    "capture_sha256": "a" * 64,
                    "capture_receipt": {"status": "verified_materialized", "source_closure_verified": True},
                    "n_mlir_operations": 1,
                    "signatures": [
                        {"ordinals": [0], "observed_signature": {**signature, "disposition": "hardware_admitted"}}
                    ],
                    "framework_universe": {
                        "source_graph_version_match": {
                            "status": "matched",
                            "catalog_status": "available",
                            "catalog_version": "fixture-torch",
                            "source_graph_versions": {
                                "original": "fixture-torch",
                                "quantized": "fixture-torch",
                                "prepared": "fixture-torch",
                            },
                        }
                    },
                    "completeness": {
                        "source_trace": trace,
                        "operation_graph_status": "complete",
                        "graph_accounting": {
                            "status": "accounted",
                            "normalized_mlir_sha256": "d" * 64,
                            "n_operations": 1,
                            "n_edges": 0,
                            "nodes": [
                                {
                                    "operation_id": "mlir:0",
                                    "ordinal": 0,
                                    "mlir_operation": "linalg.matmul",
                                    "disposition": "hardware_admitted",
                                    "accounting": "placement_obligation",
                                    "obligation_id": "application:op:0",
                                    "parent_operation_id": None,
                                }
                            ],
                            "edges": [],
                            "shape_domain": {"status": "static", "dynamic_value_ids": []},
                        },
                        "operation_obligations": [obligation],
                        "transfer_obligations": [],
                    },
                }
            }
        },
    }
    capsule = {
        "name": "functional",
        "sha256": "b" * 64,
        "program_sha256": "c" * 64,
        "status": "inventoried",
        "signatures": [signature],
    }
    _bind_graph(inputs)
    return inputs, capsule


def _report(inputs, capsules):
    return CC.build_commitment(inputs, capsules, conformance_coverage={"uncovered": [], "n_required": 0})


def test_precompiler_completeness_is_independent_and_uses_only_admitted_capsules():
    inputs, capsule = _fixture()
    complete = _report(inputs, [capsule])
    CC.require_complete(complete)
    assert complete["independent_evidence"]["status"] == "diagnostic"
    assert "not execution" in complete["qualification"]
    operation = complete["applications"]["application"]["operations"][0]
    assert operation["placement"] == "accelerator"
    assert operation["witnesses"] == ["functional"]
    basis = complete["phase1_witness_basis"]
    assert basis["universe"]["n_total"] == 1
    assert basis["uncovered_obligations"] == []
    assert basis["selection"]["claim"] == "exact_minimum"
    assert [row["name"] for row in basis["selection"]["selected_capsules"]] == ["functional"]
    assert "does not establish" in basis["qualification"]
    old_report = copy.deepcopy(complete)
    old_report["schema"] = "merlin.phase0.coverage_commitment.v1"
    with pytest.raises(ValueError, match="verified whole-workload"):
        CC.require_complete(old_report)
    # A witness left in the source pool, but not admitted, cannot cover a demand.
    absent = _report(inputs, [])
    with pytest.raises(ValueError, match="admitted-capsule coverage"):
        CC.require_complete(absent)
    phase2 = CC.build_commitment(inputs, [capsule], phase="phase2", conformance_coverage={"uncovered": []})
    with pytest.raises(ValueError, match="Phase 1"):
        CC.require_complete(phase2)


def test_unknown_source_precision_or_capability_never_becomes_verified():
    inputs, capsule = _fixture()
    for component in (
        "source",
        "source_closure",
        "precision",
        "numerics",
        "wrong_lane_numerics",
        "capability",
        "capsule",
    ):
        selected, witness = copy.deepcopy(inputs), copy.deepcopy(capsule)
        completeness = selected["accounting"]["applications"]["application"]["completeness"]
        obligation = completeness["operation_obligations"][0]
        if component == "source":
            completeness["source_trace"]["quantized_invocation_count"] = None
        elif component == "source_closure":
            selected["accounting"]["applications"]["application"]["capture_receipt"]["source_closure_verified"] = False
        elif component == "precision":
            obligation["precision"]["status"] = "unknown"
        elif component == "numerics":
            obligation["precision"].pop("numerical_contracts")
        elif component == "wrong_lane_numerics":
            contracts = obligation["precision"]["numerical_contracts"]
            contracts["host"], contracts["accelerator"] = contracts["accelerator"], contracts["host"]
        elif component == "capability":
            obligation["accelerator_admission"]["status"] = "unsupported"
        else:
            witness["signatures"][0]["ordered_result_types"][0]["dtype"] = "i8"
        report = _report(selected, [witness])
        assert report["status"] == "incomplete", component
        with pytest.raises(ValueError):
            CC.require_complete(report)
    assert _report(None, [capsule])["status"] == "incomplete"


def test_graph_totality_refuses_truncated_operations_edges_and_unproved_dynamic_shapes():
    inputs, capsule = _fixture()
    for component in (
        "missing_graph",
        "missing_node",
        "missing_obligation",
        "bad_inventory",
        "dynamic_shape",
        "dropped_edge",
    ):
        selected = copy.deepcopy(inputs)
        application = selected["accounting"]["applications"]["application"]
        completeness = application["completeness"]
        graph = completeness["graph_accounting"]
        if component == "missing_graph":
            completeness.pop("graph_accounting")
        elif component == "missing_node":
            graph["nodes"] = []
            graph["n_operations"] = 0
        elif component == "missing_obligation":
            completeness["operation_obligations"] = []
        elif component == "bad_inventory":
            application["signatures"][0]["observed_signature"]["disposition"] = "structural"
        elif component == "dropped_edge":
            graph["edges"].append(
                {
                    "id": "edge:unbound",
                    "producer_operation_id": None,
                    "consumer_operation_id": "mlir:0",
                    "value_id": "argument:0",
                    "type": "tensor<2x2xi8>",
                    "accounting": "block_argument_or_non_independent_endpoint",
                    "transfer_id": None,
                }
            )
            graph["n_edges"] = 1
        else:
            graph["shape_domain"] = {"status": "unknown", "dynamic_value_ids": ["value:0"]}
        report = _report(selected, [capsule])
        assert report["status"] == "incomplete", component
        assert any(row["component"] == "graph_totality" for row in report["blockers"])
    assert (
        _report(inputs, [capsule])["applications"]["application"]["semantic_scope"]["original_vs_quantized_equivalence"]
        == "not_established_by_capsule_coverage"
    )


def test_noncompute_nodes_and_uses_are_counted_without_inventing_a_placement():
    inputs, capsule = _fixture()
    application = inputs["accounting"]["applications"]["application"]
    application["n_mlir_operations"] = 2
    application["signatures"].append(
        {"ordinals": [1], "observed_signature": {"mlir_operation": "func.return", "disposition": "structural"}}
    )
    graph = application["completeness"]["graph_accounting"]
    graph["n_operations"] = 2
    graph["nodes"].append(
        {
            "operation_id": "mlir:1",
            "ordinal": 1,
            "mlir_operation": "func.return",
            "disposition": "structural",
            "accounting": "non_independent_compute",
            "obligation_id": None,
            "parent_operation_id": None,
        }
    )
    graph["n_edges"] = 1
    graph["edges"] = [
        {
            "id": "edge:return",
            "producer_operation_id": "mlir:0",
            "consumer_operation_id": "mlir:1",
            "value_id": "value:0",
            "type": "tensor<2x2xi32>",
            "accounting": "block_argument_or_non_independent_endpoint",
            "transfer_id": None,
        }
    ]
    _bind_graph(inputs)
    report = _report(inputs, [capsule])
    CC.require_complete(report)
    assert report["applications"]["application"]["graph_accounting"]["n_edges"] == 1


def test_support_lowering_is_graph_total_but_not_a_compute_or_transfer_endpoint():
    inputs, capsule = _fixture()
    application = inputs["accounting"]["applications"]["application"]
    completeness = application["completeness"]
    signature = {
        "mlir_operation": "tensor.empty",
        "semantic_family": "movement",
        "ordered_operand_types": [],
        "ordered_result_types": [{"shape": [2, 2], "dtype": "i32"}],
        "disposition": "support_required",
    }
    support = copy.deepcopy(completeness["operation_obligations"][0])
    support.update(
        id="mlir:1",
        operation_ids=["mlir:1"],
        mlir_ordinals=[1],
        observed_signature=signature,
        role="support_lowering",
        required_placement_choices=[],
        precision={
            "status": "resolved",
            "ordered_operand_types": [],
            "ordered_result_types": [{"shape": [2, 2], "dtype": "i32"}],
            "ordered_storage_types": [],
            "result_types": ["tensor<2x2xi32>"],
        },
        support_lowering_evidence={
            "status": "not_available",
            "source_capture_sha256": "a" * 64,
            "source_operation_id": "mlir:1",
            "operand_types": [],
            "result_types": ["tensor<2x2xi32>"],
            "operand_shapes": [],
            "result_shapes": [[2, 2]],
            "source_shape_status": "static",
        },
    )
    completeness["operation_obligations"].append(support)
    application["n_mlir_operations"] = 2
    application["signatures"].append({"ordinals": [1], "observed_signature": signature})
    graph = completeness["graph_accounting"]
    graph["n_operations"] = 2
    graph["nodes"].append(
        {
            "operation_id": "mlir:1",
            "ordinal": 1,
            "mlir_operation": "tensor.empty",
            "disposition": "support_required",
            "accounting": "support_lowering_obligation",
            "obligation_id": "mlir:1",
            "parent_operation_id": None,
        }
    )
    graph["n_edges"] = 1
    graph["edges"] = [
        {
            "id": "edge:support",
            "producer_operation_id": "mlir:1",
            "consumer_operation_id": "mlir:0",
            "value_id": "value:support",
            "type": "tensor<2x2xi32>",
            "accounting": "support_dependency",
            "transfer_id": None,
        }
    ]
    capsule["signatures"].append(signature)
    _bind_graph(inputs)
    report = _report(inputs, [capsule])
    assert report["status"] == "incomplete"
    assert [item["component"] for item in report["blockers"]] == ["support_lowering", "support_dependency"]
    assert report["applications"]["application"]["transfers"] == []
    support_report = report["applications"]["application"]["operations"][1]
    assert support_report["role"] == "support_lowering"
    assert support_report["placement"] is None
    assert support_report["witnesses"] == ["functional"]
    # A self-asserted flag in the source ledger is not a compiler artifact or
    # proof that its typed shape/value semantics survived lowering.
    support["support_lowering_evidence"]["status"] = "verified"
    assert _report(inputs, [capsule])["status"] == "incomplete"
    graph["edges"][0]["accounting"] = "block_argument_or_non_independent_endpoint"
    _bind_graph(inputs)
    assert any(item["component"] == "graph_totality" for item in _report(inputs, [capsule])["blockers"])


def test_framework_catalog_and_every_source_stage_must_match():
    inputs, capsule = _fixture()
    for component in ("missing", "catalog_unavailable", "unknown", "mismatch", "missing_stage", "false_match"):
        selected = copy.deepcopy(inputs)
        application = selected["accounting"]["applications"]["application"]
        match = application["framework_universe"]["source_graph_version_match"]
        if component == "missing":
            application.pop("framework_universe")
        elif component == "catalog_unavailable":
            match["catalog_status"] = "not_available"
        elif component in {"unknown", "mismatch"}:
            match["status"] = component
        elif component == "missing_stage":
            match["source_graph_versions"].pop("original")
        else:
            match["source_graph_versions"]["quantized"] = "different-torch"
        report = _report(selected, [capsule])
        assert report["status"] == "incomplete", component
        assert any(blocker["component"] == "framework_versions" for blocker in report["blockers"])
        with pytest.raises(ValueError):
            CC.require_complete(report)


def test_manifest_bound_inputs_refuse_tamper_and_do_not_read_checkout(tmp_path):
    inputs, _ = _fixture()
    corpus = tmp_path / "installed-run/corpus"
    record = CC.write_inputs(corpus, inputs)
    (corpus / "MANIFEST.yaml").write_text(yaml.safe_dump({"coverage_inputs": record}))
    assert CC.read_inputs(corpus) == inputs
    # The self-contained sidecar carries no original checkout path to rediscover.
    path = corpus / CC.INPUT_PATH
    path.write_bytes(json.dumps({**inputs, "software_spec": {"status": "unreviewed"}}).encode())
    with pytest.raises(ValueError, match="differ from the derivation commitment"):
        CC.read_inputs(corpus)
    with pytest.raises(ValueError, match="fresh corpus"):
        CC.write_inputs(corpus, inputs)
    assert record["sha256"] != sha256_bytes(path.read_bytes())


def test_typed_transfer_requires_review_and_an_admitted_edge_witness(monkeypatch):
    from merlin.targetgen import software_spec

    inputs, capsule = _fixture()
    completeness = inputs["accounting"]["applications"]["application"]["completeness"]
    second = copy.deepcopy(completeness["operation_obligations"][0])
    second.update(id="application:op:1", operation_ids=["mlir:1"], mlir_ordinals=[1])
    second["observed_signature"]["mlir_operation"] = "arith.trunci"
    second["accelerator_admission"]["status"] = "unsupported"
    second["host_admission"]["status"] = "admitted"
    second["precision"]["numerical_contracts"] = _numerical_contracts(
        "host", "integer_truncate", {"operation": "arith.trunci", "semantics": "retain_low_result_width_bits"}
    )
    completeness["operation_obligations"].append(second)
    application = inputs["accounting"]["applications"]["application"]
    application["n_mlir_operations"] = 2
    application["signatures"].append(
        {"ordinals": [1], "observed_signature": {**second["observed_signature"], "disposition": "host_required"}}
    )
    graph = completeness["graph_accounting"]
    graph["n_operations"] = 2
    graph["nodes"].append(
        {
            "operation_id": "mlir:1",
            "ordinal": 1,
            "mlir_operation": "arith.trunci",
            "disposition": "host_required",
            "accounting": "placement_obligation",
            "obligation_id": "application:op:1",
            "parent_operation_id": None,
        }
    )
    graph["n_edges"] = 1
    graph["edges"] = [
        {
            "id": "edge:0",
            "producer_operation_id": "mlir:0",
            "consumer_operation_id": "mlir:1",
            "value_id": "value:0",
            "type": "tensor<2x2xi32>",
            "accounting": "conditional_transfer",
            "transfer_id": "edge:0",
        }
    ]
    completeness["transfer_obligations"] = [
        {
            "id": "edge:0",
            "producer_operation_id": "mlir:0",
            "consumer_operation_id": "mlir:1",
            "value_id": "value:0",
            "type": "tensor<2x2xi32>",
            "dtype": "i32",
        }
    ]
    _bind_graph(inputs)
    truncated = copy.deepcopy(inputs)
    truncated["accounting"]["applications"]["application"]["completeness"]["transfer_obligations"] = []
    assert any(row["component"] == "graph_totality" for row in _report(truncated, [capsule])["blockers"])
    capsule["signatures"].append(copy.deepcopy(second["observed_signature"]))
    monkeypatch.setattr(
        software_spec,
        "screen_transfer_contract",
        lambda *args, **kwargs: {"status": "admitted", "reviewed": True, "matching_declarations": ["typed_transfer"]},
        raising=False,
    )
    # Two independent operations do not witness their typed boundary.
    assert _report(inputs, [capsule])["status"] == "incomplete"
    capsule["operation_signature_ids"] = {
        "witness:0": CC.signature_identity(capsule["signatures"][0]),
        "witness:1": CC.signature_identity(capsule["signatures"][1]),
    }
    capsule["operation_graph"] = {
        "edges": [
            {
                "producer_operation_id": "witness:0",
                "consumer_operation_id": "witness:1",
                "type": "tensor<2x2xi32>",
            }
        ]
    }
    CC.require_complete(_report(inputs, [capsule]))
    monkeypatch.setattr(
        software_spec, "screen_transfer_contract", lambda *args, **kwargs: {"status": "unknown", "reviewed": False}
    )
    assert _report(inputs, [capsule])["status"] == "incomplete"
    pending_inputs = copy.deepcopy(inputs)
    pending_inputs["accounting"]["applications"]["application"]["completeness"]["operation_obligations"][1][
        "host_admission"
    ]["status"] = "unknown"
    calls = []
    monkeypatch.setattr(software_spec, "screen_transfer_contract", lambda *args, **kwargs: calls.append(True))
    pending = _report(pending_inputs, [capsule])
    assert calls == []
    assert pending["status"] == "incomplete"
    edge = pending["applications"]["application"]["transfers"][0]
    assert edge["status"] == "pending_placement"
    assert edge["declaration"]["status"] == "not_screened"
    assert edge["witnesses"] == ["functional"]
    assert [row["count"] for row in pending["blockers"] if row["component"] == "transfer"] == [1]


def test_native_frozen_commitment_survives_source_removal_without_reobservation(tmp_path, monkeypatch):
    import shutil

    from merlin_experiments.phase1 import corpus_inputs as CI
    from merlin_experiments.phase1.source_inputs import fingerprint

    from merlin.common.paths import data_path
    from merlin.targetgen.sandbox import bwrap
    from merlin.targetgen.target_experiment import load_target_experiment

    inputs, witness = _fixture()
    corpus = tmp_path / "external-corpus"
    directory = corpus / "isa/functional"
    directory.mkdir(parents=True)
    program = directory / "capsule.linalg.mlir"
    program.write_text("module {}\n")
    (directory / "capsule.interface.mlir").write_text("module {}\n")
    (directory / "golden.yaml").write_text("outputs: {}\n")
    (directory / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "functional",
                "kind": "isa",
                "source_role": "handauthored_compiler_test",
                "label": "public",
                "operation": {"op": "matmul", "attributes": {}},
                "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
                "expected": {"instruction_classes": [], "modes": {}},
                "required_oracle_tiers": ["L0"],
                "interface_mlir": "capsule.interface.mlir",
                "linalg_mlir": "capsule.linalg.mlir",
            }
        )
    )
    record = CC.write_inputs(corpus, inputs)
    (corpus / "MANIFEST.yaml").write_text(yaml.safe_dump({"coverage_inputs": record}))
    witness.update(sha256=fingerprint(directory), program_sha256=sha256_bytes(program.read_bytes()))
    measured = _report(inputs, [witness])
    monkeypatch.setattr(CC, "observe_cohort", lambda *args, **kwargs: measured)
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "fixture", "capsule_corpus": str(corpus / "isa")}))
    te = load_target_experiment(descriptor)
    run = tmp_path / "run"
    run.mkdir()
    contract = data_path("contract")
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(contract))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    bundle, corpus_record = CI.stage(
        run, te, {"bundle_id": "fixture", "allowed": [], "denied": []}, contract=contract, capsules_root=corpus
    )
    workspace = tmp_path / "workspace"
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=tmp_path)
    shutil.rmtree(corpus)
    monkeypatch.setattr(CC, "observe_cohort", lambda *args, **kwargs: pytest.fail("resume rebuilt a live census"))
    view = CI.resolve(workspace, bundle, corpus_record, repo=tmp_path)
    CC.require_complete(view.workload_coverage)
    assert view.workload_coverage["cohort"]["sha256"] == measured["cohort"]["sha256"]


def test_conformance_cannot_borrow_ambient_provider_evidence(monkeypatch):
    from merlin_experiments.corpus.coverage import selected_cohort_coverage

    from merlin.targetgen import target_registry

    monkeypatch.setattr(target_registry, "resolve", lambda *args, **kwargs: pytest.fail("ambient provider was read"))
    requirement = {"target": "fixture", "cells": [], "composition": {"required": {"routing": 1}}}
    coverage = selected_cohort_coverage(requirement, [])
    assert coverage["composition"]["status"] == "not_measured"
    assert coverage["composition"]["phase"] == "phase0"
    assert coverage["composition"]["required"] == {"routing": 1}
    assert set(coverage["composition"]["phase1_receipt_required"]) == {
        "selected_capture",
        "selected_capsule",
        "compiler_execution",
        "lowering_correspondence",
        "execution",
    }
    assert CC._conformance_blockers(coverage)
