"""Coarse Phase 0 support/trace/precision checks without full-network compilation."""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from merlin.targetgen.application_inventory import application_demand_inventory
from merlin.targetgen.frontend_trace import join_frontend_trace
from merlin.targetgen.operation_accounting import build_operation_accounting
from merlin.targetgen.software_spec import admit_operation, screen_transfer_contract
from merlin.targetgen.target_experiment import HostLane, HostLaneMatrix


def _sha(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_MLIR = """builtin.module {
  func.func @main(%a: f32, %b: f32) -> f32 {
    %sum = arith.addf %a, %b {
      prov.aten = "aten.add.Tensor", prov.source_node_ids = ["g:prepared:root:n2"],
      prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"
    } : f32
    %result = arith.mulf %sum, %b {
      prov.aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n3"],
      prov.origin_node_ids = ["g:original:root:n3", "g:quantized:root:n3"], prov.trace_role = "lowering"
    } : f32
    func.return %result : f32
  }
}
"""


def _inputs(path):
    path.write_text(_MLIR)
    contract = {
        "name": "test_device",
        "compute_units": [
            {
                "name": "vector_unit",
                "kind": "vector",
                "dtypes": ["fp32"],
                "ops": ["add", "mul"],
                "semantic_capabilities": [{"family": "elementwise_map", "dtypes": ["fp32"]}],
            }
        ],
    }
    inventory = application_demand_inventory(
        {"iteration": path},
        "test_device",
        detailed=True,
        capability_contract=contract,
        include_graph=True,
        application_metadata={"iteration": {"workload_role": "iteration", "coverage_scope": "representative_subset"}},
    )
    graph = inventory["applications"]["iteration"]["operation_graph"]
    graphs = {}
    for stage in ("original", "quantized", "prepared"):
        snapshot = {
            "schema": "m2m.frontend_graph.v1",
            "stage": stage,
            "status": "complete",
            "call_count": 2,
            "nodes": [
                {
                    "id": f"g:{stage}:root:n{ordinal}",
                    "ordinal": ordinal,
                    "op": "call_function",
                    "target": operator,
                    "results": [],
                }
                for ordinal, operator in ((2, "aten.add.Tensor"), (3, "aten.mul.Tensor"))
            ],
            "edges": [],
            "runtime_versions": {"torch": "test_version"},
        }
        snapshot["sha256"] = _sha(snapshot)
        graphs[stage] = snapshot
    transformations = [
        {
            "from_stage": source,
            "to_stage": destination,
            "status": "complete",
            "relations": [
                {
                    "source_ids": [f"g:{source}:root:n{ordinal}"],
                    "destination_ids": [f"g:{destination}:root:n{ordinal}"],
                    "kind": "identity",
                }
                for ordinal in (2, 3)
            ],
        }
        for source, destination in (("original", "quantized"), ("quantized", "prepared"))
    ]
    operations = []
    for operation in graph["capture_graph"]["operations"]:
        operations.append(
            {
                "ordinal": operation["ordinal"],
                "operation": operation["mlir_operation"],
                "source_node_ids": operation["source_node_ids"],
                "origin_node_ids": operation["origin_node_ids"],
                "role": operation["trace_role"] or "structural",
                "operand_types": [value["type"] for value in operation["operands"]],
                "result_types": [value["type"] for value in operation["results"]],
            }
        )
    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "graphs": graphs,
        "transformations": transformations,
        "blockers": [],
        "mlir": {
            "sha256": graph["capture_sha256"],
            "bytes": graph["capture_bytes"],
            "operations": operations,
            "source_correspondence": [
                {"node_id": f"g:prepared:root:n{ordinal}", "mlir_ordinals": [ordinal], "status": "lowered"}
                for ordinal in (2, 3)
            ],
        },
    }
    spec = {
        "status": "reviewed",
        "operations": [
            {
                "id": "add",
                "ops": ["aten.add.Tensor"],
                "placement": "accelerator",
                "signature": {"operand_dtypes": ["fp32"]},
            },
            {"id": "mul", "ops": ["aten.mul.Tensor"], "placement": "host", "signature": {"operand_dtypes": ["fp32"]}},
        ],
    }
    host = {
        "schema": "merlin.host_capabilities.v1",
        "status": "reviewed",
        "compiler": {"package_sha256": "b" * 64, "dtype_strategy": "fp32"},
        "operations": [
            {"id": "mul", "ops": ["arith.mulf"], "placement": "host", "signature": {"operand_dtypes": ["fp32"]}}
        ],
        "evidence": {"review": "fixture declaration, not execution certification"},
    }
    selected_host = {
        "fp32": {
            "capability_spec": host,
            "package_sha256": "b" * 64,
            "capability_spec_sha256": _sha(host),
            "dtype_strategy": "fp32",
        }
    }
    return inventory, contract, spec, trace, selected_host


class Phase0SupportAccounting(unittest.TestCase):
    def test_exact_mlir_roster_does_not_hide_unlowered_prepared_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory, _, _, trace, _ = _inputs(Path(directory) / "model.mlir")
            selected = inventory["applications"]["iteration"]
            prepared = trace["graphs"]["prepared"]
            prepared["nodes"].append(
                {
                    "id": "g:prepared:root:n4",
                    "ordinal": 4,
                    "op": "call_function",
                    "target": "aten.relu.default",
                    "results": [],
                }
            )
            prepared["call_count"] = 3
            prepared["sha256"] = _sha({key: value for key, value in prepared.items() if key != "sha256"})
            transition = trace["transformations"][1]
            transition.update(status="diagnostic", unresolved_destination_ids=["g:prepared:root:n4"])
            trace["mlir"]["source_correspondence"].append(
                {"node_id": "g:prepared:root:n4", "mlir_ordinals": [], "status": "unresolved"}
            )
            trace.update(status="diagnostic", blockers=["prepared call has no final MLIR correspondence"])

            joined = join_frontend_trace(trace, selected["operation_graph"], capture_sha256=selected["capture_sha256"])
            self.assertEqual(joined["status"], "partial")
            self.assertEqual(joined["raw_mlir_correspondence"]["status"], "verified")
            self.assertEqual(joined["normalization_correspondence"]["status"], "verified")
            obligations = joined["prepared_lowering_obligations"]
            self.assertEqual(obligations["status"], "unresolved")
            self.assertEqual(obligations["unresolved_calls"][0]["node_id"], "g:prepared:root:n4")
            self.assertFalse(any("MLIR roster differs" in error for error in joined["errors"]))

    def test_unresolved_frontend_transition_exposes_typed_obligations(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory, _, _, trace, _ = _inputs(Path(directory) / "model.mlir")
            selected = inventory["applications"]["iteration"]
            graph = trace["graphs"]["quantized"]
            graph["nodes"][0]["results"] = [{"id": "g:quantized:root:n2:v0", "dtype": "int64", "shape": [1]}]
            graph["nodes"][1].update(
                target="aten.to.dtype",
                results=[{"id": "g:quantized:root:n3:v0", "dtype": "float32", "shape": [1]}],
            )
            graph["edges"] = [
                {
                    "producer_node_id": "g:quantized:root:n2",
                    "consumer_node_id": "g:quantized:root:n3",
                    "producer_value_id": "g:quantized:root:n2:v0",
                    "dtype": "int64",
                    "shape": [1],
                }
            ]
            graph["sha256"] = _sha({key: value for key, value in graph.items() if key != "sha256"})
            transition = trace["transformations"][1]
            transition["relations"] = transition["relations"][:1]
            transition.update(
                status="diagnostic",
                unresolved_source_ids=["g:quantized:root:n3"],
                unresolved_destination_ids=["g:prepared:root:n3"],
            )
            trace.update(status="diagnostic", blockers=["quantized -> prepared correspondence incomplete"])
            joined = join_frontend_trace(trace, selected["operation_graph"], capture_sha256=selected["capture_sha256"])
            self.assertEqual(joined["status"], "partial")
            unresolved = joined["transition_obligations"][1]
            self.assertEqual(unresolved["producer_unresolved_ids_status"], "matched")
            self.assertEqual(
                unresolved["unresolved_source_calls"],
                [
                    {
                        "node_id": "g:quantized:root:n3",
                        "op": "call_function",
                        "target": "aten.to.dtype",
                        "input_dtypes": ["int64"],
                        "result_dtypes": ["float32"],
                    }
                ],
            )
            self.assertEqual(unresolved["unresolved_destination_calls"][0]["node_id"], "g:prepared:root:n3")
            self.assertNotIn("eliminated", json.dumps(unresolved))

            # A producer's claimed uncovered set cannot contradict the relation roster.
            transition["unresolved_source_ids"] = []
            mismatch = join_frontend_trace(
                trace, selected["operation_graph"], capture_sha256=selected["capture_sha256"]
            )
            self.assertEqual(mismatch["transition_obligations"][1]["producer_unresolved_ids_status"], "mismatch")
            self.assertTrue(any("producer unresolved call IDs disagree" in error for error in mismatch["errors"]))

    def test_dynamic_ssa_extent_is_explicitly_unproved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mlir"
            path.write_text("""builtin.module {
              func.func @main(%n: index) -> tensor<?xi64> {
                %result = "tensor.empty"(%n) : (index) -> tensor<?xi64>
                func.return %result : tensor<?xi64>
              }
            }""")
            inventory = application_demand_inventory(
                {"dynamic": path},
                "test_device",
                detailed=True,
                capability_contract={"name": "test_device", "compute_units": []},
                include_graph=True,
            )
            accounting = build_operation_accounting(inventory)
            graph = accounting["applications"]["dynamic"]["completeness"]["graph_accounting"]
            self.assertEqual(graph["n_operations"], 4)
            self.assertEqual(graph["shape_domain"]["status"], "unknown")
            self.assertTrue(graph["shape_domain"]["dynamic_value_ids"])
            support = accounting["applications"]["dynamic"]["completeness"]["operation_obligations"][0]
            self.assertEqual(support["role"], "support_lowering")
            self.assertEqual(support["support_lowering_evidence"]["status"], "not_available")
            self.assertNotIn("numerical_contracts", support["precision"])
            self.assertEqual(support["required_placement_choices"], [])
            self.assertEqual(
                next(node for node in graph["nodes"] if node["mlir_operation"] == "tensor.empty")["accounting"],
                "support_lowering_obligation",
            )

    def test_normalization_preserves_exact_source_metadata_on_support_operations(self):
        from merlin.targetgen.application_graph import application_graph_inventory

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mlir"
            path.write_text("""builtin.module {
              func.func @main() -> tensor<4xf32> {
                %value = "tensor.empty"() {
                  prov.source_node_ids = ["g:prepared:root:n2"],
                  prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"],
                  prov.trace_role = "support", provenance_required = true
                } : () -> tensor<4xf32>
                func.return %value : tensor<4xf32>
              }
            }""")
            graph = application_graph_inventory(path)
            original = next(
                operation
                for operation in graph["capture_graph"]["operations"]
                if operation["mlir_operation"] == "tensor.empty"
            )
            normalized = next(
                operation
                for operation in graph["normalized_graph"]["operations"]
                if operation["mlir_operation"] == "tensor.empty"
            )
            self.assertEqual(original["source_node_ids"], normalized["source_node_ids"])
            self.assertEqual(original["attributes"], normalized["attributes"])
            self.assertEqual(graph["normalization_correspondence"]["status"], "serialization_equivalent")

    def test_support_to_support_ssa_use_is_not_an_independent_lane_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mlir"
            path.write_text("""builtin.module {
              func.func @main() -> tensor<2xf32> {
                %zero = arith.constant 0.0 : f32
                %value = tensor.splat %zero : tensor<2xf32>
                func.return %value : tensor<2xf32>
              }
            }""")
            inventory = application_demand_inventory(
                {"support": path},
                "test_device",
                detailed=True,
                capability_contract={"name": "test_device", "compute_units": []},
                include_graph=True,
            )
            completeness = build_operation_accounting(inventory)["applications"]["support"]["completeness"]
            graph = completeness["graph_accounting"]
            assert graph["n_operations"] == 5
            assert len(completeness["operation_obligations"]) == 2
            assert all(row["role"] == "support_lowering" for row in completeness["operation_obligations"])
            assert all(
                row["accelerator_admission"]["status"] == "not_applicable"
                for row in completeness["operation_obligations"]
            )
            assert all(
                row["support_lowering_evidence"]["status"] == "not_available"
                for row in completeness["operation_obligations"]
            )
            assert completeness["transfer_obligations"] == []
            assert [edge["accounting"] for edge in graph["edges"] if edge["accounting"] == "support_dependency"] == [
                "support_dependency"
            ]
            summary = build_operation_accounting(inventory)["overall"]
            assert summary["classification_counts"]["support_lowering_required"] == 2
            assert summary["support_partition_counts"]["not_independent_compute"] == 5
            assert summary["accelerator_admission_counts"] == {"not_applicable": 5}

    def test_joined_typed_partitions_and_conditional_transfer_are_honest(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory, contract, spec, trace, host = _inputs(Path(directory) / "model.mlir")
            report = build_operation_accounting(
                inventory,
                spec,
                capability_contract=contract,
                frontend_traces={"iteration": trace},
                host_capabilities=host,
            )
            application = report["applications"]["iteration"]
            completeness = application["completeness"]
            self.assertEqual(completeness["source_trace"]["status"], "complete")
            graph = completeness["graph_accounting"]
            self.assertEqual(graph["status"], "accounted")
            self.assertEqual(graph["n_operations"], application["n_mlir_operations"])
            self.assertEqual(graph["n_edges"], len(graph["edges"]))
            self.assertEqual(application["operation_graph_identity"]["n_edges"], graph["n_edges"])
            self.assertEqual(
                application["operation_graph_identity"]["normalized_mlir_sha256"], graph["normalized_mlir_sha256"]
            )
            self.assertEqual(
                {node["operation_id"] for node in graph["nodes"] if node["accounting"] == "placement_obligation"},
                {row["id"] for row in completeness["operation_obligations"]},
            )
            self.assertEqual(graph["shape_domain"]["status"], "static")
            self.assertEqual(report["overall"]["pytorch_provenance"]["original_pytorch_invocation_count"], 2)
            self.assertEqual(report["overall"]["pytorch_provenance"]["quantized_pytorch_invocation_count"], 2)
            self.assertEqual(application["workload_identity"]["coverage_scope"], "representative_subset")
            self.assertEqual(
                [row["required_placement_choices"] for row in completeness["operation_obligations"]],
                [["accelerator"], ["host"]],
            )
            self.assertTrue(
                all(row["precision"]["status"] == "resolved" for row in completeness["operation_obligations"])
            )
            self.assertEqual(len(completeness["transfer_obligations"]), 1)
            edge = completeness["transfer_obligations"][0]
            self.assertEqual(edge["source_type"], "f32")
            self.assertEqual(edge["status"], "conditional")
            self.assertEqual(edge["conversion"]["status"], "not_applicable")
            decomposed = deepcopy(trace)
            original = decomposed["graphs"]["original"]
            original["nodes"][0]["target"] = "aten.linear.default"
            original["nodes"][1]["target"] = "aten.relu.default"
            original["sha256"] = _sha({key: value for key, value in original.items() if key != "sha256"})
            catalog = {
                "status": "available",
                "torch": "test_version",
                "n_all_aten": 4,
                "all_ops": ["aten.linear.default", "aten.relu.default", "aten.add.Tensor", "aten.mul.Tensor"],
            }
            frontend_first = build_operation_accounting(
                inventory,
                spec,
                capability_contract=contract,
                frontend_traces={"iteration": decomposed},
                host_capabilities=host,
                framework_catalog=catalog,
            )["framework_universe"]
            self.assertEqual(
                frontend_first["observed_registered_aten_operators"], ["aten.linear.default", "aten.relu.default"]
            )
            self.assertEqual(
                frontend_first["stages"]["prepared"]["operator_names"], ["aten.add.Tensor", "aten.mul.Tensor"]
            )
            self.assertEqual(
                frontend_first["lowered_provenance"]["observed_registered_aten_operators"],
                ["aten.add.Tensor", "aten.mul.Tensor"],
            )
            # Altered bytes cannot be joined through matching operator names and counts.
            corrupted = deepcopy(trace)
            corrupted["mlir"]["sha256"] = "f" * 64
            broken = build_operation_accounting(
                inventory,
                spec,
                capability_contract=contract,
                frontend_traces={"iteration": corrupted},
                host_capabilities=host,
            )
            self.assertEqual(broken["applications"]["iteration"]["completeness"]["source_trace"]["status"], "partial")
            absent = deepcopy(trace)
            absent["graphs"]["original"] = None
            missing = build_operation_accounting(inventory, spec, frontend_traces={"iteration": absent})
            self.assertIsNone(missing["overall"]["pytorch_provenance"]["original_pytorch_invocation_count"])

    def test_exact_precision_and_transfer_constraints_do_not_default(self):
        lane = HostLane(
            "test",
            "local",
            "UNKNOWN",
            "UNKNOWN",
            "package",
            ("manifest.yaml",),
            ("package",),
            ("code",),
            dtype_strategy="fp32",
        )
        matrix = HostLaneMatrix("fp32", {"fp32": lane})
        self.assertIs(matrix.for_dtype(None), lane)
        self.assertIs(matrix.for_dtype("f32"), lane)
        with self.assertRaises(ValueError):
            matrix.for_dtype("not_a_precision")
        spec = {
            "status": "reviewed",
            "operations": [
                {
                    "id": "add",
                    "ops": ["add"],
                    "placement": "host",
                    "signature": {"ordered_operand_dtypes": ["f16", "f32"], "ordered_result_dtypes": ["f32"]},
                }
            ],
        }
        self.assertEqual(
            admit_operation(
                spec, "add", {"ordered_operand_dtypes": ["f32", "f16"], "ordered_result_dtypes": ["f32"]}, "host"
            )["status"],
            "unsupported",
        )
        self.assertEqual(
            screen_transfer_contract(
                None,
                source_placement="host",
                destination_placement="accelerator",
                operand_dtype="f32",
                result_dtype="f32",
            )["status"],
            "unknown",
        )
        self.assertEqual(
            screen_transfer_contract(
                None, source_placement="host", destination_placement="host", operand_dtype="f32", result_dtype="f32"
            )["status"],
            "not_applicable",
        )
        transfer = {
            "transfer_contracts": {
                "status": "reviewed",
                "declarations": [
                    {
                        "id": "copy",
                        "from": "host",
                        "to": "accelerator",
                        "signature": {"operand_dtype": "f32", "result_dtype": "f32"},
                        "semantics": {"mode": "preserve_values"},
                        "evidence": {"review": "fixture"},
                    }
                ],
            }
        }
        self.assertEqual(
            screen_transfer_contract(
                transfer,
                source_placement="host",
                destination_placement="accelerator",
                operand_dtype="f32",
                result_dtype="f32",
            )["status"],
            "admitted",
        )

    def test_validation_sources_cannot_hide_behind_an_iteration_alias(self):
        from merlin.targetgen.claim_models import claim_models

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mlir"
            path.write_text(_MLIR)
            for metadata in (
                {"workload_role": "validation"},
                {"workload_role": "iteration", "source_workload_id": claim_models()[0]},
            ):
                with self.assertRaises(ValueError):
                    application_demand_inventory(
                        {"independent_alias": path},
                        "test_device",
                        detailed=True,
                        capability_contract={"compute_units": []},
                        application_metadata={"independent_alias": metadata},
                    )


if __name__ == "__main__":
    unittest.main()
