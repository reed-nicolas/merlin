"""Account for typed SSA crossings around outlined model kernels.

This is a diagnostic dataflow inventory.  In particular, an SSA edge does not
choose a memory placement, transfer implementation, dispatch ABI or host
lowering.  Those must be supplied before a model can execute on a target.
"""

from __future__ import annotations

from merlin.targetgen.application_graph import _program_graph

SCHEMA = "merlin.model_stitching_inventory.v1"


def stitching_inventory(module, model_sha256: str, candidates: list[dict]) -> dict:
    """Join exact outlined operation IDs to the parsed model's complete SSA graph."""
    graph = _program_graph(module, model_sha256)
    rows = graph["operations"]
    by_id = {row["operation_id"]: row for row in rows}
    selected = {candidate["operation_id"] for candidate in candidates}
    if len(selected) != len(candidates) or any(
        operation_id not in by_id or by_id[operation_id]["mlir_operation"] != "linalg.generic"
        for operation_id in selected
    ):
        raise ValueError("outlined candidates do not identify distinct parsed contractions")

    functions = []
    for function in rows:
        if function["mlir_operation"] != "func.func":
            continue
        function_id = function["operation_id"]
        steps = []
        for row in rows:
            if row["parent_operation_id"] != function_id:
                continue
            kind = "accelerator_candidate" if row["operation_id"] in selected else "host_unlowered"
            if row["mlir_operation"] == "func.return":
                kind = "return_binding"
            steps.append(
                {
                    "operation_id": row["operation_id"],
                    "ordinal": row["ordinal"],
                    "mlir_operation": row["mlir_operation"],
                    "kind": kind,
                    "source_node_ids": row["source_node_ids"],
                    "origin_node_ids": row["origin_node_ids"],
                    "inputs": row["operands"],
                    "outputs": row["results"],
                }
            )
        functions.append(
            {
                "operation_id": function_id,
                "symbol": function["properties"].get("sym_name", function["attributes"].get("sym_name")),
                "top_level_steps": steps,
                "return_values": [
                    operand for step in steps if step["kind"] == "return_binding" for operand in step["inputs"]
                ],
                "unlowered_host_operations": [
                    step["operation_id"] for step in steps if step["kind"] == "host_unlowered"
                ],
            }
        )

    crossings = []
    for edge in graph["edges"]:
        consumer_id = edge["consumer_operation_id"]
        producer_id = edge["producer_operation_id"]
        consumer = by_id[consumer_id]
        # An exact outlined contraction consumes operands 0 and 1.  Its zero
        # accumulator initializer is an implementation detail, not a transfer.
        into_kernel = consumer_id in selected and edge["operand_index"] in (0, 1)
        from_kernel = producer_id in selected
        if not into_kernel and not from_kernel:
            continue
        if into_kernel and from_kernel:
            direction = "candidate_to_candidate"
        elif into_kernel:
            direction = "host_to_candidate"
        else:
            direction = "candidate_to_host"
        crossings.append(
            {
                "edge_id": edge["id"],
                "source_value_id": edge["value_id"],
                "producer_operation_id": producer_id,
                "consumer_operation_id": consumer_id,
                "consumer_operand_index": edge["operand_index"],
                "consumer_operation": consumer["mlir_operation"],
                "type": edge["type"],
                "shape": edge["shape"],
                "dtype": edge["dtype"],
                "direction": direction,
                "status": "requires_placement_transfer_and_dispatch",
            }
        )
    return {
        "schema": SCHEMA,
        "model_sha256": model_sha256,
        "operations_accounted": graph["n_operations"],
        "candidate_operation_ids": [candidate["operation_id"] for candidate in candidates],
        "functions": functions,
        "candidate_boundary_crossings": crossings,
        "status": "diagnostic_unexecutable",
        "missing_for_execution": [
            "host_operation_lowering",
            "memory_placement_and_typed_transfers",
            "kernel_dispatch_and_pointer_abi",
            "whole_model_numerical_equivalence",
        ],
    }
