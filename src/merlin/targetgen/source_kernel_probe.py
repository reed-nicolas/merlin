"""Bound a captured float or verified integer matrix body for a finite diagnostic.

The source operation keeps its original geometry and dtype in the record. The
window is a *new synthetic operation*: projecting its dtype or geometry does
not establish that the original model executes the projected operation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _ascii_decimal(value: str, *, positive: bool = False) -> bool:
    return bool(value) and (not positive or "1" <= value[0] <= "9") and all(
        "0" <= char <= "9" for char in value
    )


def _scalar_type(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "a" <= value[0] <= "z"
        and all("a" <= char <= "z" or "0" <= char <= "9" for char in value[1:])
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_type(spelling: str) -> tuple[int, int, str]:
    if not spelling.startswith("tensor<") or not spelling.endswith(">"):
        raise ValueError(f"expected a static rank-2 tensor, got {spelling!r}")
    parts = spelling[len("tensor<") : -1].split("x")
    if (
        len(parts) != 3
        or not _ascii_decimal(parts[0], positive=True)
        or not _ascii_decimal(parts[1], positive=True)
        or not _scalar_type(parts[2])
    ):
        raise ValueError(f"expected a static rank-2 tensor, got {spelling!r}")
    return int(parts[0]), int(parts[1]), parts[2]


def derive_kernel_window(
    capture_dir: str | Path,
    source_node_id: str,
    *,
    tile_dim: int,
    projection_types: tuple[str, str, str],
) -> dict:
    """Select one exact traced float or integer matrix body and derive one bounded test window.

    The selected trace's MLIR hash must match the model bytes. A caller selects a
    source node ID, not a shape; ambiguous nodes fail closed. The K window has
    one full tile plus the parent's K remainder (or a second full tile), so a
    parent tail remains visible without simulating its full reduction.
    """
    if type(tile_dim) is not int or tile_dim < 1:
        raise ValueError("tile_dim must be a positive integer derived from selected hardware facts")
    if len(projection_types) != 3 or any(not _scalar_type(dtype) for dtype in projection_types):
        raise ValueError("projection_types must name three MLIR scalar types")
    capture = Path(capture_dir)
    model_path = capture / "model.mlir"
    trace_path = capture / "frontend-trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    model_sha = _sha(model_path)
    if trace.get("mlir", {}).get("sha256") != model_sha:
        raise ValueError("frontend trace does not bind the selected model.mlir bytes")
    nodes = [node for node in trace["graphs"]["prepared"]["nodes"] if node.get("id") == source_node_id]
    if len(nodes) != 1:
        raise ValueError(f"source node {source_node_id!r} is absent or ambiguous in the prepared graph")
    candidates = [
        op for op in trace["mlir"]["operations"]
        if op.get("operation") in {"linalg.matmul", "linalg.generic"}
        and source_node_id in op.get("source_node_ids", [])
    ]
    integer_candidates = [row for row in candidates if row["operation"] == "linalg.generic"]
    if integer_candidates:
        from merlin.common import mlir_query as query
        from merlin.targetgen.application_inventory import exact_int_mm_generic_operation

        parsed = list(query.walk(query.parse(model_path)))
        verified = []
        for row in integer_candidates:
            ordinal = row.get("ordinal")
            if type(ordinal) is not int or ordinal < 0 or ordinal >= len(parsed):
                raise ValueError("integer matmul trace ordinal is absent from the captured MLIR")
            actual = parsed[ordinal]
            source_ids = actual.attributes.get("prov.source_node_ids")
            if (
                source_ids is None
                or source_node_id not in [getattr(value, "data", None) for value in source_ids]
                or [str(value.type) for value in actual.operands] != row.get("operand_types")
                or [str(value.type) for value in actual.results] != row.get("result_types")
                or not exact_int_mm_generic_operation(actual)
            ):
                raise ValueError("traced generic is not the exact integer matmul in captured MLIR")
            verified.append(row)
        candidates = [row for row in candidates if row["operation"] == "linalg.matmul"] + verified
    operations = candidates
    if len(operations) != 1:
        raise ValueError(f"source node {source_node_id!r} has {len(operations)} matrix bodies; expected one")
    op = operations[0]
    if len(op.get("operand_types", [])) != 3 or len(op.get("result_types", [])) != 1:
        raise ValueError("selected matrix body has an unsupported operand/result ABI")
    m, k, lhs_dtype = _matrix_type(op["operand_types"][0])
    wk, n, rhs_dtype = _matrix_type(op["operand_types"][1])
    om, on, out_dtype = _matrix_type(op["operand_types"][2])
    rm, rn, result_dtype = _matrix_type(op["result_types"][0])
    if (wk, om, on, rm, rn, out_dtype) != (k, m, n, m, n, result_dtype):
        raise ValueError("selected matrix body has inconsistent contraction geometry")
    remainder = k % tile_dim
    window = {
        "M": min(m, tile_dim),
        "K": min(k, tile_dim + (remainder if remainder else tile_dim)),
        "N": min(n, tile_dim),
    }
    metadata_path = capture / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    receipt_path = capture / "capture_receipt.json"
    return {
        "schema": "merlin.source-kernel-window.v2",
        "scope": "synthetic_matrix_body_from_captured_geometry",
        "source": {
            "model_mlir_sha256": model_sha,
            "frontend_trace_sha256": _sha(trace_path),
            "capture_receipt_sha256": _sha(receipt_path) if receipt_path.is_file() else None,
            "trace_status": trace.get("status"),
            "source_node_id": source_node_id,
            "source_target": nodes[0].get("target"),
            "module_stack": nodes[0].get("module_stack", {}),
            "mlir_operation": op["operation"],
            "mlir_ordinal": op.get("ordinal"),
            "operand_types": op["operand_types"],
            "result_types": op["result_types"],
            "origin_node_ids": op.get("origin_node_ids", []),
            "geometry": {"M": m, "K": k, "N": n},
            "dtypes": {"lhs": lhs_dtype, "rhs": rhs_dtype, "result": result_dtype},
            "capture_dtype": metadata.get("dtype"),
            "recipe_sha256": metadata.get("recipe_sha256"),
        },
        "projection": {
            "tile_dim": tile_dim,
            "geometry": window,
            "dtype": dict(zip(("lhs", "rhs", "result"), projection_types)),
            "rule": "M,N=min(parent,tile); K=min(parent,tile+(parent_K%tile or tile))",
        },
        "projected_type_body_observed": (lhs_dtype, rhs_dtype, result_dtype) == projection_types,
        "model_equivalence_claim": "none_synthetic_operands",
    }
