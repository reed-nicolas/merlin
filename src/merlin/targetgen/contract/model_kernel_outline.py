"""Outline exact integer contractions from a captured model as standalone interface kernels.

This is a model-to-kernel *slice*, not a whole-model lowering: producer SSA values
become explicit kernel arguments, and no host/device transfer or dispatch is implied.
The selected target's OOT compiler must still lower and execute each interface.
"""

from __future__ import annotations

import hashlib
from typing import Any

from merlin.common import mlir_query as mq
from merlin.targetgen.application_inventory import exact_int_mm_generic_operation
from merlin.targetgen.contract.interface_emit import emit_interface_mlir

SCHEMA = "merlin.model_kernel_outline.v1"


def _zero_initializer(value: Any) -> bool:
    """The interface MATMUL starts at zero; reject unknown accumulation inputs."""
    splat = getattr(value, "owner", None)
    if splat is None or mq.op_name(splat) != "tensor.splat" or len(splat.operands) != 1:
        return False
    constant = getattr(splat.operands[0], "owner", None)
    if constant is None or mq.op_name(constant) != "arith.constant":
        return False
    literal = constant.properties.get("value", constant.attributes.get("value"))
    scalar = getattr(getattr(literal, "value", None), "data", None)
    return type(scalar) is int and scalar == 0


def outline_integer_matmuls(model: bytes, *, target: str) -> dict:
    """Return exact-source kernel MLIR and typed SSA bindings for proven i8×i8→i32 ops.

    Only one known arithmetic form is accepted. All other operations remain in the
    model; this function never infers a placement or discharges a Phase 1 obligation.
    """
    if not target or not target.isidentifier():
        raise ValueError("target must be a nonempty identifier")
    digest = hashlib.sha256(model).hexdigest()
    module = mq.parse(model.decode("utf-8"))
    operations = list(mq.walk(module))
    identities = {id(op): f"mlir:{digest}:{ordinal}" for ordinal, op in enumerate(operations)}
    result_owners = {
        value: {
            "producer_operation_id": identities[id(op)],
            "result_index": index,
            "source_value_id": f"{identities[id(op)]}:result:{index}",
        }
        for op in operations
        for index, value in enumerate(op.results)
    }
    candidates, refused = [], []
    for ordinal, op in enumerate(operations):
        if mq.op_name(op) != "linalg.generic" or mq.attr_str(op, "prov.op") != "int_matmul":
            continue
        operation_id = identities[id(op)]
        if not exact_int_mm_generic_operation(op):
            refused.append({"operation_id": operation_id, "reason": "not the exact signed i8×i8→i32 reduction"})
            continue
        if mq.op_name(op.parent_op()) != "func.func":
            refused.append({"operation_id": operation_id, "reason": "contraction is nested in control flow"})
            continue
        if not _zero_initializer(op.operands[2]):
            refused.append({"operation_id": operation_id, "reason": "accumulator initializer is not proven zero"})
            continue
        left, right = (mq.type_shape_dtype(value.type)[0] for value in op.operands[:2])
        output = mq.type_shape_dtype(op.results[0].type)[0]
        if output != [left[0], right[1]]:
            refused.append({"operation_id": operation_id, "reason": "result shape disagrees with contraction"})
            continue
        command_buffer = {
            "abi_version": "0.1",
            "target": target,
            "tensors": {
                "A": {"shape": left, "dtype": "i8", "role": "input"},
                "W": {"shape": right, "dtype": "i8", "role": "weight"},
            },
            "commands": [
                {
                    "opcode": "RES_PACK",
                    "operands": {"src": "W", "dst": "W_res"},
                    "attributes": {"layout": "packed_rhs"},
                },
                {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "W_res", "dst": "acc"}},
                {
                    "opcode": "COMMIT",
                    "operands": {"src": "acc", "dst": "Y"},
                    "attributes": {"output_dtype": "i32", "epilogue": []},
                },
            ],
        }
        interface = emit_interface_mlir(command_buffer)

        def source(value: Any) -> dict:
            if value in result_owners:
                return result_owners[value]
            if hasattr(value, "index") and hasattr(value, "owner") and value.owner is op.parent:
                return {"source": "function_argument", "argument_index": value.index}
            return {"source": "unresolved"}

        bindings = [source(value) for value in op.operands[:2]]
        if any(binding.get("source") == "unresolved" for binding in bindings):
            refused.append(
                {"operation_id": operation_id, "reason": "operand source is not a direct SSA result or function argument"}
            )
            continue
        candidates.append(
            {
                "operation_id": operation_id,
                "ordinal": ordinal,
                "operand_bindings": bindings,
                "output_result_index": 0,
                "output_type": str(op.results[0].type),
                "interface_sha256": hashlib.sha256(interface.encode()).hexdigest(),
                "interface_mlir": interface,
            }
        )
    return {
        "schema": SCHEMA,
        "target": target,
        "model_sha256": digest,
        "candidates": candidates,
        "refused": refused,
        "qualification": "isolated exact integer kernels only; no target compilation, model stitching, or numerical proof",
    }
