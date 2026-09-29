"""A projected kernel must remain tied to one exact captured MLIR body."""

from __future__ import annotations

import hashlib
import json

import pytest

from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.contract.model_kernel_outline import outline_integer_matmuls
from merlin.targetgen.source_kernel_probe import derive_kernel_window


_INTEGER_BODY = """builtin.module {
  func.func @forward(%a: tensor<4x19xi8>, %b: tensor<19x8xi8>) -> tensor<4x8xi32> {
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %init = "tensor.splat"(%zero) : (i32) -> tensor<4x8xi32>
    %result = "linalg.generic"(%a, %b, %init) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
      %lhs32 = "arith.extsi"(%lhs) : (i8) -> i32
      %rhs32 = "arith.extsi"(%rhs) : (i8) -> i32
      %product = "arith.muli"(%lhs32, %rhs32) : (i32, i32) -> i32
      %sum = "arith.addi"(%acc, %product) : (i32, i32) -> i32
      "linalg.yield"(%sum) : (i32) -> ()
    }) {prov.op = "int_matmul", prov.source_node_ids = ["g:prepared:root:n7"]} : (tensor<4x19xi8>, tensor<19x8xi8>, tensor<4x8xi32>) -> tensor<4x8xi32>
    func.return %result : tensor<4x8xi32>
  }
}"""


def _integer_capture(tmp_path, *, body=_INTEGER_BODY):
    from merlin.common import mlir_query as query

    model = body.encode()
    (tmp_path / "model.mlir").write_bytes(model)
    module = query.parse(body)
    matches = [(index, op) for index, op in enumerate(query.walk(module))
               if query.op_name(op) == "linalg.generic"]
    assert len(matches) == 1
    ordinal, op = matches[0]
    trace = {
        "status": "complete",
        "graphs": {"prepared": {"nodes": [{"id": "g:prepared:root:n7", "target": "aten._int_mm.default"}]}},
        "mlir": {"sha256": hashlib.sha256(model).hexdigest(), "operations": [{
            "ordinal": ordinal, "operation": "linalg.generic",
            "source_node_ids": ["g:prepared:root:n7"],
            "origin_node_ids": ["g:original:root:n6"],
            "operand_types": [str(value.type) for value in op.operands],
            "result_types": [str(value.type) for value in op.results],
        }]},
    }
    (tmp_path / "frontend-trace.json").write_text(json.dumps(trace), encoding="utf-8")
    return tmp_path


def test_integerized_source_matrix_body_can_supply_a_bounded_window(tmp_path):
    capture = _integer_capture(tmp_path)
    projected = derive_kernel_window(
        capture, "g:prepared:root:n7", tile_dim=16, projection_types=("i8", "i8", "i32")
    )
    assert projected["source"]["mlir_operation"] == "linalg.generic"
    assert projected["source"]["geometry"] == {"M": 4, "K": 19, "N": 8}
    assert projected["projection"]["geometry"] == {"M": 4, "K": 19, "N": 8}
    assert projected["projected_type_body_observed"] is True


def test_exact_integer_model_body_outlines_a_compilable_interface_kernel(tmp_path):
    from merlin.targetgen.tool_cli import main

    model = _INTEGER_BODY.encode()
    (tmp_path / "model.mlir").write_bytes(model)
    output = tmp_path / "kernels"
    assert main(
        ["outline-int-mm", "--target", "example", "--mlir", str(tmp_path / "model.mlir"), "--out", str(output)]
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model_sha256"] == hashlib.sha256(model).hexdigest()
    assert len(manifest["candidates"]) == 1 and manifest["refused"] == []
    candidate = manifest["candidates"][0]
    assert candidate["operation_id"].startswith(f"mlir:{manifest['model_sha256']}:")
    assert candidate["operand_bindings"] == [
        {"source": "function_argument", "argument_index": 0},
        {"source": "function_argument", "argument_index": 1},
    ]
    interface = (output / candidate["interface_file"]).read_text()
    assert hashlib.sha256(interface.encode()).hexdigest() == candidate["interface_sha256"]
    parsed = parse_interface_mlir(interface)
    assert parsed["tensors"]["B"]["role"] == "input"
    commands = parsed["commands"]
    assert [command["opcode"] for command in commands] == ["RES_PACK", "MATMUL_RESIDENT", "COMMIT"]
    assert commands[-1]["attributes"]["output_dtype"] == "i32"

    # Neither same-typed wrong wiring nor a nonzero seed is a standalone matmul.
    wrong_product = _INTEGER_BODY.replace('"arith.muli"(%lhs32, %rhs32)', '"arith.muli"(%lhs32, %lhs32)')
    assert outline_integer_matmuls(wrong_product.encode(), target="example")["candidates"] == []
    nonzero_init = _INTEGER_BODY.replace('value = 0 : i32', 'value = 1 : i32')
    assert outline_integer_matmuls(nonzero_init.encode(), target="example")["candidates"] == []


@pytest.mark.parametrize("changed", [
    _INTEGER_BODY.replace('prov.op = "int_matmul"', 'prov.op = "elementwise"'),
    _INTEGER_BODY.replace('"arith.addi"(%acc, %product)', '"arith.subi"(%acc, %product)'),
])
def test_generic_without_exact_integer_matmul_structure_is_not_a_source_window(tmp_path, changed):
    capture = _integer_capture(tmp_path, body=changed)
    with pytest.raises(ValueError, match="integer matmul|matrix body"):
        derive_kernel_window(capture, "g:prepared:root:n7", tile_dim=16, projection_types=("i8", "i8", "i32"))


def _capture(tmp_path, *, parent_k=147):
    model = b"module { one selected operation }\n"
    (tmp_path / "model.mlir").write_bytes(model)
    trace = {
        "status": "diagnostic",
        "graphs": {"prepared": {"nodes": [{"id": "g:prepared:root:n7", "target": "aten.convolution.default"}]}},
        "mlir": {
            "sha256": hashlib.sha256(model).hexdigest(),
            "operations": [
                {
                    "ordinal": 29,
                    "operation": "linalg.matmul",
                    "source_node_ids": ["g:prepared:root:n7"],
                    "origin_node_ids": ["g:original:root:n6"],
                    "operand_types": [
                        f"tensor<64x{parent_k}xf32>",
                        f"tensor<{parent_k}x12544xf32>",
                        "tensor<64x12544xf32>",
                    ],
                    "result_types": ["tensor<64x12544xf32>"],
                }
            ],
        },
    }
    (tmp_path / "frontend-trace.json").write_text(json.dumps(trace), encoding="utf-8")
    return tmp_path


def test_source_geometry_and_k_tail_survive_bounded_integer_projection(tmp_path):
    capture = _capture(tmp_path)
    projected = derive_kernel_window(
        capture, "g:prepared:root:n7", tile_dim=16, projection_types=("i8", "i8", "i32")
    )
    assert projected["source"]["geometry"] == {"M": 64, "K": 147, "N": 12544}
    assert projected["source"]["operand_types"][0] == "tensor<64x147xf32>"
    assert projected["projection"]["geometry"] == {"M": 16, "K": 19, "N": 16}
    assert projected["projected_type_body_observed"] is False
    assert projected["model_equivalence_claim"] == "none_synthetic_operands"


def test_changed_model_bytes_refuse_stale_trace(tmp_path):
    capture = _capture(tmp_path)
    (capture / "model.mlir").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        derive_kernel_window(capture, "g:prepared:root:n7", tile_dim=16, projection_types=("i8", "i8", "i32"))


def test_projection_precision_is_selected_not_built_into_shared_geometry(tmp_path):
    capture = _capture(tmp_path)
    projected = derive_kernel_window(
        capture, "g:prepared:root:n7", tile_dim=16, projection_types=("f32", "f32", "f32")
    )
    assert projected["projection"]["dtype"] == {"lhs": "f32", "rhs": "f32", "result": "f32"}
    assert projected["projected_type_body_observed"] is True
    assert projected["model_equivalence_claim"] == "none_synthetic_operands"
