"""The installed command joins actual MLIR parsing and a selected instruction model."""

from __future__ import annotations

import builtins
import hashlib
import json

import pytest

from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.instruction_semantics import normalize_instruction_semantics
from merlin.targetgen.tool_cli import build_parser, main

_MLIR = """\
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


def test_other_operator_commands_do_not_import_host_private_search(monkeypatch):
    real_import = builtins.__import__

    def without_search(name, *args, **kwargs):
        if name == "semantic_search" or name.startswith("merlin.targetgen.semantic_search"):
            raise ImportError("host-private search implementation is masked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_search)
    args = ["fixed-boot", "--target", "fixture", "--source", "input.S", "--out", "x.o", "--clang", "clang"]
    assert build_parser().parse_args(args).command == "fixed-boot"


def test_cli_writes_byte_bound_diagnostic_receipt(tmp_path):
    mlir = tmp_path / "kernel.mlir"
    mlir.write_text(_MLIR)
    row = parse_linalg_mlir(_MLIR)["ops"][0]
    authored = {
        "schema": "merlin.instruction_semantics.v1",
        "target": "fixture",
        "memory_spaces": {},
        "instructions": [
            {
                "id": "int-add",
                "operands": [{"name": name, "type": "tensor<1xi32>"} for name in ("a", "b", "init")],
                "results": [{"name": "result", "type": "tensor<1xi32>"}],
                "software_operation": "int-add",
                "computation": {
                    "kind": "linalg.generic",
                    "indexing_maps": row["indexing_maps"],
                    "iterator_types": row["iterator_types"],
                    "scalar_body": row["scalar_body"],
                },
                "parameters": {},
                "constraints": {},
                "effects": [],
            }
        ],
    }
    model = normalize_instruction_semantics(
        authored,
        software_spec={
            "schema": "merlin.software_spec.v1",
            "target": "fixture",
            "status": "reviewed",
            "operations": [
                {
                    "id": "int-add",
                    "placement": "accelerator",
                    "signature": {
                        "ordered_operand_dtypes": ["i32", "i32", "i32"],
                        "ordered_result_dtypes": ["i32"],
                        "ranks": [1],
                    },
                }
            ],
        },
        rtl_facts={"facts": {}},
        target="fixture",
    )
    selected = tmp_path / "instructions.json"
    selected.write_text(json.dumps(model))
    output = tmp_path / "receipt.json"
    assert (
        main(
            [
                "semantic-search",
                "--target",
                "fixture",
                "--mlir",
                str(mlir),
                "--instruction-model",
                str(selected),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    receipt = json.loads(output.read_text())
    assert receipt["inputs"]["mlir_sha256"] == hashlib.sha256(mlir.read_bytes()).hexdigest()
    assert receipt["inputs"]["instruction_model_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert receipt["result"]["regions"][0]["receipt"]["status"] == "selected"
    assert receipt["result"]["regions"][0]["host_admission"] == "not_evaluated"
    assert receipt["result"]["summary"] == {
        "operations": 1,
        "device_candidates": 1,
        "unresolved": 0,
        "by_operation": {"linalg.generic": {"selected": 1}},
    }
    with pytest.raises(FileExistsError):
        main(
            [
                "semantic-search",
                "--target",
                "fixture",
                "--mlir",
                str(mlir),
                "--instruction-model",
                str(selected),
                "--out",
                str(output),
            ]
        )
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps({**model, "status": "UNKNOWN"}))
    with pytest.raises(ValueError, match="canonical_sha256"):
        main(
            [
                "semantic-search",
                "--target",
                "fixture",
                "--mlir",
                str(mlir),
                "--instruction-model",
                str(forged),
                "--out",
                str(tmp_path / "forged-receipt.json"),
            ]
        )

    stub = tmp_path / "unknown.json"
    stub.write_text(
        json.dumps(
            {
                "schema": "merlin.instruction_semantics.v1",
                "target": "fixture",
                "status": "UNKNOWN",
                "unknowns": ["no_selected_resource"],
                "instructions": [],
            }
        )
    )
    unknown_output = tmp_path / "unknown-receipt.json"
    assert (
        main(
            [
                "semantic-search",
                "--target",
                "fixture",
                "--mlir",
                str(mlir),
                "--instruction-model",
                str(stub),
                "--out",
                str(unknown_output),
            ]
        )
        == 0
    )
    assert json.loads(unknown_output.read_text())["result"]["regions"][0]["receipt"]["status"] == "unknown"
