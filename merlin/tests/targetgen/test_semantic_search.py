"""Search only selects exact typed patterns and feasible modeled allocations."""

from __future__ import annotations

import sys
from copy import deepcopy

from merlin.targetgen.semantic_search import SearchLimits, search, search_linalg_inventory


def _body(lhs: str = "arg:0", rhs: str = "arg:1") -> dict:
    return {
        "arguments": ["i32", "i32", "i32"],
        "captures": [],
        "operations": [
            {
                "op": "arith.addi",
                "operands": [lhs, rhs],
                "results": ["i32"],
                "attributes": {},
                "effects": [],
                "regions": 0,
            }
        ],
        "yields": ["op:0:0"],
        "effects": [],
    }


def _computation(body: dict | None = None) -> dict:
    return {
        "kind": "linalg.generic",
        "indexing_maps": ["map-a", "map-b", "map-out"],
        "iterator_types": ["parallel"],
        "scalar_body": _body() if body is None else body,
    }


def _kernel() -> dict:
    return {
        "schema": "merlin.semantic_kernel.v1",
        "values": [
            {"id": "a", "type": "tensor<1xi32>", "size_bytes": 4},
            {"id": "b", "type": "tensor<1xi32>", "size_bytes": 4},
            {"id": "init", "type": "tensor<1xi32>", "size_bytes": 4},
        ],
        "operations": [
            {
                "id": "add",
                "operation": "linalg.generic",
                "operands": ["a", "b", "init"],
                "results": [{"id": "r", "type": "tensor<1xi32>", "size_bytes": 4}],
                "indexing_maps": ["map-a", "map-b", "map-out"],
                "iterator_types": ["parallel"],
                "scalar_body": _body(),
                "effects": [],
            }
        ],
        "outputs": ["r"],
    }


def _instruction(identity: str, *, computation: dict | None = None, constraints: dict | None = None) -> dict:
    local = {} if constraints is None else constraints
    effects = [
        {
            "kind": kind,
            "space": space,
            field: name,
            "range": {"offset_bytes": 0, "length_bytes": 4},
        }
        for kind, field, key in (("read", "operand", "operand_spaces"), ("write", "result", "result_spaces"))
        for name, space in local.get(key, {}).items()
    ]
    return {
        "id": identity,
        "operands": [{"name": name, "type": "tensor<1xi32>"} for name in ("lhs", "rhs", "init")],
        "results": [{"name": "out", "type": "tensor<1xi32>"}],
        "computation": _computation() if computation is None else computation,
        "parameters": {},
        "effects": effects,
        "constraints": {} if constraints is None else constraints,
    }


def _model(*instructions: dict, spaces: dict | None = None) -> dict:
    return {
        "schema": "merlin.instruction_semantics.v1",
        "target": "test-target",
        "instructions": list(instructions),
        "memory_spaces": {} if spaces is None else spaces,
    }


def _space(capacity: int, *, banks: int = 1, width: int = 4) -> dict:
    return {"capacity_bytes": capacity, "alignment_bytes": 4, "banks": banks, "bank_width_bytes": width}


def test_exact_typed_match_and_integer_commutation_are_distinct() -> None:
    swapped = _computation(_body("arg:1", "arg:0"))
    selected = search(_kernel(), _model(_instruction("swapped", computation=swapped)))
    assert selected["status"] == "selected"
    assert selected["instruction_graph"] == [
        {"operation_id": "add", "instruction_id": "swapped", "rewrites": ["commute_integer_scalar:0"]}
    ]
    bad = deepcopy(_kernel())
    bad["operations"][0]["scalar_body"]["operations"][0]["results"] = ["f32"]
    assert search(bad, _model(_instruction("swapped", computation=swapped)))["status"] == "unsupported"


def test_modular_integer_add_reassociates_only_without_overflow_flags() -> None:
    kernel = _kernel()
    kernel["values"].append({"id": "c", "type": "tensor<1xi32>", "size_bytes": 4})
    operation = kernel["operations"][0]
    operation["operands"] = ["a", "b", "c", "init"]
    body = _body()
    body["arguments"] = ["i32"] * 4
    body["operations"].append(
        {
            "op": "arith.addi",
            "operands": ["op:0:0", "arg:2"],
            "results": ["i32"],
            "attributes": {},
            "effects": [],
            "regions": 0,
        }
    )
    body["yields"] = ["op:1:0"]
    operation["scalar_body"] = body
    target_body = deepcopy(body)
    target_body["operations"][0]["operands"] = ["arg:1", "arg:2"]
    target_body["operations"][1]["operands"] = ["arg:0", "op:0:0"]
    instruction = _instruction("reassociated", computation=_computation(target_body))
    instruction["operands"].insert(2, {"name": "third", "type": "tensor<1xi32>"})
    selected = search(kernel, _model(instruction))
    assert selected["status"] == "selected"
    assert "reassociate_modular_integer_add:0:1" in selected["instruction_graph"][0]["rewrites"]
    kernel["operations"][0]["scalar_body"]["operations"][0]["attributes"] = {"overflowFlags": "nsw"}
    assert search(kernel, _model(instruction))["status"] == "unsupported"


def test_floating_point_addition_is_not_rewritten() -> None:
    kernel = _kernel()
    for value in kernel["values"] + kernel["operations"][0]["results"]:
        value["type"] = "tensor<1xf32>"
    body = kernel["operations"][0]["scalar_body"]
    body["arguments"] = ["f32"] * 3
    body["operations"][0]["op"] = "arith.addf"
    body["operations"][0]["results"] = ["f32"]
    swapped = deepcopy(body)
    swapped["operations"][0]["operands"] = ["arg:1", "arg:0"]
    instruction = _instruction("swapped-float", computation=_computation(swapped))
    for value in instruction["operands"] + instruction["results"]:
        value["type"] = "tensor<1xf32>"
    assert search(kernel, _model(instruction))["status"] == "unsupported"


def test_allocator_retries_an_equivalent_instruction_graph() -> None:
    impossible = _instruction(
        "a-tight", constraints={"operand_spaces": {"lhs": "tiny"}, "result_spaces": {"out": "tiny"}}
    )
    feasible = _instruction("b-wide", constraints={"operand_spaces": {"lhs": "wide"}, "result_spaces": {"out": "wide"}})
    result = search(_kernel(), _model(impossible, feasible, spaces={"tiny": _space(4), "wide": _space(16)}))
    assert result["status"] == "selected"
    assert [attempt["allocation_status"] for attempt in result["attempts"]] == ["unsat", "sat"]
    assert result["instruction_graph"][0]["instruction_id"] == "b-wide"
    allocated = result["allocation"]["addresses"]
    assert allocated["a"]["address_bytes"] == 0
    assert allocated["r"]["address_bytes"] == 4


def test_unknown_first_candidate_does_not_hide_a_later_feasible_graph() -> None:
    unresolved = _instruction("a-unresolved", constraints={"result_spaces": {"out": "missing"}})
    feasible = _instruction("b-external")
    result = search(_kernel(), _model(unresolved, feasible))
    assert result["status"] == "selected"
    assert [attempt["allocation_status"] for attempt in result["attempts"]] == ["unknown", "sat"]


def test_bank_constraint_changes_deterministic_addresses() -> None:
    instruction = _instruction(
        "banked", constraints={"operand_spaces": {"lhs": "scratch", "rhs": "scratch"}, "distinct_banks": ["lhs", "rhs"]}
    )
    result = search(_kernel(), _model(instruction, spaces={"scratch": _space(16, banks=2)}))
    assert result["status"] == "selected"
    addresses = result["allocation"]["addresses"]
    assert addresses["a"]["address_bytes"] == 0
    assert addresses["b"]["address_bytes"] == 4
    assert (
        search(_kernel(), _model(instruction, spaces={"scratch": _space(16, banks=2)}))["allocation"]
        == result["allocation"]
    )


def test_unknown_memory_and_bounded_refusals_remain_visible() -> None:
    instruction = _instruction("local", constraints={"result_spaces": {"out": "scratch"}})
    assert search(_kernel(), _model(instruction))["status"] == "unknown"
    assert (
        search(_kernel(), _model(instruction, spaces={"scratch": {"capacity_bytes": "UNKNOWN"}}))["status"] == "unknown"
    )
    assert (
        search(_kernel(), _model(_instruction("plain")), limits=SearchLimits(max_candidates=0))["status"] == "unknown"
    )
    unknown = deepcopy(_kernel())
    unknown["operations"][0]["scalar_body"]["effects"] = None
    assert search(unknown, _model(_instruction("plain")))["status"] == "unknown"
    unknown_model = _model(_instruction("plain"))
    unknown_model["instructions"][0]["status"] = "UNKNOWN"
    assert search(_kernel(), unknown_model)["status"] == "unknown"
    uncovered = _instruction("uncovered", constraints={"result_spaces": {"out": "scratch"}})
    uncovered["effects"] = []
    assert search(_kernel(), _model(uncovered, spaces={"scratch": _space(16)}))["status"] == "unknown"


def test_optional_solver_unavailable_is_reported(monkeypatch) -> None:
    local = _instruction("local", constraints={"result_spaces": {"out": "scratch"}})
    monkeypatch.setitem(sys.modules, "z3", None)
    result = search(_kernel(), _model(local, spaces={"scratch": _space(16)}))
    assert result["status"] == "unavailable"


def test_liveness_allows_reuse_only_after_last_use() -> None:
    kernel = _kernel()
    second = deepcopy(kernel["operations"][0])
    second["id"] = "add-again"
    second["operands"][0] = "r"
    second["results"][0]["id"] = "r2"
    kernel["operations"].append(second)
    kernel["outputs"] = ["r2"]
    instruction = _instruction(
        "local", constraints={"operand_spaces": {"lhs": "scratch"}, "result_spaces": {"out": "scratch"}}
    )
    result = search(kernel, _model(instruction, spaces={"scratch": _space(8)}))
    assert result["status"] == "selected"
    assert second["operands"][0] == kernel["operations"][0]["results"][0]["id"]
    assert [node["operation_id"] for node in result["instruction_graph"]] == ["add", "add-again"]
    addresses = result["allocation"]["addresses"]
    assert addresses["a"]["address_bytes"] == addresses["r2"]["address_bytes"]
    assert addresses["r"]["address_bytes"] != addresses["r2"]["address_bytes"]


def test_parsed_inventory_surfaces_per_operation_host_gaps() -> None:
    row = {
        "id": 7,
        "kind": "linalg.generic",
        "operation": "linalg.generic",
        "ins": [
            {"source": ("arg", 0), "shape": [1], "dtype": "i32"},
            {"source": ("arg", 1), "shape": [1], "dtype": "i32"},
        ],
        "outs": [{"source": ("init", "empty"), "shape": [1], "dtype": "i32"}],
        "results": [{"shape": [1], "dtype": "i32"}],
        "indexing_maps": ["map-a", "map-b", "map-out"],
        "iterator_types": ["parallel"],
        "scalar_body": _body(),
        "source_provenance": {"source_node_ids": ["node-1"]},
    }
    unsupported = deepcopy(row)
    unsupported["id"] = 8
    unsupported["operation"] = unsupported["kind"] = "tensor.expand_shape"
    malformed = deepcopy(row)
    malformed["id"] = 9
    malformed["ins"][0]["shape"] = ["?"]
    result = search_linalg_inventory(
        {"entry": "forward", "ops": [row, unsupported, malformed]}, _model(_instruction("add"))
    )
    assert [region["placement"] for region in result["regions"]] == [
        "device_candidate",
        "unresolved",
        "unresolved",
    ]
    assert result["regions"][2]["receipt"]["status"] == "refused"
    assert all(region["host_admission"] == "not_evaluated" for region in result["regions"])
    assert result["regions"][0]["source_provenance"]["source_node_ids"] == ["node-1"]


def test_real_mlir_parser_to_search_receipt() -> None:
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
    from merlin.targetgen.instruction_semantics import normalize_instruction_semantics

    mlir = """
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
    parsed = parse_linalg_mlir(mlir)
    row = parsed["ops"][0]
    computation = {
        "kind": "linalg.generic",
        "indexing_maps": row["indexing_maps"],
        "iterator_types": row["iterator_types"],
        "scalar_body": row["scalar_body"],
    }
    description = _model(_instruction("selected", computation=computation))
    description["instructions"][0]["software_operation"] = "add"
    selected_model = normalize_instruction_semantics(
        description,
        software_spec={
            "schema": "merlin.software_spec.v1",
            "target": "test-target",
            "status": "reviewed",
            "operations": [{"id": "add", "placement": "accelerator", "signature": {"dtypes": ["i32"], "ranks": [1]}}],
        },
        rtl_facts={"facts": {}},
        target="test-target",
    )
    assert selected_model["status"] == "described", selected_model["unknowns"]
    result = search_linalg_inventory(parsed, selected_model)
    assert result["regions"][0]["receipt"]["status"] == "selected", result
    tampered = deepcopy(selected_model)
    tampered["target"] = "other-target"
    assert search_linalg_inventory(parsed, tampered)["regions"][0]["receipt"]["status"] == "refused"
