"""A selected scope obligation becomes a real, priced linalg-on-tensors program."""

from __future__ import annotations

from merlin_experiments.phase2.feedback_metrics import declared_capsule_macs, declared_reduction_depths
from xdsl.parser import Parser

from merlin.common import mlir_query
from merlin.perf.work_volume import work_from_command_buffer
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import phase_policy, scope_census
from merlin.targetgen.contract.linalg_iface import make_linalg_context


def _binding(dtype: str = "int8", accumulator: str = "i32") -> CS.CorpusBinding:
    return CS.CorpusBinding(
        target="fixture", tile_dim=4, operand_dtype=dtype, accum_dtype=accumulator,
        integer=True, tiers=["L0", "L1", "L2", "L3"], compare="exact_int",
        classes_for=lambda **_: ["CONTRACTION"],
    )


def _entry(map_count: int = 3) -> dict:
    return {
        "name": "scope_fixture", "kind": "model_slice", "source_role": "derived_sweep",
        "source_reference": "selected requirement fixture", "label": "dev", "op": "scope_chain",
        "M": 4, "K": 8, "N": 4,
        "scope_families": ["movement", "contraction"] + ["elementwise_map"] * map_count,
        "semantic": {"semantic_family": "contraction", "generalization_axis": "composition", "must_accelerate": True},
    }


def test_selected_exact_chain_is_real_verified_linalg_and_golden() -> None:
    capsule, mlir = CS.build(_entry(), _binding())
    Parser(make_linalg_context(), mlir).parse_module().verify()
    chain = scope_census.chains(mlir_query.parse(mlir))
    assert [row.signature for row in chain] == [" -> ".join(_entry()["scope_families"])]
    assert chain[0].length == 5
    env = CG.materialize_capsule_leaves(capsule)
    lhs, weight = env["A0"], env["W"]
    expected = [
        [sum(lhs.data[m * 8 + k] * weight.data[n * 8 + k] for k in range(8)) + 3
         for n in range(4)] for m in range(4)
    ]
    assert CG.golden(capsule) == {"Y0": expected}


def test_declared_price_equals_one_emitted_contraction_not_zero_mac_maps() -> None:
    capsule, _ = CS.build(_entry(), _binding())
    macs, _ = declared_capsule_macs(capsule)
    assert macs == 4 * 8 * 4
    assert phase_policy.declared_macs(capsule)[0] == macs
    assert declared_reduction_depths(capsule)[0] == (8,)
    command_buffer = {
        "tensors": {
            "A0": {"shape": [4, 8]}, "W": {"shape": [4, 8]},
            "WT": {"shape": [8, 4]}, "Y0": {"shape": [4, 4]},
        },
        "commands": [
            {"opcode": "MOVEMENT", "operands": {"src": "W", "dst": "WT"}},
            {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "WT", "dst": "Y0"}},
            *({"opcode": "VECTOR_MAP", "operands": {"lhs": "Y0", "dst": "Y0"}}
              for _ in range(capsule["operation"]["attributes"]["map_count"])),
        ],
    }
    work = work_from_command_buffer(command_buffer)
    assert not work.is_lower_bound
    assert work.exact_macs == macs


def test_map_count_is_selected_not_fixed_in_builder() -> None:
    capsule, mlir = CS.build(_entry(1), _binding())
    Parser(make_linalg_context(), mlir).parse_module().verify()
    assert [row.signature for row in scope_census.chains(mlir_query.parse(mlir))] == [
        "movement -> contraction -> elementwise_map"
    ]
    assert capsule["operation"]["attributes"]["map_count"] == 1
