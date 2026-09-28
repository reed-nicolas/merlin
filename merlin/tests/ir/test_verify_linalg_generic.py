"""The source verifier reads a generic contraction's body, not its provenance tags."""

from __future__ import annotations

import hashlib

import pytest
from xdsl.parser import Parser

from merlin.targetgen.contract.linalg_iface import make_linalg_context
from merlin.verify import HAS_XDSL, HAS_Z3
from merlin.verify.receipts import verify_transformation
from merlin.verify.smt_semantics import UnsupportedSemantics
from merlin.verify.tools import find_mlir_tool


# Exact input and output captured from the diagnostic OOT trial. Keep both in this test so the
# verifier can be qualified without importing or shipping the unqualified trial compiler.
_SOURCE_TEXT = """// Diagnostic-only Linalg input for the selected OOT compiler's second grammar.
// A and W are signed int8; the zero-filled accumulator makes the result A @ W.
builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%A: tensor<16x16xi8>, %W: tensor<16x16xi8>) -> tensor<16x16xi32> {
    %empty = tensor.empty() : tensor<16x16xi32>
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A, %W : tensor<16x16xi8>, tensor<16x16xi8>) outs(%init : tensor<16x16xi32>) attrs = {prov.region_id = "single_matmul", prov.op = "matmul", prov.family = "contraction"} {
    ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
      %lhs_wide = arith.extsi %lhs : i8 to i32
      %rhs_wide = arith.extsi %rhs : i8 to i32
      %product = arith.muli %lhs_wide, %rhs_wide : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
    } -> tensor<16x16xi32>
    func.return %result : tensor<16x16xi32>
  }
}
"""
_OOT_INTERFACE_TEXT = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %arg1 = merlin_iface.tensor {name = "arg1", role = "weight"} : tensor<16x16xi8>
  %arg0 = merlin_iface.tensor {name = "arg0", role = "input"} : tensor<16x16xi8>
  %Wp = merlin_iface.resident_pack %arg1 {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc = merlin_iface.matmul %arg0, %Wp : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %out = merlin_iface.commit %acc {name = "out", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %Wp : (!merlin_iface.resident) -> ()
}
"""
_OOT_INTERFACE_SHA256 = "e772ca1071a5f4558088f66e04c4427319841cd31d8a6adea41b674282a096a5"
_TRANSLATOR = find_mlir_tool("mlir-translate")


def _source(*, shape: int = 2, changes: tuple[tuple[str, str], ...] = ()):
    text = _SOURCE_TEXT
    if shape != 16:
        text = text.replace("16x16", f"{shape}x{shape}")
    for old, new in changes:
        assert old in text
        text = text.replace(old, new, 1)
    module = Parser(make_linalg_context(), text).parse_module()
    module.verify()
    return module


def _target(shape: int = 2):
    from merlin.xdsl_dialects.lowering.pipeline import lower_repeated_rhs_matmul

    # An independently generated interface program with the same function. The actual OOT
    # merlin_iface output is not an in-tree interface dialect module; this is an encoder check.
    return lower_repeated_rhs_matmul(reuse=1, m=shape, k=shape, n=shape).interface_module


def _oot_text(shape: int = 2):
    assert hashlib.sha256(_OOT_INTERFACE_TEXT.encode("utf-8")).hexdigest() == _OOT_INTERFACE_SHA256
    return _OOT_INTERFACE_TEXT.replace("16x16", f"{shape}x{shape}") if shape != 16 else _OOT_INTERFACE_TEXT


@pytest.mark.skipif(not (HAS_XDSL and HAS_Z3 and _TRANSLATOR), reason="needs xdsl, z3 and mlir-translate")
def test_signed_generic_matmul_matches_independent_interface():
    receipt = verify_transformation(
        "linalg_to_interface", _source(), _target(), translator=_TRANSLATOR, timeout_ms=60_000
    )
    assert receipt.status == "verified", (receipt.status, receipt.reason)


@pytest.mark.skipif(not (HAS_XDSL and HAS_Z3 and _TRANSLATOR), reason="needs xdsl, z3 and mlir-translate")
@pytest.mark.parametrize(
    "changes,expected",
    [
        ((("arith.constant 0 : i32", "arith.constant 1 : i32"),), "refuted"),
        ((("%sum = arith.addi %acc, %product", "%sum = arith.addi %product, %product"),), "unsupported"),
        ((("%lhs_wide = arith.extsi", "%lhs_wide = arith.extui"),), "unsupported"),
        ((("(d0, d2)", "(d2, d0)"),), "unsupported"),
        ((("\"parallel\", \"parallel\", \"reduction\"", "\"parallel\", \"reduction\", \"parallel\""),), "unsupported"),
    ],
)
def test_generic_mutations_cannot_receive_false_proof(changes, expected):
    receipt = verify_transformation(
        "linalg_to_interface", _source(changes=changes), _target(), translator=_TRANSLATOR, timeout_ms=60_000
    )
    assert receipt.status == expected, (receipt.status, receipt.reason)


def test_provenance_does_not_drive_generic_semantics():
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block

    from merlin.verify.linalg_semantics import encode_linalg
    from merlin.verify.smt_semantics import Encoder

    source = _source(changes=(("prov.family = \"contraction\"", "prov.family = \"elementwise_map\""),))
    with ImplicitBuilder(Block()):
        assert encode_linalg(Encoder(), source).outputs
    wrong = _source(changes=(("%product = arith.muli", "%product = arith.addi"),))
    with ImplicitBuilder(Block()):
        with pytest.raises(UnsupportedSemantics, match="generic"):
            encode_linalg(Encoder(), wrong)


@pytest.mark.skipif(not (HAS_XDSL and HAS_Z3 and _TRANSLATOR), reason="needs xdsl, z3 and mlir-translate")
def test_captured_oot_interface_text_has_replayable_16x16_receipt():
    from merlin.verify.receipts import qualify_receipt

    source = _source(shape=16)
    target = _oot_text(shape=16)
    receipt = verify_transformation(
        "linalg_to_interface", source, target, translator=_TRANSLATOR, timeout_ms=60_000
    )
    assert receipt.status == "verified", (receipt.status, receipt.reason)
    assert receipt.target == {
        "encoding": "merlin_iface_text_utf8",
        "sha256": _OOT_INTERFACE_SHA256,
    }
    assert receipt.typed_signatures["target"]["outputs"] == [
        {"name": "out", "type": "tensor<16x16xi32>"}
    ]
    assert qualify_receipt(receipt, source, target, translator=_TRANSLATOR)
    assert not qualify_receipt(receipt, source, target.replace("name = \"out\"", "name = \"other\""), translator=_TRANSLATOR)


@pytest.mark.skipif(not (HAS_XDSL and HAS_Z3 and _TRANSLATOR), reason="needs xdsl, z3 and mlir-translate")
@pytest.mark.parametrize(
    "old,new,status",
    [
        ("merlin_iface.matmul %arg0, %Wp", "merlin_iface.matmul %arg1, %Wp", "refuted"),
        ("name = \"arg1\"", "name = \"other\"", "unsupported"),
        ("merlin_iface.matmul %arg0, %Wp :", "merlin_iface.matmul %arg0, %Wp {transpose = true} :", "unsupported"),
        ("merlin_iface.evict", "merlin_iface.unmodelled", "unsupported"),
        ("epilogue = []", "epilogue = [\"relu\"]", "unsupported"),
        (
            "  merlin_iface.evict %Wp",
            "  %other = merlin_iface.commit %acc {name = \"other\", epilogue = [], "
            "output_dtype = \"i32\"} : (!merlin_iface.acc<i32>) -> tensor<2x2xi32>\n"
            "  merlin_iface.evict %Wp",
            "unsupported",
        ),
    ],
)
def test_captured_oot_interface_mutations_never_receive_false_proof(old, new, status):
    source = _source()
    text = _oot_text()
    assert old in text
    changed = text.replace(old, new, 1)
    receipt = verify_transformation(
        "linalg_to_interface", source, changed, translator=_TRANSLATOR, timeout_ms=60_000
    )
    assert receipt.status == status, (receipt.status, receipt.reason)
