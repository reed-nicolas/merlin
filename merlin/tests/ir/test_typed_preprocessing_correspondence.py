"""A source-transform map can support a narrow, hash-bound type check."""

import hashlib
import json

import pytest

from merlin.common.ir_audit import IrAudit
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.llvmlower.typed_preprocessing_correspondence import check_result_types


SOURCE = """module {
  func.func @forward(%x: f32) -> tensor<2x2xf32> {
    %c0 = arith.constant 0.0 : f32
    %t = tensor.splat %c0 : tensor<2x2xf32>
    func.return %t : tensor<2x2xf32>
  }
}"""


def _audit(tmp_path):
    with IrAudit(tmp_path, enabled="compact", producer="typed-correspondence-test", source=__file__) as audit:
        preprocessed, _ = preprocess_text(SOURCE, audit=audit)
    return preprocessed, audit.directory / "index.json"


def test_checked_result_types_are_explicitly_short_of_value_or_terminal_proof(tmp_path):
    preprocessed, index = _audit(tmp_path)
    result = check_result_types(SOURCE.encode(), preprocessed.encode(), index)
    assert result["source_operation_count"] == 2
    assert result["checked_result_count"] == 2
    assert result["claim"] == "preprocessing_result_types_only"
    assert result["value_preservation"] == "not_checked"
    assert result["terminal_lowering"] == "not_checked"


def test_checked_result_types_refuse_wrong_source_and_modified_receipt(tmp_path):
    preprocessed, index = _audit(tmp_path)
    with pytest.raises(ValueError, match="source digest"):
        check_result_types(SOURCE.replace("2x2", "3x2").encode(), preprocessed.encode(), index)

    map_path = index.parent / "source-transform-map.json"
    map_path.write_bytes(map_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt digest"):
        check_result_types(SOURCE.encode(), preprocessed.encode(), index)


def test_checked_result_types_reparse_and_reject_a_hash_consistent_wrong_mapping(tmp_path):
    preprocessed, index_path = _audit(tmp_path)
    index = json.loads(index_path.read_text())
    map_path = index_path.parent / "source-transform-map.json"
    receipt = json.loads(map_path.read_text())
    first, second = receipt["operations"]
    first["preprocessed_op_indices"], second["preprocessed_op_indices"] = (
        second["preprocessed_op_indices"], first["preprocessed_op_indices"]
    )
    first["result_map"][0]["preprocessed_op_index"] = first["preprocessed_op_indices"][0]
    second["result_map"][0]["preprocessed_op_index"] = second["preprocessed_op_indices"][0]
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    map_path.write_bytes(raw)
    index["accounting_receipts"][0]["sha256"] = digest
    index["source_transform_map"]["sha256"] = digest
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="result type"):
        check_result_types(SOURCE.encode(), preprocessed.encode(), index_path)
