"""RTL pair facts retain operation-timing provenance and checked capacity evidence."""

import copy
import hashlib
import os
from pathlib import Path

import pytest

from merlin.common.facts_view import op_pairs
from merlin.targetgen.rtl import op_pairs as OP
from merlin.targetgen.rtl.extraction_contract import op_pairs_source
from test_op_timing_facts import (
    C, DECLARATION, DIGEST, OT, artifact, declare, document as timing_document, write,
    test_declaration_normalized_and_absent, test_dump_merges_repeats_and_refuses_downgrade,
    test_in_tree_target_declares_its_source, test_malformed_declaration_fails_closed,
    test_merge_preserves_input_and_replaces_whole_section, test_read_document_reports_unreadable_and_non_json,
    test_valid_document_passes_through_unchanged,
)

PAIR = {
    "first": "scalar.add", "second": "scalar.sub", "engine": "scalar", "relation": {"destination": "different"},
    "gaps": [1, 12], "min_gap": 4, "stable_gap": 4, "assertions": {"checked": 24, "excluded": 2},
    "source": "scalar.add_sub",
}


def document(timing_path=None):
    return {
        "schema": OP.SCHEMA, "hw_ir": {"path": "/ir/design.hw.mlir", "sha256": DIGEST},
        "op_timing": {"path": str(timing_path) if timing_path else "/ir/op-timing.json",
                      "sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest() if timing_path else "cd" * 32},
        "sources": [{"path": "/ir/pairs-scalar.json", "sha256": "ef" * 32, "schema": "rtl_extract.op_pairs.v1",
                     "specs": [{"name": "scalar.add_sub", "sha256": DIGEST}]}],
        "op_pairs": [copy.deepcopy(PAIR)],
        "capacity": [{"engine": "scalar", "live": 2, "window": "next_issue_age",
                      "witness": {"source": "scalar.add_sub", "gap": 4}}],
    }


@pytest.fixture
def case(monkeypatch, tmp_path):
    declare(monkeypatch, tmp_path, {"op_timing": DECLARATION, "op_pairs": {**DECLARATION, "schema": OP.SCHEMA}})
    path = write(tmp_path, timing_document())
    return OP, op_pairs_source, document(path), OT.merge_document(artifact(), path, target="t")


@pytest.mark.parametrize("edit", [
    lambda d: d.update(schema="merlin.op_pairs.v2"), lambda d: d.update(records=d.pop("op_pairs")),
    lambda d: d.update(target="t"), lambda d: d["hw_ir"].update(sha256="AB" * 32),
    lambda d: d["op_timing"].pop("path"), lambda d: d["op_timing"].update(sha256="invalid"),
    lambda d: d.update(sources=["junk"]), lambda d: d["sources"][0].update(schema="rtl_extract.op_pairs.v2"),
    lambda d: d["sources"][0].pop("specs"), lambda d: d["sources"][0].update(specs=[]),
    lambda d: d["sources"][0].update(specs={"name": "scalar.add_sub", "sha256": DIGEST}),
    lambda d: d["sources"][0].update(specs=["junk"]),
    lambda d: d["sources"][0]["specs"][0].pop("name"),
    lambda d: d["sources"][0]["specs"][0].update(name=""),
    lambda d: d["sources"][0]["specs"][0].update(sha256="invalid"),
    lambda d: d["sources"][0]["specs"][0].update(extra=1),
    lambda d: d["sources"][0]["specs"].append(copy.deepcopy(d["sources"][0]["specs"][0])),
    lambda d: d.update(op_pairs=["junk"]), lambda d: d["op_pairs"][0].pop("second"),
    lambda d: d["op_pairs"][0].update(engine=3), lambda d: d["op_pairs"][0].update(relation={"destination": 1}),
    lambda d: d["op_pairs"][0].update(gaps=[0, 12]), lambda d: d["op_pairs"][0].update(gaps=[12, 1]),
    lambda d: d["op_pairs"][0].update(gaps=[True, 12]), lambda d: d["op_pairs"][0].update(gaps=[1, 2, 3]),
    lambda d: d["op_pairs"][0].update(min_gap=13), lambda d: d["op_pairs"][0].update(stable_gap=0),
    lambda d: d["op_pairs"][0]["assertions"].update(checked=0),
    lambda d: d["op_pairs"][0]["assertions"].update(excluded=-1),
    lambda d: d["op_pairs"].append(copy.deepcopy(d["op_pairs"][0])), lambda d: d["capacity"][0].update(live=0),
    lambda d: d["capacity"][0].update(live=True), lambda d: d["capacity"][0].update(window="busy_age"),
    lambda d: d["capacity"].append(copy.deepcopy(d["capacity"][0])),
])
def test_malformed_document_is_refused_whole(edit):
    doc = document()
    edit(doc)
    with pytest.raises(ValueError, match="op_pairs document"):
        OP.validate_document(doc)


@pytest.mark.parametrize("edit", [
    lambda b, d: d["hw_ir"].update(sha256="23" * 32), lambda b, d: d["op_timing"].update(sha256="23" * 32),
    lambda b, d: d["op_pairs"][0].update(second="scalar.missing"),
    lambda b, d: d["op_pairs"][0].update(second="scalar.mul"),
    lambda b, d: d["op_pairs"].append({**PAIR, "assertions": {"checked": 0, "excluded": 0}}),
    lambda b, d: d["capacity"][0].update(engine="missing"),
    lambda b, d: d["capacity"][0].update(window="last_event_age"),
    lambda b, d: b["facts"]["op_timing"][1].update(next_issue_age=None),
    lambda b, d: b.update(artifact()), lambda b, d: b["inputs"].pop("op_timing_sha256"),
])
def test_merge_refuses_mismatched_provenance_records_or_capacity(case, tmp_path, edit):
    _, _, doc, before = case
    edit(before, doc)
    snapshot = copy.deepcopy(before)
    with pytest.raises(ValueError):
        OP.merge_document(before, write(tmp_path, doc, "pairs.json"), target="t")
    assert before == snapshot and "op_pairs" not in before["facts"]


def test_exact_timing_bytes_and_replacement_preserve_pair_provenance(case, tmp_path):
    _, _, doc, before = case
    path, timing_path = write(tmp_path, doc, "pairs.json"), Path(doc["op_timing"]["path"])
    merged = OP.merge_document(before, path, target="t")
    snapshot = copy.deepcopy(merged)
    assert OT.merge_document(merged, timing_path, target="t") == merged
    changed = timing_document()
    changed["op_timing"][0]["first_free_age"] += 1
    with pytest.raises(ValueError):
        OT.merge_document(merged, write(tmp_path, changed, "replacement.json"), target="t")
    assert merged == snapshot
    timing_path.write_bytes(timing_path.read_bytes() + b"\n")
    regenerated = OT.merge_document(before, timing_path, target="t")
    assert regenerated["facts"] == before["facts"]
    with pytest.raises(ValueError, match="op_timing.sha256"):
        OP.merge_document(regenerated, path, target="t")


def test_dump_without_timing_is_refused(case, monkeypatch, tmp_path):
    _, _, doc, _ = case
    monkeypatch.setattr(C, "build_facts", lambda **kw: artifact())
    out = tmp_path / "facts.json"
    with pytest.raises(ValueError):
        C.dump_facts(out, target="t", op_pairs_document=write(tmp_path, doc, "pairs.json"))
    assert not out.exists()


def test_facts_view_reads_mapping_and_absence():
    assert all(op_pairs(body) is None for body in (None, {}, {"op_pairs": None}, {"op_pairs": []}))
    doc = document()
    assert op_pairs({"op_pairs": doc}) is doc


def test_real_producer_document_validates_and_merges_when_available(monkeypatch, tmp_path):
    directory = os.environ.get("MERLIN_RTL_FACTS_DIR")
    if not directory:
        pytest.skip("MERLIN_RTL_FACTS_DIR is unset")
    out = Path(directory)
    path, timing_path = out / "op-pairs.json", out / "compiler-facts.json"
    doc = OP.read_document(path)
    assert OP.validate_document(doc) is doc
    declare(monkeypatch, tmp_path, {"op_timing": DECLARATION, "op_pairs": {**DECLARATION, "schema": OP.SCHEMA}})
    before = OT.merge_document(artifact(), timing_path, target="t")
    assert op_pairs(OP.merge_document(before, path, target="t")["facts"]) == doc
