"""Shared contract, document and merge checks for RTL operation facts."""

from __future__ import annotations

import copy
import json

import pytest

from merlin.common.facts_view import op_timing, timing
from merlin.targetgen.rtl import circt_introspect as C
from merlin.targetgen.rtl import facts as F
from merlin.targetgen.rtl import op_timing as OT
from merlin.targetgen.rtl.extraction_contract import op_timing_source

DIGEST = "ab" * 32
CONTRACT_PATH = F.target_contract_path
GROUP = {"first_age": 1, "last_age": 3, "count": 3, "step": 1}
RESOLVED = {
    "name": "scalar.add", "module": "ScalarUnit", "engine": "scalar", "source": "control_simulation",
    "evidence": "simulated ScalarUnit: command accepted and writes observed",
    "assumptions": {"scratchpad_read_latency": 1, "limit": 64}, "events": {"scratchpad_read": GROUP},
    "first_free_age": 9, "next_issue_age": 4,
}
DECLARATION = {
    "schema": OT.SCHEMA, "producer": "provider extractor", "spec_set": "provider/specs",
    "hw_ir": {"requires": ["names kept", "memories inline"]},
}


def document():
    unresolved = {**copy.deepcopy(RESOLVED), "name": "scalar.mul", "events": None,
                  "first_free_age": None, "next_issue_age": None}
    return {"schema": OT.SCHEMA, "hw_ir": {"path": "/ir/design.hw.mlir", "sha256": DIGEST},
            "op_timing": [copy.deepcopy(RESOLVED), {**copy.deepcopy(RESOLVED), "name": "scalar.sub"}, unresolved]}


def declare(monkeypatch, tmp_path, block):
    import yaml

    path = tmp_path / "target_contract.yaml"
    path.write_text(yaml.safe_dump({"rtl_extraction": block}), encoding="utf-8")
    monkeypatch.setattr(F, "target_contract_path", lambda target, **kw: path)


def write(tmp_path, doc, name="op-timing.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def artifact():
    return {"schema_version": "2.0", "inputs": {"target": "t", "hw_sha256": "01" * 32},
            "facts": {"timing": [{"name": "ScalarUnit", "module": "ScalarUnit", "pipeline_depth": 2}],
                      "arrays": [{"name": "a"}]}}


@pytest.fixture
def case():
    return OT, op_timing_source, document(), artifact()


def payload(reader, doc):
    return doc["op_timing"] if reader is OT else doc


def test_declaration_normalized_and_absent(case, monkeypatch, tmp_path):
    reader, source, _, _ = case
    block = {**DECLARATION, "schema": reader.SCHEMA}
    declare(monkeypatch, tmp_path, {reader.SECTION: block})
    assert source("t") == block
    declare(monkeypatch, tmp_path, {})
    assert source("t") is None


@pytest.mark.parametrize("edit", [
    lambda b: b.update(schema="unsupported"), lambda b: b.update(producer=""), lambda b: b.pop("spec_set"),
    lambda b: b.update(spec_sett="x"), lambda b: b.update(hw_ir={"requires": []}),
    lambda b: b.update(hw_ir={"requires": ["ok", 3]}),
    lambda b: b.update(hw_ir={"requires": ["ok"], "preserve": "named"}), lambda b: b.update(hw_ir="inline"),
    ["x"],
])
def test_malformed_declaration_fails_closed(case, monkeypatch, tmp_path, edit):
    reader, source, _, _ = case
    block = copy.deepcopy(DECLARATION) | {"schema": reader.SCHEMA}
    if callable(edit):
        edit(block)
    else:
        block = edit
    declare(monkeypatch, tmp_path, {reader.SECTION: block})
    with pytest.raises(ValueError):
        source("t")


def test_in_tree_target_declares_its_source(case, monkeypatch):
    from merlin.targetgen import target_registry

    reader, source, _, _ = case
    monkeypatch.setattr(F, "target_contract_path", CONTRACT_PATH)
    declaration = target_registry.load_contract("atlas")["rtl_extraction"][reader.SECTION]
    assert declaration["schema"] == reader.SCHEMA and source("atlas") == declaration


def test_valid_document_passes_through_unchanged(case):
    reader, _, doc, _ = case
    assert reader.validate_document(doc) is doc
    if reader is not OT:
        doc["op_pairs"][0].update(min_gap=None, stable_gap=None, engine=None)
        assert reader.validate_document(doc) is doc


@pytest.mark.parametrize("edit", [
    lambda d: d.update(schema="atlas.op_timing.v0-proposal"), lambda d: d.update(records=d.pop("op_timing")),
    lambda d: d.update(target="t"), lambda d: d["hw_ir"].update(sha256="AB" * 32),
    lambda d: d["hw_ir"].pop("path"), lambda d: d["hw_ir"].update(extra=1), lambda d: d.update(op_timing=[]),
    lambda d: d["op_timing"].append(copy.deepcopy(RESOLVED)), lambda d: d["op_timing"][0].pop("name"),
    lambda d: d["op_timing"][0].update(evidence=None), lambda d: d["op_timing"][0].update(source=""),
    lambda d: d["op_timing"][0].pop("assumptions"), lambda d: d["op_timing"][0].update(first_free_age=-1),
    lambda d: d["op_timing"][0].update(first_free_age=True), lambda d: d["op_timing"][0].update(next_issue_age="4"),
    lambda d: d["op_timing"][0]["events"]["scratchpad_read"].update(count=None),
    lambda d: d["op_timing"][0]["events"]["scratchpad_read"].update(first_age=1.5),
    lambda d: d["op_timing"][0].update(events=[GROUP]), lambda d: d["op_timing"][2].update(first_free_age=7),
    lambda d: d["op_timing"][2].update(next_issue_age=0),
])
def test_malformed_document_is_refused_whole(edit):
    doc = document()
    edit(doc)
    with pytest.raises(ValueError, match="op_timing document"):
        OT.validate_document(doc)


@pytest.mark.parametrize("failure", ["missing", "json", "encoding"])
def test_read_document_reports_unreadable_and_non_json(case, tmp_path, failure):
    reader, _, doc, _ = case
    path = tmp_path / "bad.json"
    if failure != "missing":
        path.write_bytes(b"{" if failure == "json" else json.dumps(doc).encode("utf-16"))
    with pytest.raises(ValueError, match="cannot read"):
        reader.read_document(path)


def test_merge_preserves_input_and_replaces_whole_section(case, monkeypatch, tmp_path):
    reader, _, doc, before = case
    key, snapshot = reader.SECTION, copy.deepcopy(before)
    declare(monkeypatch, tmp_path, {key: {**DECLARATION, "schema": reader.SCHEMA}})
    merged = reader.merge_document(before, write(tmp_path, doc, "document.json"), target="t")
    assert before == snapshot and merged["facts"][key] == payload(reader, doc)
    assert timing(merged["facts"]) == timing(before["facts"]) and merged["facts"]["arrays"] == before["facts"]["arrays"]
    assert merged["inputs"]["hw_sha256"] == before["inputs"]["hw_sha256"]
    if reader is OT:
        suffixes = ("hw_ir_sha256", "hw_ir_path", "schema")
        assert [merged["inputs"][f"op_timing_{s}"] for s in suffixes] == [DIGEST, "/ir/design.hw.mlir", OT.SCHEMA]
        other = {**copy.deepcopy(doc), "op_timing": [{**RESOLVED, "name": "scalar.other"}]}
        keys = {("op_timing", b["name"]) for b in doc[key]} | {("timing", "ScalarUnit")}
        assert keys <= set(F._flatten_body(merged, "x"))
    else:
        assert op_timing(merged["facts"]) == op_timing(before["facts"])
        assert merged["inputs"]["op_timing_sha256"] == doc["op_timing"]["sha256"]
        assert merged["inputs"]["op_timing_path"] == doc["op_timing"]["path"]
        other = copy.deepcopy(doc)
        other[key] = [{**doc[key][0], "first": "scalar.sub", "second": "scalar.add", "source": "scalar.sub_add"}]
        other["capacity"] = []
    second = reader.merge_document(merged, write(tmp_path, other, "replacement.json"), target="t")
    assert second["facts"][key] == payload(reader, other) and merged["facts"][key] == payload(reader, doc)
    if reader is not OT:
        assert F._flatten_body(second, "x")[(key, None)]["value"] == other
        del before["inputs"]["op_timing_sha256"]
    declare(monkeypatch, tmp_path, {})
    with pytest.raises(ValueError, match=f"declares no rtl_extraction.{key}"):
        reader.merge_document(before, write(tmp_path, doc), target="t")


def test_dump_merges_repeats_and_refuses_downgrade(case, monkeypatch, tmp_path):
    reader, _, doc, _ = case
    key, calls, out = reader.SECTION, [], tmp_path / "facts.json"
    declare(monkeypatch, tmp_path, {key: {**DECLARATION, "schema": reader.SCHEMA}, "op_timing": DECLARATION})
    monkeypatch.setattr(C, "build_facts", lambda **kw: calls.append(kw) or artifact())
    kwargs = {"target": "t", f"{key}_document": write(tmp_path, doc, "document.json")}
    if reader is not OT:
        kwargs["op_timing_document"] = doc["op_timing"]["path"]
    first = C.dump_facts(out, **kwargs)
    assert first["facts"][key] == payload(reader, doc)
    assert "op_timing_document" not in calls[0] and "op_pairs_document" not in calls[0]
    assert json.loads(out.read_text())["facts"][key] == payload(reader, doc) and C.dump_facts(out, **kwargs) == first
    if reader is not OT:
        doc[key][0]["assertions"]["checked"] += 1
        write(tmp_path, doc, "document.json")
        assert C.dump_facts(out, **kwargs)["facts"][key] == doc and json.loads(out.read_text())["facts"][key] == doc
    original = out.read_bytes()
    del kwargs[f"{key}_document"]
    with pytest.raises(F.FactsDowngrade, match=key):
        C.dump_facts(out, **kwargs)
    assert out.read_bytes() == original


def test_facts_view_reads_absent_and_malformed_as_empty():
    assert op_timing(None) == () and op_timing({}) == () and op_timing({"op_timing": None}) == ()
    assert op_timing({"op_timing": ["junk", {"name": "ok"}]}) == ({"name": "ok"},)


def test_merge_refuses_a_malformed_document_without_a_partial_merge(monkeypatch, tmp_path):
    declare(monkeypatch, tmp_path, {"op_timing": DECLARATION})
    before, bad = artifact(), document()
    bad["op_timing"][2]["first_free_age"] = 3
    with pytest.raises(ValueError):
        OT.merge_document(before, write(tmp_path, bad), target="t")
    assert before == artifact()
