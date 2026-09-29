"""Equivalent CIRCT expressions must not split one decoder fan-out by SSA name."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.targetgen.rtl import mlc_bridge


class _Value:
    def __init__(self, typ: str, owner=None):
        self.type = typ
        self.owner = owner


class _Op:
    def __init__(self, name: str, operands=(), *, typ: str | None = None, **properties):
        self.op_name = SimpleNamespace(data=name)
        self.operands = tuple(operands)
        self.properties = properties
        self.results = [_Value(typ, self)] if typ is not None else []


class _Module:
    def __init__(self, name: str):
        self.name = name
        self.operations: list[_Op] = []

    def add(self, op: _Op) -> _Value:
        self.operations.append(op)
        return op.results[0]


class _Graph:
    def __init__(self, modules: list[_Module]):
        self.modules = {module.name: module for module in modules}

    def ops(self, name: str, within=None):
        modules = [within] if within is not None else self.modules.values()
        return [op for module in modules for op in module.operations if op.op_name.data == name]

    @staticmethod
    def defining_op(value):
        return value.owner

    @staticmethod
    def icmp_predicate(op):
        return op.properties["predicate"]

    @staticmethod
    def const_value(op):
        return op.properties["value"]

    @staticmethod
    def extract_range(op):
        return op.properties["lowBit"], int(op.results[0].type[1:])


def _graph_with_cloned_instruction_decode() -> _Graph:
    instruction = _Module("InstructionDecode")
    instruction_port = _Value("i32")
    for code in (87, 215, 343, 471, 599, 727, 855):
        # A newer CIRCT expansion clones both extracts and the concat for every
        # equality, though all seven dynamic operands compute the same bits.
        high = instruction.add(_Op("comb.extract", [instruction_port], typ="i7", lowBit=25))
        low = instruction.add(_Op("comb.extract", [instruction_port], typ="i7", lowBit=0))
        field = instruction.add(_Op("comb.concat", [high, low], typ="i14"))
        constant = instruction.add(_Op("hw.constant", typ="i14", value=code))
        instruction.add(_Op("comb.icmp", [field, constant], typ="i1", predicate=0))

    subdecode = _Module("UnrelatedSubdecode")
    subfield = _Value("i5")
    for code in (0, 1, 2, 3, 4, 5):
        constant = subdecode.add(_Op("hw.constant", typ="i5", value=code))
        subdecode.add(_Op("comb.icmp", [subfield, constant], typ="i1", predicate=0))

    other = _Module("OtherInstruction")
    other_port = _Value("i32")
    for code in (1087, 1215, 1343, 1471):
        high = other.add(_Op("comb.extract", [other_port], typ="i7", lowBit=25))
        low = other.add(_Op("comb.extract", [other_port], typ="i7", lowBit=0))
        field = other.add(_Op("comb.concat", [high, low], typ="i14"))
        constant = other.add(_Op("hw.constant", typ="i14", value=code))
        other.add(_Op("comb.icmp", [field, constant], typ="i1", predicate=0))
    return _Graph([instruction, subdecode, other])


def test_cloned_instruction_decode_recovers_only_one_expression_class():
    graph = _graph_with_cloned_instruction_decode()
    signals = mlc_bridge._equivalent_decode_signals(graph)
    assert signals[0].module == "InstructionDecode"
    assert signals[0].width == 14
    assert signals[0].values == (87, 215, 343, 471, 599, 727, 855)
    assert signals[0].fanout == 7
    assert signals[0].cloned_expressions == 7
    assert signals[1].module == "OtherInstruction"
    assert signals[1].values == (1087, 1215, 1343, 1471)


def test_bridge_prefers_recovered_decode_without_unioning_unrelated_modules(monkeypatch, tmp_path):
    pytest.importorskip("mlc.discover.decode")
    from merlin.targetgen.rtl import hw_graph

    graph = _graph_with_cloned_instruction_decode()
    monkeypatch.setattr(mlc_bridge, "require_mlc", lambda: None)
    monkeypatch.setattr(mlc_bridge, "core_hw_mlir", lambda target: tmp_path / "synthetic.hw.mlir")
    monkeypatch.setattr(mlc_bridge, "circt_opt_bin", lambda: None)
    monkeypatch.setattr(hw_graph, "load_hw_graph", lambda *args, **kwargs: graph)
    result = mlc_bridge.discover_legal_opcodes("synthetic")
    assert result["width"] == 14
    assert result["module"] == "InstructionDecode"
    assert result["legal_opcodes"] == [87, 215, 343, 471, 599, 727, 855]
    assert result["fanout"] == 7
    assert result["scope"] == "observed_decode_field"
    assert result["complete_isa"] is False


def test_field_local_observation_cannot_authorize_an_executable_endpoint():
    from merlin.targetgen import capability_manifests as manifests

    observed = {
        "interfaces": [{
            "name": "funct_decode_table", "legal_funct": [87, 215, 343],
            "width": 14, "scope": "observed_decode_field", "complete_isa": False,
        }]
    }
    assert manifests._endpoint_from_facts(observed) is None
    derived = manifests.derive_manifest(
        {"target": "test-chip"}, {"facts": observed},
        residual={"compute_units": [{"name": "engine", "kind": "systolic", "ops": ["matmul"], "dtypes": ["int8"]}]},
    )
    assert derived["endpoint_kind"] == "unresolved"
    assert derived["endpoint_resolution"]["status"] == "unverified"
    assert "legal_funct" not in derived.get("encoding", {})


def test_self_hosted_instruction_interface_is_separate_endpoint_evidence():
    from merlin.targetgen import capability_manifests as manifests

    observed = {"interfaces": [
        {"name": "funct_decode_table", "legal_funct": [87], "scope": "observed_decode_field"},
        {"name": "self_hosted_isa", "encoding_bits": 32, "instruction_classes": ["load", "compute"]},
    ]}
    assert manifests._endpoint_from_facts(observed) == "external_backend"


def test_field_local_values_do_not_validate_full_instruction_words(monkeypatch):
    from merlin.targetgen import isa_taxonomy
    from merlin.targetgen import rtl_check_compiler, rtl_check_runner

    facts = {"facts": {"interfaces": [{
        "name": "funct_decode_table", "legal_funct": [87], "width": 14,
        "scope": "observed_decode_field", "complete_isa": False,
    }]}}
    monkeypatch.setattr(isa_taxonomy, "taxonomy_for_target", lambda target: {})
    checks = rtl_check_compiler.compile_kernel_checks(
        {"name": "sample", "operation": {"op": "matmul"}}, facts_rec=facts, target="test-chip"
    )
    assert "ILLEGAL_OPCODE_COUNT" not in checks
    rendered = rtl_check_runner.render_kernel_decode(".word 87\n.word 0xdead0057\n", facts)
    assert "ILLEGAL_OPCODE_COUNT -" in rendered
    assert "LEGAL_OPCODE_SET_SIZE -" in rendered


def test_only_an_explicitly_complete_word_taxonomy_authorizes_legality(monkeypatch):
    from merlin.targetgen import isa_taxonomy
    from merlin.targetgen import rtl_check_compiler, rtl_check_runner

    facts = {"facts": {"interfaces": [{
        "name": "funct_decode_table", "legal_funct": [87],
        "scope": "observed_decode_field", "complete_isa": False,
    }]}}
    taxonomy = {"by_mnemonic": {"example": {
        "class": "compute", "fixed_mask": 0x7f, "fixed_value": 0x57,
    }}}
    cap = {"name": "sample", "operation": {"op": "matmul"}}
    monkeypatch.setattr(isa_taxonomy, "taxonomy_for_target", lambda target: taxonomy)
    assert "ILLEGAL_OPCODE_COUNT" not in rtl_check_compiler.compile_kernel_checks(cap, facts_rec=facts, target="chip")
    assert "ILLEGAL_OPCODE_COUNT -" in rtl_check_runner.render_kernel_decode(".word 87\n.word 0xdeadbeef\n", facts, taxonomy)

    taxonomy["complete_isa"] = True
    assert "ILLEGAL_OPCODE_COUNT 0" in rtl_check_compiler.compile_kernel_checks(cap, facts_rec=facts, target="chip")
    rendered = rtl_check_runner.render_kernel_decode(".word 87\n.word 0xdeadbeef\n", facts, taxonomy)
    assert "ILLEGAL_OPCODE_COUNT 1" in rendered


def test_incomplete_decoder_field_does_not_refute_a_command_classifier():
    from merlin.targetgen.rtl import circt_introspect

    record = {"facts": {"interfaces": [{
        "name": "funct_decode_table", "legal_funct": [1, 2],
        "scope": "observed_decode_field", "complete_isa": False,
    }]}, "source_consistency": {"status": "verified"}}
    result = circt_introspect.validate(record, rocc_funct_class={1: "one", 3: "three"})
    assert not result["diverge"]
    assert any("complete RTL RoCC funct legality unavailable" in item for item in result["unknown"])
