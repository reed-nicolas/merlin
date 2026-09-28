"""A source audit records carrier geometry, never software scale semantics."""

from merlin.targetgen.rtl.source_audit import _register_bank_geometry, build_source_audit

CHECK = {
    "id": "scale_bank",
    "kind": "hw_register_bank_geometry",
    "module": "ScaleBank",
    "write_index_port": "idx",
    "write_data_port": "data",
    "output_prefix": "reg_",
    "expected_entries": 4,
    "expected_index_type": "i2",
    "expected_data_type": "i8",
}
SOURCE = (
    "hw.module private @ScaleBank(in %idx : i2, in %data : i8, "
    "out reg_0 : i8, out reg_1 : i8, out reg_2 : i8, out reg_3 : i8) {}"
)


def test_register_bank_geometry_accepts_exact_contiguous_ports():
    status, observed = _register_bank_geometry(SOURCE, CHECK)
    assert status == "verified"
    assert observed["output_names"] == ["reg_0", "reg_1", "reg_2", "reg_3"]
    assert observed["entry_count"] == 4


def test_register_bank_geometry_rejects_missing_or_wrong_width_ports():
    for changed in (
        SOURCE.replace("out reg_2 : i8, ", ""),
        SOURCE.replace("in %idx : i2", "in %idx : i3"),
        SOURCE.replace("out reg_3 : i8", "out reg_3 : i16"),
        SOURCE.replace("out reg_3 : i8", "in %reg_3 : i8"),
        SOURCE.replace("out reg_3 : i8", "out reg_03 : i8"),
    ):
        assert _register_bank_geometry(changed, CHECK)[0] == "mismatch"
    assert _register_bank_geometry(SOURCE.replace("@ScaleBank", "@Other"), CHECK)[0] == "unknown"


def test_register_bank_audit_is_source_consistency_only(tmp_path):
    hw = tmp_path / "core.hw.mlir"
    hw.write_text(SOURCE)
    selection = {"target": "fixture", "sources": {"core_hw": {"path": str(hw), "sha256": "fixture"}}}
    report = build_source_audit(selection, {}, {"audit_checks": [CHECK]})
    assert report["status"] == "verified"
    assert report["checks"][0]["source"]["observed_geometry"]["write_data"] == {
        "direction": "in",
        "type": "i8",
    }
    assert "not operation/numerical certification" in report["qualification"]


def test_command_port_audit_rejects_reversed_direction(tmp_path):
    hw = tmp_path / "core.hw.mlir"
    hw.write_text("hw.module private @Core(in %scale : i8) {}")
    selection = {"target": "fixture", "sources": {"core_hw": {"path": str(hw), "sha256": "fixture"}}}
    check = {
        "id": "scale_command",
        "kind": "hw_port_type",
        "module": "Core",
        "port": "scale",
        "expected_type": "i8",
        "expected_direction": "out",
    }
    report = build_source_audit(selection, {}, {"audit_checks": [check]})
    assert report["status"] == "unknown"
    assert report["checks"][0]["source_status"] == "mismatch"
