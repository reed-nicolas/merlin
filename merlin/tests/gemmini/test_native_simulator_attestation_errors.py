"""The native attester keeps diagnostics and relocates only output-only annotations."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import repo_root


def _attester():
    path = repo_root() / "examples/gemmini/verification/attest_native_simulator.py"
    spec = importlib.util.spec_from_file_location("gemmini_native_attester", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_firtool_failure_reports_the_actual_diagnostic():
    module = _attester()
    script = "import sys; sys.stderr.write('error: selected source is invalid\\n' + 'x' * 2000); sys.exit(1)"
    with pytest.raises(RuntimeError, match="selected source is invalid"):
        module.run_command([sys.executable, "-c", script], timeout_s=5)


def test_hierarchy_output_relocation_changes_only_its_two_declared_filenames(tmp_path):
    module = _attester()
    selected = tmp_path / "selected"
    selected.mkdir()
    original = selected / "annotations.json"
    rows = [
        {"class": "unrelated", "target": "x"},
        {
            "class": "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation",
            "filename": str(selected / "model_module_hierarchy.json"),
        },
        {
            "class": "sifive.enterprise.firrtl.ModuleHierarchyAnnotation",
            "filename": str(selected / "top_module_hierarchy.json"),
        },
    ]
    original.write_text(json.dumps(rows))
    projected, moves = module.relocate_hierarchy_annotations(original, selected, tmp_path / "artifact")
    observed = json.loads(projected.read_text())
    assert observed[0] == rows[0]
    assert [row["filename"] for row in observed[1:]] == [
        str(tmp_path / "artifact/model_module_hierarchy.json"),
        str(tmp_path / "artifact/top_module_hierarchy.json"),
    ]
    assert len(moves) == 2
    assert json.loads(original.read_text()) == rows

    rows[1]["filename"] = str(tmp_path / "unselected/model_module_hierarchy.json")
    original.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unexpected hierarchy output"):
        module.relocate_hierarchy_annotations(original, selected, tmp_path / "other-artifact")
