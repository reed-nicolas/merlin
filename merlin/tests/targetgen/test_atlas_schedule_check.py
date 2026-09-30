"""Optional target checker preserves evidence, tool failures and compatibility gates."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checker():
    return load(repo_root() / "examples/atlas/target/rtlgraph_check.py", "atlas_schedule_check")


@pytest.fixture
def selected(tmp_path):
    fixtures = load(repo_root() / "merlin/tests/targetgen/test_atlas_scheduling_evidence.py", "scheduling_fixtures")
    contract_path, contract, _ = fixtures.bundle(tmp_path / "bundle")
    footprints = (contract_path.parent / contract["footprints"]["path"]).read_text()
    compiler = tmp_path / "selected-compiler"
    compiler.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if '--dump-footprints' in args:\n"
        f"    pathlib.Path(args[args.index('--dump-footprints') + 1]).write_text({footprints!r})\n"
        "elif '--check' in args:\n"
        "    print(args[0] + ': 17 cycles, 3 instructions issued (0 delays)')\n"
        "    print('actual check diagnostic', file=sys.stderr)\n"
        "    sys.exit(1 if '# rejected' in pathlib.Path(args[0]).read_text() else 0)\n"
        "else:\n"
        "    sys.exit(2)\n"
    )
    compiler.chmod(0o755)
    contract["compiler"] = {"path": str(compiler), "sha256": fixtures.digest(compiler.read_bytes())}
    fixtures.rewrite_contract(contract_path, contract)
    adapter = load(repo_root() / "examples/atlas/target/rtlgraph_evidence.py", "schedule_adapter")
    selection = adapter.convert_contract(contract_path, tmp_path / "selection")
    source = tmp_path / "program.S"
    source.write_text("ecall\n")
    return {"source": source, "selection": selection, "compiler": compiler}


def test_pass_keeps_actual_logs_and_same_model_scope(checker, selected, tmp_path):
    report = checker.check_schedule(**selected, output_dir=tmp_path / "checked", validation="dynamic")
    assert report["status"] == "passed"
    assert report["modeled_cycles"] == 17
    assert len(report["commands"]) == 2
    assert Path(report["commands"][1]["stderr"]["path"]).read_text() == "actual check diagnostic\n"
    assert report["verification"]["footprints_reproduced"]
    assert report["verification"]["schedule_consistency_verified"]
    assert not report["verification"]["numerical_correctness_verified"]
    assert json.loads((tmp_path / "checked/report.json").read_text()) == report
    assert (tmp_path / "checked/input.S").read_bytes() == selected["source"].read_bytes()
    assert report["profile_args"][0] == "--rtl-dma-profile"


def test_rejected_schedule_keeps_tool_verdict(checker, selected, tmp_path):
    selected["source"].write_text("ecall # rejected\n")
    report = checker.check_schedule(**selected, output_dir=tmp_path / "checked")
    assert report["status"] == "rejected" and report["returncode"] == 1
    assert report["modeled_cycles"] is None
    assert not report["verification"]["schedule_consistency_verified"]


@pytest.mark.parametrize("problem", ["compiler", "profile", "selection", "assembly", "wrong-native-dialect"])
def test_preflight_failures_never_execute_compiler(checker, selected, tmp_path, monkeypatch, problem):
    if problem == "compiler":
        selected["compiler"].write_text("changed compiler")
    elif problem == "profile":
        profile = selected["selection"].parent / "bundle/dma/atlas-dma.profile"
        profile.write_text(profile.read_text() + "channels=999\n")
    elif problem == "selection":
        document = json.loads(selected["selection"].read_text())
        document["profiles"][0]["fields"]["completion"] = "guessed"
        selected["selection"].write_text(json.dumps(document))
    elif problem == "assembly":
        selected["assembly"] = "merlin-atlas"
        selected["source"].write_text("DMA.LOAD x6, x18, x12, 3\n")
    else:
        selected["source"].write_text("DMA.LOAD x6, x18, x12, 3\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight failure launched the compiler")

    monkeypatch.setattr(checker.subprocess, "run", forbidden)
    report = checker.check_schedule(**selected, output_dir=tmp_path / "checked")
    assert report["status"] == "error" and report["error"]
    assert report["commands"] == []


def test_fresh_footprint_disagreement_stops_before_schedule(checker, selected, tmp_path, monkeypatch):
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        Path(command[command.index("--dump-footprints") + 1]).write_text("{}")
        return subprocess.CompletedProcess(command, 0, b"fresh query\n", b"")

    monkeypatch.setattr(checker.subprocess, "run", query)
    report = checker.check_schedule(**selected, output_dir=tmp_path / "checked")
    assert report["status"] == "error"
    assert "fresh compiler query" in report["error"]
    assert len(calls) == 1 and "--check" not in calls[0]


def test_timeout_keeps_partial_logs(checker, selected, tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(checker.subprocess, "run", timeout)
    report = checker.check_schedule(**selected, output_dir=tmp_path / "checked", timeout=0.01)
    assert report["status"] == "error"
    assert report["commands"][0]["timed_out"]
    assert Path(report["commands"][0]["stdout"]["path"]).read_bytes() == b"partial stdout"


def test_concurrent_input_change_invalidates_success(checker, selected, tmp_path, monkeypatch):
    actual = checker.subprocess.run

    def mutate(command, **kwargs):
        result = actual(command, **kwargs)
        if "--check" in command:
            selected["source"].write_text("ecall # changed after check\n")
        return result

    monkeypatch.setattr(checker.subprocess, "run", mutate)
    report = checker.check_schedule(**selected, output_dir=tmp_path / "checked")
    assert report["status"] == "error"
    assert "changed during check" in report["error"]
    assert not report["verification"]["schedule_consistency_verified"]
