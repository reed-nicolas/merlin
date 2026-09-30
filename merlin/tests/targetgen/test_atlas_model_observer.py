"""Independent numerical observation requires a supplied reference and complete execution."""

import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location(
        "atlas_model_observer", repo_root() / "examples/atlas/target/rtlgraph_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    expected = b"reference bytes"
    (root / "expected.bin").write_bytes(expected)
    (root / "preload.bin").write_bytes(b"input bytes")
    document = {
        "schema": "merlin.atlas_model_fixture.v1",
        "dram_size": 1024,
        "max_cycles": 100,
        "inputs": [{"base": 16, "path": "preload.bin", "sha256": hashlib.sha256(b"input bytes").hexdigest()}],
        "outputs": [
            {
                "name": "result",
                "space": "dram",
                "base": 32,
                "length": len(expected),
                "reference": {"path": "expected.bin", "sha256": hashlib.sha256(expected).hexdigest()},
            }
        ],
        "reference_provenance": "Explicit external reference fixture",
    }
    path = root / "fixture.json"
    path.write_text(json.dumps(document))
    return path, document


@pytest.fixture
def pair(adapter, fixture, tmp_path, monkeypatch):
    model = tmp_path / "selected-model"
    package = model / "npu_model"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "simulation.py").write_text("# explicitly selected source\n")
    before, after = tmp_path / "before.S", tmp_path / "after.S"
    before.write_text("delay 100\necall\n")
    after.write_text("ecall\n")
    args = dict(
        before=before,
        after=after,
        fixture=fixture[0],
        model_root=model,
        python=Path(sys.executable),
        output_dir=tmp_path / "observation",
    )

    def run(command, **kwargs):
        job = json.loads(Path(command[-3]).read_bytes())
        arm, output = command[-2], Path(command[-1])
        expected = Path(job["fixture"]["outputs"][0]["reference"]["path"]).read_bytes()
        (output / "result.bin").write_bytes(expected)
        report = {
            "status": "completed",
            "completed": True,
            "model_ticks": 20 if arm == "before" else 10,
            "outputs": {"result": adapter.identity(output / "result.bin")},
            "source": job["arms"][arm],
            "model_sources": job["model_sources"],
            "model_package": str(model / "npu_model/__init__.py"),
            "runtime": {"python": "selected"},
        }
        (output / "execution.json").write_text(json.dumps(report))
        assert "-I" in command and "-B" in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    return args


def test_byte_bound_fixture_and_fresh_paired_execution(adapter, pair):
    report = adapter.observe_pair(**pair)
    assert report["status"] == "passed"
    assert report["numerical_equivalence"] == "PASSED_SUPPLIED_REFERENCE"
    assert report["model_ticks"] == {"before": 20, "after": 10, "after_minus_before": -10}
    assert report["rtl_replayed"] is False and report["hardware_qualified"] is False
    assert report["identities"]["before"] == adapter.identity(pair["before"])
    assert report["results"]["before"]["command"] != report["results"]["after"]["command"]
    assert all(entry["source"]["sha256"] == entry["snapshot"]["sha256"] for entry in report["fixture_members"])


def test_same_wrong_outputs_do_not_pass_reference_check(adapter, pair, monkeypatch):
    original = adapter.subprocess.run

    def wrong(command, **kwargs):
        result = original(command, **kwargs)
        output = Path(command[-1])
        (output / "result.bin").write_bytes(b"same wrong output")
        execution = json.loads((output / "execution.json").read_bytes())
        execution["outputs"]["result"] = adapter.identity(output / "result.bin")
        (output / "execution.json").write_text(json.dumps(execution))
        return result

    monkeypatch.setattr(adapter.subprocess, "run", wrong)
    report = adapter.observe_pair(**pair)
    assert report["status"] == "error"
    assert "supplied reference" in report["error"]
    assert report["numerical_equivalence"] == "UNKNOWN"
    assert report["model_ticks"]["before"] is None


def test_incomplete_successful_child_is_rejected(adapter, pair, monkeypatch):
    original = adapter.subprocess.run

    def incomplete(command, **kwargs):
        result = original(command, **kwargs)
        path = Path(command[-1]) / "execution.json"
        execution = json.loads(path.read_bytes())
        execution["completed"] = False
        path.write_text(json.dumps(execution))
        return result

    monkeypatch.setattr(adapter.subprocess, "run", incomplete)
    report = adapter.observe_pair(**pair)
    assert report["status"] == "error" and "did not complete" in report["error"]


def test_runtime_timeout_preserves_unknown_numerics(adapter, pair, monkeypatch):
    def timeout(command, **kwargs):
        raise adapter.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    report = adapter.observe_pair(**pair)
    assert report["status"] == "error"
    assert report["numerical_equivalence"] == "UNKNOWN"
    assert report["results"]["before"]["returncode"] is None
    assert (pair["output_dir"] / "before/stderr.log").is_file()


def test_unavailable_runtime_records_launch_failure(adapter, pair, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("selected runtime unavailable")

    monkeypatch.setattr(adapter.subprocess, "run", unavailable)
    report = adapter.observe_pair(**pair)
    assert report["status"] == "error"
    assert report["model_ticks"]["before"] is None
    assert "selected runtime unavailable" in (pair["output_dir"] / "before/stderr.log").read_text()


def test_changed_model_invalidates_observation(adapter, pair, monkeypatch):
    original = adapter.subprocess.run

    def mutate(command, **kwargs):
        result = original(command, **kwargs)
        (pair["model_root"] / "npu_model/simulation.py").write_text("# changed\n")
        return result

    monkeypatch.setattr(adapter.subprocess, "run", mutate)
    report = adapter.observe_pair(**pair)
    assert report["status"] == "error" and "Model source changed" in report["error"]


def test_selected_source_execution_ignores_stale_bytecode(adapter, tmp_path):
    root = tmp_path / "selected"
    package = root / "npu_model"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    namespace = package / "util"
    namespace.mkdir()
    source = namespace / "helper.py"
    source.write_text("value = 'old'\n")
    original_stat = source.stat()
    py_compile.compile(str(source), doraise=True)
    source.write_text("value = 'new'\n")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    script = (
        "import importlib.util,sys; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('observer',sys.argv[1]); "
        "observer=importlib.util.module_from_spec(spec); spec.loader.exec_module(observer); "
        "root=Path(sys.argv[2]);sys.path.insert(0,str(root));observer._source_only_model(root); "
        "import npu_model.util.helper;print(npu_model.util.helper.value)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, adapter.__file__, str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "new"


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda doc: doc["outputs"][0].update(length=2000), "aperture"),
        (lambda doc: doc["inputs"][0].update(sha256="0" * 64), "bytes differ"),
        (lambda doc: doc["inputs"][0].update(path="../escape.bin"), "escapes"),
        (lambda doc: doc.update(reference_provenance=""), "provenance"),
        (lambda doc: doc.update(outputs=[]), "reference output"),
    ],
)
def test_invalid_fixture_rejected_before_execution(adapter, fixture, tmp_path, mutation, error):
    path, document = fixture
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=error):
        adapter.snapshot_fixture(path, tmp_path / "snapshot")


@pytest.mark.parametrize("document", [[], 2, None, {"schema": "merlin.atlas_model_fixture.v1"}])
def test_nonobject_or_incomplete_fixture_rejected(adapter, tmp_path, document):
    fixture = tmp_path / "bad.json"
    fixture.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        adapter.snapshot_fixture(fixture, tmp_path / "snapshot")


def test_unhashable_register_selector_rejected(adapter, fixture, tmp_path):
    path, document = fixture
    document["outputs"] = [
        {"name": "result", "space": "mrf_bf16", "registers": [[]], "reference": document["outputs"][0]["reference"]}
    ]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="register selection"):
        adapter.snapshot_fixture(path, tmp_path / "snapshot")
