from __future__ import annotations

import sys
from types import ModuleType

import pytest
from merlin_firesim_modelblaster import _check_workload, runner


def test_workload_mismatch_is_refused(tmp_path):
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    cfg = deploy / "config_runtime.yaml"
    cfg.write_text("workload:\n  workload_name: other.json\n")
    with pytest.raises(RuntimeError, match="other.json"):
        _check_workload(str(tmp_path), "selected")
    cfg.write_text("workload:\n  workload_name: selected.json\n")
    _check_workload(str(tmp_path), "selected")


def test_installed_adapter_uses_selected_checkout_and_queue(monkeypatch, tmp_path):
    source = tmp_path / "src" / "modelblaster" / "validation"
    source.mkdir(parents=True)
    (source / "firesim_runner.py").write_text("# fake selected runner\n")
    elf = tmp_path / "image.elf"
    elf.write_bytes(b"fake ELF " + runner.completion_metric_prefix.encode("ascii"))
    monkeypatch.setattr("merlin_firesim_modelblaster.env", lambda key: str(tmp_path))
    seen = {}

    def fake_run(elf, **kwargs):
        seen.update(elf=elf, **kwargs)
        return "OUT 0\nDONE\n"

    package = ModuleType("modelblaster")
    package.__path__ = []
    validation = ModuleType("modelblaster.validation")
    validation.__path__ = []
    module = ModuleType("modelblaster.validation.firesim_runner")
    module.__file__ = str(source / "firesim_runner.py")
    fake_run.__module__ = module.__name__
    module.run_firesim = fake_run
    monkeypatch.setitem(sys.modules, "modelblaster", package)
    monkeypatch.setitem(sys.modules, "modelblaster.validation", validation)
    monkeypatch.setitem(sys.modules, "modelblaster.validation.firesim_runner", module)
    assert runner.preflight() == str(source / "firesim_runner.py")
    result = runner(str(elf), firesim_root="/selected", firesim_env="/env.sh", timeout=17, queue=True)
    assert result == "OUT 0\nDONE\n"
    assert seen == {
        "elf": str(elf),
        "models": None,
        "firesim_root": "/selected",
        "firesim_env": "/env.sh",
        "timeout": 17.0,
    }
    assert sys.path[0] == str(tmp_path)


def test_adapter_rejects_image_without_its_terminal_marker(monkeypatch, tmp_path):
    source = tmp_path / "validation"
    source.mkdir()
    (source / "firesim_runner.py").write_text("# fake selected runner\n")
    elf = tmp_path / "image.elf"
    elf.write_bytes(b"fake ELF without the marker")
    monkeypatch.setattr("merlin_firesim_modelblaster.env", lambda key: str(tmp_path))
    with pytest.raises(RuntimeError, match="completion_metric_prefix"):
        runner(str(elf), firesim_root="/selected", firesim_env="/env.sh", timeout=17, queue=True)


def test_adapter_rejects_preloaded_runner_from_other_checkout(monkeypatch, tmp_path):
    source = tmp_path / "validation"
    source.mkdir()
    (source / "firesim_runner.py").write_text("# selected runner\n")
    elf = tmp_path / "image.elf"
    elf.write_bytes(runner.completion_metric_prefix.encode("ascii"))
    monkeypatch.setattr("merlin_firesim_modelblaster.env", lambda key: str(tmp_path))
    package = ModuleType("modelblaster")
    package.__path__ = []
    validation = ModuleType("modelblaster.validation")
    validation.__path__ = []
    module = ModuleType("modelblaster.validation.firesim_runner")
    module.__file__ = "/other/checkout/firesim_runner.py"

    def fake_run(*args, **kwargs):
        raise AssertionError("must reject before submission")

    fake_run.__module__ = module.__name__
    module.run_firesim = fake_run
    monkeypatch.setitem(sys.modules, "modelblaster", package)
    monkeypatch.setitem(sys.modules, "modelblaster.validation", validation)
    monkeypatch.setitem(sys.modules, "modelblaster.validation.firesim_runner", module)
    with pytest.raises(RuntimeError, match="outside selected MERLIN_MODELBLASTER"):
        runner(str(elf), firesim_root="/selected", firesim_env="/env.sh", timeout=17, queue=True)
