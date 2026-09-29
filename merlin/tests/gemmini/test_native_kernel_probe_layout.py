"""The target-owned numerical probe consumes the frozen controller's corpus layout."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def _probe():
    path = repo_root() / "examples/gemmini/verification/probe_native_kernel.py"
    spec = importlib.util.spec_from_file_location("gemmini_native_kernel_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_frozen_phase0_manifest_is_selected(tmp_path):
    probe = _probe()
    corpus = tmp_path / "phase0/capsules"
    corpus.mkdir(parents=True)
    manifest = tmp_path / "phase0/evidence-manifest.json"
    manifest.write_text("{}")
    assert probe.phase0_manifest_path(corpus) == manifest


def test_legacy_manifest_remains_selectable_but_ambiguous_inputs_fail(tmp_path):
    probe = _probe()
    corpus = tmp_path / "phase0/capsules"
    legacy = corpus / "_evidence/evidence-manifest.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")
    assert probe.phase0_manifest_path(corpus) == legacy

    current = corpus.parent / "evidence-manifest.json"
    current.write_text("{}")
    with pytest.raises(ValueError, match="ambiguous"):
        probe.phase0_manifest_path(corpus)


def test_two_engine_receipt_refuses_missing_rtl():
    probe = _probe()
    backend = SimpleNamespace(available=lambda simulator: simulator == "spike")
    with pytest.raises(RuntimeError, match="verilator unavailable"):
        probe.require_two_engine_backend(backend)
    backend.available = lambda _simulator: True
    probe.require_two_engine_backend(backend)
