"""Preflight checks for a generated Gemmini kernel probe's source evidence."""

import importlib.util
import json

import pytest

from merlin.common.paths import repo_root


def load_probe():
    source = repo_root() / "examples/gemmini/verification/probe_native_kernel.py"
    spec = importlib.util.spec_from_file_location("gemmini_native_kernel_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path):
    probe = load_probe()
    source = tmp_path / "source"
    facts = tmp_path / "facts"
    source.mkdir()
    facts.mkdir()
    selection = {"hierarchy_correspondence": {"status": "verified"}, "sources": {"core_hw": {}}}
    core_hw = source / "core.hw.mlir"
    core_hw.write_text("module {}", encoding="utf-8")
    selection["sources"]["core_hw"]["sha256"] = probe.sha(core_hw)
    write_json(source / "source-selection.json", selection)
    (facts / "facts.json").write_text("{}", encoding="utf-8")
    selection_sha = probe.sha(source / "source-selection.json")
    facts_sha = probe.sha(facts / "facts.json")
    write_json(
        facts / "validation.json",
        {"status": "verified", "facts_sha256": facts_sha, "source_selection_sha256": selection_sha},
    )
    manifest = {
        "raw_facts_sha256": facts_sha,
        "artifacts": {"selection": {"sha256": selection_sha}},
        "sources": [
            {"role": "rtl-facts", "sha256": facts_sha},
            {"role": "rtl-source:source_bundle_path", "sha256": selection_sha},
            {"role": "rtl-source:core_hw_path", "sha256": probe.sha(core_hw)},
        ],
    }
    return probe, source, facts, manifest


def test_split_facts_bind_exact_selected_source(tmp_path):
    probe, source, facts, manifest = fixture(tmp_path)
    binding = probe.checked_source_binding(manifest, source, facts)
    assert binding["facts"]["path"] == str(facts / "facts.json")
    assert binding["selection"]["path"] == str(source / "source-selection.json")
    assert binding["validation_status"] == "verified"


@pytest.mark.parametrize("change", ["missing", "mismatch"])
def test_split_facts_reject_missing_or_mismatched_facts(tmp_path, change):
    probe, source, facts, manifest = fixture(tmp_path)
    if change == "missing":
        (facts / "facts.json").unlink()
        with pytest.raises(FileNotFoundError, match="facts.json"):
            probe.checked_source_binding(manifest, source, facts)
    else:
        (facts / "facts.json").write_text('{"changed": true}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="generated corpus does not bind"):
            probe.checked_source_binding(manifest, source, facts)


def test_flat_source_bundle_remains_supported(tmp_path):
    probe, source, facts, manifest = fixture(tmp_path)
    for name in ("facts.json", "validation.json"):
        (source / name).write_bytes((facts / name).read_bytes())
    assert probe.checked_source_binding(manifest, source)["facts"]["path"] == str(source / "facts.json")


@pytest.mark.parametrize("change", ["software_spec", "support_contract"])
def test_split_evidence_rejects_changed_declared_inputs(tmp_path, change):
    probe, _, _, manifest = fixture(tmp_path)
    software_spec = tmp_path / "software-spec.yaml"
    support_contract = tmp_path / "target_contract.yaml"
    software_spec.write_text("software: selected", encoding="utf-8")
    support_contract.write_text("support: selected", encoding="utf-8")
    manifest["sources"].extend(
        [
            {"role": "software-spec", "source": str(software_spec), "sha256": probe.sha(software_spec)},
            {"role": "support-source", "source": str(support_contract), "sha256": probe.sha(support_contract)},
        ]
    )
    assert probe.checked_declared_inputs(manifest, software_spec, support_contract)
    changed = software_spec if change == "software_spec" else support_contract
    changed.write_text("changed: true", encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"does not bind {change} bytes"):
        probe.checked_declared_inputs(manifest, software_spec, support_contract)
