"""A headline RTL binding checks saved program, consoles and selected facts."""

import hashlib
import json

import pytest

from merlin.common.paths import repo_root


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def test_saved_headline_binding_refuses_changed_console_or_facts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/target"))
    from bind_native_simulator import checked_headline_artifacts

    root = tmp_path / "headline"
    root.mkdir()
    native = root / "native"
    native.mkdir()
    elf = native / "package_kernel.elf"
    elf.write_bytes(b"exact executable")
    spike = root / "spike_console.log"
    rtl = root / "verilator_console.log"
    spike.write_text("Spike output")
    rtl.write_text("Verilator output")
    capsule = root / "capsule.yaml"
    interface = root / "capsule.interface.mlir"
    capsule.write_text("capsule")
    interface.write_text("interface")
    source_window = {"model_equivalence_claim": "none_synthetic_operands"}
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    source_sha = "a" * 64
    facts = _write(facts_root / "facts.json", {"inputs": {"source_bundle_sha256": source_sha}})
    _write(
        facts_root / "validation.json",
        {"status": "verified", "facts_sha256": _sha(facts), "source_selection_sha256": source_sha},
    )
    generation = _write(
        root / "generation.json",
        {
            "schema": "merlin.gemmini-headline-window-generation.v1",
            "status": "generated_not_executed",
            "source_kernel_window": source_window,
            "facts_sha256": _sha(facts),
            "capsule_sha256": _sha(capsule),
            "interface_sha256": _sha(interface),
        },
    )
    receipt = root / "numerical_receipt.json"
    doc = {
        "status": "spike_and_verilator_passed",
        "source_model_equivalence_claim": "none_synthetic_operands",
        "diagnostic_limits": {
            "synthetic_integer_operands": True,
            "headline_numerical_equivalence": False,
            "target_admission": False,
        },
        "evidence_roots": {"output_root": str(root)},
        "source_kernel_window": source_window,
        "generated_evidence": {
            "generation_sha256": _sha(generation),
            "capsule_sha256": _sha(capsule),
            "interface_sha256": _sha(interface),
        },
        "elf_sha256": _sha(elf),
        "same_elf_sha256": _sha(elf),
        "spike_console_sha256": _sha(spike),
        "verilator_console_sha256": _sha(rtl),
    }
    _write(receipt, doc)
    assert checked_headline_artifacts(receipt, doc, facts_evidence=facts_root, source_sha=source_sha) == 1
    rtl.write_text("changed console")
    with pytest.raises(ValueError, match="Verilator console changed"):
        checked_headline_artifacts(receipt, doc, facts_evidence=facts_root, source_sha=source_sha)
    rtl.write_text("Verilator output")
    with pytest.raises(ValueError, match="facts do not bind"):
        checked_headline_artifacts(receipt, doc, facts_evidence=facts_root, source_sha="b" * 64)
