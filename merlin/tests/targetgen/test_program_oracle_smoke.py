"""The pre-flight END-TO-END oracle smoke for a self-hosted-ISA (external_backend) target: run a
KNOWN-GOOD, self-contained model program through the FULL grading path (assemble -> arc cosim -> readback)
and require a BIT-EXACT match to its own golden. This proves the oracle grades to a CORRECT verdict before a
paid run, not merely that ``arc_available`` is True (the atlas 0/11-at-$43 lesson).

Hermetic assertions (no venv/mlc): the external_backend target DECLARES a known-good smoke program (so the
name is a per-target SETUP fact, never a library literal), the loader defaults it to None when unset, and
the smoke FAILS CLOSED (raises OracleUnavailable) when the program ships no golden — never a silent pass.
A gated end-to-end assertion actually runs the smoke and checks the bit-exact verdict, skipping cleanly when
the model venv / cosim is absent.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import program_oracle as PO
from merlin.targetgen.target_experiment import load_target_experiment


def test_model_emitter_keeps_relative_output_under_the_callers_run_root(monkeypatch, tmp_path):
    """Changing cwd for an external model must not redirect generated artifacts into its checkout."""
    caller, model = tmp_path / "caller", tmp_path / "model"
    caller.mkdir()
    model.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(PO, "_program_emitter", lambda _: (model / "emit.py", []))
    monkeypatch.setattr(PO, "_model_venv_python", lambda _: model / "python")
    monkeypatch.setattr(PO, "ext_path", lambda _: model)

    def run(argv, *, cwd, **kwargs):
        output = Path(argv[argv.index("--out") + 1])
        assert output.is_absolute() and output.parent == caller / "run"
        assert Path(cwd) == model and output.parent.is_dir()
        output.write_text(json.dumps({"words": [0], "inputs": []}))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(PO.subprocess, "run", run)
    result = PO.emit_bundle(
        target="fixture_device",
        model_ext="fixture_model",
        program="Smoke",
        workdir=Path("run"),
        timeout=1,
    )
    assert result["words"] == [0]


def _external_backend_descriptor() -> Path | None:
    """The first capsule-bench descriptor whose endpoint is external_backend (self-hosted ISA) — found by
    the DERIVED endpoint kind, so no target name is hardcoded here."""
    root = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    if not root.is_dir():
        return None
    for desc in sorted(root.glob("*/target_experiment.yaml")):
        try:
            te = load_target_experiment(desc)
            if CR._endpoint_of(te.target)[0] == "external_backend":
                return desc
        except Exception:  # noqa: BLE001 — an unresolvable descriptor: keep looking
            continue
    return None


def test_external_backend_target_declares_a_known_good_smoke_program():
    """A self-hosted-ISA target must declare the known-good program the pre-flight smokes end-to-end — the
    name is DECLARED per-target setup (fail-closed: a missing declaration is a NO_GO, not a guessed name)."""
    desc = _external_backend_descriptor()
    if desc is None:
        pytest.skip("no external_backend target descriptor present")
    te = load_target_experiment(desc)
    assert te.preflight_smoke_program, (
        f"{te.target}: external_backend target declares no preflight.smoke_program — the pre-flight cannot "
        f"run an end-to-end oracle smoke and would have to guess a program (forbidden)"
    )


def test_loader_defaults_smoke_program_to_none_when_unset():
    """A descriptor with no ``preflight`` block loads ``preflight_smoke_program is None`` (opt-in) — so a
    target that omits it is honestly None, and the pre-flight fails closed rather than fabricating one."""
    gem = merlin_dir() / "experiments" / "capsule_bench" / "targets" / "gemmini" / "target_experiment.yaml"
    if not gem.is_file():
        pytest.skip("gemmini descriptor absent")
    assert load_target_experiment(gem).preflight_smoke_program is None


def test_smoke_takes_program_as_a_parameter_no_literal():
    """The reusable smoke is target-agnostic: ``program`` is a keyword PARAMETER (the concrete known-good
    name lives in the descriptor, not baked into the library)."""
    import inspect

    sig = inspect.signature(PO.run_program_oracle_smoke)
    assert "program" in sig.parameters and "target" in sig.parameters and "model_ext" in sig.parameters


def test_smoke_fails_closed_when_program_ships_no_golden(monkeypatch):
    """If the model program ships no ``golden_result``, there is nothing to compare bit-exact against — the
    smoke must raise OracleUnavailable (a NO_GO the caller surfaces), never return a silent ``ok``."""
    monkeypatch.setattr(PO, "emit_bundle", lambda **kw: {"words": [0], "inputs": [], "output": None, "golden": None})
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(PO.OracleUnavailable):
            PO.run_program_oracle_smoke(
                "any_target", model_ext="whatever", program="NoGoldenProgram", workdir=Path(td), timeout=5
            )


def test_functional_oracle_rejects_a_halted_program_with_incomplete_decode(monkeypatch, tmp_path):
    """A runner may substitute NOPs for unsupported words; halting is not proof of execution."""
    monkeypatch.setattr(PO, "_func_program_helper", lambda _: tmp_path / "runner.py")
    monkeypatch.setattr(PO, "emit_bundle", lambda **_: {"words": [0xFFFFFFFF]})
    monkeypatch.setattr(PO, "_bundle_preload", lambda *_: [])
    monkeypatch.setattr(
        PO,
        "_resolve_out_specs",
        lambda *_: {"Y0": {"base": 0, "shape": [1], "dtype": "i8", "physical": None}},
    )
    base_result = {"halted": True, "outputs": {"0": "Kg=="}}

    for decode_evidence, reason in (
        ({}, "did not report instruction decode coverage"),
        ({"unsupported": [{"index": 0, "word": 0xFFFFFFFF}]}, "substituted 1 unsupported"),
    ):
        monkeypatch.setattr(PO, "_run_func_helper", lambda *_: base_result | decode_evidence)
        with pytest.raises(PO.OracleUnavailable, match=reason):
            PO.run_program_functional_oracle(
                "fixture", model_ext="fixture", program="fixture_program", workdir=tmp_path
            )

    monkeypatch.setattr(PO, "_run_func_helper", lambda *_: base_result | {"unsupported": []})
    accepted = PO.run_program_functional_oracle(
        "fixture", model_ext="fixture", program="fixture_program", workdir=tmp_path
    )
    assert accepted["outputs"] == {"Y0": [42]}


def test_functional_debugger_refuses_substituted_instruction_trace(monkeypatch, tmp_path):
    """The agent must not inspect plausible state from a NOP-substituted program."""
    from merlin.targetgen import dram_facts

    monkeypatch.setattr(PO, "_func_program_helper", lambda _: tmp_path / "runner.py")
    monkeypatch.setattr(PO, "emit_bundle", lambda **_: {"words": [0xFFFFFFFF]})
    monkeypatch.setattr(PO, "_bundle_preload", lambda *_: [])
    monkeypatch.setattr(PO, "_resolve_out_spec", lambda *_: {"base": 0, "shape": [1], "dtype": "i8"})
    monkeypatch.setattr(dram_facts, "dram_base_for", lambda _: 0)
    result = {"halted": True, "pc": 4, "regs": [0], "dram_dumps": {}}
    kernel = tmp_path / "kernel.S"

    for decode_evidence, reason in (
        ({}, "did not report instruction decode coverage"),
        ({"unsupported": [{"index": 0, "word": 0xFFFFFFFF}]}, "substituted 1 unsupported"),
    ):
        monkeypatch.setattr(PO, "_run_func_helper", lambda *_: result | decode_evidence)
        with pytest.raises(PO.OracleUnavailable, match=reason):
            PO.run_program_debug(
                "fixture", model_ext="fixture", cb={}, kernel_s=kernel,
                dump_regions=[], workdir=tmp_path,
            )

    monkeypatch.setattr(PO, "_run_func_helper", lambda *_: result | {"unsupported": []})
    accepted = PO.run_program_debug(
        "fixture", model_ext="fixture", cb={}, kernel_s=kernel,
        dump_regions=[], workdir=tmp_path,
    )
    assert accepted["unsupported"] == [] and accepted["pc"] == 4


def test_program_oracle_smoke_bit_exact_end_to_end():
    """Gated: run the descriptor-declared known-good program through the full assemble -> arc -> readback
    path and require a bit-exact match to its own golden. Skips cleanly when the model venv / cosim is
    absent (CI without the heavy deps), so it never blocks the hermetic suite."""
    from merlin.targetgen.rtl import mlc_bridge

    desc = _external_backend_descriptor()
    if desc is None:
        pytest.skip("no external_backend target descriptor present")
    te = load_target_experiment(desc)
    target = te.target
    if not mlc_bridge.arc_available(target):
        pytest.skip(f"mlc arc cosim for {target!r} unavailable (MERLIN_MLC_DIR + built arc model)")
    program = te.preflight_smoke_program
    _, model_ext = CR._endpoint_of(target)
    if not (program and model_ext):
        pytest.skip("descriptor/contract does not declare both a smoke_program and a model_ext")

    try:
        with tempfile.TemporaryDirectory() as td:
            r = PO.run_program_oracle_smoke(target, model_ext=model_ext, program=program, workdir=Path(td), timeout=600)
    except PO.OracleUnavailable as e:
        pytest.skip(f"program oracle infra unavailable: {e}")

    assert r["ok"] is True, f"end-to-end smoke did not grade bit-exact: {r['reason']}"
    assert r["mismatches"] == 0
    assert r["program"] == program
    assert isinstance(r["cycles"], int) and r["cycles"] > 0
