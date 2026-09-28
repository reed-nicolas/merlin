"""A missing selected PT2E integerizer refuses generation before any capsule is written."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import generation


def _generation_with_entries(monkeypatch, tmp_path, entries):
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: fixture\n")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("capsules: []\n")
    upstream = tmp_path / "selected-model2mlir"
    (upstream / "m2m").mkdir(parents=True)
    (upstream / "m2m/__init__.py").write_text("# selected checkout\n")
    monkeypatch.setenv("MERLIN_MODEL2MLIR", str(upstream))
    monkeypatch.setenv("MERLIN_M2M_PYTHON", sys.executable)
    monkeypatch.setattr(generation, "validate_profile_inputs", lambda **kwargs: None)
    monkeypatch.setattr(generation, "_ensure_contract_on_path", lambda descriptor: None)
    monkeypatch.setattr(
        generation, "load_target_experiment", lambda descriptor: SimpleNamespace(target="fixture", workload_spec={})
    )
    monkeypatch.setattr(
        generation,
        "load_profile",
        lambda *args, **kwargs: {"capsules": entries, "datapath": {}, "_performance_template": {}},
    )
    monkeypatch.setattr(
        generation.CS,
        "derive_binding",
        lambda *args, **kwargs: SimpleNamespace(
            target="fixture", operand_dtype="int8", accum_dtype="i32", integer=True
        ),
    )
    monkeypatch.setattr(generation, "_performance_facts", lambda *args, **kwargs: {"sha256": "0" * 64})
    monkeypatch.setattr(generation, "expand_sweeps", lambda profile, *args, **kwargs: list(profile["capsules"]))
    monkeypatch.setattr(generation, "_resolve_flat_extents", lambda entry, binding: entry)
    monkeypatch.setattr(generation, "update_provenance_manifest", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(generation, "_write_capsule", lambda entry, *args, **kwargs: calls.append(entry["name"]))

    def run(**kwargs):
        return generation.generate_target(
            "fixture", descriptor=descriptor, recipe=recipe, output_root=tmp_path / "capsules", **kwargs
        )

    return run, calls, upstream


@pytest.mark.parametrize(
    "capture",
    [
        {"quant_recipe": {"activation": {"dtype": "int8", "mode": "static"}, "weight": {"dtype": "int8"}}},
        {"quant_scheme": "int8_static_act_int8_weight"},
    ],
)
def test_static_int8_model_missing_selected_integerizer_fails_before_all_writers(monkeypatch, tmp_path, capture):
    entries = [{"name": f"ordinary_{index}", "kind": "isa"} for index in range(84)]
    entries.append({"name": "last_static_model", "kind": "model", **capture})
    run, calls, upstream = _generation_with_entries(monkeypatch, tmp_path, entries)

    with pytest.raises(ValueError, match="pt2e_integerize.py") as error:
        run()

    assert str(upstream) in str(error.value)
    assert calls == []


def test_static_int8_model_accepts_selected_integerizer(monkeypatch, tmp_path):
    entry = {"name": "static_model", "kind": "model", "quant_scheme": "int8_static_act_int8_weight"}
    run, calls, upstream = _generation_with_entries(monkeypatch, tmp_path, [entry])
    integerizer = upstream / "m2m/capture/pt2e_integerize.py"
    integerizer.parent.mkdir(parents=True)
    integerizer.write_text("def integerize_pt2e(*args): pass\n")

    assert run() == []
    assert calls == ["static_model"]


def test_missing_capture_interpreter_keeps_optional_model_skip(monkeypatch, tmp_path):
    entry = {"name": "static_model", "kind": "model", "quant_scheme": "int8_static_act_int8_weight"}
    run, calls, _ = _generation_with_entries(monkeypatch, tmp_path, [entry])
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(tmp_path / "missing-python"))

    assert run() == []
    assert calls == ["static_model"]


def test_missing_frozen_recipe_keeps_roster_failure_in_writer(monkeypatch, tmp_path):
    entry = {
        "name": "roster_model",
        "kind": "model",
        "generalization": {"generalization_axis": "roster"},
    }
    binding = SimpleNamespace(target="fixture", operand_dtype="int8", accum_dtype="i32", integer=True)

    def missing_recipe(*args, **kwargs):
        raise ValueError("frozen capture recipe is absent")

    monkeypatch.setattr(generation, "_selected_capture_recipe", missing_recipe)
    assert (
        generation._prepare_model_capture_entry(
            entry, evidence_root=tmp_path, target="fixture", binding=binding
        )
        is entry
    )
    with pytest.raises(ValueError, match="frozen capture recipe is absent"):
        generation._with_selected_model_recipe(entry, evidence_root=tmp_path, target="fixture", binding=binding)


@pytest.mark.parametrize(
    "model",
    [
        {
            "kind": "model",
            "quant_recipe": {"activation": {"dtype": "int8", "mode": "dynamic"}, "weight": {"dtype": "int8"}},
        },
        {
            "kind": "model",
            "quant_recipe": {"activation": {"dtype": "int8", "mode": "dynamic"}, "weight": {"dtype": "int8"}},
            "quant_scheme": "int8_static_act_int8_weight",
        },
        {
            "kind": "model",
            "materialized_capture": {"path": "captured/model.mlir"},
            "quant_scheme": "int8_static_act_int8_weight",
        },
        {
            "kind": "model",
            "quant_recipe": {
                "activation": {"dtype": "fp8_e4m3", "mode": "static"},
                "weight": {"dtype": "fp8_e4m3"},
            },
        },
    ],
)
def test_other_model_routes_do_not_require_pt2e_integerizer(monkeypatch, tmp_path, model):
    run, calls, _ = _generation_with_entries(monkeypatch, tmp_path, [{"name": "other_model", **model}])

    assert run() == []
    assert calls == ["other_model"]


def test_generation_receipt_keeps_worker_exception_line(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import coverage_commitment, evidence, phase2_guards

    run, _, _ = _generation_with_entries(monkeypatch, tmp_path, [{"name": "failed_member", "kind": "isa"}])
    software = tmp_path / "software.yaml"
    software.write_text("software: selected\n")
    selected = SimpleNamespace(
        status="verified",
        software_spec=None,
        contract={},
        loaded_facts={},
        refreshed_facts={},
        isa_taxonomy={},
        raw_facts_sha256=None,
        source_snapshots=[],
    )
    monkeypatch.setattr(evidence, "select_evidence", lambda *args, **kwargs: selected)

    def export(_selected, root):
        coverage = root / "coverage"
        coverage.mkdir(parents=True)
        (coverage / "operation-accounting.json").write_text("{}\n")
        return {"sources": []}

    monkeypatch.setattr(evidence, "export_evidence", export)

    def manifest(_written, *, cap_root, **kwargs):
        (cap_root / "MANIFEST.yaml").write_text("phase_corpora: {}\n")

    monkeypatch.setattr(generation, "update_provenance_manifest", manifest)
    monkeypatch.setattr(coverage_commitment, "selected_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(coverage_commitment, "write_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        coverage_commitment,
        "observe_cohort",
        lambda *args, **kwargs: {"status": "unverified", "cohort": {"n_capsules": 0}},
    )
    monkeypatch.setattr(
        phase2_guards, "build_guard_link", lambda *args, **kwargs: {"status": "unverified", "guards": []}
    )
    private_path = str(tmp_path / "private" / "capture.py")
    worker_error = (
        "m2m capture failed for op 'model'/int8: worker exited non-zero (rc=1)\n"
        "--- last traceback ---\n"
        + f'  File "{private_path}", line 1, in capture\n    capture()\n' * 30
        + "ModuleNotFoundError: No module named 'm2m.capture.pt2e_integerize'\n"
    )

    def failed_writer(*args, **kwargs):
        raise RuntimeError(worker_error)

    monkeypatch.setattr(generation, "_write_capsule", failed_writer)
    with pytest.raises(RuntimeError, match="1 capsule"):
        run(software_spec=software)

    receipt = json.loads((tmp_path / "capsules/_evidence/coverage/generation.json").read_text())
    reason = receipt["failures"][0]["reason"]
    assert "ModuleNotFoundError: No module named 'm2m.capture.pt2e_integerize'" in reason
    assert private_path not in reason
    assert len(reason) <= 400
