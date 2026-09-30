"""Paired scheduling must validate both arms and keep modeled costs separate."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location(
        "atlas_schedule_pair", repo_root() / "examples/atlas/target/rtlgraph_compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(adapter, tmp_path, monkeypatch):
    compiler = tmp_path / "compiler"
    compiler.write_bytes(b"tool")
    source = tmp_path / "source.S"
    source.write_text("delay 10\necall\n")
    context = {
        "identities": {"compiler": "exact", "selection": "exact"},
        "profile_args": ["--rtl-dma-profile", "exact.profile"],
        "hardware": {"source_ir_sha256": "a" * 64},
        "semantics": {"dma_completion": "explicit-wait"},
        "limitations": [],
    }
    calls, results = [], [0, 0]

    def check(**kwargs):
        calls.append(kwargs)
        output = kwargs["output_dir"]
        output.mkdir()
        (output / "native.S").write_bytes(kwargs["source"].read_bytes())
        return {
            "status": "passed" if results[len(calls) - 1] == 0 else "rejected",
            "returncode": results[len(calls) - 1],
            "modeled_cycles": 20 if len(calls) == 1 else 12,
            "verification": {"schedule_consistency_verified": results[len(calls) - 1] == 0},
            "identities": {
                **context["identities"],
                "source": adapter._identity(kwargs["source"]),
                "native_source": adapter._identity(output / "native.S"),
            },
        }

    checker = SimpleNamespace(prepare_selection=lambda *_: context.copy(), check_schedule=check)
    monkeypatch.setattr(adapter, "_checker", lambda: checker)

    def run(command, **kwargs):
        assert command[1:3] == context["profile_args"]
        assert kwargs["timeout"] == 60
        Path(command[-1]).write_text("delay 2\necall\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="optimized")

    monkeypatch.setattr(adapter.subprocess, "run", run)
    return (
        {
            "source": source,
            "selection": tmp_path / "selection.json",
            "compiler": compiler,
            "output_dir": tmp_path / "paired",
        },
        calls,
        results,
        checker,
    )


@pytest.mark.parametrize("validation", ["static", "dynamic"])
def test_both_arms_same_model_and_scoped_cycles(adapter, setup, validation):
    args, calls, _, _ = setup
    report = adapter.compare_schedules(**args, validation=validation)
    assert report["status"] == "paired_checks_passed"
    assert len(calls) == 2
    for key in ("selection", "compiler", "validation"):
        assert calls[0][key] == calls[1][key]
    assert report["modeled_cost"]["after_minus_before_cycles"] == (-8 if validation == "dynamic" else None)
    assert report["hardware_timing"]["after_cycles"] is None
    assert report["numerical"]["equivalence"] == "UNKNOWN"
    assert report["functional_qualification"] == "NOT_ESTABLISHED"
    assert json.loads((args["output_dir"] / "report.json").read_text()) == report


def test_rejected_baseline_cannot_produce_optimization(adapter, setup, monkeypatch):
    args, calls, results, _ = setup
    results[0] = 1
    monkeypatch.setattr(adapter.subprocess, "run", lambda *_a, **_k: pytest.fail("must not optimize"))
    report = adapter.compare_schedules(**args)
    assert report["status"] == "baseline_rejected"
    assert len(calls) == 1
    assert report["candidate"] is None


def test_rejected_candidate_has_no_cost_delta(adapter, setup):
    args, _, results, _ = setup
    results[1] = 1
    report = adapter.compare_schedules(**args, validation="dynamic")
    assert report["status"] == "candidate_rejected"
    assert report["modeled_cost"]["after_minus_before_cycles"] is None


def test_profile_identity_drift_refuses_pair(adapter, setup):
    args, _, _, checker = setup
    original = checker.prepare_selection()
    contexts = iter([original, {**original, "identities": {"compiler": "changed"}}])
    checker.prepare_selection = lambda *_: next(contexts)
    report = adapter.compare_schedules(**args)
    assert report["status"] == "identity_error"
    assert "identities changed" in report["error"]
    assert (args["output_dir"] / "report.json").is_file()


def test_baseline_mutation_during_optimization_refuses_pair(adapter, setup, monkeypatch):
    args, _, _, _ = setup
    optimize = adapter.subprocess.run

    def mutate(command, **kwargs):
        result = optimize(command, **kwargs)
        (args["output_dir"] / "baseline/native.S").write_text("ecall\n")
        return result

    monkeypatch.setattr(adapter.subprocess, "run", mutate)
    report = adapter.compare_schedules(**args)
    assert report["status"] == "identity_error"
    assert "assembly bytes changed" in report["error"]
    assert report["candidate"] is None


def test_candidate_removed_during_check_preserves_failure_report(adapter, setup):
    args, calls, _, checker = setup
    check = checker.check_schedule

    def remove_candidate(**kwargs):
        report = check(**kwargs)
        if len(calls) == 2:
            kwargs["source"].unlink()
        return report

    checker.check_schedule = remove_candidate
    report = adapter.compare_schedules(**args)
    assert report["status"] == "identity_error"
    assert (args["output_dir"] / "report.json").is_file()


@pytest.mark.parametrize("status,verified", [("error", True), ("passed", False)])
def test_successful_process_cannot_override_invalidated_verdict(adapter, setup, monkeypatch, status, verified):
    args, _, _, checker = setup
    check = checker.check_schedule

    def invalidated(**kwargs):
        return {**check(**kwargs), "status": status, "verification": {"schedule_consistency_verified": verified}}

    checker.check_schedule = invalidated
    monkeypatch.setattr(adapter.subprocess, "run", lambda *_a, **_k: pytest.fail("must not optimize"))
    assert adapter.compare_schedules(**args)["status"] == "baseline_rejected"


@pytest.mark.parametrize("timed_out", [True, False])
def test_execution_failure_has_reviewable_report(adapter, setup, monkeypatch, timed_out):
    args, _, _, _ = setup

    def timeout(command, **kwargs):
        if timed_out:
            raise adapter.subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise PermissionError("selected compiler is not executable")

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    report = adapter.compare_schedules(**args)
    assert report["status"] == "optimization_rejected"
    assert report["optimization"].get("timed_out", False) is timed_out
    assert report["optimization"]["returncode"] is None
    assert (args["output_dir"] / "report.json").is_file()
    assert report["candidate"] is None


@pytest.fixture
def numerical(adapter, setup, monkeypatch):
    args, _, _, _ = setup
    fixture = args["source"].with_name("fixture.json")
    fixture.write_text("{}")
    args.update(model_root=fixture.parent, model_python=fixture, model_fixture=fixture)
    observations = []

    def observe(**kwargs):
        observations.append(kwargs)
        report = {
            "status": "passed", "numerical_equivalence": "PASSED_SUPPLIED_REFERENCE",
            "model_ticks": {"before": 30, "after": 25, "after_minus_before": -5},
            "identities": {key: adapter._identity(kwargs[key]) for key in ("before", "after", "fixture")},
        }
        kwargs["output_dir"].mkdir()
        (kwargs["output_dir"] / "report.json").write_text(json.dumps(report))
        return report

    observer = SimpleNamespace(observe_pair=observe)
    monkeypatch.setattr(adapter, "_observer", lambda: observer)
    return observations, observer


def test_independent_numerical_observation_keeps_timing_domains_separate(adapter, setup, numerical):
    args, _, _, _ = setup
    observations, _ = numerical
    report = adapter.compare_schedules(**args, validation="dynamic")
    assert report["status"] == "paired_checks_passed"
    assert len(observations) == 1
    assert observations[0]["before"] == args["output_dir"] / "baseline/native.S"
    assert observations[0]["after"] == args["output_dir"] / "candidate.S"
    assert report["numerical"]["equivalence"] == "PASSED_SUPPLIED_REFERENCE"
    assert report["numerical"]["model_ticks"]["after_minus_before"] == -5
    assert report["modeled_cost"]["after_minus_before_cycles"] == -8
    assert report["hardware_timing"]["status"] == "UNMEASURED"
    assert report["functional_qualification"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("status,equivalence", [("error", "UNKNOWN"), ("passed", "UNKNOWN")])
def test_numerical_failure_cannot_report_success(adapter, setup, numerical, status, equivalence):
    args, _, _, _ = setup
    _, observer = numerical
    observe = observer.observe_pair

    def failed(**kwargs):
        return {**observe(**kwargs), "status": status, "numerical_equivalence": equivalence, "error": "wrong output"}

    observer.observe_pair = failed
    report = adapter.compare_schedules(**args)
    assert report["status"] == "numerical_rejected"
    assert report["numerical"]["error"] == "wrong output"


def test_rejected_schedule_does_not_execute_numerical_observer(adapter, setup, numerical):
    args, _, results, _ = setup
    observations, _ = numerical
    results[1] = 1
    report = adapter.compare_schedules(**args)
    assert report["status"] == "candidate_rejected"
    assert not observations


@pytest.mark.parametrize("drift", ["before", "after", "fixture"])
def test_numerical_report_must_bind_checked_bytes(adapter, setup, numerical, drift):
    args, _, _, _ = setup
    _, observer = numerical
    observe = observer.observe_pair

    def wrong_identity(**kwargs):
        report = observe(**kwargs)
        report["identities"][drift]["sha256"] = "0" * 64
        return report

    observer.observe_pair = wrong_identity
    report = adapter.compare_schedules(**args)
    assert report["status"] == "numerical_rejected"
    assert report["numerical"]["equivalence"] == "UNKNOWN"


def test_fixture_mutation_during_scheduling_prevents_observation(adapter, setup, numerical, monkeypatch):
    args, _, _, _ = setup
    observations, _ = numerical
    optimize = adapter.subprocess.run

    def mutate(*arguments, **kwargs):
        result = optimize(*arguments, **kwargs)
        args["model_fixture"].write_text('{"changed":true}')
        return result

    monkeypatch.setattr(adapter.subprocess, "run", mutate)
    report = adapter.compare_schedules(**args)
    assert report["status"] == "numerical_rejected"
    assert not observations


@pytest.mark.parametrize("missing", ["model_root", "model_python", "model_fixture"])
def test_numerical_selection_requires_all_inputs(adapter, setup, numerical, missing):
    args, calls, _, _ = setup
    args.pop(missing)
    with pytest.raises(ValueError, match="supplied together"):
        adapter.compare_schedules(**args)
    assert not calls
