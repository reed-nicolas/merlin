"""Opt-in paired Atlas schedule checks; modeled costs do not qualify hardware timing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


def _module(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checker():
    return _module("rtlgraph_check.py")


def _observer():
    return _module("rtlgraph_model.py")


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare_schedules(
    *,
    source: Path,
    selection: Path,
    compiler: Path,
    output_dir: Path,
    assembly="atlas-opt-native",
    assembler=None,
    validation="static",
    timeout=60,
    model_root=None,
    model_python=None,
    model_fixture=None,
    model_timeout=120,
):
    """Check a baseline, schedule it, then check the candidate with the same model.

    Static mode intentionally supplies no cycle comparison. Dynamic costs come from the
    compiler's scheduling simulator, which does not execute tensor arithmetic or RTL.
    """
    if validation not in {"static", "dynamic"}:
        raise ValueError("validation must be static or dynamic")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    numerical_requested = any(value is not None for value in (model_root, model_python, model_fixture))
    if numerical_requested and any(value is None for value in (model_root, model_python, model_fixture)):
        raise ValueError("model_root, model_python and model_fixture must be supplied together")
    if model_timeout <= 0:
        raise ValueError("model_timeout must be positive")
    fixture_identity = _identity(model_fixture) if numerical_requested else None
    checker = _checker()
    context = checker.prepare_selection(selection, compiler)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "merlin.atlas_schedule_pair.v1",
        "status": "baseline_rejected",
        "validation": validation,
        "identities": context["identities"],
        "hardware": context["hardware"],
        "semantics": context["semantics"],
        "baseline": None,
        "candidate": None,
        "optimization": None,
        "modeled_cost": {
            "scope": "same selected partial scheduling model; DMA cycles are cost estimates",
            "before_cycles": None,
            "after_cycles": None,
            "after_minus_before_cycles": None,
        },
        "numerical": {"status": "NOT_EXECUTED", "equivalence": "UNKNOWN"},
        "hardware_timing": {"status": "UNMEASURED", "before_cycles": None, "after_cycles": None},
        "functional_qualification": "NOT_ESTABLISHED",
        "limitations": [
            *context["limitations"],
            "Schedule checks share the compiler's scheduling model and do not evaluate tensor arithmetic.",
            "Independent numerical observation is limited to the supplied fixture; no RTL was executed."
            if numerical_requested else "No runtime tensor inputs, numerical observer, or RTL execution were used.",
        ],
    }

    def finish():
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report

    def passed(arm):
        return (
            arm.get("status") == "passed"
            and arm.get("returncode") == 0
            and arm.get("verification", {}).get("schedule_consistency_verified") is True
        )

    def unchanged(*arms):
        try:
            if checker.prepare_selection(selection, compiler) != context:
                raise ValueError("selected checker identities changed during paired evaluation")
            for arm in arms:
                identities = arm["identities"]
                for key, expected in context["identities"].items():
                    if identities.get(key) != expected:
                        raise ValueError("checker report identities differ from selected pair")
                for key in ("source", "native_source"):
                    if _identity(identities[key]["path"]) != identities[key]:
                        raise ValueError("paired assembly bytes changed during evaluation")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            report["status"] = "identity_error"
            report["error"] = str(error)
            return False
        return True

    baseline = checker.check_schedule(
        source=source,
        selection=selection,
        compiler=compiler,
        output_dir=output_dir / "baseline",
        assembly=assembly,
        assembler=assembler,
        validation=validation,
        timeout=timeout,
    )
    report["baseline"] = baseline
    if not passed(baseline):
        return finish()
    if not unchanged(baseline):
        return finish()
    before = output_dir / "baseline" / "native.S"
    candidate = output_dir / "candidate.S"
    command = [
        str(Path(compiler).resolve(strict=True)),
        *context["profile_args"],
        "--validation",
        validation,
        "--dma-timing",
        "robust",
        str(before),
        "-o",
        str(candidate),
    ]
    report["status"] = "optimization_rejected"
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        report["optimization"] = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as error:
        report["optimization"] = {"command": command, "returncode": None, "timed_out": True, "stderr": str(error)}
        return finish()
    except OSError as error:
        report["optimization"] = {"command": command, "returncode": None, "stderr": str(error)}
        return finish()
    # A concurrently replaced tool or selection cannot silently join two different models.
    if not unchanged(baseline):
        return finish()
    if completed.returncode != 0 or not candidate.is_file():
        return finish()
    checked = checker.check_schedule(
        source=candidate,
        selection=selection,
        compiler=compiler,
        output_dir=output_dir / "candidate",
        assembler=assembler,
        validation=validation,
        timeout=timeout,
    )
    report["candidate"] = checked
    if not unchanged(baseline, checked):
        return finish()
    try:
        report["candidate_source"] = _identity(candidate)
    except OSError as error:
        report["status"] = "identity_error"
        report["error"] = str(error)
        return finish()
    if report["candidate_source"] != checked["identities"]["source"]:
        report["status"] = "identity_error"
        report["error"] = "candidate bytes differ from checker source identity"
        return finish()
    report["status"] = "paired_checks_passed" if passed(checked) else "candidate_rejected"
    if report["status"] == "paired_checks_passed" and validation == "dynamic":
        before_cycles, after_cycles = baseline["modeled_cycles"], checked["modeled_cycles"]
        if type(before_cycles) is int and type(after_cycles) is int:
            report["modeled_cost"].update(
                before_cycles=before_cycles,
                after_cycles=after_cycles,
                after_minus_before_cycles=after_cycles - before_cycles,
            )
    if report["status"] == "paired_checks_passed" and numerical_requested:
        try:
            if _identity(model_fixture) != fixture_identity:
                raise ValueError("numerical fixture changed during scheduling")
            observation = _observer().observe_pair(
                before=before, after=candidate, fixture=model_fixture,
                model_root=model_root, python=model_python, assembler=assembler,
                output_dir=output_dir / "numerical", timeout=model_timeout,
            )
            report["numerical"] = {
                "status": observation["status"],
                "equivalence": observation["numerical_equivalence"],
                "model_ticks": observation["model_ticks"],
                "report": _identity(output_dir / "numerical/report.json"),
            }
            if observation["status"] != "passed" or observation["numerical_equivalence"] != "PASSED_SUPPLIED_REFERENCE":
                report["status"] = "numerical_rejected"
                report["numerical"]["error"] = observation.get("error", "numerical comparison failed")
                return finish()
            expected = {"before": _identity(before), "after": report["candidate_source"], "fixture": fixture_identity}
            if any(observation["identities"].get(key) != value for key, value in expected.items()):
                raise ValueError("numerical observation identities differ from checked pair")
            if _identity(model_fixture) != fixture_identity:
                raise ValueError("numerical fixture changed during observation")
            if not unchanged(baseline, checked):
                return finish()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            report["status"] = "numerical_rejected"
            report["numerical"].update(status="error", equivalence="UNKNOWN", error=str(error))
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assembly", choices=["atlas-opt-native", "merlin-atlas"], default="atlas-opt-native")
    parser.add_argument("--assembler", type=Path)
    parser.add_argument("--validation", choices=["static", "dynamic"], default="static")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--model-fixture", type=Path)
    parser.add_argument("--model-timeout", type=float, default=120)
    args = parser.parse_args()
    report = compare_schedules(
        source=args.kernel,
        selection=args.selection,
        compiler=args.compiler,
        output_dir=args.output,
        assembly=args.assembly,
        assembler=args.assembler,
        validation=args.validation,
        timeout=args.timeout,
        model_root=args.model_root,
        model_python=args.model_python,
        model_fixture=args.model_fixture,
        model_timeout=args.model_timeout,
    )
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "paired_checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
