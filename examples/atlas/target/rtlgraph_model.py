"""Run paired native Atlas assembly in an explicitly selected independent Python model.

The supplied byte reference defines a finite numerical witness, not RTL qualification.
Each arm runs in its own process and fresh model state. Observed ticks are model ticks.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _require(value, message):
    if not value:
        raise ValueError(message)


def model_sources(model_root):
    package = Path(model_root).resolve(strict=True) / "npu_model"
    _require(package.is_dir(), "Selected root must contain npu_model")
    result = {}
    for path in sorted(package.rglob("*.py")):
        _require(path.resolve().is_relative_to(package.resolve()), "Model source symlink escapes selected package")
        result[path.relative_to(package).as_posix()] = identity(path)["sha256"]
    _require("__init__.py" in result and "simulation.py" in result, "Incomplete selected model package")
    return result


def snapshot_fixture(fixture, output_dir):
    """Verify caller-supplied raw input/reference bytes and retain an execution snapshot."""
    fixture = Path(fixture).resolve(strict=True)
    document = json.loads(fixture.read_bytes())
    _require(isinstance(document, dict), "Numerical fixture must be an object")
    _require(document.get("schema") == "merlin.atlas_model_fixture.v1", "Unsupported numerical fixture")
    _require(
        set(document) == {"schema", "dram_size", "max_cycles", "inputs", "outputs", "reference_provenance"},
        "Unexpected fixture fields",
    )
    for key in ("dram_size", "max_cycles"):
        _require(type(document[key]) is int and document[key] > 0, f"{key} must be a positive integer")
    _require(
        isinstance(document["reference_provenance"], str) and document["reference_provenance"].strip(),
        "Reference provenance is required",
    )
    _require(isinstance(document["inputs"], list), "Invalid input regions")
    _require(isinstance(document["outputs"], list) and document["outputs"], "At least one reference output is required")
    members = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    def member(record):
        _require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Invalid fixture member")
        _require(
            isinstance(record["path"], str) and isinstance(record["sha256"], str), "Invalid fixture member identity"
        )
        relative = Path(record["path"])
        _require(
            not relative.is_absolute() and ".." not in relative.parts and bool(relative.parts),
            "Fixture member escapes its directory",
        )
        source = (fixture.parent / relative).resolve(strict=True)
        _require(source.is_relative_to(fixture.parent), "Fixture member symlink escapes its directory")
        actual = identity(source)
        _require(actual["sha256"] == record["sha256"], "Fixture member bytes differ")
        destination = output_dir / f"member-{len(members)}.bin"
        destination.write_bytes(source.read_bytes())
        saved = identity(destination)
        _require(saved["sha256"] == actual["sha256"], "Fixture changed during snapshot")
        members.append({"source": actual, "snapshot": saved})
        return saved

    normalized = {**document, "inputs": [], "outputs": []}
    ranges = []
    for entry in document["inputs"]:
        _require(isinstance(entry, dict) and set(entry) == {"base", "path", "sha256"}, "Invalid preload")
        saved = member({key: entry[key] for key in ("path", "sha256")})
        length = Path(saved["path"]).stat().st_size
        base = entry["base"]
        _require(
            type(base) is int and base >= 0 and length > 0 and base + length <= document["dram_size"],
            "Preload exceeds declared DRAM aperture",
        )
        _require(all(base + length <= start or base >= end for start, end in ranges), "Overlapping input regions")
        ranges.append((base, base + length))
        normalized["inputs"].append({"base": base, **saved})
    names = set()
    for entry in document["outputs"]:
        _require(isinstance(entry, dict), "Invalid output region")
        name = entry.get("name")
        _require(isinstance(name, str) and name.isidentifier() and name not in names, "Invalid/duplicate output name")
        names.add(name)
        if entry.get("space") == "dram":
            _require(set(entry) == {"name", "space", "base", "length", "reference"}, "Invalid DRAM output")
            base, length = entry["base"], entry["length"]
            _require(
                type(base) is int
                and type(length) is int
                and base >= 0
                and length > 0
                and base + length <= document["dram_size"],
                "Output exceeds declared DRAM aperture",
            )
        elif entry.get("space") == "mrf_bf16":
            _require(set(entry) == {"name", "space", "registers", "reference"}, "Invalid MRF output")
            registers = entry["registers"]
            _require(
                isinstance(registers, list)
                and registers
                and all(type(index) is int and index >= 0 for index in registers)
                and len(set(registers)) == len(registers),
                "Invalid MRF register selection",
            )
        else:
            raise ValueError("Unsupported output space")
        saved = member(entry["reference"])
        _require(Path(saved["path"]).stat().st_size > 0, "Empty numerical reference")
        if entry["space"] == "dram":
            _require(Path(saved["path"]).stat().st_size == entry["length"], "Reference length differs from output")
        normalized["outputs"].append({**entry, "reference": saved})
    return normalized, members


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _source_only_model(model_root, expected_sources=None):
    """Load selected model source bytes, avoiding stale pre-existing pyc files."""

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            source = Path(self.path).read_bytes()
            if expected_sources is not None:
                relative = Path(self.path).resolve().relative_to(model_root / "npu_model").as_posix()
                _require(
                    hashlib.sha256(source).hexdigest() == expected_sources.get(relative),
                    "Imported model source differs from selected bytes",
                )
            return compile(source, self.path, "exec")

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "npu_model" and not fullname.startswith("npu_model."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.origin is None:
                locations = spec.submodule_search_locations
                _require(
                    locations
                    and all(Path(entry).resolve().is_relative_to(model_root / "npu_model") for entry in locations),
                    "Model namespace escaped selected source",
                )
                return spec
            _require(
                spec is not None
                and spec.origin is not None
                and Path(spec.origin).resolve().is_relative_to(model_root / "npu_model")
                and Path(spec.origin).suffix == ".py",
                "Model import escaped selected source",
            )
            spec.loader = Loader(fullname, spec.origin)
            return spec

    sys.meta_path.insert(0, Finder())


def _worker(job_path, arm, output_dir):
    """Runs only in the selected runtime; never imports an installed fallback model."""
    job = json.loads(Path(job_path).read_bytes())
    output_dir = Path(output_dir)
    report = {"status": "error", "completed": False, "model_ticks": None, "outputs": {}}
    simulation = None
    try:
        _require(identity(sys.executable) == job["runtime"], "Executing Python differs from selected runtime")
        model_root = Path(job["model_root"])
        _require(model_sources(model_root) == job["model_sources"], "Selected model source changed")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(model_root))
        _source_only_model(model_root, job["model_sources"])
        import npu_model

        _require(
            Path(npu_model.__file__).resolve() == model_root / "npu_model/__init__.py",
            "Imported model differs from selected source root",
        )
        report.update(
            source=job["arms"][arm],
            model_sources=job["model_sources"],
            model_package=str(Path(npu_model.__file__).resolve()),
        )
        import torch
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.util.converter import input_to_program

        runtime_files = [Path(torch.__file__), Path(torch._C.__file__)]
        runtime_files.extend(sorted((Path(torch.__file__).parent / "lib").glob("*.so")))
        runtime = {
            "python": sys.version,
            "torch": torch.__version__,
            "files": [identity(path) for path in runtime_files],
        }
        source = Path(job["arms"][arm]["path"])
        _require(identity(source) == job["arms"][arm], "Assembly changed before execution")
        with source.open() as stream:
            program = input_to_program(stream)
        words = list(program.assemble())
        _require(all(type(word) is int and 0 <= word < 1 << 32 for word in words), "Invalid model encodings")
        report["encoding"] = "model-only"
        if job["assembler"] is not None:
            assembler = _load(job["assembler"]["path"], "selected_assembler")
            bridge = _load(job["bridge"]["path"], "selected_assembly_bridge")
            expected = list(assembler.assemble(bridge.translate(source.read_text(), to_native=False)))
            _require(words == expected, "Native model and selected assembler encodings differ")
            report["encoding"] = "independent-words-matched"
        encoded = b"".join(word.to_bytes(4, "little") for word in words)
        (output_dir / "program.bin").write_bytes(encoded)
        report["words"] = identity(output_dir / "program.bin")
        fixture = job["fixture"]
        fixture_records = [*fixture["inputs"], *[entry["reference"] for entry in fixture["outputs"]]]
        for record in fixture_records:
            _require(identity(record["path"])["sha256"] == record["sha256"], "Execution fixture bytes changed")
        program.memory_regions = [
            (entry["base"], torch.tensor(list(Path(entry["path"]).read_bytes()), dtype=torch.uint8))
            for entry in fixture["inputs"]
        ]
        config = DefaultHardwareConfig()
        config.arch_state_config = dataclasses.replace(
            config.arch_state_config, dram_size=fixture["dram_size"], randomize_init=False
        )
        report["configuration"] = repr(config)
        report["arch_state_configuration"] = dataclasses.asdict(config.arch_state_config)
        simulation = Simulation(config, LoggerConfig(filename=str(output_dir / "trace.json")), program, verbose=False)
        simulation.run(max_cycles=fixture["max_cycles"])
        report["model_ticks"] = simulation.cycle_count
        report["completed"] = bool(simulation.core.is_finished() and simulation.cycle_count < fixture["max_cycles"])
        report["runtime_errors"] = repr(simulation.runtime_errors)
        state = simulation.core.arch_state
        for entry in fixture["outputs"]:
            if entry["space"] == "dram":
                value = state.dram[entry["base"] : entry["base"] + entry["length"]]
            else:
                _require(
                    all(index < config.arch_state_config.num_m_registers for index in entry["registers"]),
                    "MRF output register outside selected model",
                )
                value = torch.cat([state.read_mrf_bf16(index).reshape(-1) for index in entry["registers"]])
            raw = bytes(value.contiguous().view(torch.uint8).reshape(-1).tolist())
            path = output_dir / (entry["name"] + ".bin")
            path.write_bytes(raw)
            report["outputs"][entry["name"]] = identity(path)
        _require(report["completed"] and not simulation.runtime_errors, "Incomplete execution or model runtime error")
        _require(model_sources(model_root) == job["model_sources"], "Model source changed during execution")
        for record in fixture_records:
            _require(identity(record["path"])["sha256"] == record["sha256"], "Execution fixture bytes changed")
        for record in [job["arms"][arm], job["runtime"], job["bridge"], *[item for item in [job["assembler"]] if item]]:
            _require(identity(record["path"]) == record, "Selected execution input changed")
        for record in runtime["files"]:
            _require(identity(record["path"]) == record, "Runtime implementation changed")
        report.update(
            status="completed",
            runtime=runtime,
            model_package=str(Path(npu_model.__file__).resolve()),
            source=job["arms"][arm],
            model_sources=job["model_sources"],
        )
    except Exception as error:
        report["error"] = str(error)
        traceback.print_exc()
    finally:
        if simulation is not None:
            simulation.close()
        (output_dir / "execution.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "completed" else 1


def observe_pair(*, before, after, fixture, model_root, python, output_dir, assembler=None, timeout=120):
    """Execute identical preloads in fresh states, compare both arms to the supplied reference."""
    _require(timeout > 0, "Timeout must be positive")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "merlin.atlas_model_pair.v1",
        "status": "error",
        "numerical_equivalence": "UNKNOWN",
        "results": {},
        "model_ticks": {"before": None, "after": None, "after_minus_before": None},
        "scope": "Finite supplied byte reference in selected Python model; model ticks are not RTL cycles.",
        "rtl_replayed": False,
        "hardware_qualified": False,
        "limitations": [
            "Numerical scope is the supplied byte reference and selected finite inputs.",
            "Runtime records bind Python and Torch entry/extension libraries, not every transitive dependency.",
        ],
    }
    try:
        originals = {
            "before": identity(before),
            "after": identity(after),
            "fixture": identity(fixture),
            "python": identity(python),
            "observer": identity(Path(__file__)),
            "bridge": identity(Path(__file__).with_name("rtlgraph_assembly.py")),
        }
        if assembler is not None:
            originals["assembler"] = identity(assembler)
        normalized, members = snapshot_fixture(fixture, output_dir / "fixture")
        root = Path(model_root).resolve(strict=True)
        sources = model_sources(root)
        report["identities"] = originals
        report["fixture_members"] = members
        report["reference_provenance"] = normalized["reference_provenance"]
        report["model_root"] = str(root)
        report["model_sources"] = sources
        for name, key in (
            ("observer.py", "observer"),
            ("rtlgraph_assembly.py", "bridge"),
            ("before.S", "before"),
            ("after.S", "after"),
        ):
            (output_dir / name).write_bytes(Path(originals[key]["path"]).read_bytes())
            _require(
                identity(output_dir / name)["sha256"] == originals[key]["sha256"], "Source changed during snapshot"
            )
        job = {
            "model_root": str(root),
            "model_sources": sources,
            "runtime": originals["python"],
            "arms": {arm: identity(output_dir / (arm + ".S")) for arm in ("before", "after")},
            "assembler": originals.get("assembler"),
            "bridge": identity(output_dir / "rtlgraph_assembly.py"),
            "fixture": normalized,
        }
        job_path = output_dir / "job.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        report["execution_job"] = identity(job_path)
        for arm in ("before", "after"):
            arm_dir = output_dir / arm
            arm_dir.mkdir()
            command = [
                str(Path(python).absolute()),
                "-I",
                "-B",
                str(output_dir / "observer.py"),
                "--worker",
                str(job_path),
                arm,
                str(arm_dir),
            ]
            with (arm_dir / "stdout.log").open("wb") as stdout, (arm_dir / "stderr.log").open("wb") as stderr:
                try:
                    result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = None
                except OSError as error:
                    stderr.write(str(error).encode())
                    code = None
            execution_path = arm_dir / "execution.json"
            execution = json.loads(execution_path.read_bytes()) if execution_path.is_file() else {}
            report["results"][arm] = {
                "command": command,
                "returncode": code,
                "execution": execution,
                "execution_report": identity(execution_path) if execution_path.is_file() else None,
                "stdout": identity(arm_dir / "stdout.log"),
                "stderr": identity(arm_dir / "stderr.log"),
            }
        for record in originals.values():
            _require(identity(record["path"]) == record, "Selected input changed during observation")
        for entry in members:
            for record in entry.values():
                _require(identity(record["path"]) == record, "Fixture input/reference changed during observation")
        _require(model_sources(root) == sources, "Model source changed during observation")
        _require(identity(job_path) == report["execution_job"], "Execution job changed during observation")
        for arm, record in job["arms"].items():
            _require(identity(record["path"]) == record, f"{arm} snapshot changed during observation")
        completed = []
        for arm in ("before", "after"):
            result = report["results"][arm]
            execution = result["execution"]
            _require(
                result["returncode"] == 0
                and execution.get("status") == "completed"
                and execution.get("completed") is True,
                f"{arm} did not complete independent numerical execution",
            )
            _require(
                execution.get("source") == job["arms"][arm]
                and execution.get("model_sources") == sources
                and execution.get("model_package") == str(root / "npu_model/__init__.py"),
                "Execution identity differs",
            )
            values = {}
            for entry in normalized["outputs"]:
                record = execution["outputs"][entry["name"]]
                _require(
                    identity(record["path"]) == record and Path(record["path"]).resolve().parent == output_dir / arm,
                    "Observed output identity differs",
                )
                raw = Path(record["path"]).read_bytes()
                expected = Path(entry["reference"]["path"]).read_bytes()
                _require(raw == expected, f"{arm} output {entry['name']} differs from supplied reference")
                values[entry["name"]] = raw
            completed.append(values)
        _require(completed[0] == completed[1], "Paired numerical outputs differ")
        before_ticks, after_ticks = (report["results"][arm]["execution"]["model_ticks"] for arm in ("before", "after"))
        _require(
            type(before_ticks) is int
            and type(after_ticks) is int
            and 0 < before_ticks < normalized["max_cycles"]
            and 0 < after_ticks < normalized["max_cycles"],
            "Invalid completion ticks",
        )
        _require(
            report["results"]["before"]["execution"]["runtime"] == report["results"]["after"]["execution"]["runtime"],
            "Runtime identities differ between arms",
        )
        report.update(
            status="passed",
            numerical_equivalence="PASSED_SUPPLIED_REFERENCE",
            model_ticks={
                "before": before_ticks,
                "after": after_ticks,
                "after_minus_before": after_ticks - before_ticks,
            },
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report["error"] = str(error)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        return _worker(*sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "fixture", "model-root", "python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--assembler", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    report = observe_pair(
        before=args.before,
        after=args.after,
        fixture=args.fixture,
        model_root=args.model_root,
        python=args.python,
        output_dir=args.output,
        assembler=args.assembler,
        timeout=args.timeout,
    )
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
