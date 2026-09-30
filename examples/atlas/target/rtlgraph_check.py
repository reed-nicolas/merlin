"""Optional Atlas schedule checks against an explicitly selected timing bundle.

This target-owned tool establishes same-model schedule consistency. It does not
certify numerical results, replay RTL, or replace Phase 1's functional grading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import tempfile
from pathlib import Path

FLAGS = {
    "mxu0": "--experimental-mxu0-profile",
    "mxu1": "--experimental-mxu1-profile",
    "dma": "--rtl-dma-profile",
    "lsu": "--rtl-lsu-profile",
    "xlu": "--rtl-xlu-profile",
    "vpu": "--rtl-vpu-profile",
}


def identity(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _adapter():
    # Target-owned validation remains with the saved-evidence adapter.
    return runpy.run_path(str(Path(__file__).with_name("rtlgraph_evidence.py")))


def prepare_selection(selection: Path, compiler: Path) -> dict:
    """Verify selected saved bytes/model agreement and exact compiler identity.

    This function does not execute the compiler. Its profile_args are suitable
    for an explicit optional scheduling/checking subprocess.
    """
    selection, compiler = Path(selection).resolve(strict=True), Path(compiler).resolve(strict=True)
    adapter = _adapter()
    document = adapter["_json"](selection.read_bytes())
    _require(
        document.get("schema") == "merlin.scheduling_evidence.v1" and document.get("target") == "atlas",
        "Expected selected Atlas scheduling evidence",
    )
    _require(
        set(document)
        == {
            "schema",
            "target",
            "hardware",
            "producer",
            "semantics",
            "profiles",
            "artifacts",
            "limitations",
            "verification",
        },
        "Unexpected selection fields",
    )
    _require(
        isinstance(document.get("limitations"), list)
        and all(isinstance(value, str) for value in document["limitations"]),
        "Invalid selection limitations",
    )
    rows = document.get("artifacts")
    _require(isinstance(rows, list) and rows, "Missing selected artifacts")
    members, roles = {}, {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"role", "path", "sha256"}, "Invalid selected artifact")
        adapter["_identity"]({"path": row["path"], "sha256": row["sha256"]})
        relative = Path(row["path"])
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and "\\" not in row["path"]
            and relative.as_posix() == row["path"],
            "Selection path escapes its directory",
        )
        path = (selection.parent / relative).resolve(strict=True)
        _require(path.is_relative_to(selection.parent), "Selection symlink escapes its directory")
        _require(row["role"] not in roles and row["path"] not in members, "Duplicate selected artifact")
        _require(identity(path)["sha256"] == row["sha256"], f"Selected artifact changed: {row['path']}")
        roles[row["role"]], members[row["path"]] = path, row
    _require("contract" in roles, "Selection has no native contract")
    contract_path = roles["contract"]
    contract = adapter["_json"](contract_path.read_bytes())
    # Reuse the adapter's profile/evidence, semantics and footprint agreement
    # checks rather than weakening them in the executable consumer.
    with tempfile.TemporaryDirectory(prefix="selected-scheduling-") as directory:
        verified_path = adapter["convert_contract"](contract_path, Path(directory) / "verified")
        verified = adapter["_json"](verified_path.read_bytes())
    for key in ("hardware", "producer", "semantics", "profiles", "verification"):
        _require(document.get(key) == verified[key], f"Selection {key} differs from native contract")
    for row in verified["artifacts"]:
        _require(
            row["role"] in roles and identity(roles[row["role"]])["sha256"] == row["sha256"],
            f"Selection lacks matching contract member: {row['role']}",
        )
    compiler_identity = identity(compiler)
    _require(
        compiler_identity["sha256"] == contract["compiler"]["sha256"] == document["producer"]["sha256"],
        "Selected compiler bytes differ",
    )
    profile_args = []
    for entry in contract["profiles"]:
        profile_args.extend((FLAGS[entry["role"]], str(roles[f"projection:{entry['role']}"])))
    return {
        "profile_args": profile_args,
        "identities": {
            "selection": identity(selection),
            "compiler": compiler_identity,
            "contract": identity(contract_path),
        },
        "hardware": document["hardware"],
        "semantics": document["semantics"],
        "limitations": document["limitations"],
        "source": str(roles["source"]),
        "footprints": str(roles["footprints"]),
        "artifacts": {role: identity(path) for role, path in roles.items()},
    }


def run_command(command: list[str], *, output_dir: Path, label: str, timeout: float) -> dict:
    """Keep actual stdout/stderr, including partial output on timeout."""
    timed_out = False
    try:
        process = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, returncode, timed_out = error.stdout or b"", error.stderr or b"", None, True
    stdout_path, stderr_path = output_dir / f"{label}.stdout.log", output_dir / f"{label}.stderr.log"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": identity(stdout_path),
        "stderr": identity(stderr_path),
    }


def _modeled_cycles(command: dict, native: Path) -> int | None:
    # The compiler emits no JSON check summary. Accept only its exact summary
    # prefix, retain the original log, and report no metric if format differs.
    prefix = str(native) + ": "
    for line in Path(command["stdout"]["path"]).read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            words = line[len(prefix) :].split()
            if len(words) >= 2 and words[0].isdecimal() and words[1] == "cycles,":
                return int(words[0])
    return None


def check_schedule(
    *,
    source: Path,
    selection: Path,
    compiler: Path,
    output_dir: Path,
    assembly: str = "atlas-opt-native",
    assembler: Path | None = None,
    validation: str = "static",
    timeout: float = 60,
) -> dict:
    """Snapshot input, reproduce selected footprints, then check with that model.

    Failures are persisted as report.json with no passing verdict. No process is
    launched until identity and assembly compatibility checks have completed.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "merlin.atlas.schedule_check.v1",
        "status": "error",
        "returncode": None,
        "native_source": str(output_dir / "native.S"),
        "profile_args": [],
        "identities": {},
        "validation": validation,
        "assembly": assembly,
        "modeled_cycles": None,
        "commands": [],
        "limitations": [
            "Same-model schedule consistency does not establish numerical correctness or RTL behavior.",
            "Modeled cycles are cost estimates; DMA completion requires explicit matching waits.",
            "Static validation covers CFG hazards; dynamic checking follows one modeled execution.",
        ],
        "verification": {
            "compiler_identity_verified": False,
            "footprints_reproduced": False,
            "schedule_consistency_verified": False,
            "numerical_correctness_verified": False,
            "rtl_replayed": False,
        },
    }
    try:
        _require(validation in {"static", "dynamic"}, "Unsupported validation mode")
        _require(timeout > 0, "Timeout must be positive")
        context = prepare_selection(selection, compiler)
        report["identities"] = {**context["identities"], "source": identity(source)}
        report.update(
            {
                "profile_args": context["profile_args"],
                "hardware": context["hardware"],
                "semantics": context["semantics"],
                "selected_artifacts": context["artifacts"],
            }
        )
        report["limitations"].extend(context["limitations"])
        report["verification"]["compiler_identity_verified"] = True
        (output_dir / "input.S").write_bytes(Path(source).read_bytes())
        helper_path = Path(__file__).with_name("rtlgraph_assembly.py")
        report["identities"]["assembly_adapter"] = identity(helper_path)
        helper = runpy.run_path(str(helper_path))
        prepared = helper["prepare_source"](
            source=Path(source), assembly=assembly, assembler=assembler, output_dir=output_dir
        )
        native = Path(prepared["native_path"]).resolve(strict=True)
        _require(native == output_dir / "native.S", "Compatibility helper returned an unexpected source path")
        report["assembly_compatibility"] = {
            key: str(value) if isinstance(value, Path) else value for key, value in prepared.items()
        }
        _require(
            prepared["source_sha256"] == report["identities"]["source"]["sha256"],
            "Source changed during assembly compatibility check",
        )
        if assembler is not None:
            report["identities"]["assembler"] = identity(assembler)
            _require(
                report["identities"]["assembler"]["sha256"] == prepared["assembler_sha256"],
                "Assembler changed during compatibility check",
            )
        report["identities"]["native_source"] = identity(native)
        fresh_path = output_dir / "reproduced-footprints.json"
        query = run_command(
            [
                str(Path(compiler).resolve()),
                context["source"],
                "--dump-footprints",
                str(fresh_path),
                "--dma-timing",
                "robust",
                *context["profile_args"],
            ],
            output_dir=output_dir,
            label="footprints",
            timeout=timeout,
        )
        report["commands"].append(query)
        _require(query["returncode"] == 0 and not query["timed_out"], "Selected footprint reproduction failed")
        adapter = _adapter()
        _require(
            adapter["_json"](fresh_path.read_bytes()) == adapter["_json"](Path(context["footprints"]).read_bytes()),
            "Selected footprints differ from fresh compiler query",
        )
        report["verification"]["footprints_reproduced"] = True
        checked = run_command(
            [
                str(Path(compiler).resolve()),
                str(native),
                "--check",
                "--validation",
                validation,
                "--dma-timing",
                "robust",
                *context["profile_args"],
            ],
            output_dir=output_dir,
            label="check",
            timeout=timeout,
        )
        report["commands"].append(checked)
        report["returncode"] = checked["returncode"]
        report["status"] = "error" if checked["timed_out"] else "passed" if checked["returncode"] == 0 else "rejected"
        report["verification"]["schedule_consistency_verified"] = report["status"] == "passed"
        if validation == "dynamic":
            report["modeled_cycles"] = _modeled_cycles(checked, native)
        # Bind the bytes actually used; concurrent edits invalidate the result.
        for record in report["identities"].values():
            _require(identity(Path(record["path"])) == record, "Selected input changed during check")
        for record in context["artifacts"].values():
            _require(identity(Path(record["path"])) == record, "Selected artifact changed during check")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report["status"] = "error"
        report["verification"]["schedule_consistency_verified"] = False
        report["error"] = str(error)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--atlas-opt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assembly", required=True, choices=("atlas-opt-native", "merlin-atlas"))
    parser.add_argument("--assembler", type=Path)
    parser.add_argument("--validation", choices=("static", "dynamic"), default="static")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    report = check_schedule(
        source=args.source,
        selection=args.selection,
        compiler=args.atlas_opt,
        output_dir=args.output,
        assembly=args.assembly,
        assembler=args.assembler,
        validation=args.validation,
        timeout=args.timeout,
    )
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
