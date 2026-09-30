"""Explicit host Model2MLIR selection for diagnostic frozen derivations.

The selected M2M source package is copied into the run. The interpreter and
its dependency trees remain on the host, but their exact bytes and membership
are checked before freezing and every launch/resume. This is a byte-checked
selection on a managed host, *not* a sandbox or Phase 0 capture admission.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from pathlib import Path

import yaml

from merlin_experiments.capture_execution.sealed_m2m import (
    _capture_api_missing,
    _frontend_trace_api_missing,
    _source_tree,
    _static_integer_reference_api_missing,
    _venv_home,
)
from merlin_experiments.capture_execution.sealed_static import _canonical_path, _file_digest

SCHEMA = "merlin.phase0.selected_m2m_runtime.v1"
_MAX_SOURCE_COPY_BYTES = 15_000_000_000


def _workload_names(root: Path, synth_profile: Path | None) -> tuple[str, ...]:
    if synth_profile is None or not synth_profile.is_file():
        return ()
    from merlin.targetgen.capsule_source import resolve_model_loader

    document = yaml.safe_load(synth_profile.read_bytes()) or {}
    if not isinstance(document, dict) or not isinstance(document.get("capsules", []), list):
        raise ValueError("selected synthesis profile has no valid capsule list")
    names = set()
    for entry in document.get("capsules", []):
        if not isinstance(entry, dict):
            raise ValueError("selected synthesis profile has a malformed capsule")
        if entry.get("materialized_capture") or entry.get("micro_model"):
            continue
        if entry.get("kind") != "model" and entry.get("op") != "model":
            continue
        loader = resolve_model_loader(entry, root)
        try:
            relative = loader.relative_to(root / "workloads")
        except ValueError as exc:
            raise ValueError(
                "live model loader is outside selected Model2MLIR workloads; materialize it first"
            ) from exc
        if len(relative.parts) != 2 or relative.name != "loader.py" or not loader.is_file():
            raise ValueError(f"selected Model2MLIR workload loader is absent or indirect: {loader}")
        names.add(relative.parts[0])
    return tuple(sorted(names))


def _check_workload_declarations(root: Path, names: tuple[str, ...], python: Path) -> None:
    for name in names:
        directory = root / "workloads" / name
        declaration = directory / "capture.toml"
        if declaration.is_symlink():
            raise ValueError(f"selected Model2MLIR capture declaration is indirect: {declaration}")
        document = tomllib.loads(declaration.read_text()) if declaration.is_file() else {}
        configured = document.get("venv")
        if configured:
            path = Path(str(configured))
            path = path if path.is_absolute() else directory / path
            if (path / "bin/python").absolute() != python:
                raise ValueError(
                    f"Model2MLIR workload {name} selects another Python; materialize it or select a separate runtime"
                )
        if document.get("upstream"):
            raise ValueError(f"Model2MLIR workload {name} selects external source; materialize its capture first")
        locations = document.get("env") or {}
        if not isinstance(locations, dict):
            raise ValueError(f"Model2MLIR workload {name} has an invalid capture environment")
        if any(isinstance(value, str) and Path(value).is_dir() for value in locations.values()):
            raise ValueError(f"Model2MLIR workload {name} selects an unbound external directory")


def observe(
    root: Path, python: Path, *, synth_profile: Path | None = None, workload_names: tuple[str, ...] | None = None
) -> dict:
    """Bind the source package, venv and base Python selected by an operator."""
    root = _canonical_path(Path(root), exists=True)
    python = Path(python)
    if python.name != "python":
        raise ValueError("selected Model2MLIR Python must be named bin/python")
    python = _canonical_path(python.parent, exists=True) / python.name
    package = root / "m2m"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"selected Model2MLIR package is absent: {package}")
    names = _workload_names(root, synth_profile) if workload_names is None else workload_names
    if len(names) != len(set(names)) or any(Path(name).name != name or name in (".", "..") for name in names):
        raise ValueError("invalid selected Model2MLIR workload names")
    _check_workload_declarations(root, names, python)
    workloads = {name: _source_tree(root / "workloads" / name) for name in names}
    package_inventory = _source_tree(package)
    missing_capture_api = _capture_api_missing(root)
    missing_frontend_trace_api = _frontend_trace_api_missing(root)
    missing_static_integer_api = _static_integer_reference_api_missing(root)
    copy_bytes = package_inventory["bytes"] + sum(row["bytes"] for row in workloads.values())
    if copy_bytes > _MAX_SOURCE_COPY_BYTES:
        raise ValueError("selected Model2MLIR package and workload bytes exceed the 15 GB source-copy limit")
    return {
        "schema": SCHEMA,
        "status": "diagnostic_host_runtime",
        "root": str(root),
        "python": str(python),
        "package": package_inventory,
        "same_conversion_capture_api": {
            "status": "available_for_sealed_preflight" if not missing_capture_api else "incompatible",
            "missing": list(missing_capture_api),
            "phase0_admission": "not_granted",
        },
        "frontend_trace_api": {
            "status": "available_for_sealed_preflight" if not missing_frontend_trace_api else "incompatible",
            "missing": list(missing_frontend_trace_api),
            "phase0_admission": "not_granted",
        },
        "static_integer_reference_api": {
            "status": "available_for_sealed_preflight" if not missing_static_integer_api else "incompatible",
            "missing": list(missing_static_integer_api),
            "phase0_admission": "not_granted",
        },
        "workloads": workloads,
        "source_copy_bytes": copy_bytes,
        **_runtime(python),
        "phase0_admission": "not_granted",
    }


def _runtime(python: Path) -> dict:
    venv = python.parent.parent
    if python != venv / "bin/python" or not (venv / "pyvenv.cfg").is_file():
        raise ValueError("selected Model2MLIR Python must be a venv's bin/python")
    base = _venv_home(venv).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("selected Model2MLIR Python is absent or not executable")
    if python.resolve() != (base / "bin/python3.12").resolve():
        raise ValueError("selected Model2MLIR venv does not use its declared base Python")
    return {
        "base": str(base),
        "venv": _source_tree(venv, skip_lib64=True),
        "base_python": _source_tree(base),
        "python_sha256": _file_digest(python),
    }


def stage(selection: dict, destination: Path) -> dict:
    """Copy selected package bytes once; keep the audited host runtime explicit."""
    if selection.get("schema") != SCHEMA or observe(
        Path(selection["root"]), Path(selection["python"]), workload_names=tuple(selection["workloads"])
    ) != selection:
        raise ValueError("selected Model2MLIR runtime changed before freezing")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("frozen Model2MLIR source destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required_space = selection["source_copy_bytes"] + max(64_000_000, selection["source_copy_bytes"] // 10)
    if required_space > shutil.disk_usage(destination.parent).free:
        raise ValueError("selected Model2MLIR source exceeds available run storage")
    destination.mkdir(parents=True)
    shutil.copytree(Path(selection["root"]) / "m2m", destination / "m2m", symlinks=False)
    if _source_tree(destination / "m2m") != selection["package"]:
        raise ValueError("selected Model2MLIR source changed while staging")
    for name, expected in selection["workloads"].items():
        source = Path(selection["root"]) / "workloads" / name
        copied = destination / "workloads" / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, copied, symlinks=False)
        if _source_tree(copied) != expected:
            raise ValueError(f"selected Model2MLIR workload changed while staging: {name}")
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    for path in (p for p in destination.rglob("*") if p.is_dir()):
        path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    return {
        **selection,
        "frozen_root": str(destination),
        "frozen_package": _source_tree(destination / "m2m"),
        "frozen_workloads": {name: _source_tree(destination / "workloads" / name) for name in selection["workloads"]},
    }


def verify(frozen: dict) -> None:
    """Reject changed copied source or live runtime before every attempt."""
    if (
        frozen.get("schema") != SCHEMA
        or frozen.get("status") != "diagnostic_host_runtime"
        or frozen.get("phase0_admission") != "not_granted"
    ):
        raise ValueError("unsupported frozen Model2MLIR selection")
    selected = {
        key: value for key, value in frozen.items() if key not in {"frozen_root", "frozen_package", "frozen_workloads"}
    }
    # The original package/workload source is not reopened: only copied source
    # is authoritative after staging. The selected host venv is still live.
    current = _runtime(Path(frozen["python"]))
    if any(current[key] != selected[key] for key in ("venv", "base_python", "python_sha256", "base")):
        raise ValueError("selected Model2MLIR host runtime changed; freeze a new run")
    copied = Path(frozen["frozen_root"])
    if _source_tree(copied / "m2m") != frozen.get("frozen_package"):
        raise ValueError("frozen Model2MLIR source package changed")
    if not isinstance(frozen.get("frozen_workloads"), dict) or set(frozen["frozen_workloads"]) != set(
        selected["workloads"]
    ):
        raise ValueError("frozen Model2MLIR workload membership changed")
    for name, expected in frozen["frozen_workloads"].items():
        if _source_tree(copied / "workloads" / name) != expected:
            raise ValueError(f"frozen Model2MLIR workload changed: {name}")


def environment(frozen: dict) -> dict[str, str]:
    """Route all historical M2M aliases to one selected source and interpreter."""
    root = frozen["frozen_root"]
    python = frozen["python"]
    return {
        "MERLIN_M2M_DIR": root,
        "MERLIN_MODEL2MLIR": root,
        "MERLIN_M2M_PYTHON": python,
        "MERLIN_PHASE0_M2M_REQUIRED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def receipt(frozen: dict) -> bytes:
    """Small, inspectable run artifact identifying this diagnostic selection."""
    return (json.dumps(frozen, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
