"""Pre-execution selection for one fresh, checkpoint-free CPU M2M capture.

The selection is created before its run directory exists and is supplied again
by exact digest to issuance and derivation. A reproducible sandbox replay binds
these selected bytes to a capture, but this v1 selection is not a verified
Phase 0 execution issuer or a claim that the target compiler ran the model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import (
    _bwrap_binary,
    _canonical_path,
    _digest,
    _file_digest,
    _json,
)

SCHEMA = "merlin.phase0.capture_selection.v1"
MEMBER = "capture-selection.json"


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _libraries(plan: dict) -> list[dict[str, Any]]:
    return [
        {"path": name, "bytes": Path(name).stat().st_size, "sha256": _file_digest(Path(name))}
        for name in plan["system_libs"]
    ]


def _selected_bytes(plan: dict, run_dir: Path, bwrap: Path) -> dict:
    output = run_dir / "capture"
    command = sealed_m2m._command_v2(output, dtype=plan["dtype"], recipe=plan.get("recipe") is not None)
    return {
        "schema": SCHEMA,
        "status": "preselected_before_capture",
        "run_dir": str(run_dir),
        "plan": plan,
        "plan_sha256": _digest(_json(plan)),
        "checkpoint": {"kind": "none"},
        "system_libraries": _libraries(plan),
        "bwrap": {"path": str(bwrap), "sha256": _file_digest(bwrap)},
        "issuer_source_sha256": _file_digest(Path(sealed_m2m.__file__)),
        "sandbox_policy_sha256": sealed_m2m._policy(command, output),
        "phase0_admission": "not_granted",
    }


def select(
    *,
    m2m_root: Path,
    workload_root: Path,
    worker: Path,
    venv: Path,
    schemas_root: Path,
    run_dir: Path,
    output_dir: Path,
    dtype: str = "fp32",
    recipe: Path | None = None,
    checkpoint: Path | None = None,
    bwrap_binary: Path | None = None,
) -> dict:
    """Write one owner-only selection before any capture output exists.

    External checkpoints require a separately mounted and inventoried guest
    input, so this first policy supports only loaders with no checkpoint. The
    absence is explicit and verifiable instead of silently reading a host path.
    """
    if checkpoint is not None:
        raise ValueError("external checkpoints require a selected guest mount; v1 supports explicit none only")
    run = _canonical_path(Path(run_dir), exists=False)
    destination = _canonical_path(Path(output_dir), exists=False)
    if run.exists() or destination.exists():
        raise ValueError("capture run and selection output must both be fresh")
    if run == destination or run.is_relative_to(destination) or destination.is_relative_to(run):
        raise ValueError("capture output and selection output may not overlap")
    if not run.parent.is_dir() or not destination.parent.is_dir():
        raise ValueError("capture and selection output parents must already exist")
    plan = sealed_m2m.prepare_plan(
        m2m_root=m2m_root,
        workload_root=workload_root,
        worker=worker,
        venv=venv,
        schemas_root=schemas_root,
        dtype=dtype,
        recipe=recipe,
    )
    selected_inputs = [Path(plan[name]) for name in (
        "m2m_root", "workload_root", "worker", "merlin_root", "schemas_root", "venv", "base"
    )]
    if plan.get("recipe"):
        selected_inputs.append(Path(plan["recipe"]["path"]))
    if any(
        output == item or output.is_relative_to(item) or item.is_relative_to(output)
        for output in (run, destination)
        for item in selected_inputs
    ):
        raise ValueError("capture run or selection output overlaps a selected input")
    bwrap = _bwrap_binary(bwrap_binary)
    selected = _selected_bytes(plan, run, bwrap)
    raw = _json(selected) + b"\n"
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    path = destination / MEMBER
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    destination.chmod(0o700)
    return {"path": str(path), "sha256": _digest(raw), "schema": SCHEMA, "phase0_admission": "not_granted"}


def load(path: Path, *, expected_sha256: str) -> dict:
    """Open only the independently chosen selection bytes, never receipt.plan."""
    if not _sha(expected_sha256):
        raise ValueError("capture selection requires an independently supplied SHA-256")
    selected_path = _canonical_path(Path(path), exists=True)
    if selected_path.name != MEMBER or not selected_path.is_file() or selected_path.is_symlink():
        raise ValueError("capture selection must name an ordinary capture-selection.json")
    if (
        selected_path.stat().st_uid != os.getuid()
        or selected_path.parent.stat().st_uid != os.getuid()
        or selected_path.stat().st_mode & 0o077
        or selected_path.parent.stat().st_mode & 0o077
    ):
        raise ValueError("capture selection must remain owner-only")
    raw = selected_path.read_bytes()
    if _digest(raw) != expected_sha256:
        raise ValueError("capture selection bytes differ from the pre-execution identity")
    try:
        selected = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("capture selection is unreadable") from exc
    if (
        not isinstance(selected, dict)
        or selected.get("schema") != SCHEMA
        or selected.get("status") != "preselected_before_capture"
        or selected.get("phase0_admission") != "not_granted"
        or selected.get("checkpoint") != {"kind": "none"}
        or selected.get("plan_sha256") != _digest(_json(selected.get("plan")))
    ):
        raise ValueError("capture selection has an unsupported or inconsistent policy")
    return selected


def issue(path: Path, *, expected_sha256: str) -> Path:
    """Execute only a fresh run selected by the owner-controlled artifact."""
    selected = load(path, expected_sha256=expected_sha256)
    run = _canonical_path(Path(selected["run_dir"]), exists=False)
    if run.exists():
        raise ValueError("capture run already exists; a selection cannot adopt historical output")
    plan = selected["plan"]
    current = sealed_m2m.prepare_plan(
        m2m_root=Path(plan["m2m_root"]),
        workload_root=Path(plan["workload_root"]),
        worker=Path(plan["worker"]),
        venv=Path(plan["venv"]),
        schemas_root=Path(plan["schemas_root"]),
        dtype=plan["dtype"],
        recipe=Path(plan["recipe"]["path"]) if plan.get("recipe") else None,
        max_snapshot_bytes=plan["max_snapshot_bytes"],
    )
    bwrap = _bwrap_binary(Path(selected["bwrap"]["path"]))
    if current != plan or _selected_bytes(current, run, bwrap) != selected:
        raise ValueError("selected capture source, runtime, checkpoint absence, tool or sandbox policy changed")
    if _digest(Path(path).read_bytes()) != expected_sha256:
        raise ValueError("capture selection changed before execution")
    receipt = sealed_m2m.issue(
        plan, run, bwrap_binary=bwrap, capture_selection_sha256=expected_sha256,
        selected_system_libraries=selected["system_libraries"],
        selected_bwrap_sha256=selected["bwrap"]["sha256"],
    )
    if _digest(Path(path).read_bytes()) != expected_sha256:
        raise ValueError("capture selection changed during execution")
    return receipt


def verify(path: Path, *, expected_sha256: str, model_path: Path) -> dict:
    """Read-only independent binding of selected bytes to a fresh replay.

    The returned record intentionally remains nonadmissible: no verified issuer
    or Phase 0 release authority is granted by this v1 evidence alone.
    """
    selected = load(path, expected_sha256=expected_sha256)
    run = _canonical_path(Path(selected["run_dir"]), exists=True)
    if run.stat().st_uid != os.getuid() or run.stat().st_mode & 0o077:
        raise ValueError("selected sealed capture run must remain owner-only")
    model = _canonical_path(Path(model_path), exists=True)
    if model != run / "capture/model.mlir" or not model.is_file():
        raise ValueError("selected capture model is not this fresh run's exact output")
    pending = run / "sealed_m2m_pending.json"
    if pending.is_symlink() or not pending.is_file():
        raise ValueError("selected sealed M2M receipt is absent or indirect")
    receipt = json.loads(pending.read_bytes())
    if (
        receipt.get("schema") != sealed_m2m.SCHEMA
        or receipt.get("capture_selection_sha256") != expected_sha256
        or receipt.get("plan") != selected["plan"]
        or receipt.get("policy_sha256") != selected["sandbox_policy_sha256"]
        or receipt.get("bwrap_sha256") != selected["bwrap"]["sha256"]
        or receipt.get("issuer_sha256") != selected["issuer_source_sha256"]
    ):
        raise ValueError("sealed capture does not bind the independently selected plan and policy")
    guest_root = run / "snapshots/guest-root"
    if [library["path"] for library in selected["system_libraries"]] != selected["plan"]["system_libs"]:
        raise ValueError("selected system libraries differ from the planned runtime")
    for library in selected["system_libraries"]:
        source = Path(library["path"])
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("selected system library path is unsafe")
        copied = guest_root / source.relative_to("/")
        if (
            copied.is_symlink()
            or not copied.is_file()
            or copied.stat().st_size != library["bytes"]
            or _file_digest(copied) != library["sha256"]
        ):
            raise ValueError("sealed runtime differs from preselected system library bytes")
    replay = sealed_m2m.replay_verify(run, bwrap_binary=Path(selected["bwrap"]["path"]))
    if (
        replay.get("status") != "verified_sandbox_replay"
        or replay.get("sealed_source_closure_replayed") is not True
        or replay.get("receipt_sha256") != _file_digest(pending)
    ):
        raise ValueError("fresh sandbox replay did not verify the selected capture")
    materialized = run / "capture/capture_receipt.json"
    result = {
        "schema": "merlin.phase0.preselected_capture_replay.v1",
        "status": "verified_preselected_replay",
        "selection_sha256": expected_sha256,
        "model_sha256": _file_digest(model),
        "capture_receipt_sha256": _file_digest(materialized),
        "sealed_receipt_sha256": replay["receipt_sha256"],
        "source_closure_verified": False,
        "phase0_admission": "not_granted",
    }
    if _digest(Path(path).read_bytes()) != expected_sha256:
        raise ValueError("capture selection changed during replay")
    return result
