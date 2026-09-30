"""Target-bound, operator-selected Phase 1 oracle timing input.

This record is produced only after a real cert-tier observation. Its path is a
resource owned by the selected target, never an alias through shared scripts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def timing_path(resources: Path, target: str) -> Path:
    if not target or not all(char.isascii() and (char.isalnum() or char in "_-") for char in target):
        raise ValueError("timing target must be a nonempty path-safe identity")
    return Path(resources) / f".oracle_timing.{target}.json"


def requires_chipyard_timing(descriptor: Path | None) -> bool:
    """Require byte-bound timing for Chipyard or an unclassifiable descriptor (fail closed)."""
    if descriptor is None:
        return True
    from merlin.targetgen.target_experiment import load_target_experiment

    try:
        return load_target_experiment(descriptor).sim_via == "chipyard"
    except (OSError, ValueError):
        return True


def read_timing(path: Path, *, target: str) -> dict:
    """Read a measured record, refusing aliases and foreign/invalid observations."""
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"oracle timing path contains a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"oracle timing record is absent: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"oracle timing record is unreadable: {path}") from exc
    if not isinstance(record, dict) or record.get("target") != target:
        raise ValueError(f"oracle timing record is not bound to target {target!r}: {path}")
    if not isinstance(record.get("config"), str) or not record["config"].strip():
        raise ValueError(f"oracle timing record has no simulator config: {path}")
    if not isinstance(record.get("measured_by"), str) or not record["measured_by"].strip():
        raise ValueError(f"oracle timing record has no measurement producer: {path}")
    digest = record.get("simulator_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"oracle timing record has no simulator byte identity: {path}")
    seconds = record.get("verilator_per_capsule_s")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"oracle timing record has no positive finite observation: {path}")
    return record


def read_verified_timing(path: Path, *, descriptor: Path, target: str) -> dict:
    """Match the observation to this target's selected Chipyard simulator bytes."""
    from merlin.common.digest import sha256_file
    from merlin.common.paths import ext_path
    from merlin.targetgen.target_experiment import (
        declared_vs_resolved_contract,
        load_capability_manifest,
        load_target_experiment,
    )

    if descriptor is None:
        raise ValueError("oracle timing has no selected target descriptor")
    record = read_timing(path, target=target)
    try:
        selected = load_target_experiment(descriptor)
        if selected.target != target or selected.sim_via != "chipyard":
            raise ValueError("selected descriptor does not declare this target's Chipyard simulator")
        _, contract_path, agreement = declared_vs_resolved_contract(selected)
        if agreement != "agree" or contract_path is None:
            raise ValueError(f"selected target contract is not agreed and resolvable: {agreement}")
        config = (
            load_capability_manifest(target, contract_path=contract_path).contract.get("runtime") or {}
        ).get("rtl_sim_config")
        if not isinstance(config, str) or record["config"] != config:
            raise ValueError("oracle timing config differs from the selected target contract")
        simulator = ext_path("chipyard") / "sims" / "verilator" / f"simulator-chipyard.harness-{config}"
        if not simulator.is_file():
            raise ValueError(f"measured simulator binary is absent: {simulator}")
        if record["simulator_sha256"] != sha256_file(simulator):
            raise ValueError(f"measured simulator bytes changed: {simulator}")
    except (OSError, KeyError) as exc:
        raise ValueError(f"cannot verify selected simulator for oracle timing: {exc}") from exc
    return record
