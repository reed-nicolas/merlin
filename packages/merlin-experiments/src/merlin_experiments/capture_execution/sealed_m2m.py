"""One bounded, independently replayable FP32 Model2MLIR capture.

This is a *process and byte-closure* proof for a selected CPU workload, not a
general Python purity theorem and not Phase 0 admission.  The guest has an
empty filesystem root apart from copied runtime/source, private devices and
temporary/output mounts; it has no network or host checkout mount.  Reviewers
must separately qualify the workload, framework numerics and Phase 0 policy.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .python_preflight import _loader_env_reads
from .sealed_python import _FLAGS, _TIMEOUT_SECONDS
from .sealed_static import _bwrap_binary, _canonical_path, _digest, _file_digest, _json, _tree

SCHEMA = "merlin.sealed_m2m_fp32.v1"
_MAX_SNAPSHOT_BYTES = 15_000_000_000
_LAUNCH_PREFIX = (
    "import runpy,sys;"
    "sys.path[:0]=['/source/m2m-src','/opt/capture-venv/lib/python3.12/site-packages'];"
    "sys.argv=['/source/worker.py','--m2m-dir','/source/m2m-src','--loader',"
    "'/source/workload/loader.py','--dtype','fp32','--seed','0',"
    "'--materialize-bundle','--out',"
)
_LAUNCH_SUFFIX = (
    "];import structlog;"
    "structlog.configure(processors=[structlog.processors.KeyValueRenderer(sort_keys=True)]);"
    "runpy.run_path('/source/worker.py',run_name='__main__')"
)


def _command(output_mount: Path) -> tuple[str, ...]:
    return ("/opt/capture-venv/bin/python", "-I", "-S", "-B", "-c",
            _LAUNCH_PREFIX + repr(str(output_mount)) + _LAUNCH_SUFFIX)


def _policy(command: tuple[str, ...], output_mount: Path) -> str:
    return _digest(_json({"flags": _FLAGS, "guest_env": {"USER": "capture", "LOGNAME": "capture",
                                                     "XDG_CACHE_HOME": str(output_mount / "cache")},
                          "command": command, "output_mount": str(output_mount),
                          "mounts": ["guest-root:ro", "source:ro", "capture:rw", "tmp:tmpfs", "dev:private"],
                          "timeout_seconds": _TIMEOUT_SECONDS}))


def _execute(bwrap: Path, runtime: Path, source: Path, output: Path,
             command: tuple[str, ...], output_mount: Path) -> dict[str, Any]:
    argv = [str(bwrap), *_FLAGS, "--setenv", "USER", "capture", "--setenv", "LOGNAME", "capture",
            "--setenv", "XDG_CACHE_HOME", str(output_mount / "cache"),
            "--ro-bind", str(runtime), "/", "--ro-bind", str(source), "/source",
            "--bind", str(output), str(output_mount), "--tmpfs", "/tmp", "--dev", "/dev", "--", *command]
    try:
        result = subprocess.run(argv, env={}, cwd="/", stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealedM2MError(f"sandbox execution failed: {type(exc).__name__}") from exc
    if result.returncode:
        raise SealedM2MError(f"sandboxed M2M exited {result.returncode}: "
                             f"{result.stderr.decode('utf-8', errors='replace')[-8000:]}")
    return {"returncode": 0,
            "stdout": {"bytes": len(result.stdout), "sha256": _digest(result.stdout)},
            "stderr": {"bytes": len(result.stderr), "sha256": _digest(result.stderr)}}


class SealedM2MError(ValueError):
    """The proposed capture or replay does not satisfy this narrow policy."""


def _source_tree(root: Path, *, skip_lib64: bool = False) -> dict[str, Any]:
    """Digest normalized file bytes, names and modes without retaining a huge manifest.

    Only the venv's known directory alias is skipped.  File symlinks are
    dereferenced by copytree and thus by this inventory; outside targets are
    permitted only for the three CPython executable aliases, whose selected
    base interpreter is independently snapshotted.
    """
    if not root.is_dir() or root.is_symlink():
        raise SealedM2MError(f"selected tree is absent or indirect: {root}")
    records: list[tuple[Any, ...]] = []
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        here = Path(current)
        relative = here.relative_to(root).as_posix()
        if relative == "." and skip_lib64:
            if (here / "lib64").is_symlink() and (here / "lib64").resolve() == (here / "lib").resolve():
                directories.remove("lib64")
            elif (here / "lib64").exists():
                raise SealedM2MError("venv /lib64 is not the expected /lib alias")
        for name in directories:
            if (here / name).is_symlink():
                raise SealedM2MError(f"directory link is outside the supported snapshot policy: {here / name}")
        records.append((relative, *sorted({"kind": "directory", "mode": stat.S_IMODE(here.stat().st_mode),
                                            "members": sorted([*directories, *files])}.items())))
        for name in sorted(files):
            path = here / name
            member = path.relative_to(root).as_posix()
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if not resolved.is_file():
                    raise SealedM2MError(f"non-file link in source: {member}")
                if not resolved.is_relative_to(root) and not (skip_lib64 and member in {
                    "bin/python", "bin/python3", "bin/python3.12"
                }):
                    raise SealedM2MError(f"external source link: {member}")
            if not path.is_file():
                raise SealedM2MError(f"non-regular source member: {member}")
            info = path.stat()
            total += info.st_size
            records.append((member, *sorted({"kind": "file", "mode": stat.S_IMODE(info.st_mode),
                                            "bytes": info.st_size, "sha256": _file_digest(path)}.items())))
    return {"members": len(records), "bytes": total, "sha256": _digest(_json(sorted(records)))}


def _snapshot_tree(root: Path) -> dict[str, Any]:
    # _tree rejects every symlink and checks file identity during hashing.
    rows = _tree(root)
    compact = [(name, *sorted(info.items())) for name, info in rows.items()]
    return {
        "members": len(rows),
        "bytes": sum(row.get("bytes", 0) for row in rows.values()),
        "sha256": _digest(_json(compact)),
    }


def _venv_home(venv: Path) -> Path:
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file():
        raise SealedM2MError("selected interpreter has no pyvenv.cfg")
    homes = [line.partition("=")[2].strip() for line in cfg.read_text().splitlines()
             if line.partition("=")[0].strip() == "home"]
    if len(homes) != 1 or not Path(homes[0]).is_absolute() or ".." in Path(homes[0]).parts:
        raise SealedM2MError("unsupported venv home")
    base = Path(homes[0]).parent
    if not (base / "bin/python3.12").resolve().is_file():
        raise SealedM2MError("venv base CPython is absent")
    return base


def _system_libs(interpreter: Path, torch_so: Path, numpy_so: Path) -> tuple[Path, ...]:
    external: set[Path] = set()
    for binary in (interpreter, torch_so, numpy_so):
        result = subprocess.run(["/usr/bin/ldd", str(binary)], capture_output=True, text=True,
                                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"}, timeout=30)
        if result.returncode or "not found" in result.stdout:
            raise SealedM2MError(f"ELF dependencies unavailable for {binary.name}")
        for line in result.stdout.splitlines():
            match = re.search(r"(?:^|\s)(?:=>\s+)?(/(?:lib|usr/lib)[^\s()]*)\s+\(", line)
            if match:
                path = Path(match.group(1))
                if not path.is_file() or not path.resolve().is_file():
                    raise SealedM2MError(f"unavailable system ELF library: {path}")
                external.add(path)
    return tuple(sorted(external))


def prepare_plan(*, m2m_root: Path, workload_root: Path, worker: Path,
                 venv: Path, max_snapshot_bytes: int = _MAX_SNAPSHOT_BYTES) -> dict[str, Any]:
    """Read-only selection and space bound; no capture or admission claim."""
    if type(max_snapshot_bytes) is not int or not 0 < max_snapshot_bytes <= _MAX_SNAPSHOT_BYTES:
        raise SealedM2MError("snapshot cap must be a positive bound no larger than 15 GB")
    m2m_root = _canonical_path(m2m_root, exists=True)
    workload_root = _canonical_path(workload_root, exists=True)
    worker = _canonical_path(worker, exists=True)
    venv = _canonical_path(venv, exists=True)
    if not (m2m_root / "m2m/api.py").is_file() or not (workload_root / "loader.py").is_file():
        raise SealedM2MError("M2M package or workload loader is absent")
    if _loader_env_reads((workload_root / "loader.py").read_text()):
        raise SealedM2MError("this first sealed policy rejects environment-reading loaders")
    if not worker.is_file() or worker.suffix != ".py":
        raise SealedM2MError("worker must be a selected Python source file")
    selected = subprocess.run(["git", "rev-parse", "HEAD"], cwd=m2m_root,
                              capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "m2m"], cwd=m2m_root,
                           capture_output=True, text=True, timeout=5, check=True).stdout
    if dirty or len(selected) != 40:
        raise SealedM2MError("selected M2M package must have a clean pinned commit")
    base = _venv_home(venv)
    interpreter = venv / "bin/python"
    if not interpreter.is_file() or interpreter.resolve() != (base / "bin/python3.12").resolve():
        raise SealedM2MError("venv interpreter does not resolve to selected base CPython")
    site = venv / "lib/python3.12/site-packages"
    torch_so = next(site.glob("torch/_C*.so"), None)
    numpy_so = next(site.glob("numpy/_core/_multiarray_umath*.so"), None)
    if torch_so is None or numpy_so is None:
        raise SealedM2MError("selected torch/NumPy ELF roots are absent")
    libs = _system_libs(interpreter, torch_so, numpy_so)
    # Inventory *before* copying: `du` does not dereference a directory link,
    # whereas copytree does.  These exact normalized bytes and topology are
    # part of the plan and are rechecked at issuance, before any 9 GB copy.
    selected_trees = {
        "venv": _source_tree(venv, skip_lib64=True),
        "base": _source_tree(base.resolve()),
        "m2m": _source_tree(m2m_root / "m2m"),
        "workload": _source_tree(workload_root),
    }
    estimate = sum(row["bytes"] for row in selected_trees.values())
    estimate += worker.stat().st_size + sum(path.stat().st_size for path in libs)
    if estimate > max_snapshot_bytes or estimate > shutil.disk_usage(venv).free:
        raise SealedM2MError(f"normalized snapshot bytes {estimate} exceed selected cap or free space")
    return {
        "schema": SCHEMA, "status": "plan_only", "m2m_root": str(m2m_root),
        "m2m_commit": selected, "workload_root": str(workload_root),
        "worker": str(worker), "venv": str(venv), "base": str(base),
        "worker_sha256": _file_digest(worker),
        "loader_sha256": _file_digest(workload_root / "loader.py"),
        "system_libs": [str(path) for path in libs], "estimate_bytes": estimate,
        "selected_trees": selected_trees,
        "max_snapshot_bytes": max_snapshot_bytes,
        "command_template_sha256": _digest((_LAUNCH_PREFIX + _LAUNCH_SUFFIX).encode()),
        "scope": "one FP32 CPU capture; no env-reading loader, checkpoint or target recipe",
    }


def _validate_snapshots(source: Path, runtime: Path, output_mount: Path) -> None:
    for name in ("source", "capture-out", "tmp", "dev"):
        path = runtime / name
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise SealedM2MError(f"guest mount point /{name} is not empty")
    executable = runtime / "opt/capture-venv/bin/python"
    if executable.is_symlink() or not executable.is_file() or executable.read_bytes()[:4] != b"\x7fELF":
        raise SealedM2MError("snapshotted interpreter is not a direct ELF")
    if not (source / "worker.py").is_file() or not (source / "workload/loader.py").is_file():
        raise SealedM2MError("selected source entrypoints are absent")
    mounted = runtime / output_mount.relative_to("/")
    if mounted.is_symlink() or not mounted.is_dir() or any(mounted.iterdir()):
        raise SealedM2MError("host-resolvable guest output mount point is not empty")


def _materialized(output: Path, source: Path, output_mount: Path) -> dict[str, Any]:
    from merlin.targetgen.application_inventory import verify_capture_receipt

    result = verify_capture_receipt(output / "model.mlir")
    if result.get("status") != "verified_materialized":
        raise SealedM2MError(f"M2M materialized receipt is unverified: {result.get('errors')}")
    mlir = (output / "model.mlir").read_text()
    pointer = "prov.weights_file = " + json.dumps(str(output_mount / "weights.safetensors"))
    if mlir.count("prov.weights_file") != 1 or pointer not in mlir:
        raise SealedM2MError("saved MLIR does not identify host-resolvable, receipt-bound weights")
    payload = json.loads((output / "capture_receipt.json").read_bytes())
    loader = payload.get("source") or {}
    entry = (payload.get("tool") or {}).get("executed_entrypoint") or {}
    if (loader.get("path") != "/source/workload/loader.py"
            or loader.get("sha256") != _file_digest(source / "workload/loader.py")
            or entry.get("path") != "/source/worker.py"
            or entry.get("sha256") != _file_digest(source / "worker.py")):
        raise SealedM2MError("M2M receipt does not bind the snapshotted entrypoints")
    sources = (payload.get("tool") or {}).get("source_sha256")
    if (
        not isinstance(sources, dict)
        or not sources
        or (payload.get("tool") or {}).get("source_inventory_status") != "complete"
    ):
        raise SealedM2MError("M2M receipt lacks its complete direct source inventory")
    for name, digest in sources.items():
        path = source / "m2m-src" / name
        if not isinstance(name, str) or ".." in Path(name).parts or not path.is_file() or _file_digest(path) != digest:
            raise SealedM2MError(f"M2M source digest differs: {name}")
    return {"status": result["status"], "receipt_sha256": result["receipt_sha256"]}


def issue(plan: dict[str, Any], run_dir: Path, *, bwrap_binary: Path | None = None) -> Path:
    """Make one private snapshot and capture; receipt remains pending replay."""
    if plan.get("schema") != SCHEMA or plan.get("status") != "plan_only":
        raise SealedM2MError("unsupported M2M plan")
    selected = prepare_plan(m2m_root=Path(plan["m2m_root"]), workload_root=Path(plan["workload_root"]),
                            worker=Path(plan["worker"]), venv=Path(plan["venv"]),
                            max_snapshot_bytes=plan["max_snapshot_bytes"])
    if selected != plan:
        raise SealedM2MError("selected M2M plan changed")
    run_dir = _canonical_path(run_dir, exists=False)
    inputs = [Path(plan[key]) for key in ("m2m_root", "workload_root", "worker", "venv", "base")]
    if any(run_dir == path or run_dir.is_relative_to(path) or path.is_relative_to(run_dir) for path in inputs):
        raise SealedM2MError("run directory overlaps a selected input")
    bwrap = _bwrap_binary(bwrap_binary)
    run_dir.mkdir(parents=False, exist_ok=False)
    source = run_dir / "snapshots/source"
    runtime = run_dir / "snapshots/guest-root"
    source.mkdir(parents=True)
    runtime.mkdir()
    shutil.copytree(Path(plan["m2m_root"]) / "m2m", source / "m2m-src/m2m", symlinks=False)
    shutil.copytree(Path(plan["workload_root"]), source / "workload", symlinks=False)
    shutil.copy2(plan["worker"], source / "worker.py")
    venv_copy = runtime / "opt/capture-venv"
    venv_copy.parent.mkdir(parents=True)
    shutil.copytree(plan["venv"], venv_copy, symlinks=False,
                    ignore=lambda directory, names: {"lib64"} if Path(directory) == Path(plan["venv"]) else set())
    base = Path(plan["base"])
    base_copy = runtime / base.relative_to("/")
    base_copy.parent.mkdir(parents=True)
    shutil.copytree(base.resolve(), base_copy, symlinks=False)
    for name in plan["system_libs"]:
        path = Path(name)
        target = runtime / path.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    output = run_dir / "capture"
    for name in ("source", "capture-out", "tmp", "dev"):
        (runtime / name).mkdir()
    (runtime / output.relative_to("/")).mkdir(parents=True)
    _validate_snapshots(source, runtime, output)
    # A copied runtime is direct files only; compare its selected large roots
    # to independently normalized source inventories before execution.
    if _snapshot_tree(venv_copy) != _source_tree(Path(plan["venv"]), skip_lib64=True):
        raise SealedM2MError("venv snapshot differs from selected normalized bytes")
    if _snapshot_tree(base_copy) != _source_tree(base.resolve()):
        raise SealedM2MError("base CPython snapshot differs from selected bytes")
    if _snapshot_tree(source / "m2m-src/m2m") != _source_tree(Path(plan["m2m_root"]) / "m2m"):
        raise SealedM2MError("M2M package snapshot differs from selected bytes")
    if _snapshot_tree(source / "workload") != _source_tree(Path(plan["workload_root"])):
        raise SealedM2MError("workload snapshot differs from selected bytes")
    if _file_digest(source / "worker.py") != _file_digest(Path(plan["worker"])):
        raise SealedM2MError("worker snapshot differs from selected bytes")
    for name in plan["system_libs"]:
        if _file_digest(runtime / Path(name).relative_to("/")) != _file_digest(Path(name)):
            raise SealedM2MError(f"system ELF snapshot differs from selected bytes: {name}")
    source_digest = _snapshot_tree(source)
    runtime_digest = _snapshot_tree(runtime)
    if source_digest["bytes"] + runtime_digest["bytes"] > plan["max_snapshot_bytes"]:
        raise SealedM2MError("actual snapshot bytes exceed selected cap")
    output.mkdir()
    command = _command(output)
    process = _execute(bwrap, runtime, source, output, command, output)
    materialized = _materialized(output, source, output)
    if (_snapshot_tree(source), _snapshot_tree(runtime)) != (source_digest, runtime_digest):
        raise SealedM2MError("sealed source or runtime changed during capture")
    receipt = run_dir / "sealed_m2m_pending.json"
    payload = {
        "schema": SCHEMA, "status": "pending_replay", "issuer_sha256": _file_digest(Path(__file__)),
        "nonce": secrets.token_hex(16), "plan": plan, "command": list(command),
        "policy_sha256": _policy(command, output), "source": source_digest,
        "guest_root": runtime_digest, "output": _snapshot_tree(output),
        "process": process, "materialized": materialized,
        "bwrap_sha256": _file_digest(bwrap),
        "scope": "isolated selected FP32 CPU M2M capture; no Phase 0 admission",
    }
    with receipt.open("xb") as stream:
        stream.write(_json(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt.chmod(0o444)
    return receipt


def replay_verify(run_dir: Path, *, bwrap_binary: Path | None = None) -> dict[str, Any]:
    """Independent replay from exact private bytes; no Phase 0 status is granted."""
    run_dir = _canonical_path(run_dir, exists=True)
    receipt = run_dir / "sealed_m2m_pending.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise SealedM2MError("pending M2M receipt is absent or indirect")
    try:
        doc = json.loads(receipt.read_bytes())
    except (ValueError, UnicodeDecodeError) as exc:
        raise SealedM2MError("pending M2M receipt is unreadable") from exc
    plan = doc.get("plan") or {}
    if (doc.get("schema") != SCHEMA or doc.get("status") != "pending_replay"
            or doc.get("issuer_sha256") != _file_digest(Path(__file__))
            or not isinstance(doc.get("nonce"), str) or len(doc["nonce"]) != 32
            or plan.get("schema") != SCHEMA or plan.get("status") != "plan_only"
            or plan.get("command_template_sha256") != _digest((_LAUNCH_PREFIX + _LAUNCH_SUFFIX).encode())
            or doc.get("command") != list(_command(run_dir / "capture"))
            or doc.get("policy_sha256") != _policy(_command(run_dir / "capture"), run_dir / "capture")
            or doc.get("scope") != "isolated selected FP32 CPU M2M capture; no Phase 0 admission"):
        raise SealedM2MError("pending M2M receipt has an unsupported policy")
    source, runtime, output = (run_dir / "snapshots/source", run_dir / "snapshots/guest-root", run_dir / "capture")
    _validate_snapshots(source, runtime, output)
    if (_snapshot_tree(source), _snapshot_tree(runtime), _snapshot_tree(output)) != (
        doc.get("source"), doc.get("guest_root"), doc.get("output")
    ):
        raise SealedM2MError("sealed M2M snapshot or capture bytes differ")
    if _materialized(output, source, output) != doc.get("materialized"):
        raise SealedM2MError("materialized M2M receipt differs")
    bwrap = _bwrap_binary(bwrap_binary)
    if _file_digest(bwrap) != doc.get("bwrap_sha256"):
        raise SealedM2MError("bubblewrap bytes differ from M2M receipt")
    with tempfile.TemporaryDirectory(prefix="m2m-replay-", dir=run_dir) as temporary:
        replay = Path(temporary)
        replay.chmod(output.stat().st_mode & 0o777)
        if _execute(bwrap, runtime, source, replay, _command(output), output) != doc.get("process"):
            raise SealedM2MError("fresh M2M process output differs")
        if _materialized(replay, source, output) != doc.get("materialized") or _snapshot_tree(replay) != doc["output"]:
            raise SealedM2MError("fresh M2M materialized bytes differ")
    if (_snapshot_tree(source), _snapshot_tree(runtime), _snapshot_tree(output)) != (
        doc["source"], doc["guest_root"], doc["output"]
    ):
        raise SealedM2MError("sealed M2M evidence changed during replay")
    return {
        "schema": SCHEMA, "status": "verified_sandbox_replay", "sealed_source_closure_replayed": True,
        "scope": "selected FP32 CPU workload in copied empty-root Python runtime only",
        "phase0_admission": "not_granted", "receipt_sha256": _file_digest(receipt),
    }
