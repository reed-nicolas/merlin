"""One bounded, independently replayable CPU Model2MLIR capture.

This is a *process and byte-closure* proof for a selected CPU workload, not a
general Python purity theorem and not Phase 0 admission.  The guest has an
empty filesystem root apart from copied runtime/source, private devices and
temporary/output mounts; it has no network or host checkout mount.  Reviewers
must separately qualify the workload, framework numerics and Phase 0 policy.
"""

from __future__ import annotations

import ast
import json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .python_preflight import _loader_env_reads
from .sealed_python import _FLAGS, _TIMEOUT_SECONDS
from .sealed_static import _bwrap_binary, _canonical_path, _digest, _file_digest, _json, _tree

SCHEMA_V1 = "merlin.sealed_m2m_fp32.v1"
SCHEMA = "merlin.sealed_m2m_cpu.v2"
# The only historical v1 issuer whose policy this verifier knows. New captures
# use v2; an old unsigned receipt remains a diagnostic replay claim only.
_V1_ISSUER_SHA256 = "f8ca017999a5cb40d44ed29bc9412bef842fe8f3edd8d85170879d1df15d1dd6"
_HISTORICAL_V2_ISSUER_SHA256 = "36aa1528481a9630e2e31394f45fdde62a0bfea738b8e1855085029f9574344b"
_V1_SCOPE = "isolated selected FP32 CPU M2M capture; no Phase 0 admission"
_V2_SCOPE = "isolated selected CPU M2M capture; no Phase 0 admission"
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


def _command_v2(output_mount: Path, *, dtype: str, recipe: bool) -> tuple[str, ...]:
    if (dtype, recipe) not in {("fp32", False), ("int8", True)}:
        raise SealedM2MError("CPU capture requires fp32 without a recipe or int8 with a selected recipe")
    worker = "/source/merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    argv = [worker, "--m2m-dir", "/source/m2m-src", "--loader",
            "/source/workload/loader.py", "--dtype", dtype, "--seed", "0",
            "--materialize-bundle", "--out", str(output_mount)]
    if recipe:
        argv += ["--recipe", "/source/inputs/quant_recipe.json"]
    program = (
        "import runpy,sys;"
        "sys.path[:0]=['/source/m2m-src','/source/merlin-src',"
        "'/opt/capture-venv/lib/python3.12/site-packages'];"
        "sys.argv=" + repr(argv) + ";import structlog;"
        "structlog.configure(processors=[structlog.processors.KeyValueRenderer(sort_keys=True)]);"
        "runpy.run_path(" + repr(worker) + ",run_name='__main__')"
    )
    return ("/opt/capture-venv/bin/python", "-I", "-S", "-B", "-c", program)


def _recipe_selection(path: Path | None, *, dtype: str) -> dict[str, Any] | None:
    if dtype == "fp32":
        if path is not None:
            raise SealedM2MError("fp32 capture must not select a quantization recipe")
        return None
    if dtype != "int8" or path is None:
        raise SealedM2MError("CPU capture supports only fp32 or int8 with an explicit recipe")
    path = _canonical_path(path, exists=True)
    if not path.is_file() or path.is_symlink():
        raise SealedM2MError("selected quantization recipe is absent or indirect")
    try:
        recipe = json.loads(path.read_bytes())
        from merlin.targetgen.quant_recipe import digest as recipe_digest
        valid = (
            isinstance(recipe, dict)
            and recipe.get("schema") == "quant_recipe_v1"
            and recipe.get("status") == "derived"
            and recipe.get("recipe_sha256") == recipe_digest(recipe)
            and recipe.get("software_numerical_engine") == "integer_reference"
            and all((recipe.get(part) or {}).get("dtype") == "int8" for part in ("activation", "weight"))
            and (recipe.get("activation") or {}).get("mode") == "static"
        )
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise SealedM2MError("selected int8 recipe is unreadable or malformed") from exc
    if not valid:
        raise SealedM2MError("selected int8 recipe lacks a derived static W8A8 integer-reference contract")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": _file_digest(path), "recipe_sha256": recipe["recipe_sha256"]}


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


def _capture_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Check the selected source API without importing its heavyweight runtime.

    This is only a compatibility gate. The fresh sandbox execution and replay,
    not source signatures, establish whether a selected implementation works.
    """
    return _source_api_missing(m2m_root, {
        "m2m/api.py": {
            "convert": {"backend", "quantization", "quantization_preapplied", "level", "func_name", "weights_path"}
        },
        "m2m/capture/bundle.py": {
            "write_bundle": {"source_path", "capture_trace", "conversion_result"}
        },
        "m2m/capture/provenance.py": {"write_capture_receipt": {"source_path"}},
    })


def _frontend_trace_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Report the exact optional APIs needed for frontend-op and precision evidence."""
    return _source_api_missing(m2m_root, {
        "m2m/api.py": {"convert": {"capture_trace", "original_frontend_snapshot"}},
        "m2m/capture/trace.py": {
            "capture_frontend_snapshot": {"stage"},
            "materialize_frontend_precision": {"dtype", "original_frontend_snapshot"},
        },
    })


def _static_integer_reference_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Report APIs needed before a static W8A8 capture can claim integer arithmetic."""
    return _source_api_missing(m2m_root, {
        "m2m/capture/pt2e_integerize.py": {"integerize_pt2e": set()},
        "m2m/capture/pt2e_integer_reference.py": {"run_pt2e_integer_reference": set()},
    })


def _source_api_missing(m2m_root: Path, required: dict[str, dict[str, set[str]]]) -> tuple[str, ...]:
    """Inspect selected source signatures only; neither execution nor provenance proof."""
    missing: list[str] = []
    for member, functions in required.items():
        source = m2m_root / member
        if not source.is_file() or source.is_symlink():
            missing.append(member)
            continue
        try:
            module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeError, SyntaxError):
            missing.append(f"{member}: readable Python source")
            continue
        top_level = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        for function, parameters in functions.items():
            node = top_level.get(function)
            if node is None:
                missing.append(f"{member}:{function}")
                continue
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            declared = {argument.arg for argument in arguments}
            missing.extend(f"{member}:{function}({name})" for name in sorted(parameters - declared))
    return tuple(missing)


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


def _ldd_library_path(line: str) -> Path | None:
    """Parse the two ordinary `ldd` dependency forms without matching line text."""
    fields = line.partition("(")[0].split()
    if len(fields) == 3 and fields[1] == "=>":
        raw = fields[2]
    elif len(fields) == 1:
        raw = fields[0]
    else:
        return None
    if raw.startswith(("/lib", "/usr/lib")):
        return Path(raw)
    return None


def _system_libs(interpreter: Path, torch_so: Path, numpy_so: Path) -> tuple[Path, ...]:
    external: set[Path] = set()
    for binary in (interpreter, torch_so, numpy_so):
        result = subprocess.run(["/usr/bin/ldd", str(binary)], capture_output=True, text=True,
                                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"}, timeout=30)
        if result.returncode or "not found" in result.stdout:
            raise SealedM2MError(f"ELF dependencies unavailable for {binary.name}")
        for line in result.stdout.splitlines():
            path = _ldd_library_path(line)
            if path is not None:
                if not path.is_file() or not path.resolve().is_file():
                    raise SealedM2MError(f"unavailable system ELF library: {path}")
                external.add(path)
    return tuple(sorted(external))


def prepare_plan(*, m2m_root: Path, workload_root: Path, worker: Path,
                 venv: Path, schemas_root: Path, dtype: str = "fp32", recipe: Path | None = None,
                 max_snapshot_bytes: int = _MAX_SNAPSHOT_BYTES) -> dict[str, Any]:
    """Read-only selection and space bound; no capture or admission claim."""
    if type(max_snapshot_bytes) is not int or not 0 < max_snapshot_bytes <= _MAX_SNAPSHOT_BYTES:
        raise SealedM2MError("snapshot cap must be a positive bound no larger than 15 GB")
    m2m_root = _canonical_path(m2m_root, exists=True)
    workload_root = _canonical_path(workload_root, exists=True)
    worker = _canonical_path(worker, exists=True)
    venv = _canonical_path(venv, exists=True)
    schemas_root = _canonical_path(schemas_root, exists=True)
    selected_recipe = _recipe_selection(recipe, dtype=dtype)
    merlin_root = worker.parents[1]
    if worker != merlin_root / "targetgen/_m2m_capture_worker.py" or not (merlin_root / "__init__.py").is_file():
        raise SealedM2MError("selected worker must belong to the selected Merlin source package")
    if schemas_root not in {merlin_root / "_data/schemas", merlin_root.parent.parent / "merlin/schemas"}:
        raise SealedM2MError("selected schemas must belong to the selected Merlin package or source checkout")
    if not schemas_root.is_dir() or any(
        not (schemas_root / name).is_file()
        for name in ("quant_formats.registry.yaml", "quant_format.schema.yaml")
    ):
        raise SealedM2MError("selected Merlin schema tree lacks the quant-format registry and validator")
    if not (m2m_root / "m2m/api.py").is_file() or not (workload_root / "loader.py").is_file():
        raise SealedM2MError("M2M package or workload loader is absent")
    missing_api = _capture_api_missing(m2m_root)
    if missing_api:
        raise SealedM2MError(
            "selected Model2MLIR lacks same-conversion materialization/receipt API: "
            + ", ".join(missing_api)
        )
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
        "merlin": _source_tree(merlin_root),
        "schemas": _source_tree(schemas_root),
    }
    estimate = sum(row["bytes"] for row in selected_trees.values())
    estimate += sum(path.stat().st_size for path in libs)
    if selected_recipe is not None:
        estimate += selected_recipe["bytes"]
    if estimate > max_snapshot_bytes or estimate > shutil.disk_usage(venv).free:
        raise SealedM2MError(f"normalized snapshot bytes {estimate} exceed selected cap or free space")
    return {
        "schema": SCHEMA, "status": "plan_only", "m2m_root": str(m2m_root),
        "m2m_commit": selected, "workload_root": str(workload_root),
        "worker": str(worker), "merlin_root": str(merlin_root), "schemas_root": str(schemas_root),
        "venv": str(venv), "base": str(base), "dtype": dtype, "recipe": selected_recipe,
        "worker_sha256": _file_digest(worker),
        "loader_sha256": _file_digest(workload_root / "loader.py"),
        "system_libs": [str(path) for path in libs], "estimate_bytes": estimate,
        "selected_trees": selected_trees,
        "max_snapshot_bytes": max_snapshot_bytes,
        "command_template_sha256": _digest(_command_v2(Path("/capture-out"), dtype=dtype,
                                                       recipe=selected_recipe is not None)[-1].encode()),
        "scope": "one selected CPU capture; no env-reading loader or unselected checkpoint",
    }


def _validate_snapshots(source: Path, runtime: Path, output_mount: Path, *, schema: str = SCHEMA_V1) -> None:
    for name in ("source", "capture-out", "tmp", "dev"):
        path = runtime / name
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise SealedM2MError(f"guest mount point /{name} is not empty")
    executable = runtime / "opt/capture-venv/bin/python"
    if executable.is_symlink() or not executable.is_file() or executable.read_bytes()[:4] != b"\x7fELF":
        raise SealedM2MError("snapshotted interpreter is not a direct ELF")
    worker = (source / "worker.py" if schema == SCHEMA_V1 else
              source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py")
    if not worker.is_file() or not (source / "workload/loader.py").is_file():
        raise SealedM2MError("selected source entrypoints are absent")
    if schema == SCHEMA and any(
        not (source / "merlin-src/merlin/_data/schemas" / name).is_file()
        for name in ("quant_formats.registry.yaml", "quant_format.schema.yaml")
    ):
        raise SealedM2MError("selected Merlin package data is absent from the snapshot")
    mounted = runtime / output_mount.relative_to("/")
    if mounted.is_symlink() or not mounted.is_dir() or any(mounted.iterdir()):
        raise SealedM2MError("host-resolvable guest output mount point is not empty")


def _materialized(output: Path, source: Path, output_mount: Path,
                  *, worker_member: str = "worker.py") -> dict[str, Any]:
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
    worker = source / worker_member
    if (loader.get("path") != "/source/workload/loader.py"
            or loader.get("sha256") != _file_digest(source / "workload/loader.py")
            or entry.get("path") != "/source/" + worker_member
            or entry.get("sha256") != _file_digest(worker)):
        raise SealedM2MError("M2M receipt does not bind the snapshotted entrypoints")
    sources = (payload.get("tool") or {}).get("source_sha256")
    if (
        not isinstance(sources, dict)
        or not sources
        or (payload.get("tool") or {}).get("source_inventory_status") != "complete"
    ):
        raise SealedM2MError("M2M receipt lacks its complete direct source inventory")
    for name, digest in sources.items():
        if (not isinstance(name, str) or not name.startswith("m2m/")
                or PurePosixPath(name).as_posix() != name or ".." in PurePosixPath(name).parts
                or "\\" in name or "\x00" in name):
            raise SealedM2MError(f"unsafe M2M source member: {name!r}")
        path = source / "m2m-src" / name
        if not path.is_file() or _file_digest(path) != digest:
            raise SealedM2MError(f"M2M source digest differs: {name}")
    return {"status": result["status"], "receipt_sha256": result["receipt_sha256"]}


def _materialized_v2(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    result = _materialized(output, source, output_mount,
                           worker_member="merlin-src/merlin/targetgen/_m2m_capture_worker.py")
    metadata = json.loads((output / "meta.json").read_bytes())
    if not isinstance(metadata, dict) or metadata.get("dtype") != plan.get("dtype"):
        raise SealedM2MError("capture metadata does not identify the selected dtype")
    recipe = plan.get("recipe")
    if plan.get("dtype") == "fp32":
        if recipe is not None or metadata.get("recipe_sha256") is not None:
            raise SealedM2MError("fp32 capture unexpectedly selected a quantization recipe")
    elif plan.get("dtype") == "int8":
        selected = source / "inputs/quant_recipe.json"
        if not isinstance(recipe, dict) or not selected.is_file():
            raise SealedM2MError("int8 capture did not retain the selected recipe bytes")
        observed = _recipe_selection(selected, dtype="int8")
        if any(observed[key] != recipe.get(key) for key in ("bytes", "sha256", "recipe_sha256")):
            raise SealedM2MError("int8 recipe snapshot differs from the selected plan")
        stats = metadata.get("quantization_stats") or {}
        agreement = ((metadata.get("integerization_receipt") or {}).get("golden_agreement") or {})
        if (
            metadata.get("scheme") != "int8_static_act_int8_weight"
            or metadata.get("recipe_sha256") != recipe.get("recipe_sha256")
            or stats.get("recipe_sha256") != recipe.get("recipe_sha256")
            or agreement.get("status") != "passed"
            or agreement.get("reference") != "pt2e_integer"
            or metadata.get("software_numerical_engine") not in (None, "integer_reference")
        ):
            raise SealedM2MError("int8 capture lacks selected recipe and independent integer-reference agreement")
    else:
        raise SealedM2MError("unsupported selected capture dtype")
    return result


def _stage_source(plan: dict[str, Any], source: Path) -> None:
    """Copy exactly the source members named by the selected v2 plan."""
    shutil.copytree(Path(plan["m2m_root"]) / "m2m", source / "m2m-src/m2m", symlinks=False)
    shutil.copytree(Path(plan["workload_root"]), source / "workload", symlinks=False)
    shutil.copytree(Path(plan["merlin_root"]), source / "merlin-src/merlin", symlinks=False)
    if _snapshot_tree(source / "merlin-src/merlin") != plan["selected_trees"]["merlin"]:
        raise SealedM2MError("Merlin package snapshot differs from selected bytes")
    bundled_schemas = source / "merlin-src/merlin/_data/schemas"
    if not bundled_schemas.exists():
        shutil.copytree(Path(plan["schemas_root"]), bundled_schemas, symlinks=False)
    if plan.get("recipe"):
        selected_recipe = source / "inputs/quant_recipe.json"
        selected_recipe.parent.mkdir()
        shutil.copy2(plan["recipe"]["path"], selected_recipe)


def _verify_staged_selection(plan: dict[str, Any], source: Path, runtime: Path) -> dict[str, Any]:
    """Reject bytes copied after selection changed, before executing any guest code."""
    roots = {
        "venv": runtime / "opt/capture-venv",
        "base": runtime / Path(plan["base"]).relative_to("/"),
        "m2m": source / "m2m-src/m2m",
        "workload": source / "workload",
        "schemas": source / "merlin-src/merlin/_data/schemas",
    }
    selected_trees = plan["selected_trees"]
    for name, path in roots.items():
        if _snapshot_tree(path) != selected_trees[name]:
            raise SealedM2MError(f"staged {name} bytes differ from the pre-execution selection")
    # External schemas are inserted after the original Merlin package snapshot.
    if plan["schemas_root"] == str(Path(plan["merlin_root"]) / "_data/schemas"):
        if _snapshot_tree(source / "merlin-src/merlin") != selected_trees["merlin"]:
            raise SealedM2MError("staged Merlin bytes differ from the pre-execution selection")
    return selected_trees["schemas"]


def issue(plan: dict[str, Any], run_dir: Path, *, bwrap_binary: Path | None = None,
          capture_selection_sha256: str | None = None,
          selected_system_libraries: list[dict[str, Any]] | None = None,
          selected_bwrap_sha256: str | None = None) -> Path:
    """Make one private snapshot and capture; receipt remains pending replay."""
    if plan.get("schema") != SCHEMA or plan.get("status") != "plan_only":
        raise SealedM2MError("unsupported M2M plan")
    if capture_selection_sha256 is not None and (
        not isinstance(capture_selection_sha256, str)
        or len(capture_selection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in capture_selection_sha256)
    ):
        raise SealedM2MError("capture selection requires an exact SHA-256 identity")
    if capture_selection_sha256 is not None and (
        selected_system_libraries is None or selected_bwrap_sha256 is None
    ):
        raise SealedM2MError("selected capture requires antecedent system-library and bubblewrap bytes")
    if selected_system_libraries is not None and (
        not isinstance(selected_system_libraries, list)
        or [row.get("path") for row in selected_system_libraries] != plan.get("system_libs")
    ):
        raise SealedM2MError("selected system-library roster differs from the plan")
    selected = prepare_plan(m2m_root=Path(plan["m2m_root"]), workload_root=Path(plan["workload_root"]),
                            worker=Path(plan["worker"]), venv=Path(plan["venv"]),
                            schemas_root=Path(plan["schemas_root"]),
                            dtype=plan["dtype"],
                            recipe=Path(plan["recipe"]["path"]) if plan.get("recipe") else None,
                            max_snapshot_bytes=plan["max_snapshot_bytes"])
    if selected != plan:
        raise SealedM2MError("selected M2M plan changed")
    run_dir = _canonical_path(run_dir, exists=False)
    inputs = [Path(plan[key]) for key in (
        "m2m_root", "workload_root", "worker", "merlin_root", "schemas_root", "venv", "base"
    )]
    if plan.get("recipe"):
        inputs.append(Path(plan["recipe"]["path"]))
    if any(run_dir == path or run_dir.is_relative_to(path) or path.is_relative_to(run_dir) for path in inputs):
        raise SealedM2MError("run directory overlaps a selected input")
    bwrap = _bwrap_binary(bwrap_binary)
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes differ from the pre-execution selection")
    run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    source = run_dir / "snapshots/source"
    runtime = run_dir / "snapshots/guest-root"
    source.mkdir(parents=True)
    runtime.mkdir()
    _stage_source(plan, source)
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
    _validate_snapshots(source, runtime, output, schema=SCHEMA)
    # Rehash the copied bytes against the antecedent selection, not mutable
    # host paths: a source change after prepare_plan must never be executed.
    selected_schemas = _verify_staged_selection(plan, source, runtime)
    if _file_digest(source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py") != plan["worker_sha256"]:
        raise SealedM2MError("worker snapshot differs from selected bytes")
    if plan.get("recipe") and _file_digest(source / "inputs/quant_recipe.json") != plan["recipe"]["sha256"]:
        raise SealedM2MError("selected recipe snapshot differs from selected bytes")
    for index, name in enumerate(plan["system_libs"]):
        copied = runtime / Path(name).relative_to("/")
        expected = selected_system_libraries[index] if selected_system_libraries is not None else None
        if expected is not None:
            if copied.stat().st_size != expected.get("bytes") or _file_digest(copied) != expected.get("sha256"):
                raise SealedM2MError(f"system ELF snapshot differs from the pre-execution selection: {name}")
        elif _file_digest(copied) != _file_digest(Path(name)):
            raise SealedM2MError(f"system ELF snapshot differs from selected bytes: {name}")
    source_digest = _snapshot_tree(source)
    runtime_digest = _snapshot_tree(runtime)
    if source_digest["bytes"] + runtime_digest["bytes"] > plan["max_snapshot_bytes"]:
        raise SealedM2MError("actual snapshot bytes exceed selected cap")
    output.mkdir()
    command = _command_v2(output, dtype=plan["dtype"], recipe=plan.get("recipe") is not None)
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes changed before sandbox execution")
    process = _execute(bwrap, runtime, source, output, command, output)
    materialized = _materialized_v2(output, source, output, plan)
    if (_snapshot_tree(source), _snapshot_tree(runtime)) != (source_digest, runtime_digest):
        raise SealedM2MError("sealed source or runtime changed during capture")
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes changed during sandbox execution")
    receipt = run_dir / "sealed_m2m_pending.json"
    payload = {
        "schema": SCHEMA, "status": "pending_replay", "issuer_sha256": _file_digest(Path(__file__)),
        "nonce": secrets.token_hex(16), "plan": plan, "command": list(command),
        "policy_sha256": _policy(command, output), "source": source_digest,
        "guest_root": runtime_digest, "output": _snapshot_tree(output),
        "schemas": selected_schemas,
        "process": process, "materialized": materialized,
        "bwrap_sha256": _file_digest(bwrap),
        "scope": _V2_SCOPE,
    }
    if capture_selection_sha256 is not None:
        payload["capture_selection_sha256"] = capture_selection_sha256
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
    schema = doc.get("schema")
    if schema == SCHEMA_V1:
        expected_issuer = _V1_ISSUER_SHA256
        expected_template = _digest((_LAUNCH_PREFIX + _LAUNCH_SUFFIX).encode())
        expected_command = _command(run_dir / "capture")
        expected_scope = _V1_SCOPE
    elif schema == SCHEMA:
        dtype, recipe = plan.get("dtype"), plan.get("recipe")
        if dtype not in {"fp32", "int8"} or (recipe is not None) != (dtype == "int8"):
            raise SealedM2MError("pending M2M receipt has an unsupported dtype or recipe selection")
        selected_digest = doc.get("capture_selection_sha256")
        if selected_digest is not None and (
            not isinstance(selected_digest, str)
            or len(selected_digest) != 64
            or any(character not in "0123456789abcdef" for character in selected_digest)
        ):
            raise SealedM2MError("pending M2M receipt has a malformed capture selection identity")
        expected_issuer = _file_digest(Path(__file__))
        expected_template = _digest(_command_v2(Path("/capture-out"), dtype=dtype,
                                                recipe=recipe is not None)[-1].encode())
        expected_command = _command_v2(run_dir / "capture", dtype=dtype, recipe=recipe is not None)
        expected_scope = _V2_SCOPE
    else:
        raise SealedM2MError("pending M2M receipt has an unsupported schema")
    supported_issuer = (
        {expected_issuer, _HISTORICAL_V2_ISSUER_SHA256}
        if schema == SCHEMA and doc.get("capture_selection_sha256") is None
        else {expected_issuer}
    )
    if (doc.get("status") != "pending_replay"
            or doc.get("issuer_sha256") not in supported_issuer
            or not isinstance(doc.get("nonce"), str) or len(doc["nonce"]) != 32
            or plan.get("schema") != schema or plan.get("status") != "plan_only"
            or plan.get("command_template_sha256") != expected_template
            or doc.get("command") != list(expected_command)
            or doc.get("policy_sha256") != _policy(expected_command, run_dir / "capture")
            or doc.get("scope") != expected_scope):
        raise SealedM2MError("pending M2M receipt has an unsupported policy")
    source, runtime, output = (run_dir / "snapshots/source", run_dir / "snapshots/guest-root", run_dir / "capture")
    _validate_snapshots(source, runtime, output, schema=schema)
    if (_snapshot_tree(source), _snapshot_tree(runtime), _snapshot_tree(output)) != (
        doc.get("source"), doc.get("guest_root"), doc.get("output")
    ):
        raise SealedM2MError("sealed M2M snapshot or capture bytes differ")
    if schema == SCHEMA:
        selected_trees = plan.get("selected_trees") or {}
        if not isinstance(selected_trees, dict):
            raise SealedM2MError("v2 plan lacks exact selected source/runtime tree identities")
        commit = plan.get("m2m_commit")
        if (not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef" for character in commit)):
            raise SealedM2MError("v2 plan lacks a pinned M2M revision identity")
        selected_roots = {
            "venv": runtime / "opt/capture-venv",
            "m2m": source / "m2m-src/m2m",
            "workload": source / "workload",
            "schemas": source / "merlin-src/merlin/_data/schemas",
        }
        base = plan.get("base")
        if (not isinstance(base, str) or not base.startswith("/") or base == "/"
                or Path(base).as_posix() != base or ".." in Path(base).parts):
            raise SealedM2MError("selected base Python path is absent or unsafe")
        selected_roots["base"] = runtime / base.lstrip("/")
        if set(selected_trees) != set(selected_roots) | {"merlin"}:
            raise SealedM2MError("v2 plan lacks exact selected source/runtime tree identities")
        # The external-schema policy adds the selected schema tree to the
        # Merlin package after its original selected-tree digest was taken.
        if plan.get("schemas_root") == str(Path(str(plan.get("merlin_root"))) / "_data/schemas"):
            selected_roots["merlin"] = source / "merlin-src/merlin"
        for name, path in selected_roots.items():
            if _snapshot_tree(path) != selected_trees[name]:
                raise SealedM2MError(f"selected {name} bytes differ from the v2 plan")
        if _file_digest(source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py") != plan.get("worker_sha256"):
            raise SealedM2MError("selected Merlin worker bytes differ from the v2 plan")
        if selected_trees["schemas"] != doc.get("schemas"):
            raise SealedM2MError("selected Merlin schema bytes differ from the v2 receipt")
    materialized = (_materialized(output, source, output) if schema == SCHEMA_V1
                    else _materialized_v2(output, source, output, plan))
    if materialized != doc.get("materialized"):
        raise SealedM2MError("materialized M2M receipt differs")
    bwrap = _bwrap_binary(bwrap_binary)
    if _file_digest(bwrap) != doc.get("bwrap_sha256"):
        raise SealedM2MError("bubblewrap bytes differ from M2M receipt")
    with tempfile.TemporaryDirectory(prefix="m2m-replay-", dir=run_dir) as temporary:
        replay = Path(temporary)
        replay.chmod(output.stat().st_mode & 0o777)
        if _execute(bwrap, runtime, source, replay, expected_command, output) != doc.get("process"):
            raise SealedM2MError("fresh M2M process output differs")
        replay_materialized = (_materialized(replay, source, output) if schema == SCHEMA_V1
                               else _materialized_v2(replay, source, output, plan))
        if replay_materialized != doc.get("materialized") or _snapshot_tree(replay) != doc["output"]:
            raise SealedM2MError("fresh M2M materialized bytes differ")
    if (_snapshot_tree(source), _snapshot_tree(runtime), _snapshot_tree(output)) != (
        doc["source"], doc["guest_root"], doc["output"]
    ):
        raise SealedM2MError("sealed M2M evidence changed during replay")
    result = {
        "schema": schema, "status": "verified_sandbox_replay", "sealed_source_closure_replayed": True,
        "scope": ("selected FP32 CPU workload in copied empty-root Python runtime only"
                  if schema == SCHEMA_V1 else "selected CPU workload in copied empty-root Python runtime only"),
        "phase0_admission": "not_granted", "receipt_sha256": _file_digest(receipt),
    }
    if schema == SCHEMA:
        result["capture_dtype"] = plan["dtype"]
    return result
