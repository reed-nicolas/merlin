"""Small policy checks; the real M2M/PyTorch process smoke is run separately."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from merlin.targetgen import application_inventory
from merlin.targetgen.quant_recipe import digest as recipe_digest
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_m2m import (
    SealedM2MError,
    _command,
    _command_v2,
    _ldd_library_path,
    _policy,
    _recipe_selection,
    _snapshot_tree,
    _source_tree,
    prepare_plan,
)
from merlin_experiments.phase0.capture_execution_attestation import AttestationNotVerified, require_verified_execution


def test_normalized_venv_inventory_matches_copied_bytes(tmp_path):
    selected = tmp_path / "selected"
    (selected / "lib").mkdir(parents=True)
    (selected / "lib/value.py").write_bytes(b"value = 1\n")
    (selected / "lib64").symlink_to("lib", target_is_directory=True)
    expected = _source_tree(selected, skip_lib64=True)
    copied = tmp_path / "copied"
    shutil.copytree(selected, copied, symlinks=False,
                    ignore=lambda directory, names: {"lib64"} if Path(directory) == selected else set())
    assert _snapshot_tree(copied) == expected
    (copied / "lib/value.py").write_bytes(b"value = 2\n")
    assert _snapshot_tree(copied) != expected


def test_other_directory_alias_is_rejected(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SealedM2MError, match="directory link"):
        _source_tree(selected)


def test_guest_output_path_is_host_resolvable_and_command_is_fixed(tmp_path):
    output = tmp_path / "capture"
    command = _command(output)
    assert command[:5] == ("/opt/capture-venv/bin/python", "-I", "-S", "-B", "-c")
    assert "sys.path[:0]" in command[-1]
    assert "KeyValueRenderer(sort_keys=True)" in command[-1]
    assert repr(str(output)) in command[-1]
    assert "--out" in command[-1]
    assert _policy(command, output) != _policy(command, tmp_path / "other")


def test_int8_command_binds_selected_recipe_and_never_uses_host_path(tmp_path):
    command = _command_v2(tmp_path / "capture", dtype="int8", recipe=True)
    assert "'/source/merlin-src'" in command[-1]
    assert "'--dtype', 'int8'" in command[-1]
    assert "'--recipe', '/source/inputs/quant_recipe.json'" in command[-1]
    assert str(tmp_path / "recipe.json") not in command[-1]
    assert _policy(command, tmp_path / "capture") != _policy(
        _command_v2(tmp_path / "capture", dtype="fp32", recipe=False), tmp_path / "capture"
    )
    with pytest.raises(SealedM2MError, match="requires fp32 without a recipe or int8"):
        _command_v2(tmp_path / "capture", dtype="int8", recipe=False)


def test_v2_guest_command_imports_worker_sibling_from_selected_package(tmp_path):
    source = tmp_path / "source"
    worker = source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).parent))\n"
                      "import _recipe_quantizer\nprint(_recipe_quantizer.MARKER)\n")
    (worker.parent / "_recipe_quantizer.py").write_text("MARKER = 'selected-sibling'\n")
    stub = source / "m2m-src/structlog.py"
    stub.parent.mkdir()
    stub.write_text("class processors:\n"
                    "    @staticmethod\n"
                    "    def KeyValueRenderer(**kwargs): return object()\n"
                    "def configure(**kwargs): pass\n")
    command = _command_v2(Path("/capture-out"), dtype="int8", recipe=True)
    program = command[-1].replace("/source", str(source))
    observed = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program],
                              capture_output=True, text=True, timeout=10, check=False)
    assert observed.returncode == 0, observed.stderr
    assert observed.stdout.strip() == "selected-sibling"


def test_v2_materialized_receipt_binds_the_executed_package_worker(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "capture"
    worker_member = "merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    for member, content in ((worker_member, b"worker\n"), ("workload/loader.py", b"loader\n"),
                            ("m2m-src/m2m/api.py", b"api\n")):
        path = source / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    output.mkdir()
    (output / "model.mlir").write_text(
        "prov.weights_file = " + json.dumps(str(output / "weights.safetensors")) + "\n"
    )
    receipt = {
        "source": {"path": "/source/workload/loader.py",
                   "sha256": sealed_m2m._file_digest(source / "workload/loader.py")},
        "tool": {"executed_entrypoint": {"path": "/source/" + worker_member,
                                           "sha256": sealed_m2m._file_digest(source / worker_member)},
                 "source_inventory_status": "complete",
                 "source_sha256": {"m2m/api.py": sealed_m2m._file_digest(source / "m2m-src/m2m/api.py")}},
    }
    (output / "capture_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(application_inventory, "verify_capture_receipt", lambda *_: {
        "status": "verified_materialized", "receipt_sha256": "bound"
    })
    assert sealed_m2m._materialized(output, source, output, worker_member=worker_member)["status"] == (
        "verified_materialized"
    )
    receipt["tool"]["executed_entrypoint"]["path"] = "/source/worker.py"
    (output / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(SealedM2MError, match="snapshotted entrypoints"):
        sealed_m2m._materialized(output, source, output, worker_member=worker_member)


def test_int8_recipe_selection_is_explicit_and_content_bound(tmp_path):
    recipe = {
        "schema": "quant_recipe_v1", "status": "derived", "software_numerical_engine": "integer_reference",
        "activation": {"dtype": "int8", "mode": "static"}, "weight": {"dtype": "int8"},
    }
    recipe["recipe_sha256"] = recipe_digest(recipe)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    selected = _recipe_selection(path, dtype="int8")
    assert selected["path"] == str(path)
    assert selected["recipe_sha256"] == recipe["recipe_sha256"]
    with pytest.raises(SealedM2MError, match="must not select"):
        _recipe_selection(path, dtype="fp32")
    with pytest.raises(SealedM2MError, match="explicit recipe"):
        _recipe_selection(None, dtype="int8")
    recipe["activation"]["mode"] = "dynamic"
    path.write_text(json.dumps(recipe))
    with pytest.raises(SealedM2MError, match="static W8A8"):
        _recipe_selection(path, dtype="int8")
    path.unlink()
    path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        _recipe_selection(path, dtype="int8")


def test_int8_materialization_refuses_recipe_or_integer_reference_mismatch(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "capture"
    (source / "merlin-src/merlin/targetgen").mkdir(parents=True)
    (source / "inputs").mkdir()
    output.mkdir()
    (source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py").write_bytes(b"worker\n")
    recipe = {"schema": "quant_recipe_v1", "status": "derived", "software_numerical_engine": "integer_reference",
              "activation": {"dtype": "int8", "mode": "static"}, "weight": {"dtype": "int8"}}
    recipe["recipe_sha256"] = recipe_digest(recipe)
    (source / "inputs/quant_recipe.json").write_text(json.dumps(recipe))
    (output / "meta.json").write_text(json.dumps({
        "dtype": "int8", "scheme": "int8_static_act_int8_weight", "recipe_sha256": recipe["recipe_sha256"],
        "quantization_stats": {"recipe_sha256": recipe["recipe_sha256"]},
        "integerization_receipt": {"golden_agreement": {"status": "passed", "reference": "pt2e_integer"}},
    }))
    monkeypatch.setattr(sealed_m2m, "_materialized", lambda *_, **__: {"status": "verified_materialized"})
    plan = {"dtype": "int8", "recipe": {"sha256": sealed_m2m._file_digest(source / "inputs/quant_recipe.json"),
                                      "bytes": (source / "inputs/quant_recipe.json").stat().st_size,
                                      "recipe_sha256": recipe["recipe_sha256"]}}
    assert sealed_m2m._materialized_v2(output, source, output, plan)["status"] == "verified_materialized"
    plan["recipe"]["recipe_sha256"] = "different"
    with pytest.raises(SealedM2MError, match="selected plan"):
        sealed_m2m._materialized_v2(output, source, output, plan)
    plan["recipe"]["recipe_sha256"] = recipe["recipe_sha256"]
    meta = json.loads((output / "meta.json").read_text())
    meta["integerization_receipt"]["golden_agreement"]["status"] = "failed"
    (output / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SealedM2MError, match="integer-reference agreement"):
        sealed_m2m._materialized_v2(output, source, output, plan)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_cpu_receipts_replay_only_under_their_selected_policy(tmp_path, monkeypatch, version):
    run = tmp_path / "run"
    source, runtime, output = run / "snapshots/source", run / "snapshots/guest-root", run / "capture"
    for path in (source, runtime, output):
        path.mkdir(parents=True)
    input_identity, output_identity = {"sha256": "input"}, {"sha256": "output"}
    process, materialized = {"returncode": 0}, {"status": "verified_materialized"}
    old = version == "v1"
    schema = sealed_m2m.SCHEMA_V1 if old else sealed_m2m.SCHEMA
    command = _command(output) if old else _command_v2(output, dtype="int8", recipe=True)
    template = (sealed_m2m._LAUNCH_PREFIX + sealed_m2m._LAUNCH_SUFFIX) if old else _command_v2(
        Path("/capture-out"), dtype="int8", recipe=True
    )[-1]
    plan = {"schema": schema, "status": "plan_only",
            "command_template_sha256": sealed_m2m._digest(template.encode())}
    if not old:
        plan.update({"dtype": "int8", "recipe": {"sha256": "selected"}})
    receipt = {
        "schema": schema, "status": "pending_replay",
        "issuer_sha256": sealed_m2m._V1_ISSUER_SHA256 if old else "current-issuer", "nonce": "0" * 32,
        "plan": plan,
        "command": list(command), "policy_sha256": _policy(command, output),
        "scope": sealed_m2m._V1_SCOPE if old else sealed_m2m._V2_SCOPE,
        "source": input_identity, "guest_root": input_identity,
        "output": output_identity, "process": process, "materialized": materialized,
        "bwrap_sha256": "bwrap",
    }
    (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(sealed_m2m, "_validate_snapshots", lambda *_, **__: None)
    monkeypatch.setattr(sealed_m2m, "_snapshot_tree", lambda path: (
        input_identity if Path(path) in (source, runtime) else output_identity
    ))
    monkeypatch.setattr(sealed_m2m, "_materialized", lambda *_: materialized)
    monkeypatch.setattr(sealed_m2m, "_materialized_v2", lambda *_: materialized)
    monkeypatch.setattr(sealed_m2m, "_execute", lambda *_: process)
    monkeypatch.setattr(sealed_m2m, "_bwrap_binary", lambda *_: tmp_path / "bwrap")
    monkeypatch.setattr(sealed_m2m, "_file_digest", lambda path: (
        "bwrap" if Path(path).name == "bwrap" else
        "current-issuer" if Path(path).name == "sealed_m2m.py" else "receipt"
    ))
    result = sealed_m2m.replay_verify(run)
    assert result["schema"] == schema
    assert result.get("capture_dtype") == (None if old else "int8")
    assert result["phase0_admission"] == "not_granted"
    assert result["status"] == "verified_sandbox_replay"


def test_plan_cannot_raise_snapshot_cap(tmp_path):
    with pytest.raises(SealedM2MError, match="no larger than 15 GB"):
        prepare_plan(m2m_root=tmp_path, workload_root=tmp_path, worker=tmp_path,
                     venv=tmp_path, max_snapshot_bytes=15_000_000_001)


def test_ldd_dependency_parser_accepts_only_structural_library_paths():
    assert _ldd_library_path("libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x123)") == Path(
        "/lib/x86_64-linux-gnu/libc.so.6"
    )
    assert _ldd_library_path("/lib64/ld-linux-x86-64.so.2 (0x123)") == Path("/lib64/ld-linux-x86-64.so.2")
    assert _ldd_library_path("linux-vdso.so.1 (0x123)") is None
    assert _ldd_library_path("libmissing.so => not found") is None


@pytest.mark.parametrize("schema", [sealed_m2m.SCHEMA_V1, sealed_m2m.SCHEMA])
def test_scoped_replay_proof_cannot_be_used_as_phase0_admission(schema):
    with pytest.raises(AttestationNotVerified):
        require_verified_execution({
            "schema": schema,
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "phase0_admission": "not_granted",
        })
