"""The Python capture inventory must never issue a sealed-execution claim."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from merlin_experiments.capture_execution import python_preflight
from merlin_experiments.capture_execution.python_preflight import inspect


def _selection(tmp_path):
    m2m = tmp_path / "m2m-checkout"
    (m2m / "m2m").mkdir(parents=True)
    (m2m / "m2m/__init__.py").write_text("")
    worker = tmp_path / "worker.py"
    worker.write_text("print('not run')\n")
    workload = m2m / "workloads/example"
    workload.mkdir(parents=True)
    loader = workload / "loader.py"
    loader.write_text('import os\nINPUT = os.environ.get("MODEL_INPUT_NPZ")\n')
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.symlink_to(sys.executable)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = venv / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    missing = tmp_path / "missing-editable-source"
    (site / "external.pth").write_text(str(missing) + "\n")
    (venv / "pyvenv.cfg").write_text(f"home = {Path(sys.executable).resolve().parent}\n")
    (workload / "capture.toml").write_text(f'venv = "{venv}"\npython = "{version[6:]}"\n')
    return worker, loader, m2m, python, missing


def test_preflight_names_missing_editable_and_env_without_execution(tmp_path):
    worker, loader, m2m, python, missing = _selection(tmp_path)
    result = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python)
    assert result["status"] == "blocked_unsealed_python_capture"
    assert result["source_closure_verified"] is False
    assert result["fresh_execution"] is False
    assert result["issued_capture"] is False
    assert str(missing) in result["missing_paths"]
    assert {row["name"] for row in result["loader_environment_reads"]} == {"MODEL_INPUT_NPZ"}
    assert result["inputs"]["venv_python"]["kind"] == "symlink"
    assert not (tmp_path / "capture").exists()
    absent_data = tmp_path / "missing-input.npz"
    selected = inspect(
        worker=worker,
        loader=loader,
        m2m_root=m2m,
        python=python,
        selected_env={"MODEL_INPUT_NPZ": str(absent_data)},
    )
    assert str(absent_data) in selected["missing_paths"]
    delegated = inspect(
        worker=worker,
        loader=loader,
        m2m_root=m2m,
        python=python,
        required_env_names=["MODEL_TOKEN_IDS"],
    )
    assert delegated["caller_required_environment_names"] == ["MODEL_TOKEN_IDS"]
    assert "MODEL_TOKEN_IDS" in delegated["unselected_loader_environment"]

    optional_manifest = loader.parent / "capture.toml"
    optional_manifest.unlink()
    without_manifest = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python)
    assert without_manifest["inputs"]["loader_manifest"]["kind"] == "missing"
    assert str(optional_manifest) not in without_manifest["missing_paths"]


def test_preflight_cli_writes_once_and_returns_blocked(tmp_path):
    worker, loader, m2m, python, _ = _selection(tmp_path)
    owner = m2m / "m2m/__init__.py"
    receipt = tmp_path / "capture_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "source": {"path": str(loader), "sha256": hashlib.sha256(loader.read_bytes()).hexdigest()},
                "tool": {"source_sha256": {"m2m/__init__.py": hashlib.sha256(owner.read_bytes()).hexdigest()}},
                "source_closure_verified": True,
            }
        )
    )
    output = tmp_path / "evidence" / "preflight.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    command = [
        sys.executable,
        "-m",
        "merlin_experiments.capture_execution.python_preflight",
        "--worker",
        str(worker),
        "--loader",
        str(loader),
        "--m2m-root",
        str(m2m),
        "--python",
        str(python),
        "--capture-receipt",
        str(receipt),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    assert first.returncode == 2, first.stderr
    saved = json.loads(output.read_text())
    assert saved["source_closure_verified"] is False
    assert saved["fresh_execution"] is False
    assert saved["inputs"]["worker"]["sha256"]
    assert saved["capture_receipt_audit"]["status"] == "current_direct_sources_match"
    assert saved["capture_receipt_audit"]["historical_execution_verified"] is False
    assert subprocess.run(command, capture_output=True, text=True, env=env, check=False).returncode != 0
    assert json.loads(output.read_text()) == saved


def test_receipt_audit_reports_current_direct_source_drift_without_upgrading(tmp_path):
    worker, loader, m2m, python, _ = _selection(tmp_path)
    owner = m2m / "m2m/api.py"
    owner.write_text("VERSION = 1\n")
    delegated = m2m / "m2m/causal_session.py"
    delegated.write_text("DELEGATED = True\n")
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "loader_dependency_sources": [
                    {
                        "module": "m2m.causal_session",
                        "path": str(delegated),
                        "sha256": hashlib.sha256(delegated.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    receipt = tmp_path / "old-capture-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "source": {"path": str(loader), "sha256": sha(loader)},
                "tool": {"source_sha256": {"m2m/api.py": sha(owner)}},
                "artifacts": {"meta.json": {"sha256": sha(meta)}},
                "source_closure_verified": False,
            }
        )
    )
    matched = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    audit = matched["capture_receipt_audit"]
    assert audit["status"] == "current_direct_sources_match"
    assert audit["historical_execution_verified"] is False
    assert matched["source_closure_verified"] is False
    inventory = matched["m2m_package_source_inventory"]
    assert inventory["status"] == "current_tree_inventoried"
    assert inventory["unlisted_python_sources_by_receipt"] == [
        "m2m/__init__.py",
        "m2m/causal_session.py",
    ]
    observed = audit["observed_imports"]
    assert observed["status"] == "observed_imports_inventoried"
    assert observed["selected_m2m_sources"][0]["name"] == "m2m/causal_session.py"
    assert observed["selected_m2m_sources"][0]["named_direct_owner"] is False
    first_tree = inventory["tree_sha256"]
    owner.write_text("VERSION = 2\n")
    drifted = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    assert drifted["capture_receipt_audit"]["status"] == "current_direct_sources_drift"
    assert drifted["capture_receipt_audit"]["tool_sources"][0]["status"] == "drift_or_missing"
    assert drifted["source_closure_verified"] is False
    assert drifted["m2m_package_source_inventory"]["tree_sha256"] != first_tree
    meta.write_text("{}")
    changed_meta = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    assert changed_meta["capture_receipt_audit"]["observed_imports"]["status"] == "meta_drift_or_missing"


def test_preflight_reports_delegated_checkout_loader_observed_by_capture(tmp_path):
    worker, delegated_loader, m2m, python, _ = _selection(tmp_path)
    wrapper = tmp_path / "tiny_adapter.py"
    wrapper.write_text("from workloads.example.loader import get_model_and_inputs\n")
    owner = m2m / "m2m/__init__.py"

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "loader_dependency_sources": [
                    {
                        "module": "workloads.example.loader",
                        "path": str(delegated_loader),
                        "sha256": sha(delegated_loader),
                    }
                ]
            }
        )
    )
    receipt = tmp_path / "capture_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "source": {"path": str(wrapper), "sha256": sha(wrapper)},
                "tool": {"source_sha256": {"m2m/__init__.py": sha(owner)}},
                "artifacts": {"meta.json": {"sha256": sha(meta)}},
                "source_closure_verified": False,
            }
        )
    )

    result = inspect(worker=worker, loader=wrapper, m2m_root=m2m, python=python, capture_receipt=receipt)
    observed = result["capture_receipt_audit"]["observed_imports"]
    assert observed["selected_checkout_sources"] == [
        {
            "module": "workloads.example.loader",
            "name": "workloads/example/loader.py",
            "observed_sha256": sha(delegated_loader),
            "current_sha256": sha(delegated_loader),
            "named_direct_owner": False,
            "status": "match",
        }
    ]
    assert result["source_closure_verified"] is False
    delegated_loader.write_text("CHANGED = True\n")
    drifted = inspect(worker=worker, loader=wrapper, m2m_root=m2m, python=python, capture_receipt=receipt)
    assert drifted["capture_receipt_audit"]["observed_imports"]["selected_checkout_sources"][0]["status"] == (
        "drift_or_missing"
    )
    assert drifted["source_closure_verified"] is False


def test_receipt_audit_rejects_escape_and_wrong_loader(tmp_path):
    worker, loader, m2m, python, _ = _selection(tmp_path)
    receipt = tmp_path / "untrusted-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "source": {"path": str(loader), "sha256": hashlib.sha256(loader.read_bytes()).hexdigest()},
                "tool": {"source_sha256": {"../outside.py": "0" * 64}},
            }
        )
    )
    result = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    assert result["capture_receipt_audit"]["status"] == "invalid_receipt"
    assert result["source_closure_verified"] is False
    payload = json.loads(receipt.read_text())
    payload["source"]["path"] = str(tmp_path / "another-loader.py")
    receipt.write_text(json.dumps(payload))
    result = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    assert "differs from the selected loader" in result["capture_receipt_audit"]["errors"][0]
    (m2m / "m2m/ambient.py").symlink_to(loader)
    result = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python)
    assert result["m2m_package_source_inventory"]["status"] == "unusable_source_tree"
    assert result["source_closure_verified"] is False


def test_observed_import_symlink_never_reads_external_target(tmp_path, monkeypatch):
    worker, loader, m2m, python, _ = _selection(tmp_path)
    outside = tmp_path / "outside-secret.py"
    outside.write_text("SECRET = 1\n")
    bridge = m2m / "m2m/bridge.py"
    bridge.symlink_to(outside)
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps({"loader_dependency_sources": [{"module": "m2m.bridge", "path": str(bridge), "sha256": "0" * 64}]})
    )

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    receipt = tmp_path / "capture_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "source": {"path": str(loader), "sha256": sha(loader)},
                "tool": {"source_sha256": {"m2m/__init__.py": sha(m2m / "m2m/__init__.py")}},
                "artifacts": {"meta.json": {"sha256": sha(meta)}},
            }
        )
    )
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == outside:
            raise AssertionError("outside source bytes must not be read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    observed = result["capture_receipt_audit"]["observed_imports"]
    assert observed["status"] == "invalid_observed_imports"
    assert observed["selected_m2m_sources"][0]["status"] == "unsafe_path"
    assert observed["selected_m2m_sources"][0]["current_sha256"] is None
    assert result["m2m_package_source_inventory"]["status"] == "unusable_source_tree"
    assert result["source_closure_verified"] is False


def test_package_inventory_rejects_symlinked_root_ancestor(tmp_path, monkeypatch):
    worker, loader, m2m, python, _ = _selection(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(m2m.parent, target_is_directory=True)
    original_sha = python_preflight._sha

    def guarded_sha(path):
        if path.absolute().is_relative_to(alias):
            raise AssertionError("symlinked M2M root must not be hashed")
        return original_sha(path)

    monkeypatch.setattr(python_preflight, "_sha", guarded_sha)
    result = inspect(worker=worker, loader=loader, m2m_root=alias / m2m.name, python=python)
    assert result["m2m_package_source_inventory"]["status"] == "unusable_source_tree"
    assert result["source_closure_verified"] is False
