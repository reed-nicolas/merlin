"""Small policy checks; the real M2M/PyTorch process smoke is run separately."""

import shutil
from pathlib import Path

import pytest
from merlin_experiments.capture_execution.sealed_m2m import (
    SealedM2MError,
    _command,
    _policy,
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


def test_plan_cannot_raise_snapshot_cap(tmp_path):
    with pytest.raises(SealedM2MError, match="no larger than 15 GB"):
        prepare_plan(m2m_root=tmp_path, workload_root=tmp_path, worker=tmp_path,
                     venv=tmp_path, max_snapshot_bytes=15_000_000_001)


def test_scoped_replay_proof_cannot_be_used_as_phase0_admission():
    with pytest.raises(AttestationNotVerified):
        require_verified_execution({
            "schema": "merlin.sealed_m2m_fp32.v1",
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "phase0_admission": "not_granted",
        })
