"""Process-level check for the narrow, non-admissible Python sandbox."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.capture_execution.sealed_python import SealedPythonError, issue, replay_verify
from merlin_experiments.phase0.capture_execution_attestation import AttestationNotVerified, require_verified_execution

_PROGRAM = """\
import os

if os.path.exists('/etc/passwd') or os.path.exists('/no-home') or os.path.exists('/proc'):
    raise SystemExit('ambient host path visible')
with open('/source/input.txt', 'rb') as source:
    data = source.read()
try:
    with open('/source/input.txt', 'wb'):
        pass
except OSError:
    pass
else:
    raise SystemExit('source was writable')
with open('/capture-out/model.mlir', 'xb') as output:
    output.write(data)
print('python-capture-ok')
"""


def _guest_root(tmp_path: Path) -> Path:
    python = Path("/usr/bin/python3.12")
    stdlib = Path("/usr/lib/python3.12")
    if not python.is_file() or not stdlib.is_dir() or not shutil.which("bwrap") or not shutil.which("ldd"):
        pytest.skip("system CPython 3.12, standard library, ldd and bubblewrap are required")
    guest = tmp_path / "guest"
    for member in ("source", "capture-out", "tmp", "dev"):
        (guest / member).mkdir(parents=True)
    selected = guest / python.relative_to("/")
    selected.parent.mkdir(parents=True)
    shutil.copy2(python, selected)
    shutil.copytree(stdlib, guest / stdlib.relative_to("/"), symlinks=False)
    dependencies = subprocess.run(["ldd", str(python)], capture_output=True, text=True, check=True)
    for line in dependencies.stdout.splitlines():
        for name in re.findall(r"(?:^|\s)(/[^\s()]+)", line):
            dependency = Path(name)
            if dependency.is_file():
                destination = guest / dependency.relative_to("/")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dependency, destination)
    return guest


def test_fresh_python_capture_replays_but_cannot_claim_phase0_admission(tmp_path):
    guest = _guest_root(tmp_path)
    source = tmp_path / "input"
    source.mkdir()
    (source / "capture.py").write_text(_PROGRAM)
    (source / "input.txt").write_bytes(b"module { from sealed source }\n")
    run = tmp_path / "run"
    receipt = issue(source, guest, "capture.py", run, interpreter="/usr/bin/python3.12")
    document = json.loads(receipt.read_text())
    assert document["status"] == "local_sandbox_execution_nonadmissible"
    assert document["source_closure_verified"] is False
    assert document["phase0_admissible"] is False
    assert (run / "capture/model.mlir").read_bytes() == b"module { from sealed source }\n"
    assert replay_verify(run)["status"] == "replay_verified_nonadmissible"
    with pytest.raises(AttestationNotVerified):
        require_verified_execution(document)

    (source / "input.txt").write_bytes(b"live source changed")
    assert replay_verify(run)["source_closure_verified"] is False
    (run / "capture/model.mlir").write_bytes(b"tampered output")
    with pytest.raises(SealedPythonError, match="output bytes differ"):
        replay_verify(run)
    (run / "capture/model.mlir").write_bytes(b"module { from sealed source }\n")
    (run / "snapshots/source/input.txt").write_bytes(b"tampered")
    with pytest.raises(SealedPythonError, match="source or guest-root bytes differ"):
        replay_verify(run)


def test_python_receipt_forgery_and_unsafe_selection_are_rejected(tmp_path):
    guest = _guest_root(tmp_path)
    source = tmp_path / "input"
    source.mkdir()
    (source / "capture.py").write_text(_PROGRAM)
    (source / "input.txt").write_bytes(b"module {}\n")
    with pytest.raises(SealedPythonError, match="unsafe selected member"):
        issue(source, guest, "../capture.py", tmp_path / "invalid", interpreter="/usr/bin/python3.12")
    receipt = issue(source, guest, "capture.py", tmp_path / "run", interpreter="/usr/bin/python3.12")
    original = json.loads(receipt.read_text())
    receipt.chmod(0o644)
    receipt.write_text(json.dumps(dict(original, source_closure_verified=True, phase0_admissible=True)))
    with pytest.raises(SealedPythonError, match="non-admissible Python issuer"):
        replay_verify(tmp_path / "run")
