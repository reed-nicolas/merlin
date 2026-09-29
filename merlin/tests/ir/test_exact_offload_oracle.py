"""An exact device selection needs a native oracle pass, not a host-only pass."""

from __future__ import annotations

import pytest

from merlin.common.digest import sha256_text
from merlin.llvmlower.exact_offload import ExactOffloadSelection, SelectedKernel
from merlin.targetgen import oot_runner


@pytest.mark.parametrize(
    ("oracle_result", "trace_status", "drives", "accepted"),
    [
        ("pass", "pass", True, True),
        ("skipped", "pass", True, False),
        ("fail", "pass", True, False),
        ("pass", "fail", False, False),
        ("pass", "pass", None, False),
    ],
)
def test_exact_offload_requires_executed_oracle_and_accelerator_trace(
    tmp_path, monkeypatch, oracle_result, trace_status, drives, accepted
):
    interface = "module { func.func @kernel() }\n"
    selection = ExactOffloadSelection(
        target="test_device",
        model_sha256="a" * 64,
        package_sha256="b" * 64,
        transport="test_transport",
        abi_sha256="c" * 64,
        kernels=(SelectedKernel("selected_operation", interface, sha256_text(interface)),),
        software_spec_sha256="d" * 64,
        capability_contract_sha256="e" * 64,
    )
    monkeypatch.setattr(ExactOffloadSelection, "check_release", lambda self: None)
    monkeypatch.setattr(ExactOffloadSelection, "check_package", lambda self, _package: None)
    monkeypatch.setattr(ExactOffloadSelection, "check_backend_contract", lambda self: None)

    def certify(_package, selected_interface, **kwargs):
        assert selected_interface.read_text() == interface
        assert kwargs["require_accelerator_trace"] is True
        assert kwargs["target"] == "test_device"
        return {
            "status": "pass",
            "oracle": {"result": oracle_result},
            "trace_check": {"status": trace_status, "drives_accelerator": drives},
        }

    monkeypatch.setattr(oot_runner, "certify", certify)
    if accepted:
        certified = selection.certify(tmp_path / "package", runs_root=tmp_path / "runs", simulator="spike", timeout=5)
        assert certified.certified
    else:
        with pytest.raises(ValueError, match="did not pass a running accelerator oracle"):
            selection.certify(tmp_path / "package", runs_root=tmp_path / "runs", simulator="spike", timeout=5)
