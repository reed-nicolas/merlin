"""One joined capture/ABI/lineage check over an actual recurrent three-stage model."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.session_bundle import load
from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen.application_inventory import verify_capture_receipt

LOADER = """
import torch
from torch import nn
from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session

class Prefix(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
    def forward(self, x):
        return self.layer(x)

class Step(nn.Module):
    def forward(self, context, state):
        result = context + state
        return result, result

class Final(nn.Module):
    def forward(self, state):
        return state[:, :2]

class Capture:
    def external_runtime_session(self):
        prefix, step, final = Prefix().eval(), Step().eval(), Final().eval()
        value = torch.randn(1, 4)
        with torch.no_grad():
            context = prefix(value)
        state = torch.zeros_like(context)
        child = dict(kind="generic_recurrent", paper_ready=False, steps=2,
                     stages=["step"],
                     stage_schedule=[dict(name="step", steps=2, execution="compiled_recurrent", timed=True)],
                     states=[dict(name="state", input_index=1, output_index=1)],
                     streams=[], quality=dict(output_index=0),
                     provenance=dict(synthetic_inputs=True, full_checkpoint=False))
        programs = (ExternalRuntimeProgram("prefix", prefix, (value,), 1),
                    ExternalRuntimeProgram("step", step, (context, state), 2, child),
                    ExternalRuntimeProgram("final", final, (context * 2,), 1))
        metadata = dict(kind="generic_recurrent", paper_ready=False,
                        stages=[p.name for p in programs], quality_program="step", states=["state"],
                        stage_schedule=[dict(name=p.name, steps=p.steps, execution="compiled", timed=True)
                                        for p in programs],
                        provenance=dict(synthetic_inputs=True, full_checkpoint=False),
                        bindings=[dict(name="context", **{"from":dict(program="prefix", output_index=0),
                                       "to":dict(program="step", input_index=0)}),
                                  dict(name="result", **{"from":dict(program="step", output_index=0),
                                       "to":dict(program="final", input_index=0)})])
        return make_external_runtime_session(version=2, programs=programs, metadata=metadata)

def get_model_and_inputs():
    return Capture(), ()
"""


@pytest.mark.slow
def test_worker_captures_the_entire_declared_session_with_owned_sidecars(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text(LOADER)
    out = tmp_path / "capture"
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "fp32",
            "--seed",
            "7",
            "--materialize-bundle",
            "--out",
            "capture",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    session = load(out)  # Checks ordered stage/weight-prefixed ABI and carried-state bindings.
    assert session.program_names == ("prefix", "step", "final")
    assert len(session.bindings) == 2
    receipt = json.loads((out / "session-receipt.json").read_bytes())
    assert receipt["agentic"] is False
    assert receipt["determinism"]["seed"] == 7
    for program in session.programs:
        stage = program.bundle
        assert verify_capture_receipt(stage / "model.mlir")["status"] == "verified_materialized"
        assert f'prov.weights_file = "{stage / "weights.safetensors"}"' in (stage / "model.mlir").read_text()
        assert json.loads((stage / "frontend-trace.json").read_bytes())["status"] == "complete"
        metadata = json.loads((stage / "meta.json").read_bytes())
        assert metadata["input_abi"]
        assert len(metadata["output_abi"]) == (2 if program.name == "step" else 1)
        assert all(row["dtype"] == "f32" for row in metadata["output_abi"])
        assert metadata["ok"] and metadata["opaque"] == 0
        assert metadata["loader_provenance"]["synthetic_inputs"] is True
        assert metadata["framework_catalog"]["status"] == "available"
