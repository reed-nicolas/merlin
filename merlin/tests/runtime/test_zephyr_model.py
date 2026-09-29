"""Whole-model execution on Zephyr (SMP) — spike gate.

Bridges merlin's bare-metal spike path to a real Zephyr image that also runs on
the 2-tile FireSim board (``runtime.backends.zephyr_model``). The model ``.o`` (rv64gcv)
+ the data-driven C runtime are linked into a Zephyr app whose single worker thread is
``k_thread_cpu_pin``-ed to the RVV tile and calls ``merlin_run``; output is dumped over
the console with the same OUT/ARGMAX/METRIC/DONE protocol the bare-metal harness uses.

Gated on the Zephyr+spike toolchain (``zephyr_model.available()``) and ``MERLIN_RUN_SLOW``
(it lowers the model, builds a Zephyr image with west/cmake/ninja, and runs spike — a
minute-plus). The FireSim leg is driven separately (needs the FPGA + queue daemon).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = repo_root()


def _zm():
    from merlin.runtime.backends import zephyr_model

    return zephyr_model


@pytest.mark.skipif(
    not os.environ.get("MERLIN_RUN_SLOW"), reason="set MERLIN_RUN_SLOW=1 (builds a Zephyr image + runs spike)"
)
@pytest.mark.skipif(not _zm().available(), reason="Zephyr/spike toolchain unavailable")
@pytest.mark.parametrize("rvv_hart", [0, 1])
def test_small_llama_on_zephyr_spike(rvv_hart, tmp_path):
    """small_llama whole-model on Zephyr/spike (-p2), worker pinned to ``rvv_hart``.

    Both pinnings are exercised: hart 0 is the simple case; hart 1 mirrors the FireSim
    board where only tile 1 carries the Saturn vector unit (so the RVV worker MUST land
    on hart 1 there). Gate is the host==torch threshold (cos>0.9999, rel<1e-3)."""
    zm = _zm()
    bundle = REPO / "out/artifacts" / "recaptures" / "small_consistent"
    if not (bundle / "model.mlir").is_file():
        pytest.skip("small_consistent not captured")
    golden = np.load(bundle / "golden.npy")
    res = zm.build_and_run(
        bundle, tmp_path, board="spike_riscv64", rvv_hart=rvv_hart, harts=2, arena_mb=64, reference=golden, timeout=1800
    )
    assert res["ok"], (res.get("cos"), res.get("rel"))
    assert res["metrics"].get("cycles", 0) > 0
    assert "MODELBLASTER_WALL_CYCLES" in res["console"]


@pytest.mark.skipif(not _zm().available(), reason="Zephyr/spike toolchain unavailable")
def test_zephyr_app_builds_for_chipyard_board(tmp_path):
    """The FireSim image (board ``chipyard_riscv64``, worker on the RVV tile = hart 1)
    builds — the disable-cpu@2..7 overlay applies and the rv64gcv model object links into
    the rv64gc Zephyr image. Build-only (no FPGA); the run is driven by run_on_firesim."""
    zm = _zm()
    bundle = REPO / "out/artifacts" / "recaptures" / "small_consistent"
    if not (bundle / "model.mlir").is_file():
        pytest.skip("small_consistent not captured")
    b = zm.build_app(bundle, tmp_path, board="chipyard_riscv64", rvv_hart=1, arena_mb=64, cpus=2)
    assert Path(b["elf"]).is_file()


def test_gate_multi_tier_and_legacy():
    """``_gate`` emits per-tier cos/rel/argmax and ``ok = T1(w8a8) or T2(fp32)``; a single
    reference array keeps the legacy strict fp32 gate."""
    from merlin.runtime.backends.zephyr_model import _gate

    fp32 = np.array([1.0, 2, 3, 4, 5], np.float32)
    w8a8 = np.array([1.01, 2.02, 2.98, 4.01, 5.0], np.float32)
    spike = np.array([1.011, 2.019, 2.979, 4.012, 4.998], np.float32)

    g = _gate(spike, {"w8a8": w8a8, "fp32": fp32})
    assert g["w8a8_cos"] > 0.999 and g["w8a8_rel"] < 1e-2  # T1
    assert g["fp32_cos"] > 0.99 and g["fp32_argmax"]  # T2
    assert g["ok"] is True
    assert {"w8a8_cos", "fp32_cos", "cos", "rel"} <= set(g)

    # legacy single-reference: strict fp32 threshold
    near = fp32 * np.float32(1.00001)
    assert _gate(near, fp32)["ok"] is True
    far = np.array([5.0, 4, 3, 2, 1], np.float32)  # argmax flipped, low cos
    assert _gate(far, {"fp32": fp32})["ok"] is False


def test_firesim_workload_mismatch_is_refused(tmp_path):
    """A ``config_runtime.yaml`` naming someone else's workload must raise, not run.

    The staging name and the booted name are two different settings, and when they disagree FireSim
    boots the OTHER workload's leftover binary and reports nothing unusual -- the run just produces the
    wrong program's output, or none. Only the no-queue path consults this file; the queue daemon is
    passed the workload explicitly and writes its own config.
    """
    from merlin.runtime.backends.zephyr_model import _check_firesim_workload

    deploy = tmp_path / "deploy"
    deploy.mkdir()
    cfg = deploy / "config_runtime.yaml"

    cfg.write_text("workload:\n  workload_name: modelblaster-firesim.json\n  terminate_on_completion: true\n")
    with pytest.raises(RuntimeError) as e:
        _check_firesim_workload(str(tmp_path), "merlin-oscar")
    assert "modelblaster-firesim.json" in str(e.value) and "merlin-oscar" in str(e.value)

    cfg.write_text("workload:\n  workload_name: merlin-oscar.json\n")
    _check_firesim_workload(str(tmp_path), "merlin-oscar")  # agrees -> silent

    # An absent config is not this check's business: FireSim itself reports it, with its own message.
    _check_firesim_workload(str(tmp_path / "nowhere"), "merlin-oscar")


def test_firesim_runner_seam_parses_and_gates_without_legacy_setup(monkeypatch):
    """An OOT runner owns execution; Merlin consumes only its UART text."""
    import sys

    zm = _zm()
    before_path = list(sys.path)
    before_env = dict(os.environ)
    received = {}

    def runner(elf, *, firesim_root, firesim_env, timeout, queue):
        received.update(elf=elf, firesim_root=firesim_root, firesim_env=firesim_env, timeout=timeout, queue=queue)
        return "OUT 2 1065353216 1073741824\nMETRIC cycles 1234\nDONE\n"

    def no_legacy(*args, **kwargs):
        raise AssertionError("the selected runner must not load the legacy adapter")

    monkeypatch.setattr(zm, "_legacy_modelblaster_firesim", no_legacy)
    result = zm.run_on_firesim(
        "image.elf",
        reference=np.array([1.0, 2.0], dtype=np.float32),
        firesim_root="/selected/firesim",
        firesim_env="/selected/env.sh",
        timeout=23,
        runner=runner,
    )
    assert received == {
        "elf": "image.elf",
        "firesim_root": "/selected/firesim",
        "firesim_env": "/selected/env.sh",
        "timeout": 23,
        "queue": True,
    }
    assert result["ok"] is True
    assert result["metrics"]["cycles"] == 1234
    assert sys.path == before_path
    assert dict(os.environ) == before_env


def test_firesim_runner_requires_uart_text():
    zm = _zm()

    def wrong_result(elf, *, firesim_root, firesim_env, timeout, queue):
        return {"ok": True}

    with pytest.raises(zm.ZephyrModelError, match="captured UART text"):
        zm.run_on_firesim("image.elf", runner=wrong_result)


def test_firesim_legacy_default_keeps_existing_call_shape(monkeypatch):
    zm = _zm()
    calls = []

    def legacy(elf, *, firesim_root, firesim_env, timeout, queue):
        calls.append((elf, timeout, queue))
        return "OUT 2 1065353216 1073741824\nMETRIC cycles 7\nDONE\n"

    monkeypatch.setattr(zm, "_legacy_modelblaster_firesim", legacy)
    result = zm.run_on_firesim("image.elf", timeout=900)
    assert calls == [("image.elf", 900, True)]
    assert result["metrics"]["cycles"] == 7


def test_console_parse_surfaces_per_op_profile():
    """A console from an ``op_profile`` build carries ``PROF <id> <ticks> <hits>``; the parser must
    hand them back, since per-op cycles are what price a unit. Absent PROF lines, the key stays absent
    rather than becoming an empty dict a caller could mistake for "measured, and it was zero"."""
    from merlin.runtime.backends.zephyr_model import _parse_console

    base = "OUT 2 1065353216 1073741824\nMETRIC cycles 1234\nDONE\n"
    res = _parse_console(base + "PROF 3 900 2\nPROF 7 100 1\n", 0)
    assert res["op_prof"] == {3: (900, 2), 7: (100, 1)}
    assert "op_prof" not in _parse_console(base, 0)
