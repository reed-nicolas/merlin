"""L2 provenance is selected by the simulator, not by an accelerator name."""

from __future__ import annotations

from pathlib import Path

import pytest

from merlin.targetgen import evaluation_cohort as cohort
from merlin.targetgen import oracle_policy as policy


def _bound_engine(root: Path, *, engine: str, target: str) -> dict:
    root.mkdir()
    binary = root / "simulator"
    source = root / "source"
    config = root / "config.toml"
    config_tree = source / "config"
    config_tree.mkdir(parents=True)
    binary.write_bytes(engine.encode())
    config.write_text(f"engine = {engine!r}\n")
    (config_tree / "timing.toml").write_text("latency = 4\n")
    binding = cohort.configured_executable_binding(
        engine=engine, binary=binary, source=source, config=config, config_tree=config_tree
    )
    binding["target"] = target
    binding["binding_sha256"] = cohort._canonical_json_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )
    return binding


@pytest.fixture
def selected_oracles(monkeypatch):
    monkeypatch.setattr(policy, "_SIM_ORACLES", {})
    monkeypatch.setattr(policy, "_ensure_sim_oracles_discovered", lambda: None)
    engines = {"alpha_target": "engine_a", "beta_target": "engine_b"}
    monkeypatch.setattr(cohort, "_declared_l2_engine", engines.__getitem__)
    return engines


def _register(engine: str, binding):
    policy.register_sim_oracle(
        engine,
        adapters=lambda _target: {},
        available=lambda _target: (True, "synthetic"),
        exclusive=True,
        l2_binding=binding,
    )


def test_distinct_selected_engines_bind_distinct_executable_bytes(tmp_path, monkeypatch, selected_oracles):
    alpha = _bound_engine(tmp_path / "alpha", engine="engine_a", target="alpha_target")
    beta = _bound_engine(tmp_path / "beta", engine="engine_b", target="beta_target")
    seen = []
    _register("engine_a", lambda target: (seen.append(("a", target)), alpha)[1])
    _register("engine_b", lambda target: (seen.append(("b", target)), beta)[1])

    from merlin.runtime.backends import base

    monkeypatch.setattr(base, "get_backend", lambda _name: pytest.fail("L2 provenance must use the oracle seam"))
    assert cohort.selected_l2_engine_binding("alpha_target") == alpha
    assert cohort.selected_l2_engine_binding("beta_target") == beta
    assert seen == [("a", "alpha_target"), ("b", "beta_target")]
    assert alpha["binary"]["sha256"] != beta["binary"]["sha256"]


def test_missing_or_wrong_selected_binding_refuses(tmp_path, selected_oracles):
    _register("engine_a", None)
    with pytest.raises(ValueError, match="no registered l2_binding"):
        cohort.selected_l2_engine_binding("alpha_target")

    wrong = _bound_engine(tmp_path / "wrong", engine="engine_b", target="alpha_target")
    _register("engine_a", lambda _target: wrong)
    with pytest.raises(ValueError, match="no valid selected L2 engine binding"):
        cohort.selected_l2_engine_binding("alpha_target")


def test_selected_binding_refuses_changed_executable_and_config(tmp_path, selected_oracles):
    binding = _bound_engine(tmp_path / "alpha", engine="engine_a", target="alpha_target")
    _register("engine_a", lambda _target: binding)
    assert cohort.selected_l2_engine_binding("alpha_target") == binding

    binary = Path(binding["binary"]["path"])
    binary.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="executable content digest mismatch"):
        cohort.selected_l2_engine_binding("alpha_target")

    binary.write_bytes(b"engine_a")
    config_member = Path(binding["config"]["tree"]) / "timing.toml"
    config_member.write_text("latency = 9\n")
    with pytest.raises(ValueError, match="config tree content digest mismatch"):
        cohort.selected_l2_engine_binding("alpha_target")
