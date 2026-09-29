"""Installed phase children cannot inherit unrecorded board facts."""

from __future__ import annotations

import pytest

from merlin_experiments.measured_launch import execution_environment
from merlin_experiments.spec import SpecError


@pytest.mark.parametrize("module", ["merlin_experiments.phase1", "merlin_experiments.phase2.portfolio_cli"])
def test_installed_phase_child_strips_ambient_board_catalog(module, monkeypatch):
    monkeypatch.setenv("MERLIN_BOARD_CATALOG", "/mutable/board.yaml")
    command = {"module": module, "env": {"MERLIN_REPO_ROOT": "/sealed/source"}}
    assert "MERLIN_BOARD_CATALOG" not in execution_environment(command)
    command["env"]["MERLIN_BOARD_CATALOG"] = "/mutable/board.yaml"
    with pytest.raises(SpecError, match="unsealed board catalog"):
        execution_environment(command)


def test_historical_native_command_keeps_its_original_environment(monkeypatch):
    monkeypatch.setenv("MERLIN_BOARD_CATALOG", "/historical/board.yaml")
    assert execution_environment({"module": None, "env": {}})["MERLIN_BOARD_CATALOG"] == "/historical/board.yaml"
