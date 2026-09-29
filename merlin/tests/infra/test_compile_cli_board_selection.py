"""A model run may not infer an accelerator board from its execution mode."""

from __future__ import annotations

import pytest

from merlin.compile_cli import compile_rvv
from merlin.runtime.boards import BoardRegistryError


def _compile(board: str | None):
    return compile_rvv(
        "not_a_capture",
        "int8",
        run="spike",
        verify=False,
        package=None,
        auto_capture=False,
        timeout=5,
        board=board,
    )


def test_missing_board_refuses_before_capture():
    result = _compile(None)
    assert result["status"] == "not_run"
    assert "--board" in result["reason"]


def test_unknown_board_refuses_before_capture():
    with pytest.raises(BoardRegistryError, match="not in the selected catalog"):
        _compile("not_in_catalog")
