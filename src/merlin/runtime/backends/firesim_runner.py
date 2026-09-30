"""Select an installed FireSim runner without importing target-owned code in core.

The runner owns staging, queue submission, and simulator-specific environment.
Merlin owns UART parsing and numerical gating after the runner returns text.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol

from merlin.common.paths import env

RUNNER_GROUP = "merlin.firesim_runners"


class FireSimRunner(Protocol):
    def __call__(
        self,
        elf: str,
        *,
        firesim_root: str,
        firesim_env: str,
        timeout: int,
        queue: bool,
    ) -> str: ...


class FireSimRunnerError(ValueError):
    """An installed runner was not explicitly or unambiguously selected."""


def select_runner(name: str | None = None) -> FireSimRunner:
    """Load exactly one named, installed runner; never infer one from a checkout.

    The caller may pass ``name`` or set ``MERLIN_FIRESIM_RUNNER`` in the
    environment/.env. An installed package alone does not select a runner.
    """
    selected = name or env("MERLIN_FIRESIM_RUNNER")
    if not selected:
        raise FireSimRunnerError(
            "install a FireSim runner distribution with `python -m pip install /path/to/runner.whl` "
            "(entry-point group merlin.firesim_runners), then select its name with runner_name= "
            "or MERLIN_FIRESIM_RUNNER; no checkout or target is an implicit execution provider"
        )
    providers = tuple(entry_points(group=RUNNER_GROUP))
    matches = tuple(provider for provider in providers if provider.name == selected)
    if len(matches) != 1:
        available = sorted(provider.name for provider in providers)
        raise FireSimRunnerError(
            f"FireSim runner {selected!r} has {len(matches)} installed providers "
            f"(expected exactly one; available: {available}). Install a runner wheel if missing, "
            "or remove duplicate installed providers"
        )
    try:
        runner = matches[0].load()
    except Exception as exc:  # noqa: BLE001 - identify selected distribution failure
        raise FireSimRunnerError(f"cannot load FireSim runner {selected!r}: {type(exc).__name__}: {exc}") from exc
    if not callable(runner):
        raise FireSimRunnerError(f"FireSim runner {selected!r} is not callable")
    return runner
