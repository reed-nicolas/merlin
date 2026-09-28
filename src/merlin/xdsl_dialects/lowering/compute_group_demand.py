"""State accelerator compute groups as deduplicated capsule demands."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from typing import Any


def demand(groups: Sequence[Any], *, weight_args: Collection[int] | None = None) -> dict[str, Any]:
    """Report every distinct accelerator program and every unstateable group.

    The capsule uses device form: its stored right operand, target convolution
    form, and readout stages. The capture's framework form is not the backend's
    actual program. Imports are local to keep group formation independent of
    capsule serialization.
    """
    from . import compute_groups, group_command

    entries: dict[str, dict[str, Any]] = {}
    unstated: Counter = Counter()
    for group in groups:
        if group.placement == compute_groups.HOST:
            continue
        if group.root is None:
            unstated["a region placed on a unit has no contraction root to state"] += 1
            continue
        try:
            entry = group_command.program(group, weight_args=weight_args).entry
        except compute_groups.NoCapsuleForm as error:
            unstated[str(error)] += 1
            continue
        numbers = ("name", "acc_scale", "lhs_scale", "rhs_scale")
        key = repr(sorted((k, repr(v)) for k, v in entry.items() if k not in numbers))
        entries.setdefault(key, {**entry, "count": 0})["count"] += 1
    return {"entries": sorted(entries.values(), key=lambda e: -e["count"]), "unstated": dict(unstated)}
