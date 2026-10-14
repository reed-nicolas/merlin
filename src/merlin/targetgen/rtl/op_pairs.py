"""Validate external RTL pair facts, keeping timing, capacity and source provenance together."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .op_timing import _Document, _pin, _text

SCHEMA = "merlin.op_pairs.v1"
SECTION = "op_pairs"


def _integer(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _check_document(document: dict[str, Any]) -> None:
    _pin(document["op_timing"], "op_timing")
    for key in ("sources", SECTION, "capacity"):
        if not isinstance(document[key], list):
            raise ValueError(f"{key} must be a list")
    for source in document["sources"]:
        _pin(source, "sources entry", source=True)
    seen = set()
    for i, block in enumerate(document[SECTION]):
        where = f"{SECTION}[{i}]"
        required = {"first", "second", "engine", "relation", "gaps", "min_gap", "stable_gap", "assertions", "source"}
        if not isinstance(block, dict) or set(block) != required:
            raise ValueError(f"{where} needs exactly {sorted(required)}")
        if any(not _text(block[key]) for key in ("first", "second", "source")):
            raise ValueError(f"{where} needs nonempty first, second and source strings")
        if block["engine"] is not None and not _text(block["engine"]):
            raise ValueError(f"{where}: engine must be a nonempty string or None")
        relation, gaps, assertions = block["relation"], block["gaps"], block["assertions"]
        if not isinstance(relation, dict) or any(not _text(k) or not isinstance(v, str) for k, v in relation.items()):
            raise ValueError(f"{where}: relation must map strings to strings")
        if not isinstance(gaps, list) or len(gaps) != 2 or not all(_integer(g, 1) for g in gaps) or gaps[0] > gaps[1]:
            raise ValueError(f"{where}: gaps must be an ordered pair of positive integers")
        for key in ("min_gap", "stable_gap"):
            gap = block[key]
            if gap is not None and (not _integer(gap, 1) or not gaps[0] <= gap <= gaps[1]):
                raise ValueError(f"{where}: {key} must be within gaps or None")
        if block["stable_gap"] is not None and (block["min_gap"] is None or block["min_gap"] > block["stable_gap"]):
            raise ValueError(f"{where}: stable_gap requires a min_gap no greater than it")
        if (
            not isinstance(assertions, dict) or set(assertions) != {"checked", "excluded"}
            or not _integer(assertions["checked"], 1) or not _integer(assertions["excluded"])
        ):
            raise ValueError(f"{where}: assertions needs positive checked and non-negative excluded integers")
        key = (block["first"], block["second"], tuple(sorted(relation.items())))
        if key in seen:
            raise ValueError(f"{where}: duplicate first, second and relation")
        seen.add(key)
    engines = set()
    for entry in document["capacity"]:
        if (
            not isinstance(entry, dict) or not {"engine", "live", "window"} <= set(entry)
            or set(entry) - {"engine", "live", "window", "witness"} or not _text(entry["engine"])
            or not _integer(entry["live"], 1) or entry["window"] not in ("next_issue_age", "last_event_age")
        ):
            raise ValueError("capacity entry needs engine, live >= 1 and a known window")
        if entry["engine"] in engines:
            raise ValueError("capacity engines must be unique")
        engines.add(entry["engine"])
        if "witness" in entry:
            witness = entry["witness"]
            if (
                not isinstance(witness, dict) or set(witness) != {"source", "gap"}
                or not _text(witness["source"]) or not _integer(witness["gap"], 1)
            ):
                raise ValueError("capacity witness needs source and a positive gap")


_DOCUMENT = _Document(
    SECTION, {"schema", "hw_ir", "op_timing", "sources", SECTION, "capacity"}, _check_document, whole=True
)


def validate_document(document: Any, *, origin: str = "op_pairs document") -> dict[str, Any]:
    """Return a valid pair document unchanged; refuse malformed records as a whole."""
    return _DOCUMENT.validate(document, origin=origin)


def read_document(path: str | Path) -> dict[str, Any]:
    """Parse and validate a producer document; unreadable input is an error, not absence."""
    return _DOCUMENT.read(path)[0]


def _check_timing(document: dict[str, Any], blocks: list[dict[str, Any]], hw_digest: str, digest: str) -> None:
    validate_document(document)
    if document["hw_ir"]["sha256"] != hw_digest:
        raise ValueError("op_pairs document: hw_ir.sha256 differs from the merged op_timing document")
    if document["op_timing"]["sha256"] != digest:
        raise ValueError("op_pairs document: op_timing.sha256 differs from the merged op_timing document")
    resolved = {b["name"]: b for b in blocks if isinstance(b, dict) and isinstance(b.get("events"), dict)}
    for record in document[SECTION]:
        if any(name not in resolved for name in (record["first"], record["second"])):
            raise ValueError("op_pairs document: first and second must name resolved op_timing blocks")
        a, b = (resolved[record[key]].get("engine") for key in ("first", "second"))
        if record["engine"] != (a if a == b else None):
            raise ValueError("op_pairs document: engine differs from the resolved op_timing blocks")
    for entry in document["capacity"]:
        windows = {
            "next_issue_age" if b.get("next_issue_age") is not None else "last_event_age"
            for b in resolved.values() if b.get("engine") == entry["engine"]
        }
        if windows != {entry["window"]}:
            raise ValueError("op_pairs document: capacity window differs from the resolved engine blocks")


def _check_merge(artifact: dict[str, Any], document: dict[str, Any], digest: str) -> None:
    inputs, facts = artifact.get("inputs") or {}, artifact.get("facts") or {}
    if not facts.get("op_timing") or not inputs.get("op_timing_sha256"):
        raise ValueError("op_pairs document: requires a merged op_timing document with its SHA-256")
    _check_timing(document, facts["op_timing"], inputs.get("op_timing_hw_ir_sha256"), inputs["op_timing_sha256"])


def merge_document(artifact: dict[str, Any], path: str | Path, *, target: str) -> dict[str, Any]:
    """Copy the artifact, replacing the whole pair document after checking both provenance pins."""
    return _DOCUMENT.merge(artifact, path, target, _check_merge)
