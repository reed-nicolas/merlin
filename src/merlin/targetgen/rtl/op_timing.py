"""Validate external per-operation RTL timing; unresolved events carry no ages."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA = "merlin.op_timing.v1"
SECTION = "op_timing"
_EVENT_AGES = ("first_age", "last_age", "step")


def _age(value: Any) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_block(block: Any, index: int) -> str:
    where = f"{SECTION}[{index}]"
    if not isinstance(block, dict):
        raise ValueError(f"{where} must be a mapping")
    name = block.get("name")
    if not _text(name):
        raise ValueError(f"{where} needs a nonempty name")
    where = f"{SECTION} block {name!r}"
    for key in ("module", "source", "evidence"):
        if not _text(block.get(key)):
            raise ValueError(f"{where} needs a nonempty {key}")
    if not isinstance(block.get("assumptions"), dict):
        raise ValueError(f"{where} needs an assumptions mapping")
    events = block.get("events")
    ages = [block.get("first_free_age"), block.get("next_issue_age")]
    if not all(_age(value) for value in ages):
        raise ValueError(f"{where}: first_free_age and next_issue_age are non-negative integers or None")
    if events is None:
        if any(value is not None for value in ages):
            raise ValueError(f"{where} is unresolved (events None) but carries ages; unknown is None throughout")
        return name
    if not isinstance(events, dict):
        raise ValueError(f"{where}: events must be a mapping or None")
    for group, event in events.items():
        count = event.get("count") if isinstance(event, dict) else None
        if not (isinstance(count, int) and not isinstance(count, bool) and count >= 0) or not all(
            _age(event.get(key)) for key in _EVENT_AGES
        ):
            raise ValueError(f"{where}: event {group!r} needs an integer count and non-negative integer or None ages")
    return name


def _check_document(document: dict[str, Any]) -> None:
    blocks = document[SECTION]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{SECTION} must be a nonempty list: an empty section would read as 'no operations'")
    names = [_check_block(block, i) for i, block in enumerate(blocks)]
    if len(set(names)) != len(names):
        raise ValueError(f"{SECTION} block names must be unique")


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _pin(value: Any, where: str, *, source: bool = False) -> None:
    keys = {"path", "sha256", "schema", "specs"} if source else {"path", "sha256"}
    if (
        not isinstance(value, dict) or set(value) != keys or not _text(value["path"]) or not _sha256(value["sha256"])
        or (source and value["schema"] not in ("rtl_extract.op_pairs.v1", "rtl_extract.op_sequences.v1"))
    ):
        raise ValueError(f"{where} needs {sorted(keys)} with a lowercase hex SHA-256 and a supported schema")
    if source:
        specs = value["specs"]
        if not isinstance(specs, list) or not specs or not all(
                isinstance(s, dict) and set(s) == {"name", "sha256"} and _text(s["name"]) and _sha256(s["sha256"]) for s in specs):
            raise ValueError(f"{where}: specs must be a nonempty list of {{name, sha256}} with lowercase hex SHA-256")
        if len({s["name"] for s in specs}) != len(specs):
            raise ValueError(f"{where}: specs names must be unique")


class _Document:
    """Shared envelope validation, exact-byte reading and atomic merge for operation facts."""

    def __init__(self, section: str, keys: set[str], check: Callable, *, whole: bool = False):
        self.section, self.schema = section, f"merlin.{section}.v1"
        self.keys, self.check, self.whole = keys, check, whole

    def validate(self, document: Any, *, origin: str) -> dict[str, Any]:
        try:
            if not isinstance(document, dict) or set(document) != self.keys:
                raise ValueError(f"must be a mapping with exactly {sorted(self.keys)}")
            if document["schema"] != self.schema:
                raise ValueError(f"schema must be {self.schema!r}")
            _pin(document["hw_ir"], "hw_ir")
            self.check(document)
        except ValueError as error:
            raise ValueError(f"{origin}: {error}") from None
        return document

    def read(self, path: str | Path) -> tuple[dict[str, Any], str]:
        try:
            raw = Path(path).read_bytes()
            document = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{path}: cannot read an {self.section} document: {error}") from None
        return self.validate(document, origin=str(path)), hashlib.sha256(raw).hexdigest()

    def merge(self, artifact: dict[str, Any], path: str | Path, target: str, check: Callable) -> dict[str, Any]:
        from .extraction_contract import _operation_source

        if _operation_source(target, self.section, self.schema) is None:
            raise ValueError(
                f"{target}: the target contract declares no rtl_extraction.{self.section}; refusing {path}"
            )
        document, digest = self.read(path)
        check(artifact, document, digest)
        merged = copy.deepcopy(artifact)
        payload = document if self.whole else document[self.section]
        merged["facts"] = {**(merged.get("facts") or {}), self.section: copy.deepcopy(payload)}
        inputs = {"schema": document["schema"], "path": str(path), "sha256": digest}
        for key in (("hw_ir", "op_timing") if self.whole else ("hw_ir",)):
            inputs.update({f"{key}_{field}": value for field, value in document[key].items()})
        merged["inputs"] = {**(merged.get("inputs") or {}), **{f"{self.section}_{k}": v for k, v in inputs.items()}}
        return merged


_DOCUMENT = _Document(SECTION, {"schema", "hw_ir", SECTION}, _check_document)


def validate_document(document: Any, *, origin: str = "op_timing document") -> dict[str, Any]:
    """Return a valid timing document unchanged; refuse malformed blocks as a whole."""
    return _DOCUMENT.validate(document, origin=origin)


def read_document(path: str | Path) -> dict[str, Any]:
    """Parse and validate a producer document; unreadable input is an error, not absence."""
    return _DOCUMENT.read(path)[0]


def _check_merge(artifact: dict[str, Any], document: dict[str, Any], digest: str) -> None:
    pairs = (artifact.get("facts") or {}).get("op_pairs")
    if pairs is not None:
        from .op_pairs import _check_timing

        _check_timing(pairs, document[SECTION], document["hw_ir"]["sha256"], digest)


def merge_document(artifact: dict[str, Any], path: str | Path, *, target: str) -> dict[str, Any]:
    """Copy the artifact, replacing timing only when any existing pair provenance still matches."""
    return _DOCUMENT.merge(artifact, path, target, _check_merge)
