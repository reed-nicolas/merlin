"""Selected OOT instruction semantics for exploratory, target-independent selection.

This is an authored description, not a proof that RTL implements the arithmetic.
The selected software specification owns numerical legality; the selected CIRCT
facts own only the hardware facts they actually extracted.  Both input identities
travel with the normalized description so Phase 0 can qualify claims separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from merlin.common.schemas import validate_or_raise

SCHEMA = "merlin.instruction_semantics.v1"
UNKNOWN = "UNKNOWN"
_SPACE_FIELDS = ("capacity_bytes", "alignment_bytes", "banks", "bank_width_bytes")
_PARAM_FIELDS = frozenset({"type", "min", "max", "multiple_of", "choices"})
_CONSTRAINT_FIELDS = frozenset({"operand_spaces", "result_spaces", "distinct_banks"})
_CHECKED_SW_SIGNATURE_FIELDS = frozenset(
    {
        "operand_dtypes",
        "dtypes",
        "accumulator_dtype",
        "readout_dtype",
        "ranks",
        "ordered_operand_dtypes",
        "ordered_result_dtypes",
    }
)
_DTYPE_ALIASES = {"int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64", "fp32": "f32"}


def _canonical(value: Any) -> bytes:
    """Canonical JSON identity; rejects YAML-only scalars and non-finite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _snapshot(value: bytes | Mapping[str, Any] | None, *, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bytes):
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(f"{label}: cannot parse selected bytes: {exc}") from exc
        digest = hashlib.sha256(value).hexdigest()
    elif isinstance(value, Mapping):
        parsed = copy.deepcopy(dict(value))
        digest = hashlib.sha256(_canonical(parsed)).hexdigest()
    else:
        raise ValueError(f"{label}: expected selected bytes or a mapping")
    if not isinstance(parsed, dict):
        raise ValueError(f"{label}: expected a mapping")
    _canonical(parsed)
    return parsed, digest


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _names(rows: Any, *, label: str) -> list[dict[str, str]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "type"}:
            raise ValueError(f"{label}[{index}] requires only name and type")
        name = _nonempty(row["name"], f"{label}[{index}].name")
        typ = _nonempty(row["type"], f"{label}[{index}].type")
        if name in seen:
            raise ValueError(f"{label}: duplicate name {name!r}")
        seen.add(name)
        result.append({"name": name, "type": typ})
    return result


def _scalar_body(body: Any, *, label: str, unknowns: list[str]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError(f"{label} must be a mapping")
    arguments = body.get("arguments")
    captures = body.get("captures", [])
    operations = body.get("operations")
    yields = body.get("yields")
    if not isinstance(arguments, list) or any(not isinstance(t, str) or not t for t in arguments):
        raise ValueError(f"{label}.arguments must be MLIR type strings")
    if not isinstance(captures, list):
        raise ValueError(f"{label}.captures must be a list")
    normalized_captures = []
    for index, capture in enumerate(captures):
        site = f"{label}.captures[{index}]"
        if not isinstance(capture, dict) or set(capture) - {"source", "shape", "dtype", "result_index", "const_value"}:
            raise ValueError(f"{site} has unsupported fields")
        source = capture.get("source")
        shape = capture.get("shape")
        if (
            not isinstance(source, (list, tuple))
            or len(source) != 2
            or source[0] not in {"arg", "op", "init", "const", "other"}
            or not isinstance(source[1], (int, str))
        ):
            raise ValueError(f"{site}.source must identify a captured value")
        if not isinstance(shape, list) or any(type(dim) is not int for dim in shape):
            raise ValueError(f"{site}.shape must be integer dimensions")
        dtype = _nonempty(capture.get("dtype"), f"{site}.dtype")
        result_index = capture.get("result_index")
        if result_index is not None and (type(result_index) is not int or result_index < 0):
            raise ValueError(f"{site}.result_index must be nonnegative")
        const_value = capture.get("const_value")
        if "const_value" in capture and (type(const_value) not in (int, float) or isinstance(const_value, bool)):
            raise ValueError(f"{site}.const_value must be numeric")
        normalized_capture = {"source": list(source), "shape": list(shape), "dtype": dtype}
        if result_index is not None:
            normalized_capture["result_index"] = result_index
        if "const_value" in capture:
            normalized_capture["const_value"] = const_value
        normalized_captures.append(normalized_capture)
    if not isinstance(operations, list) or not isinstance(yields, list):
        raise ValueError(f"{label} requires operations and yields lists")
    available = {f"arg:{index}" for index in range(len(arguments))}
    available.update(f"capture:{index}" for index in range(len(captures)))
    normalized_ops = []
    for index, operation in enumerate(operations):
        site = f"{label}.operations[{index}]"
        if not isinstance(operation, dict):
            raise ValueError(f"{site} must be a mapping")
        op = _nonempty(operation.get("op"), f"{site}.op")
        operands = operation.get("operands")
        results = operation.get("results")
        attrs = operation.get("attributes", {})
        regions = operation.get("regions", 0)
        if not isinstance(operands, list) or any(ref not in available for ref in operands):
            raise ValueError(f"{site}.operands must reference preceding scalar values")
        if not isinstance(results, list) or any(not isinstance(t, str) or not t for t in results):
            raise ValueError(f"{site}.results must be MLIR type strings")
        if not isinstance(attrs, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in attrs.items()
        ):
            raise ValueError(f"{site}.attributes must map names to MLIR attribute strings")
        if type(regions) is not int or regions < 0:
            raise ValueError(f"{site}.regions must be a nonnegative integer")
        effects = operation.get("effects")
        if effects is not None and (
            not isinstance(effects, list) or any(not isinstance(effect, str) or not effect for effect in effects)
        ):
            raise ValueError(f"{site}.effects must be a string list or null")
        if effects is None:
            unknowns.append(f"{site}.effects")
        if regions:
            unknowns.append(f"{site}.nested_regions")
        normalized_ops.append(
            {
                "op": op,
                "operands": list(operands),
                "results": list(results),
                "attributes": dict(sorted(attrs.items())),
                "effects": effects,
                "regions": regions,
            }
        )
        available.update(f"op:{index}:{result}" for result in range(len(results)))
    if any(ref not in available for ref in yields):
        raise ValueError(f"{label}.yields must reference scalar values")
    effects = body.get("effects")
    if effects is not None and (
        not isinstance(effects, list) or any(not isinstance(effect, str) or not effect for effect in effects)
    ):
        raise ValueError(f"{label}.effects must be a string list or null")
    if effects is None:
        unknowns.append(f"{label}.effects")
    return {
        "arguments": list(arguments),
        "captures": normalized_captures,
        "operations": normalized_ops,
        "yields": list(yields),
        "effects": effects,
    }


def _computation(value: Any, *, label: str, unknowns: list[str]) -> dict[str, Any]:
    if value is None:
        unknowns.append(f"{label}.missing")
        return {"status": UNKNOWN, "reason": "instruction computation was not described"}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if value.get("status") == UNKNOWN:
        reason = _nonempty(value.get("reason"), f"{label}.reason")
        unknowns.append(f"{label}.{reason}")
        return {"status": UNKNOWN, "reason": reason}
    kind = _nonempty(value.get("kind"), f"{label}.kind")
    if kind != "linalg.generic":
        raise ValueError(f"{label}.kind {kind!r} is not described by v1")
    maps = value.get("indexing_maps")
    maps_explicit = value.get("indexing_maps_explicit")
    iterators = value.get("iterator_types")
    if maps is not None and (not isinstance(maps, list) or any(not isinstance(item, str) or not item for item in maps)):
        raise ValueError(f"{label}.indexing_maps must be MLIR attribute strings or null")
    if iterators is not None and (
        not isinstance(iterators, list) or any(item not in {"parallel", "reduction"} for item in iterators)
    ):
        raise ValueError(f"{label}.iterator_types must be parallel/reduction strings or null")
    if maps is None:
        unknowns.append(f"{label}.indexing_maps")
    if maps_explicit is not None and type(maps_explicit) is not bool:
        raise ValueError(f"{label}.indexing_maps_explicit must be Boolean or null")
    if iterators is None:
        unknowns.append(f"{label}.iterator_types")
    return {
        "kind": kind,
        "indexing_maps": maps,
        "indexing_maps_explicit": maps_explicit,
        "iterator_types": iterators,
        "scalar_body": _scalar_body(value.get("scalar_body"), label=f"{label}.scalar_body", unknowns=unknowns),
    }


def _parameters(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    result = {}
    for name, domain in sorted(value.items()):
        _nonempty(name, f"{label} name")
        if not isinstance(domain, dict) or set(domain) - _PARAM_FIELDS or domain.get("type") != "integer":
            raise ValueError(f"{label}.{name} requires an integer domain")
        for field in ("min", "max", "multiple_of"):
            if field in domain and (type(domain[field]) is not int or (field == "multiple_of" and domain[field] < 1)):
                raise ValueError(f"{label}.{name}.{field} must be an integer of the required range")
        if "min" in domain and "max" in domain and domain["min"] > domain["max"]:
            raise ValueError(f"{label}.{name}: min exceeds max")
        if "choices" in domain and (
            not isinstance(domain["choices"], list)
            or not domain["choices"]
            or any(type(choice) is not int for choice in domain["choices"])
        ):
            raise ValueError(f"{label}.{name}.choices must be nonempty integer list")
        result[name] = copy.deepcopy(domain)
    return result


def _memory_spaces(value: Any, unknowns: list[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("memory_spaces must be a mapping")
    result = {}
    for name, declaration in sorted(value.items()):
        _nonempty(name, "memory space name")
        if not isinstance(declaration, dict) or set(declaration) - set(_SPACE_FIELDS):
            raise ValueError(f"memory_spaces.{name} has unsupported fields")
        row = {}
        for field in _SPACE_FIELDS:
            item = declaration.get(field, UNKNOWN)
            if item != UNKNOWN and (type(item) is not int or item < 1):
                raise ValueError(f"memory_spaces.{name}.{field} must be positive or UNKNOWN")
            if item == UNKNOWN:
                unknowns.append(f"memory_spaces.{name}.{field}")
            row[field] = item
        result[name] = row
    return result


def _constraints(value: Any, *, label: str, names: set[str], spaces: dict[str, Any]) -> dict[str, Any]:
    if value is None:
        return {"operand_spaces": {}, "result_spaces": {}, "distinct_banks": []}
    if not isinstance(value, dict) or set(value) - _CONSTRAINT_FIELDS:
        raise ValueError(f"{label} has unsupported fields")
    result = {}
    for field in ("operand_spaces", "result_spaces"):
        locations = value.get(field, {})
        if not isinstance(locations, dict):
            raise ValueError(f"{label}.{field} must be a mapping")
        for name, space in locations.items():
            if name not in names or space not in spaces:
                raise ValueError(f"{label}.{field} references undeclared value or space")
        result[field] = dict(sorted(locations.items()))
    distinct = value.get("distinct_banks", [])
    if (
        not isinstance(distinct, list)
        or any(name not in names for name in distinct)
        or len(set(distinct)) != len(distinct)
    ):
        raise ValueError(f"{label}.distinct_banks must name distinct declared values")
    result["distinct_banks"] = list(distinct)
    return result


def _effects(
    value: Any, *, label: str, names: set[str], spaces: dict[str, Any], unknowns: list[str]
) -> list[dict[str, Any]] | None:
    if value is None:
        unknowns.append(f"{label}.missing")
        return None
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list or null")
    result = []
    for index, effect in enumerate(value):
        site = f"{label}[{index}]"
        if not isinstance(effect, dict) or set(effect) - {"kind", "space", "operand", "result", "range"}:
            raise ValueError(f"{site} has unsupported fields")
        kind = effect.get("kind")
        space = effect.get("space")
        if kind not in {"read", "write"} or space not in spaces:
            raise ValueError(f"{site} requires read/write and a declared memory space")
        reference = effect.get("operand" if kind == "read" else "result")
        if reference not in names or ("operand" in effect and "result" in effect):
            raise ValueError(f"{site} requires one declared read operand or write result")
        span = effect.get("range", UNKNOWN)
        if span == UNKNOWN:
            unknowns.append(f"{site}.range")
        elif (
            not isinstance(span, dict)
            or set(span) != {"offset_bytes", "length_bytes"}
            or any(
                type(span[field]) is not int or span[field] < (1 if field == "length_bytes" else 0)
                for field in ("offset_bytes", "length_bytes")
            )
        ):
            raise ValueError(f"{site}.range requires nonnegative offset and positive length")
        result.append(
            {"kind": kind, "space": space, "operand" if kind == "read" else "result": reference, "range": span}
        )
    return result


def _effect_coverage_unknowns(constraints: dict[str, Any], effects: list[dict[str, Any]] | None) -> list[str]:
    if effects is None:
        return []
    observed_reads = {(row["operand"], row["space"]) for row in effects if row["kind"] == "read"}
    observed_writes = {(row["result"], row["space"]) for row in effects if row["kind"] == "write"}
    unknowns = [
        f"effects.uncovered_read.{name}"
        for name, space in constraints["operand_spaces"].items()
        if (name, space) not in observed_reads
    ]
    unknowns.extend(
        f"effects.uncovered_write.{name}"
        for name, space in constraints["result_spaces"].items()
        if (name, space) not in observed_writes
    )
    return unknowns


def _mlir_dtype_rank(type_string: str) -> tuple[str, int] | None:
    """Read simple ranked MLIR tensor/vector types; unfamiliar syntax stays unknown."""
    source = type_string.strip()
    if source.startswith(("tensor<", "vector<")) and source.endswith(">"):
        body = source[source.index("<") + 1 : -1]
        if "," in body or body.startswith("*"):
            return None
        parts = body.split("x")
        if len(parts) < 2 or any(not part or (part != "?" and not part.isdecimal()) for part in parts[:-1]):
            return None
        return parts[-1], len(parts) - 1
    if "<" in source or ">" in source or not source:
        return None
    return source, 0


def _software_signature_unknowns(
    declaration: dict[str, Any], operands: list[dict[str, str]], results: list[dict[str, str]]
) -> list[str]:
    """Screen only type/rank axes we can compare without guessing numerics."""
    signature = declaration.get("signature")
    if not isinstance(signature, dict):
        signature = {name: declaration[name] for name in _CHECKED_SW_SIGNATURE_FIELDS if name in declaration}
    if not signature:
        return ["software_signature_missing"]
    unknowns = [
        f"software_signature_unchecked.{field}" for field in sorted(set(signature) - _CHECKED_SW_SIGNATURE_FIELDS)
    ]
    if any(field in declaration for field in ("ops", "families", "family")):
        unknowns.append("software_selector_unchecked")
    parsed_operands = [_mlir_dtype_rank(row["type"]) for row in operands]
    parsed_results = [_mlir_dtype_rank(row["type"]) for row in results]
    if any(item is None for item in parsed_operands + parsed_results):
        return [*unknowns, "software_signature_unparsed_mlir_type"]

    def _normalize_dtype(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return _DTYPE_ALIASES.get(value, value)

    def _check_types(rows: list[tuple[str, int]], *, ordered_field: str, allowed_fields: tuple[str, ...], role: str):
        ordered = signature.get(ordered_field)
        if ordered is not None:
            if not isinstance(ordered, list) or len(ordered) != len(rows):
                unknowns.append(f"software_signature_{role}_arity_mismatch")
                return
            for index, (observed, _) in enumerate(rows):
                if _normalize_dtype(ordered[index]) != observed:
                    unknowns.append(f"software_signature_{role}_dtype_mismatch.{index}")
            return
        allowed = set()
        for field in allowed_fields:
            declared = signature.get(field)
            if isinstance(declared, list):
                allowed.update(_normalize_dtype(item) for item in declared)
            elif declared is not None:
                allowed.add(_normalize_dtype(declared))
        if not allowed:
            unknowns.append(f"software_signature_{role}_dtype_unconstrained")
            return
        for index, (observed, _) in enumerate(rows):
            if observed not in allowed:
                unknowns.append(f"software_signature_{role}_dtype_mismatch.{index}")

    _check_types(
        parsed_operands,
        ordered_field="ordered_operand_dtypes",
        allowed_fields=("operand_dtypes", "dtypes"),
        role="operand",
    )
    _check_types(
        parsed_results,
        ordered_field="ordered_result_dtypes",
        allowed_fields=("readout_dtype", "accumulator_dtype", "dtypes"),
        role="result",
    )
    ranks = signature.get("ranks")
    if ranks is not None:
        if not isinstance(ranks, list) or any(type(rank) is not int or rank < 0 for rank in ranks):
            unknowns.append("software_signature_ranks_invalid")
        else:
            for role, rows in (("operand", parsed_operands), ("result", parsed_results)):
                for index, (_, rank) in enumerate(rows):
                    if rank not in ranks:
                        unknowns.append(f"software_signature_{role}_rank_mismatch.{index}")
    return unknowns


def normalize_instruction_semantics(
    document: Mapping[str, Any],
    *,
    software_spec: bytes | Mapping[str, Any] | None,
    rtl_facts: bytes | Mapping[str, Any] | None,
    target: str,
    source_bytes: bytes | None = None,
    software_source_bytes: bytes | None = None,
    rtl_source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate an authored v1 description against exactly supplied selected inputs.

    No target lookup or RTL extraction occurs here. Mapping inputs use canonical JSON
    identities; callers with frozen input bytes should pass those bytes for exact hashes.
    Unknowns are retained so exploratory search cannot silently become admission.
    """
    if not isinstance(document, Mapping):
        raise ValueError("instruction semantics must be a mapping")
    source = copy.deepcopy(dict(document))
    validate_or_raise(source, "instruction_semantics")
    if source.get("schema") != SCHEMA:
        raise ValueError(f"instruction semantics must declare schema {SCHEMA}")
    if source.get("target") != _nonempty(target, "target"):
        raise ValueError("instruction semantics target differs from selected target")
    if set(source) - {"schema", "target", "instructions", "memory_spaces"}:
        raise ValueError("instruction semantics has unsupported top-level fields")
    sw, sw_digest = _snapshot(software_spec, label="selected software spec")
    facts, facts_digest = _snapshot(rtl_facts, label="selected CIRCT facts")
    # Phase 0's absent optional observations are represented as empty mappings.
    if sw == {} and software_source_bytes is None:
        sw, sw_digest = None, None
    if facts == {} and rtl_source_bytes is None:
        facts, facts_digest = None, None
    if sw is not None and (sw.get("target") != target or sw.get("schema") != "merlin.software_spec.v1"):
        raise ValueError("selected software spec does not match the target or schema")
    if facts is not None and not isinstance(facts.get("facts"), dict):
        raise ValueError("selected CIRCT facts require a facts mapping")
    if software_source_bytes is not None:
        if sw is None:
            raise ValueError("software source bytes require a selected software spec")
        from merlin.targetgen.software_spec import validate_software_spec

        raw_software = yaml.safe_load(software_source_bytes)
        if validate_software_spec(raw_software, target=target) != sw:
            raise ValueError("selected software spec differs from supplied source bytes")
        sw_digest = hashlib.sha256(software_source_bytes).hexdigest()
    if rtl_source_bytes is not None:
        if facts is None or yaml.safe_load(rtl_source_bytes) != facts:
            raise ValueError("selected CIRCT facts differ from supplied source bytes")
        facts_digest = hashlib.sha256(rtl_source_bytes).hexdigest()
    unknowns: list[str] = []
    if sw is None:
        unknowns.append("selected_software_spec_missing")
    elif sw.get("status") != "reviewed":
        unknowns.append("selected_software_spec_unreviewed")
    if facts is None:
        unknowns.append("selected_circt_facts_missing")
    spaces = _memory_spaces(source.get("memory_spaces"), unknowns)
    instructions = source.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("instructions must be a nonempty list")
    declared_operations = sw.get("operations") if sw is not None else None
    if isinstance(declared_operations, dict):
        sw_operations = {
            identity: declaration
            for identity, declaration in declared_operations.items()
            if isinstance(identity, str) and isinstance(declaration, dict)
        }
    elif isinstance(declared_operations, list):
        sw_operations = {
            row["id"]: row for row in declared_operations if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
    else:
        sw_operations = {}
    normalized = []
    identities = set()
    for index, instruction in enumerate(instructions):
        site = f"instructions[{index}]"
        if not isinstance(instruction, dict) or set(instruction) - {
            "id",
            "operands",
            "results",
            "computation",
            "parameters",
            "constraints",
            "effects",
            "software_operation",
        }:
            raise ValueError(f"{site} has unsupported fields")
        identity = _nonempty(instruction.get("id"), f"{site}.id")
        if identity in identities:
            raise ValueError(f"duplicate instruction id {identity!r}")
        identities.add(identity)
        operands = _names(instruction.get("operands"), label=f"{site}.operands")
        results = _names(instruction.get("results"), label=f"{site}.results")
        names = {row["name"] for row in operands + results}
        if len(names) != len(operands) + len(results):
            raise ValueError(f"{site}: operands and results share a name")
        row_unknowns: list[str] = []
        computation = _computation(instruction.get("computation"), label=f"{site}.computation", unknowns=row_unknowns)
        software_operation = instruction.get("software_operation")
        software_declaration = None
        if software_operation is None:
            row_unknowns.append("software_operation_unlinked")
        elif sw is None and isinstance(software_operation, str) and software_operation:
            row_unknowns.append("software_operation_unverifiable")
        elif not isinstance(software_operation, str) or software_operation not in sw_operations:
            raise ValueError(f"{site}.software_operation must reference a selected SW operation ID")
        else:
            software_declaration = copy.deepcopy(sw_operations[software_operation])
            software_declaration["id"] = software_operation
            if software_declaration.get("placement") not in {"accelerator", "fused_accelerator"}:
                row_unknowns.append("software_operation_not_accelerator")
            row_unknowns.extend(_software_signature_unknowns(software_declaration, operands, results))
        constraints = _constraints(
            instruction.get("constraints"), label=f"{site}.constraints", names=names, spaces=spaces
        )
        effects = _effects(
            instruction.get("effects"), label=f"{site}.effects", names=names, spaces=spaces, unknowns=row_unknowns
        )
        row_unknowns.extend(_effect_coverage_unknowns(constraints, effects))
        row = {
            "id": identity,
            "operands": operands,
            "results": results,
            "computation": computation,
            "parameters": _parameters(instruction.get("parameters"), label=f"{site}.parameters"),
            "constraints": constraints,
            "effects": effects,
            "software_operation": software_operation,
            "software_declaration": software_declaration,
            "status": UNKNOWN if row_unknowns else "described",
            "unknowns": sorted(set(row_unknowns)),
        }
        normalized.append(row)
        unknowns.extend(f"{identity}.{item}" for item in row_unknowns)
    normalized.sort(key=lambda row: row["id"])
    digest = hashlib.sha256(source_bytes if source_bytes is not None else _canonical(source)).hexdigest()
    model = {
        "schema": SCHEMA,
        "target": target,
        "source_sha256": digest,
        "software_spec_sha256": sw_digest,
        "circt_facts_sha256": facts_digest,
        "memory_spaces": spaces,
        "instructions": normalized,
        "status": UNKNOWN if unknowns else "described",
        "unknowns": sorted(set(unknowns)),
    }
    model["canonical_sha256"] = hashlib.sha256(_canonical(model)).hexdigest()
    return model


def load_instruction_semantics(
    path: str | Path,
    *,
    software_spec: bytes | Mapping[str, Any] | None,
    rtl_facts: bytes | Mapping[str, Any] | None,
    target: str,
    software_source_bytes: bytes | None = None,
    rtl_source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Read only one explicitly selected OOT resource path."""
    selected = Path(path)
    source_bytes = selected.read_bytes()
    try:
        document = yaml.safe_load(source_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"{selected}: invalid instruction semantics: {exc}") from exc
    return normalize_instruction_semantics(
        document,
        software_spec=software_spec,
        rtl_facts=rtl_facts,
        target=target,
        source_bytes=source_bytes,
        software_source_bytes=software_source_bytes,
        rtl_source_bytes=rtl_source_bytes,
    )


def _digest_field(value: Any, *, label: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


def _unknown_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a string list")
    if value != sorted(set(value)):
        raise ValueError(f"{label} must be sorted and unique")
    return value


def validate_normalized_instruction_model(
    model: Mapping[str, Any],
    *,
    expected_target: str | None = None,
    expected_source_sha256: str | None = None,
    expected_software_spec_sha256: str | None = None,
    expected_circt_facts_sha256: str | None = None,
) -> dict[str, Any]:
    """Check a frozen model's structure and self-consistency before search.

    Optional expected identities bind the model to independently selected bytes.
    Without them, a self-asserted digest establishes no source authenticity.
    """
    if not isinstance(model, Mapping):
        raise ValueError("normalized instruction model must be a mapping")
    result = copy.deepcopy(dict(model))
    required = {
        "schema",
        "target",
        "source_sha256",
        "software_spec_sha256",
        "circt_facts_sha256",
        "memory_spaces",
        "instructions",
        "status",
        "unknowns",
        "canonical_sha256",
    }
    if set(result) != required or result.get("schema") != SCHEMA:
        raise ValueError("normalized instruction model has unsupported schema or fields")
    target = _nonempty(result["target"], "normalized instruction target")
    if expected_target is not None and target != expected_target:
        raise ValueError("normalized instruction model target differs from selected target")
    for field in ("source_sha256", "software_spec_sha256", "circt_facts_sha256", "canonical_sha256"):
        _digest_field(result[field], label=field, optional=field in {"software_spec_sha256", "circt_facts_sha256"})
    for field, expected in (
        ("source_sha256", expected_source_sha256),
        ("software_spec_sha256", expected_software_spec_sha256),
        ("circt_facts_sha256", expected_circt_facts_sha256),
    ):
        if expected is not None and result[field] != expected:
            raise ValueError(f"normalized instruction model {field} differs from selected bytes")
    checksum = result.pop("canonical_sha256")
    if hashlib.sha256(_canonical(result)).hexdigest() != checksum:
        raise ValueError("normalized instruction model canonical_sha256 mismatch")
    result["canonical_sha256"] = checksum
    generated_unknowns: list[str] = []
    spaces = _memory_spaces(result["memory_spaces"], generated_unknowns)
    if spaces != result["memory_spaces"]:
        raise ValueError("normalized instruction model has noncanonical memory spaces")
    instructions = result["instructions"]
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("normalized instruction model requires instructions")
    ids = []
    for index, row in enumerate(instructions):
        site = f"instructions[{index}]"
        if not isinstance(row, dict) or set(row) != {
            "id",
            "operands",
            "results",
            "computation",
            "parameters",
            "constraints",
            "effects",
            "software_operation",
            "software_declaration",
            "status",
            "unknowns",
        }:
            raise ValueError(f"{site} has noncanonical fields")
        identity = _nonempty(row["id"], f"{site}.id")
        ids.append(identity)
        operands = _names(row["operands"], label=f"{site}.operands")
        outputs = _names(row["results"], label=f"{site}.results")
        names = {item["name"] for item in operands + outputs}
        if len(names) != len(operands) + len(outputs):
            raise ValueError(f"{site} shares operand/result names")
        locally_generated: list[str] = []
        if (
            _computation(row["computation"], label=f"{site}.computation", unknowns=locally_generated)
            != row["computation"]
        ):
            raise ValueError(f"{site} has noncanonical computation")
        if _parameters(row["parameters"], label=f"{site}.parameters") != row["parameters"]:
            raise ValueError(f"{site} has noncanonical parameters")
        if (
            _constraints(row["constraints"], label=f"{site}.constraints", names=names, spaces=spaces)
            != row["constraints"]
        ):
            raise ValueError(f"{site} has noncanonical constraints")
        if (
            _effects(row["effects"], label=f"{site}.effects", names=names, spaces=spaces, unknowns=locally_generated)
            != row["effects"]
        ):
            raise ValueError(f"{site} has noncanonical effects")
        locally_generated.extend(_effect_coverage_unknowns(row["constraints"], row["effects"]))
        software_operation = row["software_operation"]
        if software_operation is None:
            locally_generated.append("software_operation_unlinked")
        else:
            _nonempty(software_operation, f"{site}.software_operation")
        software_declaration = row["software_declaration"]
        if software_declaration is not None:
            if not isinstance(software_declaration, dict) or software_declaration.get("id") != software_operation:
                raise ValueError(f"{site}.software_declaration must match its SW operation ID")
            if software_declaration.get("placement") not in {"accelerator", "fused_accelerator"}:
                locally_generated.append("software_operation_not_accelerator")
            locally_generated.extend(_software_signature_unknowns(software_declaration, operands, outputs))
        elif software_operation is not None:
            locally_generated.append("software_operation_unverifiable")
        row_unknowns = _unknown_list(row["unknowns"], label=f"{site}.unknowns")
        if not set(locally_generated) <= set(row_unknowns):
            raise ValueError(f"{site} suppresses semantic unknowns")
        if row["status"] != (UNKNOWN if row_unknowns else "described"):
            raise ValueError(f"{site} status differs from unknowns")
        generated_unknowns.extend(f"{identity}.{item}" for item in row_unknowns)
    if ids != sorted(set(ids)):
        raise ValueError("normalized instruction IDs must be sorted and unique")
    if result["software_spec_sha256"] is None:
        generated_unknowns.append("selected_software_spec_missing")
    if result["circt_facts_sha256"] is None:
        generated_unknowns.append("selected_circt_facts_missing")
    unknowns = _unknown_list(result["unknowns"], label="normalized instruction unknowns")
    if not set(generated_unknowns) <= set(unknowns):
        raise ValueError("normalized instruction model suppresses selected-input or instruction unknowns")
    if result["status"] != (UNKNOWN if unknowns else "described"):
        raise ValueError("normalized instruction model status differs from unknowns")
    return result
