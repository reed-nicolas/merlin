"""Copy a current Atlas timing handoff into read-only scheduling evidence.

This target-owned adapter checks saved bytes and agreement, not compiler or RTL
execution. The original footprint document remains the authority for unknowns,
row streams, reservations, and completion-dependent accesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROLES = {"mxu0", "mxu1", "dma", "lsu", "xlu", "vpu"}
METADATA = {"schema", "config", "source_ir_sha256", "evidence_sha256"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def finite_float(value):
        result = float(value)
        _require(math.isfinite(result), "Nonfinite JSON number")
        return result

    document = json.loads(raw, object_pairs_hook=_object, parse_constant=reject_constant, parse_float=finite_float)
    _require(isinstance(document, dict), "Expected a JSON object")
    return document


def _identity(record):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Invalid artifact identity")
    _require(
        isinstance(record["path"], str) and bool(record["path"]) and _is_hash(record["sha256"]),
        "Invalid artifact path or hash",
    )


def _fields(raw):
    fields = {}
    for line in raw.decode("utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        _require(separator and key and key not in fields, "Malformed or duplicate profile field")
        fields[key] = value
    return fields


def _profile(role, raw, evidence_raw):
    fields, report = _fields(raw), _json(evidence_raw)
    schemas = {f"atlas-{role}-profile-v1"}
    if role in {"mxu0", "dma", "lsu", "xlu"}:
        schemas.add(f"atlas-{role}-profile-v2")
    _require(fields.get("schema") in schemas, f"Unsupported {role} profile schema")
    _require(fields.get("config") == report.get("config") == "EE290SimConfig", "Profile config differs")
    version = fields["schema"].rsplit("-", 1)[1]
    if role in {"mxu0", "mxu1"}:
        versions = {1, 2} if role == "mxu1" else ({2} if version == "v2" else {1})
        _require(
            type(report.get("schema_version")) is int
            and report["schema_version"] in versions
            and report.get("kind") == f"atlas-partial-{role}-profile",
            "Unsupported MXU evidence",
        )
    else:
        _require(
            report.get("schema") == f"atlas.rtlgraph.{role}-profile.{version}", "Evidence schema differs from profile"
        )
    _require(fields.get("evidence_sha256") == _hash(evidence_raw), "Profile/evidence byte identity differs")
    hardware = fields.get("source_ir_sha256")
    inputs = report.get("inputs")
    _require(isinstance(inputs, dict) and isinstance(inputs.get("hardware_ir"), dict), "Missing hardware identity")
    _require(_is_hash(hardware) and inputs["hardware_ir"].get("sha256") == hardware, "Profile hardware differs")
    overrides = report.get("compiler_overrides")
    _require(isinstance(overrides, dict), "Missing compiler overrides")
    native = {key: value for key, value in fields.items() if key not in METADATA}
    _require(native == {key: str(value) for key, value in overrides.items()}, "Profile/evidence settings differ")
    for key in ("assumptions", "limitations"):
        values = report.get(key, [])
        _require(
            isinstance(values, list) and all(isinstance(value, str) for value in values), f"Invalid evidence {key}"
        )
    return fields, report


def _model_agreement(model, selected):
    for role in ("dma", "lsu", "xlu", "vpu"):
        flag = f"rtl_{role}"
        _require(type(model.get(flag)) is bool and model[flag] == (role in selected), f"Model {flag} differs")
    for flag, role in (("rtl_dma_ranges", "dma"), ("rtl_scalar_lsu", "lsu")):
        expected = role in selected and selected[role]["schema"].endswith("-v2")
        _require(type(model.get(flag)) is bool and model[flag] == expected, f"Model {flag} differs")
    for role, fields in selected.items():
        for key, value in fields.items():
            if role in {"mxu0", "mxu1"} and key in {"first_write_age", "overwrite_acc_read_hold"}:
                model_key = f"{role}_{key}"
            elif role == "lsu" and key.startswith(("vload_", "vstore_", "scalar_")):
                model_key = key
            elif role in {"xlu", "vpu"} and key.endswith("_age"):
                model_key = f"{role}_{key}"
            else:
                continue
            if key == "overwrite_acc_read_hold":
                _require(
                    value in {"0", "1"} and type(model.get(model_key)) is bool and model[model_key] == (value == "1"),
                    f"Model {model_key} differs",
                )
            else:
                _require(
                    value.isdecimal() and type(model.get(model_key)) is int and model[model_key] == int(value),
                    f"Model {model_key} differs",
                )


def convert_contract(contract_path: Path, output_dir: Path) -> Path:
    """Validate and snapshot saved members without accessing the compiler binary."""
    contract_path = Path(contract_path).resolve(strict=True)
    root = contract_path.parent
    raw_contract = contract_path.read_bytes()
    contract = _json(raw_contract)
    _require(
        contract.get("schema") == "atlas.rtlgraph.contract.v1" and contract.get("config") == "EE290SimConfig",
        "Unsupported contract schema/config",
    )
    hardware = contract.get("source_ir_sha256")
    _require(_is_hash(hardware), "Invalid contract hardware identity")
    _identity(contract.get("compiler"))  # Validate the record only; never open this path.
    members = {"bundle/contract.json": raw_contract}
    artifacts = [{"role": "contract", "path": "bundle/contract.json", "sha256": _hash(raw_contract)}]

    def member(role, record):
        _identity(record)
        relative = Path(record["path"])
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.parts
            and "\\" not in record["path"]
            and record["path"] == relative.as_posix(),
            "Bundle path escapes its directory",
        )
        path = (root / relative).resolve(strict=True)
        _require(path.is_relative_to(root), "Bundle symlink escapes its directory")
        raw = path.read_bytes()
        _require(_hash(raw) == record["sha256"], f"Artifact changed: {record['path']}")
        destination = (Path("bundle") / relative).as_posix()
        _require(destination not in members, "Duplicate bundle member path")
        members[destination] = raw
        artifacts.append({"role": role, "path": destination, "sha256": _hash(raw)})
        return raw

    member("source", contract.get("source"))
    footprints = _json(member("footprints", contract.get("footprints")))
    profiles, selected = [], {}
    entries = contract.get("profiles")
    _require(isinstance(entries, list) and entries, "Missing selected profiles")
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == {"role", "projection", "evidence"}, "Invalid profile entry")
        role = entry["role"]
        _require(isinstance(role, str) and role in ROLES and role not in selected, "Unsupported or duplicate role")
        fields, report = _profile(
            role, member(f"projection:{role}", entry["projection"]), member(f"evidence:{role}", entry["evidence"])
        )
        _require(fields["source_ir_sha256"] == hardware, "Mixed hardware identities")
        selected[role] = fields
        profiles.append(
            {
                "component": role,
                "fields": {k: v for k, v in fields.items() if k not in METADATA},
                "assumptions": report.get("assumptions", []),
                "limitations": report.get("limitations", []),
            }
        )
    _require(
        footprints.get("schema") == "atlas.footprints.v1" and footprints.get("schedule_validated") is False,
        "Unsupported or qualified footprint query",
    )
    model, native = footprints.get("model"), footprints.get("semantics")
    _require(isinstance(model, dict) and model.get("source_ir_sha256") == hardware, "Footprint hardware differs")
    _require(isinstance(native, dict), "Missing footprint semantics")
    _model_agreement(model, selected)
    rtl_dma = "dma" in selected
    semantics = {
        "assembly": "atlas-opt-native",
        "age_zero": "instruction-issue",
        "hold_interval": "inclusive",
        "dma_completion": "explicit-wait" if rtl_dma else "modeled-completion",
    }
    _require(contract.get("semantics") == semantics, "Unsupported contract semantics")
    completion = (
        "memory lifetime until explicit matching DMA.WAIT; age is not a bound"
        if rtl_dma
        else "access at modeled DMA completion"
    )
    _require(
        native.get("age_origin") == "instruction issue"
        and native.get("hold_end") == "inclusive"
        and native.get("at_completion") == completion,
        "Footprint completion/age conventions differ",
    )
    _require(
        "unknown_value" in native
        and native["unknown_value"] is None
        and native.get("dma_cycles") == "cost estimate, not a completion guarantee",
        "Footprint unknown/estimate conventions differ",
    )
    if rtl_dma:
        _require(selected["dma"].get("completion") == "explicit-wait", "DMA profile completion differs")
    _require(isinstance(footprints.get("blocks"), list), "Missing native footprint blocks")
    limitations = contract.get("limitations", [])
    _require(
        isinstance(limitations, list) and all(isinstance(value, str) for value in limitations),
        "Invalid contract limitations",
    )
    envelope = {
        "schema": "merlin.scheduling_evidence.v1",
        "target": "atlas",
        "hardware": {"config": contract["config"], "source_ir_sha256": hardware},
        "producer": {"name": "atlas-opt", "sha256": contract["compiler"]["sha256"]},
        "semantics": semantics,
        "profiles": profiles,
        "artifacts": artifacts,
        "limitations": [
            *limitations,
            "Saved artifact agreement does not qualify a schedule or hardware target.",
            "Compiler bytes, footprint reproduction, schedule legality and RTL execution are unverified.",
        ],
        "verification": {
            "artifact_bytes_verified": True,
            "projection_evidence_agreement_verified": True,
            "compiler_identity_verified": False,
            "footprints_reproduced": False,
            "schedule_legality_verified": False,
            "rtl_replayed": False,
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for relative, raw in members.items():
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    output = output_dir / "scheduling-evidence.json"
    output.write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(convert_contract(args.contract, args.output))


if __name__ == "__main__":
    main()
