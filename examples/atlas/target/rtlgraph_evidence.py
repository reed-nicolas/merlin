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

# Authored assembler mnemonics and the native report's instruction labels are
# different vocabularies. Keep this translation at the target-owned edge.
VPU_COMMANDS = {
    "VADD_BF16": "add",
    "VSUB_BF16": "sub",
    "VMUL_BF16": "mul",
    "VMINIMUM_BF16": "pairmin",
    "VMAXIMUM_BF16": "pairmax",
    "VMOV": "mov",
    "VRECIP_BF16": "rcp",
    "VEXP_BF16": "exp",
    "VEXP2_BF16": "exp2",
    "VPACK_BF16_FP8": "fp8pack",
    "VUNPACK_FP8_BF16": "fp8unpack",
    "VRELU_BF16": "relu",
    "VSIN_BF16": "sin",
    "VCOS_BF16": "cos",
    "VTANH_BF16": "tanh",
    "VLOG2_BF16": "log",
    "VSQRT_BF16": "sqrt",
    "VSQUARE_BF16": "square",
    "VCUBE_BF16": "cube",
    "VREDSUM_BF16": "csum",
    "VREDMIN_BF16": "cmin",
    "VREDMAX_BF16": "cmax",
    "VREDSUM_ROW_BF16": "rsum",
    "VREDMIN_ROW_BF16": "rmin",
    "VREDMAX_ROW_BF16": "rmax",
    "VLI_ALL": "vliAll",
    "VLI_ROW": "vliRow",
    "VLI_COL": "vliCol",
    "VLI_ONE": "vliOne",
}
FREE_AGES = {
    "VLOAD": ("lsu", "vload_first_free_age"),
    "VSTORE": ("lsu", "vstore_first_free_age"),
    "VTRPOSE_XLU": ("xlu", "first_free_age"),
}


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


def _schedule(raw):
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            _require(isinstance(key, str) and key not in result, "Invalid or duplicate schedule YAML key")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    document = yaml.load(raw, Loader=UniqueLoader)
    _require(
        isinstance(document, dict) and type(document.get("version")) is int and document["version"] == 1,
        "Unsupported authored schedule contract",
    )
    return document


def compare_schedule_assumptions(schedule, profiles, reports, semantics):
    """Compare scoped accepted-command spacing, never first-write age to latency.

    Same-operation engine spacing is a diagnostic projection of a resource rule.
    It cannot establish cross-operation acceptance, frontend assertions, operand
    hazards, or the conditions under which instruction issue reaches the engine.
    """
    rows, coverage = [], []
    _require(
        semantics.get("age_zero") == "instruction-issue" and semantics.get("hold_interval") == "inclusive",
        "Unsupported comparison age conventions",
    )

    def spacing(mnemonic):
        if mnemonic in FREE_AGES:
            component, field = FREE_AGES[mnemonic]
            value = profiles.get(component, {}).get(field)
            value = int(value) if isinstance(value, str) and value.isdecimal() else None
            return component, field, value
        if mnemonic in VPU_COMMANDS:
            field = f"instructions.{VPU_COMMANDS[mnemonic]}.same_op_next_issue"
            instructions = reports.get("vpu", {}).get("instructions", {})
            _require(isinstance(instructions, dict), "Invalid VPU instruction timing observations")
            command = instructions.get(VPU_COMMANDS[mnemonic], {})
            _require(isinstance(command, dict), "Invalid VPU command timing observation")
            value = command.get("same_op_next_issue")
            if value is not None:
                _require(type(value) is int and value >= 0, "Invalid accepted-command spacing")
            return "vpu", field, value
        return None, None, None

    for kind in ("minimum_issue_gap", "register_dependency_gap"):
        rules = schedule.get(kind, [])
        _require(isinstance(rules, list), "Invalid authored schedule rules")
        names = set()
        for rule in rules:
            _require(
                isinstance(rule, dict) and isinstance(rule.get("name"), str) and rule["name"] not in names,
                "Invalid or duplicate authored rule",
            )
            names.add(rule["name"])
            producers, consumers = rule.get("producers"), rule.get("consumers")
            _require(
                isinstance(producers, list)
                and producers
                and isinstance(consumers, list)
                and consumers
                and all(isinstance(op, str) and op for op in [*producers, *consumers])
                and len(set(producers)) == len(producers)
                and len(set(consumers)) == len(consumers),
                "Invalid authored instruction scope",
            )
            cycles = rule.get("cycles")
            _require(type(cycles) is int and cycles >= 0, "Invalid authored issue gap")
            compared = 0
            for producer in producers:
                row = {
                    "kind": kind,
                    "rule": rule["name"],
                    "producer": producer,
                    "consumer": producer if producer in consumers else None,
                    "authored_cycles": cycles,
                    "observed_cycles": None,
                    "component": None,
                    "evidence_field": None,
                    "status": "unsupported",
                }
                if kind == "register_dependency_gap":
                    row["reason"] = (
                        "Row writes and engine occupancy do not establish register-sensitive visibility gaps."
                    )
                elif producer not in consumers:
                    row["reason"] = "The rule contains no same-operation pair; cross-operation spacing is not compared."
                else:
                    component, field, value = spacing(producer)
                    row.update(component=component, evidence_field=field, observed_cycles=value)
                    if component is None:
                        row["reason"] = (
                            "No accepted-command spacing mapping; MXU first-write ages are not resource gaps."
                        )
                    elif value is None:
                        row["status"] = "unknown"
                        row["reason"] = "Selected evidence has no accepted-command spacing for this operation."
                    else:
                        compared += 1
                        row["status"] = "matched" if value == cycles else "mismatch"
                        row["relation"] = (
                            "equal"
                            if value == cycles
                            else "authored_more_conservative"
                            if cycles > value
                            else "authored_below_observed_spacing"
                        )
                        row["reason"] = (
                            "Conditional same-operation engine spacing; frontend acceptance "
                            "and logical reservations remain obligations."
                        )
                rows.append(row)
            total = len(producers) * len(consumers)
            coverage.append(
                {
                    "kind": kind,
                    "rule": rule["name"],
                    "authored_pairs": total,
                    "compared_same_operation_pairs": compared,
                    "uncompared_pairs": total - compared,
                    "complete": compared == total,
                }
            )
    statuses = ("matched", "mismatch", "unknown", "unsupported")
    return {
        "schema": "atlas.scheduling_assumptions_comparison.v1",
        "scope": "Conditional same-operation accepted-engine-command spacing versus authored issue-distance minima.",
        "qualified_for_use": False,
        "rows": rows,
        "rule_coverage": coverage,
        "summary": {status: sum(row["status"] == status for row in rows) for status in statuses},
        "unsupported_scope": [
            "Cross-operation pairs, whole-frontend acceptance and assertion legality.",
            "Register dependencies, row visibility, physical-port conflicts and logical reservations.",
            "MXU first-write ages are operand events, not resource release or completion bounds.",
            "DMA explicit waits are completion policy, not fixed issue or completion latency.",
            "Control flow and delay encoding; numerical correctness and performance predictions.",
        ],
        "action": (
            "Review mismatches in a new authored contract after independent acceptance "
            "and hazard checks; this report changes no rules."
        ),
    }


def convert_contract(contract_path: Path, output_dir: Path, *, schedule_contract: Path | None = None) -> Path:
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
    profiles, selected, reports = [], {}, {}
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
        reports[role] = report
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
    if schedule_contract is not None:
        schedule_raw = Path(schedule_contract).read_bytes()
        comparison = compare_schedule_assumptions(_schedule(schedule_raw), selected, reports, semantics)
        comparison["inputs"] = {
            "native_contract_sha256": _hash(raw_contract),
            "schedule_contract_sha256": _hash(schedule_raw),
            "hardware": {"config": contract["config"], "source_ir_sha256": hardware},
            "profile_evidence": [row for row in artifacts if row["role"].startswith("evidence:")],
        }
        for role, name, raw in (
            ("authored-schedule-contract", "bundle/schedule_contract.yaml", schedule_raw),
            (
                "schedule-assumption-comparison",
                "bundle/assumption-comparison.json",
                (json.dumps(comparison, indent=2, allow_nan=False) + "\n").encode(),
            ),
        ):
            _require(name not in members, "Comparison path collides with native bundle")
            members[name] = raw
            artifacts.append({"role": role, "path": name, "sha256": _hash(raw)})
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
    parser.add_argument(
        "--schedule-contract",
        type=Path,
        help="Snapshot authored schedule YAML and compare scoped assumptions (requires PyYAML)",
    )
    args = parser.parse_args()
    print(convert_contract(args.contract, args.output, schedule_contract=args.schedule_contract))


if __name__ == "__main__":
    main()
