"""Read selected scheduling observations without extracting or qualifying hardware."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.scheduling_evidence.v1"
MANIFEST_ROLE = "scheduling-manifest"
ARTIFACT_ROLE = "scheduling-artifact:"


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate scheduling JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite scheduling JSON value: {value}")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite scheduling JSON number")
    return result


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"scheduling {label} must be a nonempty string")


def _hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"scheduling {label} must be a lowercase SHA-256 digest")


def _strings(value, label):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"scheduling {label} must be a list of strings")


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"scheduling {label} must be a mapping")


def decode(raw, *, target):
    document = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)
    _mapping(document, "manifest")
    if document.get("schema") != SCHEMA or document.get("target") != target:
        raise ValueError("scheduling evidence schema or target differs from selection")
    expected = {
        "schema",
        "target",
        "hardware",
        "producer",
        "semantics",
        "profiles",
        "artifacts",
        "limitations",
        "verification",
    }
    if set(document) != expected:
        raise ValueError("scheduling manifest fields differ from v1 schema")
    hardware, producer = document.get("hardware"), document.get("producer")
    _mapping(hardware, "hardware")
    _mapping(producer, "producer")
    if set(hardware) != {"config", "source_ir_sha256"} or set(producer) != {"name", "sha256"}:
        raise ValueError("scheduling hardware or producer fields differ from v1 schema")
    _text(hardware.get("config"), "hardware.config")
    _hash(hardware.get("source_ir_sha256"), "hardware.source_ir_sha256")
    _text(producer.get("name"), "producer.name")
    _hash(producer.get("sha256"), "producer.sha256")
    _mapping(document.get("semantics"), "semantics")
    _strings(document.get("limitations"), "limitations")
    profiles = document.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("scheduling profiles must be a nonempty list")
    components = set()
    for row in profiles:
        _mapping(row, "profile")
        if set(row) != {"component", "fields", "assumptions", "limitations"}:
            raise ValueError("scheduling profile fields differ from v1 schema")
        _text(row.get("component"), "profile.component")
        if row["component"] in components:
            raise ValueError("duplicate scheduling profile component")
        components.add(row["component"])
        _mapping(row.get("fields"), "profile.fields")
        _strings(row.get("assumptions"), "profile.assumptions")
        _strings(row.get("limitations"), "profile.limitations")
    verification = document.get("verification")
    _mapping(verification, "verification")
    required = {
        "artifact_bytes_verified": True,
        "projection_evidence_agreement_verified": True,
        "compiler_identity_verified": False,
        "footprints_reproduced": False,
        "schedule_legality_verified": False,
        "rtl_replayed": False,
    }
    if set(verification) != set(required) or any(type(value) is not bool for value in verification.values()):
        raise ValueError("scheduling verification fields must be boolean producer assertions")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("scheduling artifacts must be a nonempty list")
    roles, paths = set(), set()
    for row in artifacts:
        _mapping(row, "artifact")
        if set(row) != {"role", "path", "sha256"}:
            raise ValueError("scheduling artifact fields differ from v1 schema")
        _text(row.get("role"), "artifact.role")
        _text(row.get("path"), "artifact.path")
        _hash(row.get("sha256"), "artifact.sha256")
        path = PurePosixPath(row["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in row["path"]
            or str(path) != row["path"]
            or not path.parts
        ):
            raise ValueError("scheduling artifact path must be a contained canonical relative path")
        if path.parts[0] in ("manifest.json", "ingestion.json"):
            raise ValueError("scheduling artifact path is reserved")
        if row["role"] in roles or row["path"] in paths:
            raise ValueError("duplicate scheduling artifact role or path")
        roles.add(row["role"])
        paths.add(row["path"])
    return document


def compare_hardware(document, descriptor, source_consistency):
    declared = document["hardware"]
    config = ((descriptor.get("rtl") or {}).get("elaboration") or {}).get("config")
    sources = source_consistency.get("sources") or []
    hashes = (
        {row.get("sha256") for row in sources if isinstance(row, dict) and row.get("role") in ("core_hw", "soc_hw")}
        if source_consistency.get("status") == "verified"
        else set()
    )
    generic = source_consistency.get("genericization") or {}
    # A production-validated generic serialization receipt binds the original HW input,
    # never its serialized output, as the scheduling extractor's hardware identity.
    if source_consistency.get("status") == "verified" and generic.get("kind") == "circt_generic_serialization":
        hashes.add((generic.get("input") or {}).get("sha256"))
    hashes.discard(None)
    if config is not None and config != declared["config"]:
        status, reason = "mismatch", "selected elaboration configuration differs"
    elif hashes and declared["source_ir_sha256"] not in hashes:
        status, reason = "mismatch", "scheduling source IR differs from selected production sources"
    elif config is None or not hashes:
        status, reason = "unknown", "selected configuration and verified production source identity are both required"
    else:
        status, reason = "matched", "configuration and exact production source digest match"
    return {"status": status, "reason": reason, "selected_config": config, "selected_source_ir_sha256": sorted(hashes)}


def load_selection(path, *, target, descriptor, source_consistency, observe):
    path = Path(path).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("scheduling manifest traverses a symlink")
    raw = observe(path, MANIFEST_ROLE, required=True)
    document = decode(raw, target=target)
    for row in document["artifacts"]:
        member = path.parent / row["path"]
        if member.is_symlink() or any(parent.is_symlink() for parent in member.parents):
            raise ValueError("scheduling artifact traverses a symlink")
        content = observe(member, ARTIFACT_ROLE + row["path"], required=True)
        if _digest(content) != row["sha256"]:
            raise ValueError(f"scheduling artifact digest differs: {row['path']}")
    return {
        "manifest": document,
        "manifest_sha256": _digest(raw),
        "hardware_comparison": compare_hardware(document, descriptor, source_consistency),
        "qualified_for_use": False,
        "qualification": (
            "Artifact hashes checked by this reader; projection agreement and other "
            "verification fields remain producer assertions. Hardware matching is an "
            "identity comparison, not an execution, correctness or performance qualification."
        ),
    }


def snapshot_outputs(view, sources, *, target):
    """Check the imported view against frozen source bytes and build export members."""
    manifests = [source.content for source in sources if source.role == MANIFEST_ROLE]
    if len(manifests) != 1:
        raise ValueError("scheduling evidence requires one manifest snapshot")
    raw = manifests[0]
    document = decode(raw, target=target)
    if (
        view.get("manifest") != document
        or view.get("manifest_sha256") != _digest(raw)
        or view.get("qualified_for_use") is not False
    ):
        raise ValueError("scheduling view differs from manifest snapshot")
    members = {}
    for source in sources:
        if source.role.startswith(ARTIFACT_ROLE):
            name = source.role[len(ARTIFACT_ROLE) :]
            if name in members:
                raise ValueError("duplicate scheduling artifact snapshot")
            members[name] = source.content
    if set(members) != {row["path"] for row in document["artifacts"]}:
        raise ValueError("scheduling artifact snapshot inventory differs")
    outputs = {"hardware/scheduling/manifest.json": raw}
    for row in document["artifacts"]:
        content = members[row["path"]]
        if _digest(content) != row["sha256"]:
            raise ValueError("scheduling artifact snapshot digest differs")
        outputs["hardware/scheduling/" + row["path"]] = content
    outputs["hardware/scheduling/ingestion.json"] = (json.dumps(view, sort_keys=True, indent=2) + "\n").encode()
    return outputs
