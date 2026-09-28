"""Per-operation host support declarations beside an immutable compiler package.

A package precision lane is not an operation-support manifest. These declarations
are separately selected and digest-bound; no compiler payload is rewritten.
"""

from __future__ import annotations

import copy

from merlin.common.digest import is_sha256
from merlin.targetgen.software_spec import admit_operation

SCHEMA = "merlin.host_capabilities.v1"


def validate_host_capabilities(
    document: dict, *, package_sha256: str | None = None, dtype_strategy: str | None = None
) -> dict:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"host capability spec must declare schema {SCHEMA}")
    if document.get("status") not in {"reviewed", "unreviewed"}:
        raise ValueError("host capability spec must declare review status")
    compiler = document.get("compiler")
    if not isinstance(compiler, dict) or (
        not is_sha256(compiler.get("package_sha256"))
        and not (document["status"] == "unreviewed" and compiler.get("package_sha256") is None)
    ):
        raise ValueError("host capability spec requires an exact compiler package SHA256")
    if not isinstance(compiler.get("dtype_strategy"), str) or not compiler["dtype_strategy"]:
        raise ValueError("host capability spec requires an explicit precision lane")
    if (
        package_sha256 is not None
        and compiler["package_sha256"] is not None
        and compiler["package_sha256"] != package_sha256
    ):
        raise ValueError("host capability spec is bound to a different compiler package")
    if dtype_strategy is not None and compiler["dtype_strategy"] != dtype_strategy:
        raise ValueError("host capability spec is bound to a different precision lane")
    operations = document.get("operations")
    if not isinstance(operations, list):
        raise ValueError("host capability spec operations must be a list")
    seen = set()
    for row in operations:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("host capability operations require unique IDs")
        seen.add(row["id"])
        if row.get("placement") != "host" or not isinstance(row.get("signature"), dict) or not row["signature"]:
            raise ValueError("host capability operations require host placement and typed signature constraints")
        for selector in ("ops", "families"):
            if selector in row and (
                not isinstance(row[selector], list)
                or any(not isinstance(value, str) or not value for value in row[selector])
            ):
                raise ValueError(f"host capability {selector} must be explicit string lists")
    if not isinstance(document.get("evidence"), dict):
        raise ValueError("host capability spec must record evidence and unresolved obligations")
    return document


def admit_host_operation(selected: dict | None, row: dict, signature: dict) -> dict:
    """Screen every selected host profile independently from accelerator admission."""
    if selected is None:
        return {
            "status": "unknown",
            "review_status": "unknown",
            "reviewed": False,
            "reason": "no selected per-operation host capability spec",
            "profiles": [],
        }
    if not isinstance(selected, dict):
        raise ValueError("selected host capabilities must be a profile mapping")
    profiles = []
    for name, selection in sorted(selected.items()):
        if not isinstance(selection, dict):
            raise ValueError("selected host capability profile must be a mapping")
        document = selection.get("capability_spec")
        if document is None:
            profiles.append(
                {
                    "profile": name,
                    "status": "unknown",
                    "reviewed": False,
                    "review_status": "unknown",
                    "reason": "host profile has no operation capability declaration",
                }
            )
            continue
        validate_host_capabilities(
            document, package_sha256=selection.get("package_sha256"), dtype_strategy=selection.get("dtype_strategy")
        )
        pinned = (
            is_sha256(selection.get("package_sha256"))
            and is_sha256(selection.get("capability_spec_sha256"))
            and document["compiler"].get("package_sha256") == selection["package_sha256"]
        )
        decisions = []
        for declaration in document["operations"]:
            # A named selector is narrower than a semantic family. A package
            # schedule matching linalg.matmul must not admit an unrelated
            # linalg.generic merely because both describe contractions. Family
            # selectors remain available when no exact ops were declared.
            identities = (row.get("frontend_op"), row["mlir_operation"])
            exact_ops = declaration.get("ops") or []
            if exact_ops and not any(identity in exact_ops for identity in identities):
                continue
            operation = next(
                (
                    identity
                    for identity in identities
                    if identity in exact_ops
                ),
                row["mlir_operation"],
            )
            decision = admit_operation({**document, "operations": [declaration]}, operation, signature, "host")
            if "declaration" in decision:
                decisions.append(decision)
        verdict = next(
            (decision for decision in decisions if decision["status"] == "admitted"),
            next((decision for decision in decisions if decision["status"] == "unknown"), None),
        )
        status = verdict["status"] if verdict else "unsupported"
        reason = verdict["reason"] if verdict else "no host operation declaration admits the observed signature"
        if document["status"] != "reviewed" and status == "unsupported":
            status, reason = "unknown", "unreviewed host declarations cannot establish absence of operation support"
        if not pinned:
            status, reason = "unknown", "host package and capability spec byte identities are not selected together"
        profiles.append(
            {
                "profile": name,
                "status": status,
                "reason": reason,
                "review_status": document["status"],
                "reviewed": document["status"] == "reviewed" and pinned,
                "package_sha256": selection.get("package_sha256"),
                "capability_spec_sha256": selection.get("capability_spec_sha256"),
                "dtype_strategy": selection.get("dtype_strategy"),
                "decisions": decisions,
            }
        )
    verdict = next(
        (profile for profile in profiles if profile["status"] == "admitted"),
        next((profile for profile in profiles if profile["status"] == "unknown"), None),
    )
    if verdict is None:
        verdict = {
            "status": "unsupported" if profiles else "unknown",
            "review_status": "unknown",
            "reviewed": False,
            "reason": "no selected host profile admits the observed operation signature",
        }
    return {
        **copy.deepcopy({key: verdict[key] for key in ("status", "review_status", "reviewed", "reason")}),
        "profiles": profiles,
        "qualification": "selected declaration screen; host lowering remains unverified",
    }
