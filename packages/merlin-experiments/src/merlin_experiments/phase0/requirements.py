"""Deterministic, installed requirement derivation from explicit iteration captures.

No agent, candidate compiler or headline evaluation is involved. Authored inputs
are declarations; generated inventories and synthesis plans are not certificates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen import application_inventory, conformance, corpus_synth, target_registry
from merlin.targetgen.rtl.facts import observed_facts
from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.spec import load_spec

from .declarations import from_definition
from .evidence import _materialize_evidence, export_evidence, select_evidence
from .performance_scope import derive_performance_scope
from .profiles import selected_software_spec_path, synthesis_input_identity
from .software_screen import diagnostic_entry, intersect_requirement, screen_entry
from .typed_scope import typed_required_instances


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _materialized_iteration_capsules(full: dict, digest: str) -> tuple[list[dict], dict[str, bytes]]:
    """Select full iteration programs, not a family representative or a held-out model.

    Save all producer-receipt members before exposing entries. Reopening the
    source loader or claiming the target executes the program is not involved.
    """
    from merlin.targetgen.capsule_source import materialized_model_artifacts

    entries, outputs = [], {}
    for label, application in sorted(full["applications"].items()):
        if not label or Path(label).name != label or label in {".", ".."}:
            raise ValueError("application identity must be a single safe path component")
        source = Path(application["capture_source_path"])
        selection = {
            "path": str(source),
            "capture_sha256": application["capture_sha256"],
            "receipt_sha256": application["capture_receipt"]["receipt_sha256"],
            "workload_id": label,
            "workload_role": "iteration",
            "coverage_scope": "full_capture",
            "full_inventory_sha256": digest,
            "operation_count": application["n_operations"],
        }
        artifact = materialized_model_artifacts(selection)
        receipt_raw = (source.parent / "capture_receipt.json").read_bytes()
        if hashlib.sha256(receipt_raw).hexdigest() != selection["receipt_sha256"]:
            raise ValueError(f"capture receipt changed while copying {label}")
        receipt = json.loads(receipt_raw)
        members = set(receipt["artifacts"]) | {"capture_receipt.json"}
        if artifact.meta.get("framework_catalog"):
            members.add("pytorch-opset.json")
        for member in sorted(members):
            path = source.parent / member
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"materialized member missing or symlinked: {path}")
            outputs[f"materialized/{label}/{member}"] = path.read_bytes()
        # Independent recheck closes a producer mutation during the copy.
        copied = outputs[f"materialized/{label}/model.mlir"]
        if hashlib.sha256(copied).hexdigest() != selection["capture_sha256"]:
            raise ValueError(f"capture bytes changed while copying {label}")
        for member, identity in receipt["artifacts"].items():
            raw = outputs[f"materialized/{label}/{member}"]
            if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
                raise ValueError(f"receipt-bound bytes changed while copying {label}/{member}")
        if (
            artifact.meta.get("framework_catalog")
            and hashlib.sha256(outputs[f"materialized/{label}/pytorch-opset.json"]).hexdigest()
            != artifact.meta["framework_catalog"]["sha256"]
        ):
            raise ValueError(f"framework catalog changed while copying {label}")
        entries.append(
            {
                "name": f"SY_source_{label}",
                "cat": "model",
                "kind": "model",
                "op": "model",
                "model": label,
                "label": "public",
                "operand_dtype": artifact.dtype,
                "source_role": "materialized_iteration_capture",
                "source_reference": "full saved iteration capture; target compile and execution unverified",
                "materialized_capture": {**selection, "path": f"materialized/{label}/model.mlir"},
                "generalization": {"generalization_axis": "composition"},
            }
        )
    return entries, outputs


def capture_selections(selections: list[str]) -> dict[str, Path]:
    result = {}
    for item in selections:
        label, separator, location = item.partition("=")
        if not separator or not label or not location or label in result:
            raise ValueError(f"invalid/duplicate capture selection {item!r}; use LABEL=PATH")
        path = Path(location).expanduser().absolute()
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"capture selection traverses a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[label] = path
    return result


def _selected_file_specs(selections: list[str], role: str) -> dict[str, tuple[Path, str]]:
    result = {}
    for item in selections:
        label, separator, location = item.partition("=")
        name, digest_separator, digest = location.rpartition("@")
        if (
            not separator or not digest_separator or not label or not name or label in result
            or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid/duplicate {role} {item!r}; use LABEL=PATH@SHA256")
        path = Path(name).expanduser().absolute()
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file():
            raise ValueError(f"{role} is absent or indirect: {path}")
        result[label] = (path, digest)
    return result


def capture_selection_specs(selections: list[str]) -> dict[str, tuple[Path, str]]:
    """Parse independent pre-execution capture selections and byte digests."""
    return _selected_file_specs(selections, "capture preselection")


def quantization_policy_specs(selections: list[str]) -> dict[str, tuple[Path, str]]:
    """Parse operator-selected external quantization policies and byte digests."""
    return _selected_file_specs(selections, "quantization policy")


def _validate_capture_recipes(
    captures: dict[str, Path], selected_recipe_hashes: set[str], *, software_spec_sha256: str | None = None,
    policy_selections: dict[str, tuple[Path, str]] | None = None,
) -> dict[str, str]:
    """A realized quantized graph must use a recipe this provider actually derived."""
    selected_policies = {}
    policy_selections = policy_selections or {}
    for label, path in sorted(captures.items()):
        meta_path = path.with_name("meta.json")
        if meta_path.is_symlink():
            raise ValueError(f"{label}: capture metadata may not be a symlink")
        meta = json.loads(meta_path.read_bytes())
        if not isinstance(meta, dict):
            raise ValueError(f"{label}: capture metadata must be a mapping")
        stats = meta.get("quantization_stats") or {}
        if not isinstance(stats, dict):
            raise ValueError(f"{label}: quantization statistics must be a mapping")
        actual = stats.get("recipe_sha256")
        if actual is not None and (
            not isinstance(actual, str)
            or len(actual) != 64
            or any(char not in "0123456789abcdef" for char in actual)
            or actual not in selected_recipe_hashes
        ):
            raise ValueError(
                f"{label}: capture used a different quantization recipe from the selected provider; "
                "regenerate the capture from its selected Phase 0 recipe"
            )
        manifest_path = path.with_name("quantization-manifest.json")
        if manifest_path.exists() or meta.get("quantization_manifest") is not None or (
            b"prov.quantization_manifest_sha256" in path.read_bytes()
        ):
            verified = application_inventory.verify_capture_receipt(path)
            if verified["status"] != "verified_materialized":
                raise ValueError(f"{label}: external quantization manifest is not byte-bound to the capture")
            manifest = json.loads(manifest_path.read_bytes())
            if software_spec_sha256 is None or manifest.get("contract_sha256") != software_spec_sha256:
                raise ValueError(f"{label}: external quantization contract differs from selected software spec")
            selected_policy = policy_selections.get(label)
            if selected_policy is None:
                raise ValueError(f"{label}: external quantization requires an independent policy selection")
            policy_path, policy_sha256 = selected_policy
            if (policy_path.is_symlink() or any(parent.is_symlink() for parent in policy_path.parents)
                    or not policy_path.is_file()
                    or hashlib.sha256(policy_path.read_bytes()).hexdigest() != policy_sha256
                    or manifest.get("policy_sha256") != policy_sha256):
                raise ValueError(f"{label}: selected quantization policy differs from capture manifest")
            selected_policies[label] = policy_sha256
    if set(policy_selections) != set(selected_policies):
        raise ValueError("quantization policy selections must name exactly the external captures")
    return selected_policies


def derive(
    definition: str | Path,
    captures: dict[str, Path],
    *,
    rtl_facts: str | Path,
    output_root: str | Path,
    native_qualifications: dict[str, Path] | None = None,
    capture_preselections: dict[str, tuple[Path, str]] | None = None,
    quantization_policies: dict[str, tuple[Path, str]] | None = None,
) -> dict:
    """Write a byte-bound requirement, complete census and diagnostic candidate plan.

    All declared iteration applications must be supplied. Old synthesis/private
    profiles and historical corpus members are deliberately not inputs. Repeating
    the same selection produces identical bytes; changed inputs need a new root.
    """
    declaration = from_definition(definition)
    te = load_target_experiment(declaration.descriptor)
    declared = (te.workload_spec or {}).get("applications")
    if not isinstance(declared, (list, tuple)) or not declared or len(declared) != len(set(declared)):
        raise ValueError("deterministic derivation needs an explicit nonempty, unique application roster")
    if set(captures) != set(declared):
        raise ValueError(
            f"iteration roster mismatch: missing={sorted(set(declared) - set(captures))}, "
            f"extra={sorted(set(captures) - set(declared))}"
        )
    if len({str(path.resolve()) for path in captures.values()}) != len(captures):
        raise ValueError("distinct application labels cannot select the same capture path")
    capture_preselections = capture_preselections or {}
    if capture_preselections and set(capture_preselections) != set(captures):
        raise ValueError("capture preselection must cover the entire declared iteration roster")
    selected_capture_evidence = {}
    if capture_preselections:
        from .capture_selection import verify

        for label, (selection_path, selected_sha256) in sorted(capture_preselections.items()):
            selected_capture_evidence[label] = verify(
                selection_path, expected_sha256=selected_sha256, model_path=captures[label]
            )
    spec = load_spec(definition)
    config = spec.document["phases"]["0"]["config"]
    software = selected_software_spec_path(
        declaration.recipe, spec.resolve(config["software_spec"]) if config.get("software_spec") else None
    )
    capability_contract_path = (
        spec.resolve(config["capability_contract"]) if config.get("capability_contract") else None
    )
    hardware = spec.resolve(config["hardware_spec"]) if config.get("hardware_spec") else None
    if software is None:
        raise ValueError("deterministic derivation requires an explicit software spec in the recipe")
    selected = select_evidence(
        te.target,
        descriptor=declaration.descriptor,
        capability_contract_path=capability_contract_path,
        software_spec=software,
        hardware_spec=hardware,
        facts_path=rtl_facts,
    )
    options = {
        "capability_contract": selected.contract,
        "include_graph": True,
        "application_metadata": {
            label: {"workload_id": label, "workload_role": "iteration", "coverage_scope": "full_capture"}
            for label in captures
        },
    }
    # Legacy readers use target names; these scopes prevent a second live
    # contract/facts selection or silent extraction from an ambient cache.
    with (
        target_registry.observed_contract(te.target, selected.contract),
        observed_facts(te.target, selected.refreshed_facts, Path(rtl_facts)),
    ):
        full = application_inventory.application_demand_inventory(captures, te.target, detailed=True, **options)
        requirement = conformance.derive_spec(
            te.target,
            captures,
            applications=captures,
            oracle_tiers=[],
            corpus_roots=[],
            cert_budget_s=(te.workload_spec or {}).get("cert_budget_s"),
            application_inventory_options=options,
        )
    if full["status"] != "inventoried":
        raise ValueError("one or more declared iteration captures could not be fully inventoried")
    digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != requirement["application_demands"]["full_inventory_sha256"]:
        raise ValueError("capture bytes changed while deriving requirements")
    # Retain the exact operation IDs and SSA types behind the family-only raw
    # scope census. This is source evidence, not an accelerator-eligible Phase 2
    # requirement; placement and compiler correspondence remain unresolved.
    requirement["scope"]["typed_required_instances"] = typed_required_instances(requirement["scope"], full)
    from merlin.targetgen.quantization_spec import build_quantization_contract, capture_recipe_candidates

    quantization = build_quantization_contract(
        selected.software_spec,
        {
            "contract": selected.contract,
            "quantization_candidates": selected.quantization_snapshot.get("quantization_candidates", []),
            "readout_facets": selected.readout_facets,
            "readout_numerics": selected.readout_numerics,
        },
    )
    selected_recipes = capture_recipe_candidates(selected.software_spec, quantization)
    selected_policies = _validate_capture_recipes(
        captures, {row["recipe"]["recipe_sha256"] for row in selected_recipes},
        software_spec_sha256=hashlib.sha256(software.read_bytes()).hexdigest(),
        policy_selections=quantization_policies,
    )
    if selected_policies:
        requirement["quantization_policy_selections"] = {
            "schema": "merlin.phase0.quantization_policy_selections.v1",
            "status": "byte_selected_not_numerically_reviewed",
            "applications": selected_policies,
        }
    requirement["application_demands"]["sidecar"] = "application-demands.json"
    if selected_capture_evidence:
        requirement["capture_execution_preselections"] = {
            "schema": "merlin.phase0.capture_preselections.v1",
            "status": "replay_verified_nonadmissible",
            "applications": selected_capture_evidence,
            "phase0_admission": "not_granted",
        }
    requirement = intersect_requirement(requirement, selected.software_spec, selected.contract)
    # The recipe's tier ladder is an authored PLAN, not evidence that an oracle
    # was constructed. Keep it separate from ``oracle_tiers`` (which remains
    # observed-only) so synthesis can cap unaffordable members to a declared
    # functional screen without claiming that screen executed at derivation.
    recipe_doc = yaml.safe_load(declaration.recipe.read_text(encoding="utf-8")) or {}
    planned_tiers = (
        (recipe_doc.get("datapath") or {}).get("required_oracle_tiers") if isinstance(recipe_doc, dict) else None
    )
    if planned_tiers is not None and (
        not isinstance(planned_tiers, list)
        or any(
            not isinstance(tier, str) or not tier.startswith("L") or not tier[1:].isdigit()
            for tier in planned_tiers
        )
    ):
        raise ValueError("selected recipe required_oracle_tiers must be a list of fidelity tiers")
    requirement["oracle_tiers_declared"] = list(planned_tiers or [])
    requirement["scope"]["performance"] = derive_performance_scope(requirement["scope"], selected.software_spec)
    requirement["derivation"]["phase0_execution"] = {
        "agentic": False,
        "policy": "deterministic from selected inputs",
        "definition_sha256": hashlib.sha256(spec.path.read_bytes()).hexdigest(),
        "oracle_tiers": "not constructed during derivation; establish in execution qualification",
        "historical_corpus": "not selected",
        "headline_workloads": "held out",
        **selected.derivation_identity,
    }
    root = Path(output_root).absolute()
    outputs = {
        "requirements.yaml": yaml.safe_dump(requirement, sort_keys=False).encode(),
        "application-demands.json": _json(full),
    }
    if selected_capture_evidence:
        outputs["capture-preselections.json"] = _json(requirement["capture_execution_preselections"])
    # Save the census even when an exact writer cannot express every signature.
    # Such a plan is diagnostic and must not become a selectable verified corpus.
    try:
        plan = corpus_synth.synthesize(
            requirement,
            workload_spec=te.workload_spec,
            application_inventory=full,
            capability_contract=selected.contract,
        )
    except corpus_synth.SynthesisError as exc:
        plan = {"status": "blocked", "reason": str(exc), "capsules": [], "provenance": {}}
    screens = []
    for index, entry in enumerate(plan.get("capsules") or []):
        decision = screen_entry(
            selected.software_spec,
            entry,
            defaults=selected.software_spec["numerical_semantics"],
            host_capabilities=selected.host_capabilities,
        )
        screens.append({"capsule": entry.get("name"), **decision})
        if decision["status"] == "unsupported":
            plan["capsules"][index] = diagnostic_entry(entry, decision)
    plan.setdefault("provenance", {})["software_intersection"] = {
        **requirement["software_intersection"],
        "candidate_screens": screens,
    }
    # Exact source obligations are covered by complete materialized source
    # programs, independently of the target capability-axis representatives.
    source_entries, source_outputs = _materialized_iteration_capsules(full, digest)
    outputs.update(source_outputs)
    plan.setdefault("provenance", {})["materialized_iteration_captures"] = {
        "status": "byte_verified",
        "applications": sorted(captures),
        "full_inventory_sha256": digest,
        "scope": "full capture, not headline validation",
        "qualification": "host reference and source coverage only; target support and execution unverified",
    }
    if plan.get("status") != "blocked":
        plan["capsules"] = [*plan.get("capsules", []), *source_entries]
    outputs["synthesis-plan.json"] = _json(plan)
    _materialize_evidence(root, outputs)
    selected = select_evidence(
        te.target,
        descriptor=declaration.descriptor,
        capability_contract_path=capability_contract_path,
        software_spec=software,
        hardware_spec=hardware,
        facts_path=rtl_facts,
        conformance_spec=root / "requirements.yaml",
        native_qualifications=native_qualifications,
    )
    manifest = export_evidence(selected, root / "evidence")
    identity = synthesis_input_identity(
        conformance_spec=root / "requirements.yaml",
        recipe=declaration.recipe,
        descriptor=declaration.descriptor,
        software_spec=software,
    )
    if plan.get("status") != "blocked":
        profile = {
            "provenance": {
                **plan.get("provenance", {}),
                "selected_inputs": identity,
                "qualification": "diagnostic candidate generation, not a reviewed corpus",
            },
            "capsules": plan.get("capsules", []),
        }
        _materialize_evidence(root, {"synthesis.yaml": yaml.safe_dump(profile, sort_keys=False).encode()})
    accounting = json.loads((root / "evidence/coverage/operation-accounting.json").read_bytes())
    operation_plan = plan.get("provenance", {}).get("application_operation_plan") or {}
    report = {
        "schema": "merlin.phase0_derivation.v1",
        "target": te.target,
        "status": "diagnostic",
        "agentic": False,
        "selected_inputs": identity,
        "raw_facts_sha256": selected.raw_facts_sha256,
        "evidence_artifacts": len(manifest["artifacts"]),
        "applications": sorted(captures),
        "native_baseline_observations": {
            label: {
                key: value
                for key, value in observation.items()
                if key
                in {"status", "executor", "capture_sha256", "receipt_sha256", "max_absolute_error", "target_executed"}
            }
            for label, observation in selected.native_baseline_observations.items()
        },
        "mlir_operations": full["n_operations"],
        "source_trace_statuses": {
            label: app.get("pytorch_provenance", {}).get("source_trace_status", "unknown")
            for label, app in accounting.get("applications", {}).items()
        },
        "candidate_capsules": len(plan.get("capsules", [])),
        "synthesis_profile": "synthesis.yaml" if plan.get("status") != "blocked" else None,
        "application_operation_plan": {
            key: value for key, value in operation_plan.items() if key not in {"obligations", "missing_mapping"}
        },
        "blockers": [*selected.qualification_blockers, *([plan["reason"]] if plan.get("status") == "blocked" else [])],
        "qualification": "derivation only; no compiler execution, oracle or release approval",
    }
    _materialize_evidence(root, {"derivation.json": _json(report)})
    return report
