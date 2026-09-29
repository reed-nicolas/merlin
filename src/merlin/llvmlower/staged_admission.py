"""Byte-bound development evidence for an unreviewed whole-model kernel candidate.

This is deliberately separate from ``ExactOffloadSelection``.  Source tensor SSA
establishes logical types and dataflow, not physical layout, tail materialization,
or buffer aliasing.  Emitting a package artifact and a host shim helps develop
those obligations, but cannot review a software declaration or certify execution.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from subprocess import TimeoutExpired

from merlin.common.digest import sha256_text
from merlin.targetgen import package_runtime
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.contract.model_kernel_outline import outline_integer_matmuls
from merlin.targetgen.contract.model_kernel_route import _emit
from merlin.targetgen.contract.resident_interface_abi import bind_single_resident_matmul
from merlin.targetgen.rtl.facts import validate_facts

from .device_shim import emit_translation_unit, kernel_abi_for
from .exact_offload import _package_sha256

SCHEMA = "merlin.staged_model_kernel_admission.v1"


def _tile_edge_from_exact_facts(raw: bytes, target: str) -> tuple[int, str | None]:
    try:
        document = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("selected RTL facts are not JSON") from exc
    inputs = document.get("inputs") if isinstance(document, dict) else None
    if not isinstance(inputs, dict) or inputs.get("target") != target:
        raise ValueError("selected RTL facts do not identify the exact candidate target")
    problems = validate_facts(document, target=target)
    if problems:
        raise ValueError("selected RTL facts fail validation: " + "; ".join(problems[:3]))
    body = document.get("facts")
    arrays = body.get("arrays") if isinstance(body, dict) else None
    if not isinstance(arrays, list):
        raise ValueError("selected RTL facts have no arrays list")
    edges = set()
    for array in arrays:
        if not isinstance(array, dict):
            raise ValueError("selected RTL facts contain an array without dimensions")
        rows, cols = array.get("rows"), array.get("cols")
        if type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0:
            raise ValueError("selected RTL facts contain an array without positive integer dimensions")
        # The current shim has one tile edge, not independent row/column edges.
        # min(rows, cols) would silently mis-stage a rectangular mesh.
        if rows != cols:
            raise ValueError("selected RTL facts contain a non-square array; one tile edge is insufficient")
        edges.add(rows)
    if len(edges) != 1:
        raise ValueError("selected RTL facts do not determine one unambiguous tile edge")
    consistency = document.get("source_consistency")
    reported_status = consistency.get("status") if isinstance(consistency, dict) else None
    return edges.pop(), reported_status


def stage_integer_model_admission(
    model: bytes,
    *,
    target: str,
    software_spec: bytes,
    capability_contract: bytes,
    package_dir: str | Path,
    operation_id: str,
    rtl_facts: bytes | None = None,
    timeout: int = 30,
) -> dict:
    """Observe one exact kernel while accounting for every model candidate.

    This function cannot return an offload selection or raise an admission status.
    The reviewed-SW, release, running accelerator and model-numeric gates remain
    owned by the existing exact selection/certification path.
    """
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    outline = outline_integer_matmuls(
        model, target=target, software_spec=software_spec, capability_contract=capability_contract
    )
    by_id = {row["operation_id"]: row for row in outline["candidates"]}
    candidate = by_id.get(operation_id)
    if candidate is None or not operation_id.startswith(f"mlir:{outline['model_sha256']}:"):
        raise ValueError(f"{operation_id!r} is not a candidate of these exact model bytes")
    interface = candidate["interface_mlir"]
    interface_sha = sha256_text(interface)
    if interface_sha != candidate["interface_sha256"]:
        raise ValueError("selected candidate interface bytes changed after outlining")
    resident = bind_single_resident_matmul(interface, target=target)
    m, n, k = resident.m, resident.n, resident.k

    package = package_runtime.load_package(package_dir)
    if package.target != target:
        raise ValueError("selected package target differs from exact candidate target")
    package_sha = _package_sha256(Path(package.directory))
    abi = kernel_abi_for(target)
    abi_sha = sha256_text(repr(abi)) if abi is not None else None
    if abi is not None and abi.symbol != resident.kernel_symbol:
        raise ValueError("selected shim kernel name disagrees with pointer ABI contract")
    edge, facts_consistency = _tile_edge_from_exact_facts(rtl_facts, target) if rtl_facts is not None else (None, None)
    identity = {
        "target": target,
        "model_sha256": outline["model_sha256"],
        "operation_id": operation_id,
        "interface_sha256": interface_sha,
        "software_spec_sha256": outline["software_spec_sha256"],
        "capability_contract_sha256": outline["capability_contract_sha256"],
        "package_sha256": package_sha,
        "kernel_abi_sha256": abi_sha,
        "kernel_abi_contract_sha256": resident.contract_sha256,
        "rtl_facts_sha256": hashlib.sha256(rtl_facts).hexdigest() if rtl_facts is not None else None,
    }
    identity["binding_sha256"] = sha256_text(json.dumps(identity, sort_keys=True, separators=(",", ":")))

    with tempfile.TemporaryDirectory(prefix="merlin_staged_admission_") as scratch:
        path = Path(scratch) / "selected.interface.mlir"
        path.write_text(interface, encoding="utf-8")
        command_path = Path(scratch) / "command_buffer.json"
        command = _emit(
            package, path, command_path, text=interface, target=target, timeout=timeout
        )
        if command["status"] == "isolated_kernel_emitted":
            generated = json.loads(command_path.read_bytes())
            expected = parse_interface_mlir(interface)
            inputs_match = all(
                (generated.get("tensors") or {}).get(name, {}).get(field) == spec[field]
                for name, spec in expected["tensors"].items()
                for field in ("shape", "dtype")
            )
            if generated.get("commands") != expected["commands"] or not inputs_match:
                command = {
                    "status": "interface_disagreement",
                    "reason": "package command wiring or external tensor types differ from the exact interface",
                    "command_buffer_sha256": command["command_buffer_sha256"],
                }
        artifact: dict = {"status": "not_attempted", "reason": "command buffer did not emit"}
        if command["status"] == "isolated_kernel_emitted":
            try:
                process = package_runtime.run_entrypoint(package, "emit_target_artifact", path, timeout=timeout)
                if process.returncode != 0:
                    artifact = {"status": "compiler_error", "reason": (process.stderr or "")[-500:]}
                elif not process.stdout or f"llvm.func @{resident.kernel_symbol}(" not in process.stdout:
                    artifact = {"status": "invalid_artifact", "reason": "no contract-named LLVM kernel entry"}
                else:
                    artifact = {
                        "status": "emitted_unverified",
                        "sha256": sha256_text(process.stdout),
                        "bytes": len(process.stdout.encode("utf-8")),
                    }
            except (OSError, TimeoutExpired) as exc:
                artifact = {"status": "compiler_unavailable", "reason": str(exc)[:500]}
    if _package_sha256(Path(package.directory)) != package_sha:
        raise ValueError("selected OOT package changed during staged admission")

    # No ambient RTL cache lookup: a shim emitted from an unpinned tile edge
    # could not be reproduced or bound to this exact review identity.
    unit = None if edge is None else emit_translation_unit(
        target, {"merlin_staged_kernel": (m, n, k)},
        {"merlin_staged_kernel": resident.dtypes},
        kernel_symbol_for=(lambda _symbol: abi.symbol) if abi is not None else None,
        tile_edge=edge,
    )
    shim_generated = unit is not None and bool(unit.symbols)
    shim = {
        "status": "generated_unexecuted" if shim_generated else "declined",
        "binding_sha256": identity["binding_sha256"],
        "kernel_symbol": resident.kernel_symbol,
        "pointer_order": list(resident.pointer_order),
        "tile_edge": edge,
        "rtl_facts_provenance": {
            "reported_source_consistency": facts_consistency or "not_reported",
            "status": "reported_verified_not_rechecked" if facts_consistency == "verified" else "unverified",
            "qualification": "source consistency is reported by supplied facts, not independently checked here",
        },
        "padding_required": bool(edge and any(extent % edge for extent in (m, n, k))),
        "sha256": sha256_text(unit.text) if shim_generated else None,
        "source": unit.text if shim_generated else None,
        "declines": list(unit.skipped) if unit is not None else [
            ["merlin_staged_kernel", "explicit same-target RTL facts are required to derive a bound tile edge"]
        ],
    }
    compiler = {
        "status": "emitted_unverified" if command["status"] == "isolated_kernel_emitted"
        and artifact["status"] == "emitted_unverified" else "incomplete",
        "binding_sha256": identity["binding_sha256"],
        "package_id": package.package_id,
        "command_buffer": command,
        "target_artifact": artifact,
    }
    obligations = {
        "layouts": {
            "status": "requires_review",
            "source_fact": "ranked tensor SSA has no physical strides",
            "codegen_observation": (
                "row-major descriptor guard generated, not executed" if shim_generated else "shim unavailable"
            ),
            "review_question": "Are the selected model's realized memrefs contiguous, or does the guard reject them?",
        },
        "tails": {
            "status": "requires_review",
            "source_fact": {"M": m, "N": n, "K": k},
            "codegen_observation": (
                "padding shim generated, not numerically certified" if shim_generated else "shim unavailable"
            ),
            "review_question": "Do the selected package kernel and shim preserve zero-padded valid-window results?",
        },
        "aliasing": {
            "status": "requires_review",
            "source_fact": "SSA identity does not establish buffer non-overlap",
            "codegen_observation": (
                "output non-overlap guard generated, not executed" if shim_generated else "shim unavailable"
            ),
            "review_question": "Are valid selected-model buffers disjoint at dispatch, without silent host fallback?",
        },
    }
    if bind_single_resident_matmul(interface, target=target).contract_sha256 != resident.contract_sha256:
        raise ValueError("kernel pointer ABI contract changed during staged admission")
    counts = Counter(row["software_admission"]["status"] for row in outline["candidates"])
    physical_constraints = sorted(
        set(candidate["software_admission"].get("unresolved_constraints", ()))
        & {"layouts", "tails", "aliasing"}
    )
    return {
        "schema": SCHEMA,
        "exact_binding": identity,
        "selected_interface_mlir": interface,
        "candidate_count": len(outline["candidates"]),
        "candidate_admission_counts": dict(sorted(counts.items())),
        "refused_count": len(outline["refused"]),
        "source_facts": {
            "logical_shapes": {
                resident.lhs_tensor: [m, k], resident.weight_tensor: [k, n], resident.output_tensor: [m, n],
            },
            "logical_dtypes": dict(zip(
                (resident.lhs_tensor, resident.weight_tensor, resident.output_tensor), resident.dtypes,
                strict=True,
            )),
            "operation": "exact_signed_i8xi8_to_i32_matmul_with_zero_initializer",
            "operand_bindings": candidate["operand_bindings"],
            "output_result_id": f"{operation_id}:result:{candidate['output_result_index']}",
            "physical_layout": "not_observed_in_tensor_ssa",
            "physical_aliasing": "not_observed_in_tensor_ssa",
        },
        "software_admission": candidate["software_admission"],
        "admission_boundary": {
            "phase0_semantic_screen": {
                "status": "diagnostic_observation_not_phase0_admission",
                "logical_operation": "exact_signed_i8xi8_to_i32_matmul",
                "authored_review_status": candidate["software_admission"].get("review_status"),
                "software_admission_status": candidate["software_admission"]["status"],
                "unresolved_physical_constraints": physical_constraints,
                "qualification": (
                    "logical source observation only; this receipt is not a Phase 0 admission "
                    "artifact or reviewed accelerator support"
                ),
            },
            "phase1_physical_plan": {
                "status": "generated_not_executed" if shim_generated else "unavailable",
                "transport": "host pointer call plus guarded contiguous descriptors and edge staging",
                "build_time_seam": "merlin.llvmlower.device_build.build_device_objects(expected_interfaces=...)",
                "required_proofs": [
                    "compiled selected OOT artifact and shim with exact pointer ABI",
                    "runtime descriptor layout and non-overlap guards on actual model buffers",
                    "valid-window tail numerics on an executing target kernel",
                ],
                "qualification": "a generated C plan is not a physical-layout proof or SW review",
            },
            "promotion_policy": "this staged artifact cannot upgrade software admission or certify whole-model offload",
        },
        "compiler_evidence": compiler,
        "shim_evidence": shim,
        "codegen_obligations": obligations,
        "review_required": True,
        "whole_model_offload_verified": False,
        "qualification": (
            "diagnostic codegen evidence only; no reviewed SW admission, accelerator run, or whole-model proof"
        ),
    }
