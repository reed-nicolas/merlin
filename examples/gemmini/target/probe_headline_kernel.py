#!/usr/bin/env python3
"""Generate a bounded Gemmini integer probe from one captured headline matrix body.

The source capture supplies geometry and identity. The emitted i8 operands are
synthetic; a passing native run checks this kernel window, not model quantization
or whole-model execution. Phase 0's iteration corpus is never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml
from probe_native_kernel import inputs_and_scalar

from merlin.runtime.backends.base import get_backend
from merlin.targetgen import corpus_spec
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.source_kernel_probe import derive_kernel_window
from merlin.targetgen.target_experiment import load_target_experiment

REPO = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_binding(facts_path: Path, contract_path: Path):
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    mesh = [row for row in facts.get("facts", {}).get("arrays", []) if row.get("name") == "mesh"]
    if len(mesh) != 1 or mesh[0].get("rows") != mesh[0].get("cols") or int(mesh[0].get("rows") or 0) < 1:
        raise ValueError("selected facts lack one positive square mesh")
    # The example contract adds the Phase 0 corpus issue order. The out-of-tree
    # provider contract owns the executable backend but need not repeat that
    # authoring field. Require their shared machine declaration to agree.
    binding_contract_path = REPO / "examples/gemmini/target/contracts/target_contract.yaml"
    contract = yaml.safe_load(binding_contract_path.read_text(encoding="utf-8"))
    support_contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if support_contract.get("name") != contract.get("name") or support_contract.get("compute_units") != contract.get(
        "compute_units"
    ):
        raise ValueError("selected OOT support and example binding contracts disagree on compute units")
    for key, value in support_contract.get("encoding", {}).items():
        if key in contract.get("encoding", {}) and contract["encoding"][key] != value:
            raise ValueError(f"selected OOT support and example binding contracts disagree on encoding.{key}")
    recipe = yaml.safe_load((REPO / "examples/gemmini/phase0/recipe.yaml").read_text(encoding="utf-8"))
    te = load_target_experiment(REPO / "examples/gemmini/target/descriptor.yaml")
    binding = corpus_spec.derive_binding(te, recipe["datapath"], contract=contract, facts=facts)
    tile = int(mesh[0]["rows"])
    if binding.tile_dim != tile or binding.operand_dtype != "int8" or binding.accum_dtype != "i32":
        raise ValueError("selected contract, facts and integer corpus binding disagree")
    return binding


def generate(
    capture: Path, source_node_id: str, facts_path: Path, contract_path: Path, output_root: Path
) -> tuple[dict, Path]:
    root = output_root.resolve()
    if not root.is_relative_to(REPO / "out") or root.exists() or root.is_symlink():
        raise ValueError("output root must be a fresh path beneath this checkout's out/")
    binding = _selected_binding(facts_path, contract_path)
    projection = derive_kernel_window(
        capture,
        source_node_id,
        tile_dim=binding.tile_dim,
        projection_types=(
            binding.mlir_dtype(binding.operand_dtype),
            binding.mlir_dtype(binding.operand_dtype),
            binding.mlir_dtype(binding.accum_dtype),
        ),
    )
    geometry = projection["projection"]["geometry"]
    identity = hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    name = f"headline_window_{identity}"
    entry = {
        "name": name,
        "kind": "layer",
        "op": "matmul",
        "source_role": "headline_geometry_derived",
        "source_reference": (
            f"{source_node_id} at traced MLIR ordinal {projection['source']['mlir_ordinal']}; "
            "bounded synthetic i8 body"
        ),
        "label": "diagnostic",
        "M": geometry["M"],
        "K": geometry["K"],
        "N": geometry["N"],
        "lhs": "A0",
        "weight": "W",
        "out": "Y0",
        "output_dtype": "i32",
    }
    capsule, interface = corpus_spec.build_matmul(entry, binding)
    capsule["source_kernel_window"] = projection
    capsule["diagnostic_limits"] = {
        "synthetic_integer_operands": True,
        "headline_numerical_equivalence": False,
        "target_admission": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    capsule_path = root / "capsule.yaml"
    capsule_path.write_text(yaml.safe_dump(capsule, sort_keys=False), encoding="utf-8")
    interface_path = root / "capsule.interface.mlir"
    interface_path.write_text(interface, encoding="utf-8")
    _write(
        root / "generation.json",
        {
            "schema": "merlin.gemmini-headline-window-generation.v1",
            "status": "generated_not_executed",
            "source_kernel_window": projection,
            "diagnostic_limits": capsule["diagnostic_limits"],
            "facts_sha256": _sha(facts_path),
            "binding_contract_sha256": _sha(REPO / "examples/gemmini/target/contracts/target_contract.yaml"),
            "support_contract_sha256": _sha(contract_path),
            "capsule_sha256": _sha(capsule_path),
            "interface_sha256": _sha(interface_path),
        },
    )
    return projection, root


def _verified_generation(projection: dict, root: Path) -> dict:
    """Bind a native run to the exact diagnostic files it is about to execute."""
    generation_path = root / "generation.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    capsule_path = root / "capsule.yaml"
    interface_path = root / "capsule.interface.mlir"
    capsule = yaml.safe_load(capsule_path.read_text(encoding="utf-8"))
    if (
        generation.get("schema") != "merlin.gemmini-headline-window-generation.v1"
        or generation.get("status") != "generated_not_executed"
        or generation.get("source_kernel_window") != projection
        or capsule.get("source_kernel_window") != projection
        or capsule.get("diagnostic_limits") != generation.get("diagnostic_limits")
        or generation.get("capsule_sha256") != _sha(capsule_path)
        or generation.get("interface_sha256") != _sha(interface_path)
    ):
        raise ValueError("native probe inputs differ from their generated source binding")
    return {
        "generation_sha256": _sha(generation_path),
        "capsule_sha256": generation["capsule_sha256"],
        "interface_sha256": generation["interface_sha256"],
    }


def run_native(projection: dict, root: Path, *, rtl: bool) -> dict:
    generated_evidence = _verified_generation(projection, root)
    support = Path(os.environ["MERLIN_TARGET_PATH"]).resolve()
    if not (support / "backend/gemmini.py").is_file():
        raise ValueError("selected OOT Gemmini support has no backend/gemmini.py")
    backend = get_backend("gemmini")
    if not backend.available("spike"):
        raise RuntimeError("selected native Gemmini Spike is unavailable")
    interface = root / "capsule.interface.mlir"
    cb = parse_interface_mlir(interface.read_text(encoding="utf-8"))
    work = root / "native"
    work.mkdir()
    spike = backend.run_command_buffer(cb, workdir=work, simulator="spike", timeout=180)
    elf = Path(spike["elf"])
    m, k, n = (projection["projection"]["geometry"][axis] for axis in ("M", "K", "N"))
    a, w, scalar, _, _ = inputs_and_scalar(
        (work / "main.c").read_text(encoding="utf-8"), m, k, n, tile=projection["projection"]["tile_dim"]
    )
    if spike["outputs"].get("Y0") != scalar:
        raise AssertionError("Gemmini Spike differs from scalar arithmetic on compiled input literals")
    observed_partial = 0
    for row in a:
        for column in range(n):
            total = 0
            for index in range(k):
                total += row[index] * w[index][column]
                observed_partial = max(observed_partial, abs(total))
    spec = yaml.safe_load((REPO / "examples/gemmini/target/software-spec.yaml").read_text(encoding="utf-8"))
    signed_bits = spec["numerical_semantics"]["internal_arithmetic"]["mac_result_bits"]
    if observed_partial >= 1 << (signed_bits - 1):
        raise AssertionError("compiled input literals exceed the selected internal MAC exactness bound")
    result = {
        "status": "spike_passed",
        "scope": "bounded_synthetic_i8_matrix_body_only",
        "source_model_equivalence_claim": projection["model_equivalence_claim"],
        "diagnostic_limits": {
            "synthetic_integer_operands": True,
            "headline_numerical_equivalence": False,
            "target_admission": False,
        },
        "source_kernel_window": projection,
        "generated_evidence": generated_evidence,
        "compiled_c_sha256": _sha(work / "main.c"),
        "elf_sha256": _sha(elf),
        "scalar_output_sha256": hashlib.sha256(json.dumps(scalar, separators=(",", ":")).encode()).hexdigest(),
        "max_absolute_partial_sum": observed_partial,
        "spike_cycles": spike["raw_metrics"].get("cycles"),
    }
    (root / "spike_console.log").write_text(spike["console"], encoding="utf-8")
    if rtl:
        if not backend.available("verilator"):
            raise RuntimeError("selected native Gemmini Verilator is unavailable")
        console = backend.run_elf(elf, simulator="verilator", timeout=180)
        outputs, metrics = backend.parse_output(console)
        if outputs.get("Y0") != scalar:
            raise AssertionError("same-ELF Gemmini Verilator differs from scalar arithmetic")
        (root / "verilator_console.log").write_text(console, encoding="utf-8")
        result["status"] = "spike_and_verilator_passed"
        result["verilator_cycles"] = metrics.get("cycles")
    if _verified_generation(projection, root) != generated_evidence:
        raise ValueError("native probe inputs changed during execution")
    _write(root / "numerical_receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--source-node-id", required=True)
    parser.add_argument("--rtl-facts", required=True, type=Path)
    parser.add_argument("--support-contract", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--native", action="store_true", help="execute on native Gemmini Spike")
    parser.add_argument("--rtl", action="store_true", help="also execute the same ELF on Verilator")
    args = parser.parse_args()
    if args.rtl and not args.native:
        parser.error("--rtl requires --native")
    if args.native:
        selected_support = Path(os.environ["MERLIN_TARGET_PATH"]).resolve()
        if args.support_contract.resolve() != (selected_support / "contracts/target_contract.yaml").resolve():
            parser.error("native execution requires the selected OOT support contract")
    projection, root = generate(
        args.capture, args.source_node_id, args.rtl_facts, args.support_contract, args.output_root
    )
    result = run_native(projection, root, rtl=args.rtl) if args.native else {"status": "generated_not_executed"}
    print(json.dumps({"output_root": str(root), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
