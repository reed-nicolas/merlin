"""One captured W8A8 integer operation becomes one inspectable, verified PyTorch slice."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from copy import deepcopy

import pytest
import yaml

from merlin.targetgen import capsule_inputs
from merlin.targetgen.application_inventory import (
    exact_int_mm_geometry, verified_static_integerization, verify_capture_receipt,
)


def test_integer_reference_projection_requires_exact_accounted_artifact():
    agreement = {
        "status": "passed", "finite": True, "samples": 1, "reference": "pt2e_integer",
        "max_abs": 0.0, "max_rel": 0.0, "atol": 0.0, "rtol": 0.0,
        "source": {"sha256": "a" * 64},
        "output": {"path": "integer-reference.json", "sha256": "b" * 64},
        "executed_contractions": {"linear": 1, "conv2d": 0, "matmul": 0,
                                  "total": 1, "selected": 1, "observed": 1},
        "outputs": [{"finite": True, "within_tolerance": True,
                     "max_abs": 0.0, "max_rel": 0.0, "atol": 0.0, "rtol": 0.0}],
    }
    projection = {
        "schema": "merlin.capture_integerization.v1", "status": "byte_bound_metadata",
        "source_quantization": "int8_static_act_int8_weight",
        "software_numerical_engine": "integer_reference",
        "capture_receipt_sha256": "c" * 64,
        "metadata": {"sha256": "d" * 64, "bytes": 1},
        "capture": {"sha256": "e" * 64, "bytes": 1},
        "reference_artifact": {"sha256": "b" * 64, "bytes": 1},
        "integerization_receipt": {
            "schema": "m2m.pt2e-integerize.v1", "accumulator_bound_checked": True,
            "quantized_contractions_seen": 1, "quantized_contractions_integerized": 1,
            "quantized_contractions_remaining": 0, "exported_integer_mm_count": 1,
            "integer_mm_emitted": 1, "refusals": [], "golden_agreement": agreement,
            "quantized_by_kind": {"linear": {"seen": 1}, "conv2d": {"seen": 0},
                                  "matmul": {"seen": 0}},
        },
    }
    assert verified_static_integerization(projection)
    broken = deepcopy(projection)
    broken["reference_artifact"]["sha256"] = "f" * 64
    assert not verified_static_integerization(broken)
    broken = deepcopy(projection)
    broken["integerization_receipt"]["golden_agreement"]["outputs"][0]["max_abs"] = 0.001
    assert not verified_static_integerization(broken)
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.capsule_source import (
    M2MUnavailable,
    PytorchRefSource,
    _freeze_selected_m2m_tool,
    write_pytorch_capsule,
)
from merlin.targetgen.corpus_synth import SynthesisError, _application_operation_plan, exact_int_mm_entries


def _demand_and_inventory():
    row = {
        "operation": "aten._int_mm.default",
        "mlir_operation": "linalg.generic",
        "frontend_op": "aten._int_mm.default",
        "provenance_op": "int_matmul",
        "semantic_family": "contraction",
        "operand_format": "int8",
        "accumulator_dtypes": ["i32"],
        "ordered_operand_types": [
            {"shape": [2, 32], "dtype": "i8"},
            {"shape": [32, 64], "dtype": "i8"},
            {"shape": [2, 64], "dtype": "i32"},
        ],
        "ordered_result_types": [{"shape": [2, 64], "dtype": "i32"}],
        "result_shapes": [[2, 64]],
        "contraction_shape": {"M": 2, "K": 32, "N": 64, "rank": 3},
        "indexing_maps": [
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
        ],
        "iterator_types": [
            "#linalg.iterator_type<parallel>",
            "#linalg.iterator_type<parallel>",
            "#linalg.iterator_type<reduction>",
        ],
        "body_operations": ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"],
        "quant_evidence": {"prov.quant_inner_1": "weights.int_data"},
        "disposition": "hardware_admitted",
        "count": 1,
        "ordinals": [4],
    }
    app = {
        "capture_sha256": "b" * 64,
        "capture_normalization": {"normalized_sha256": "c" * 64},
        "capture_quantization": "int8_dyn_act_int8_weight",
        "signatures": [row],
    }
    full = {"schema_version": 1, "applications": {"app": app}}
    digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    demands = {
        "full_inventory_sha256": digest,
        "status": "inventoried",
        "n_operations": 1,
        "applications": {
            "app": {
                **{key: app[key] for key in ("capture_sha256", "capture_normalization", "capture_quantization")},
                "counts": {"hardware_admitted": 1},
            }
        },
        "operation_groups": [
            {
                "operation": "aten._int_mm.default",
                "mlir_operation": "linalg.generic",
                "semantic_family": "contraction",
                "operand_format": "int8",
                "disposition": "hardware_admitted",
                "shape_class": "contraction:rank_3",
                "count": 1,
                "sources": {"app": {"capture_sha256": app["capture_sha256"], "count": 1}},
            }
        ],
    }
    return demands, full


def test_exact_integer_matmul_bridge_refuses_weaker_signatures():
    demands, full = _demand_and_inventory()
    entries, refused = exact_int_mm_entries(demands, full)
    assert len(entries) == 1 and not refused
    entry = entries[0]
    assert (entry["M"], entry["K"], entry["N"]) == (2, 32, 64)
    assert entry["capture_op"] == "int_matmul" and entry["output_dtype"] == "i32"
    assert entry["application_signature_match"]["status"] == "candidate_unverified"
    assert entry["application_signature_match"]["sources"][0]["source_index"] == 0
    assert "application" not in entry["application_signature_match"]["sources"][0]

    row = full["applications"]["app"]["signatures"][0]
    assert exact_int_mm_geometry(row) == (2, 32, 64)
    for changed in ("ordered_operand_types", "indexing_maps", "body_operations", "quant_evidence"):
        broken = {**row, changed: None}
        assert exact_int_mm_geometry(broken) is None, changed
    with pytest.raises(SynthesisError, match="digest differs"):
        exact_int_mm_entries({**demands, "full_inventory_sha256": "a" * 64}, full)

    plan = _application_operation_plan(demands, exact_entries=entries)
    assert plan["status"] == "obligations_pending" and plan["blocked_operations"] == 0
    obligation = plan["obligations"][0]
    assert obligation["status"] == "candidate_unverified"
    assert obligation["coverage_status"] == "unverified"
    assert obligation["capsule_candidates"] == [entry["name"]]
    entry["application_signature_match"]["sources"][0]["ordinals"] = []
    partial = _application_operation_plan(demands, exact_entries=entries)
    assert partial["status"] == "blocked" and partial["blocked_operations"] == 1
    assert partial["obligations"][0]["status"] == "refused"


def test_exact_integer_slice_uses_private_roster_ordinals_without_losing_group_identity():
    demands, full = _demand_and_inventory()
    private_name = "heldout_secret_model"
    full["applications"][private_name] = deepcopy(full["applications"]["app"])
    demands["applications"][private_name] = deepcopy(demands["applications"]["app"])
    demands["operation_groups"][0]["count"] = 2
    demands["operation_groups"][0]["sources"][private_name] = {
        "capture_sha256": full["applications"][private_name]["capture_sha256"],
        "count": 1,
    }
    demands["n_operations"] = 2
    demands["full_inventory_sha256"] = hashlib.sha256(
        json.dumps(full, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    entries, refused = exact_int_mm_entries(demands, full)
    assert len(entries) == 1 and not refused
    assert [source["source_index"] for source in entries[0]["application_signature_match"]["sources"]] == [0, 1]
    assert private_name not in yaml.safe_dump(entries)
    plan = _application_operation_plan(demands, exact_entries=entries)
    assert plan["blocked_operations"] == 0
    assert plan["obligations"][0]["capsule_candidates"] == [entries[0]["name"]]


def test_integer_matmul_capsule_has_matching_linalg_and_independent_golden(tmp_path, monkeypatch):
    source = PytorchRefSource()
    if not source.available():
        pytest.skip("model2MLIR capture interpreter is unavailable")
    from merlin.targetgen.corpus_spec import CorpusBinding

    demands, full = _demand_and_inventory()
    entry = exact_int_mm_entries(demands, full)[0][0]
    binding = CorpusBinding(
        target="test",
        tile_dim=16,
        operand_dtype="i8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0", "L1"],
        compare="exact_int",
        classes_for=lambda **_: [],
    )
    original_capture = source.capture
    saved_capture = {}

    def capture_once(spec):
        result = original_capture(spec)
        saved_capture["artifact"] = result
        return result

    monkeypatch.setattr(source, "capture", capture_once)
    directory = write_pytorch_capsule(entry, binding, tmp_path, source=source)
    capsule = load_capsule(directory)
    golden = yaml.safe_load((directory / "golden.yaml").read_text())
    assert capsule["application_signature_match"]["status"] == "verified_capture_match"
    assert capsule["semantic"]["generalization_axis"] == "application_operation"
    assert capsule["source_role"] == "model_derived"
    assert capsule["numeric_policy"] == {"compare": "exact_int", "dtype": "i32"}
    assert "tensor<2x64xi32>" in (directory / "capsule.interface.mlir").read_text()
    assert 'prov.aten = "aten._int_mm.default"' in (directory / "capsule.linalg.mlir").read_text()
    assert golden["golden_source"] == "host_torch_eager"
    assert len(golden["outputs"]["Y0"]) == 2
    from merlin.targetgen import capsule_golden

    assert capsule_inputs.is_exact_pytorch_integer_source(capsule)
    leaves = capsule_inputs.materialize_capsule_leaves(capsule)
    values = capsule_inputs.materialized_input_values(capsule)
    assert set(values) == {"A0", "W"}
    for name, tensor in leaves.items():
        saved = golden["oracle_provenance"]["inputs"][name]
        assert saved["dtype"] == tensor.dtype == "i8"
        assert bytes(value & 0xFF for value in tensor.data).hex() == saved["integer_bytes_hex"]
        assert values[name]["values"] == tensor.data
        assert capsule_inputs.canonical_input_values(capsule, directory)[name]["values"] == tensor.data
    assert capsule_golden.golden(capsule, directory) == golden["outputs"]
    from merlin.runtime.commandbuffer import materialize_inputs

    # A command-buffer reference and a positional whole-program harness both receive the
    # exact captured bytes, never their own deterministic fill.
    for positional in (False, True):
        selected = list(values.items())
        names = [f"arg{i}" for i in range(len(selected))] if positional else [name for name, _ in selected]
        cb = {
            "tensors": {
                alias: {"role": "input", "shape": value["shape"], "dtype": "i8"}
                for alias, (_, value) in zip(names, selected, strict=True)
            },
            "canonical_inputs": {alias: deepcopy(value) for alias, (_, value) in zip(names, selected, strict=True)},
        }
        capsule_inputs.bind_exact_integer_stimulus(capsule, cb)
        assert [materialize_inputs(cb)[alias].data for alias in names] == [value["values"] for _, value in selected]
        cb["canonical_inputs"][names[0]]["values"][0] += 1
        with pytest.raises(ValueError, match="changed input binding"):
            capsule_inputs.bind_exact_integer_stimulus(capsule, cb)

    tampered = tmp_path / "tampered-source"
    shutil.copytree(directory, tampered)
    tampered_golden = deepcopy(golden)
    del tampered_golden["oracle_provenance"]["inputs"]["A0"]["integer_bytes_hex"]
    (tampered / "golden.yaml").write_text(yaml.safe_dump(tampered_golden))
    with pytest.raises(ValueError, match="captured bytes"):
        capsule_golden.golden(load_capsule(tampered), tampered)
    tampered_golden = deepcopy(golden)
    tampered_golden["oracle_provenance"]["inputs"]["A0"]["integer_bytes_hex"] = "00" * len(leaves["A0"].data)
    (tampered / "golden.yaml").write_text(yaml.safe_dump(tampered_golden))
    with pytest.raises(ValueError, match="projection differs"):
        capsule_golden.golden(load_capsule(tampered), tampered)
    tampered_golden = deepcopy(golden)
    tampered_golden["outputs"]["Y0"][0][0] += 1
    (tampered / "golden.yaml").write_text(yaml.safe_dump(tampered_golden))
    with pytest.raises(ValueError, match="host-eager output differs"):
        capsule_golden.golden(load_capsule(tampered), tampered)

    # The same public writer, not just the standalone tool observer, must keep
    # the exact installed package bytes and a bounded identity statement.
    install = tmp_path / "selected-m2m"
    module = install / "m2m" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"# selected package\n")
    wheel = tmp_path / "m2m-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("m2m/__init__.py", module.read_bytes())
    dist = install / "m2m-0.0.1.dist-info"
    dist.mkdir()
    (dist / "direct_url.json").write_text(json.dumps({"url": wheel.as_uri()}))
    installed_source = PytorchRefSource(m2m_dir=install)
    monkeypatch.setattr(installed_source, "capture", lambda spec: saved_capture["artifact"])
    bound = write_pytorch_capsule(entry, binding, tmp_path / "bound", source=installed_source)
    tool = load_capsule(bound)["capture_tool"]
    assert tool["status"] == "verified_selected_package"
    sidecar = (bound / tool["path"]).read_bytes()
    assert hashlib.sha256(sidecar).hexdigest() == tool["sha256"]
    observed = json.loads(sidecar)
    assert hashlib.sha256((bound / observed["wheel"]["path"]).read_bytes()).hexdigest() == observed["wheel"]["sha256"]
    assert observed["modules"]["m2m/__init__.py"]["sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()
    from merlin.targetgen.contract.materialize import materialize_public_capsules

    admitted = materialize_public_capsules(tmp_path / "cohort", corpus_roots=[bound.parent])
    assert bound.name in admitted
    copied = tmp_path / "cohort" / bound.name
    assert "capture_tool" not in load_capsule(copied)
    assert not (copied / "capture-tool.json").exists()
    assert not (copied / "capture-tool.whl").exists()
    (bound / "capture-tool.whl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="wheel bytes changed"):
        load_capsule(bound)
    with pytest.raises(ValueError, match="absent or indirect"):
        (bound / "capture-tool.whl").unlink()
        materialize_public_capsules(tmp_path / "refused-cohort", corpus_roots=[bound.parent])
    module.write_bytes(b"# changed after wheel installation\n")
    with pytest.raises(M2MUnavailable, match="differs from its selected wheel"):
        _freeze_selected_m2m_tool(install, tmp_path / "refused")


def test_static_integer_slices_require_saved_byte_bound_complete_finite_conversion(tmp_path, monkeypatch):
    """A real receipt projection feeds planning; missing/tampered observations cannot lend proof."""
    demands, full = _demand_and_inventory()
    agreement = {
        "status": "passed",
        "samples": 1,
        "finite": True,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "atol": 0.001,
        "rtol": 0.001,
        "outputs": [
            {"finite": True, "within_tolerance": True, "max_abs": 0.0, "max_rel": 0.0, "atol": 0.001, "rtol": 0.001}
        ],
    }
    metadata = {
        "scheme": "int8_static_act_int8_weight",
        "integerization_receipt": {
            "schema": "m2m.pt2e-integerize.v1",
            "accumulator_bound_checked": True,
            "quantized_contractions_seen": 1,
            "quantized_contractions_integerized": 1,
            "quantized_contractions_remaining": 0,
            "exported_integer_mm_count": 1,
            "integer_mm_emitted": 1,
            "refusals": [],
            "golden_agreement": agreement,
        },
    }
    artifacts = {}
    for name, content in {
        "model.mlir": b"module {}",
        "weights.safetensors": b"weights",
        "weights.safetensors.manifest.json": b"{}",
        "meta.json": json.dumps(metadata, sort_keys=True).encode(),
    }.items():
        (tmp_path / name).write_bytes(content)
        artifacts[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    receipt = {
        "schema": "m2m.capture-receipt.v1",
        "materialized_abi": {"complete": True},
        "artifacts": artifacts,
        "source_closure_verified": False,
    }
    (tmp_path / "capture_receipt.json").write_text(json.dumps(receipt))
    observed = verify_capture_receipt(tmp_path / "model.mlir")
    projection = observed.pop("capture_integerization")
    assert observed["status"] == "verified_materialized" and observed["source_closure_verified"] is False
    # A producer can edit its own receipt. Materialization verification must not
    # upgrade that claim to independently verified source closure.
    receipt["source_closure_verified"] = True
    (tmp_path / "capture_receipt.json").write_text(json.dumps(receipt))
    claimed = verify_capture_receipt(tmp_path / "model.mlir")
    assert claimed["status"] == "verified_materialized"
    assert claimed["source_closure_verified"] is False
    receipt["source_closure_verified"] = False
    (tmp_path / "capture_receipt.json").write_text(json.dumps(receipt))
    app = full["applications"]["app"]
    app.update(
        capture_sha256=artifacts["model.mlir"]["sha256"],
        capture_quantization=metadata["scheme"],
        capture_receipt=observed,
        capture_integerization=projection,
    )
    app["signatures"][0]["quant_evidence"] = None  # Only verified static materialization supplies the origin.

    def candidates(inventory):
        selected = deepcopy(demands)
        selected["full_inventory_sha256"] = hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        selected["applications"]["app"]["capture_quantization"] = metadata["scheme"]
        selected["applications"]["app"]["capture_sha256"] = app["capture_sha256"]
        selected["operation_groups"][0]["sources"]["app"]["capture_sha256"] = app["capture_sha256"]
        entries, refused = exact_int_mm_entries(selected, inventory)
        return selected, entries, refused

    selected, entries, refused = candidates(full)
    assert len(entries) == 1 and not refused
    match = entries[0]["application_signature_match"]
    assert match["source_quantization"] == metadata["scheme"]
    assert match["sources"][0]["capture_integerization"] == projection
    assert _application_operation_plan(selected, exact_entries=entries)["blocked_operations"] == 0
    from merlin_experiments.phase0.writer import _write_capsule_inner

    from merlin.targetgen import capsule_source as CSRC
    from merlin.targetgen.corpus_spec import CorpusBinding

    class AvailableSource:
        def available(self):
            return True

    observed_entries = []
    monkeypatch.setattr(CSRC, "PytorchRefSource", AvailableSource)
    monkeypatch.setattr(CSRC, "write_pytorch_capsule", lambda entry, *_args, **_kwargs: observed_entries.append(entry))
    binding = CorpusBinding(
        target="test",
        tile_dim=16,
        operand_dtype="i8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0", "L1"],
        compare="exact_int",
        classes_for=lambda **_: [],
    )
    _write_capsule_inner(entries[0], binding, tmp_path)
    assert observed_entries == entries
    for failure in (
        "missing",
        "receipt_mismatch",
        "capture_mismatch",
        "remaining",
        "nonfinite",
        "failed_agreement",
        "wrong_body",
    ):
        broken = deepcopy(full)
        record = broken["applications"]["app"]
        if failure == "missing":
            del record["capture_integerization"]
        elif failure == "receipt_mismatch":
            record["capture_integerization"]["capture_receipt_sha256"] = "d" * 64
        elif failure == "capture_mismatch":
            record["capture_integerization"]["capture"]["sha256"] = "e" * 64
        elif failure == "wrong_body":
            record["signatures"][0]["body_operations"] = ["arith.addi"]
        else:
            proof = record["capture_integerization"]["integerization_receipt"]
            if failure == "remaining":
                proof["quantized_contractions_remaining"] = 1
            elif failure == "nonfinite":
                proof["golden_agreement"]["outputs"][0]["max_abs"] = float("inf")
            else:
                proof["golden_agreement"]["status"] = "failed"
        assert candidates(broken)[1] == [], failure
        assert len(candidates(broken)[2]) == 1, failure
    (tmp_path / "meta.json").write_text(json.dumps({**metadata, "scheme": "other"}))
    tampered = verify_capture_receipt(tmp_path / "model.mlir")
    assert tampered["status"] == "unverified" and "capture_integerization" not in tampered


def test_phase0_live_model_uses_only_frozen_sw_scoped_recipe(tmp_path):
    """A broad HW recipe cannot silently re-enable an SW-excluded operation."""
    from merlin_experiments.phase0.generation import _selected_capture_recipe

    from merlin.targetgen.quant_recipe import digest

    recipe = {
        "schema": "quant_recipe_v1",
        "target": "test",
        "status": "derived",
        "families": ["contraction"],
        "weight": {"dtype": "int8"},
        "activation": {"dtype": "int8"},
        "accumulator_dtype": "int32",
        "unquantized": {"window_mean": "not selected by software operation scope"},
    }
    recipe["recipe_sha256"] = digest(recipe)
    raw = json.dumps(recipe, sort_keys=True).encode()
    byte_sha = hashlib.sha256(raw).hexdigest()
    folder = tmp_path / "software" / "quantization-recipes"
    folder.mkdir(parents=True)
    (folder / f"{byte_sha}.json").write_bytes(raw)
    index = {
        "schema": "merlin.phase0.capture_recipes.v1",
        "target": "test",
        "recipes": [
            {
                "path": f"software/quantization-recipes/{byte_sha}.json",
                "sha256": byte_sha,
                "recipe_sha256": recipe["recipe_sha256"],
            }
        ],
    }
    index_path = tmp_path / "software" / "quantization-recipes.json"
    index_path.write_text(json.dumps(index))
    selected = _selected_capture_recipe(tmp_path, target="test", operand_dtype="i8", accumulator_dtype="i32")
    assert selected["families"] == ["contraction"] and "window_mean" in selected["unquantized"]
    with pytest.raises(ValueError, match="unambiguous"):
        _selected_capture_recipe(tmp_path, target="test", operand_dtype="i4", accumulator_dtype="i32")
    broad = deepcopy(recipe)
    broad["families"].append("window_mean")
    broad["recipe_sha256"] = digest(broad)
    broad_raw = json.dumps(broad, sort_keys=True).encode()
    broad_sha = hashlib.sha256(broad_raw).hexdigest()
    (folder / f"{broad_sha}.json").write_bytes(broad_raw)
    index["recipes"].append(
        {
            "path": f"software/quantization-recipes/{broad_sha}.json",
            "sha256": broad_sha,
            "recipe_sha256": broad["recipe_sha256"],
        }
    )
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unambiguous"):
        _selected_capture_recipe(tmp_path, target="test", operand_dtype="i8", accumulator_dtype="i32")
    index["recipes"] = index["recipes"][:1]
    index["recipes"][0]["sha256"] = "0" * 64
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="bytes differ"):
        _selected_capture_recipe(tmp_path, target="test", operand_dtype="i8", accumulator_dtype="i32")
