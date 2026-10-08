"""RANDOM-INITIALIZED STRUCTURAL DIAGNOSTIC: ResNet50 v1.5 FP32 on the host CPU."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


RTOL, ATOL = 1e-4, 1e-4


def safe_path(value):
    """Validate the helper's import path before importing its path validator."""
    path = Path(os.path.abspath(value))
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise ValueError("Restricted path component")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            target = Path(os.readlink(current))
            current = safe_path(target if target.is_absolute() else current.parent / target)
    return current


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_identity(path, allowed_path):
    path = allowed_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def onnx_inventory(model, onnx):
    def tensor_info(tensor):
        return {"dtype": onnx.TensorProto.DataType.Name(tensor.data_type), "shape": list(tensor.dims)}

    def attribute(value):
        if isinstance(value, onnx.TensorProto):
            return tensor_info(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (list, tuple)):
            return [attribute(item) for item in value]
        return value

    values = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        kind = value.type.tensor_type
        shape = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None for dim in kind.shape.dim]
        values[value.name] = {"dtype": onnx.TensorProto.DataType.Name(kind.elem_type), "shape": shape}
    initializers = {value.name: tensor_info(value) for value in model.graph.initializer}
    values.update(initializers)
    nodes, counts = [], Counter()
    for index, node in enumerate(model.graph.node):
        op = f"{node.domain or 'ai.onnx'}::{node.op_type}"
        counts[op] += 1
        attrs = {attr.name: attribute(onnx.helper.get_attribute_value(attr)) for attr in node.attribute}
        nodes.append({"index": index, "name": node.name, "operator": op, "attributes": attrs, "inputs": [{"name": name, **values.get(name, {})} for name in node.input], "outputs": [{"name": name, **values.get(name, {})} for name in node.output]})
        if node.op_type == "Conv" and len(node.input) > 1 and node.input[1] in initializers:
            initializers[node.input[1]]["layout"] = "OIHW"
        if node.op_type == "Gemm" and len(node.input) > 1 and node.input[1] in initializers:
            initializers[node.input[1]]["layout"] = "OI" if attrs.get("transB", 0) else "IO"
    return {"operator_counts": dict(sorted(counts.items())), "nodes": nodes, "initializers": initializers, "values": values}


def relax_inventory(mod, tvm):
    calls, counts = [], Counter()

    def visit(expr):
        if isinstance(expr, tvm.relax.Call):
            op = getattr(expr.op, "name", getattr(expr.op, "name_hint", str(expr.op)))
            counts[op] += 1
            calls.append({"operator": op, "inputs": [str(arg.struct_info) for arg in expr.args], "output": str(expr.struct_info), "attributes": str(expr.attrs)})

    for function in mod.functions.values():
        if isinstance(function, tvm.relax.Function):
            tvm.relax.analysis.post_order_visit(function.body, visit)
    return {"operator_counts": dict(sorted(counts.items())), "calls": calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", required=True, type=Path)
    parser.add_argument("--tvm-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    output = safe_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    allowed_path = safe_path
    report = {"status": "failed", "stage": "initialization", "scope": __doc__, "trained_model": False, "paper_validation": False, "seed": 194, "rtol": RTOL, "atol": ATOL, "images": [], "verifier_source": file_identity(__file__, allowed_path)}

    def write_json(name, data):
        allowed_path(output / name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    def stage(name):
        report["stage"] = name
        write_json("results.json", report)
        print(name, flush=True)

    try:
        stage("load_frontend_helpers")
        helper = load_module("resnet_frontend_helpers", safe_path(args.tvm_source) / "apps/gemmini/verify_onnx.py")
        allowed_path = helper.allowed_path
        report["frontend_helper_source"] = helper.identity(helper.__file__)
        stage("dependencies")
        import numpy as np
        import onnx
        import torch
        import torchvision
        import torchvision.models.resnet as resnet_source
        import tvm

        torch.set_num_threads(1)
        torch.manual_seed(194)
        from_onnx, metadata = helper.provenance(tvm, torch, onnx, np, None)
        report.update(metadata)
        report["versions"]["torchvision"] = torchvision.__version__
        report["torchvision_resnet_source"] = helper.identity(resnet_source.__file__)
        loader_path = allowed_path(allowed_path(args.model2mlir_root) / "workloads/resnet50_v1_5/loader.py")
        report["loader_source"] = helper.identity(loader_path)
        stage("load_random_initialized_model")
        inherited = {key: value for key, value in os.environ.items() if key.startswith("M2M_RESNET_") or key == "M2M_SESSION_STEPS"}
        for key in inherited:
            os.environ.pop(key)
        isolated = {"M2M_RESNET_RANDOM": "1", "M2M_RESNET_PRETRAINED": "0", "M2M_RESNET_PAPER_READY": "0", "M2M_SESSION_STEPS": "2"}
        os.environ.update(isolated)
        try:
            loader = load_module("resnet_structural_loader", loader_path)
            model, _ = loader.get_model_and_inputs()
        finally:
            for key in isolated:
                os.environ.pop(key, None)
            os.environ.update(inherited)
        model = model.cpu().eval()
        images = model.session_images
        if images.dtype != torch.float32 or tuple(images.shape) != (2, 1, 3, 224, 224) or torch.equal(images[0], images[1]) or not torch.isfinite(images).all():
            raise AssertionError("Expected two distinct finite FP32 NCHW loader images")
        if model.paper_ready or model.session_provenance["checkpoint"] != "random_init":
            raise AssertionError("Expected explicit random initialization without paper-ready attribution")
        backbone = model.model
        blocks = [len(getattr(backbone, f"layer{i}")) for i in range(1, 5)]
        strides = {name: {"stride": list(module.stride), "weight_shape": list(module.weight.shape), "weight_dtype": str(module.weight.dtype), "input_layout": "NCHW", "weight_layout": "OIHW"} for name, module in backbone.named_modules() if isinstance(module, torch.nn.Conv2d)}
        if blocks != [3, 4, 6, 3] or any(strides[f"layer{i}.0.conv1"]["stride"] != [1, 1] or strides[f"layer{i}.0.conv2"]["stride"] != [2, 2] for i in (2, 3, 4)):
            raise AssertionError("Expected ResNet50 v1.5 stage structure and stride in the 3x3 convolution")
        if any(value.dtype != torch.float32 for value in model.parameters()):
            raise AssertionError("Expected original full FP32 model parameters")
        state_hash = loader._state_dict_sha256(backbone)
        report["model"] = {"architecture": "ResNet50 v1.5", "state_dict_sha256": state_hash, "parameter_count": sum(value.numel() for value in backbone.parameters()), "module_count": sum(1 for _ in backbone.modules()), "stage_blocks": blocks, "conv_modules": strides, "loader_provenance": model.session_provenance, "loader_environment": isolated, "input_layout": "NCHW", "parameter_perturbation": False}
        expected_outputs = []
        for index, image in enumerate(images):
            stage(f"torch_execute_image_{index}")
            with torch.no_grad():
                expected = model(image).numpy()
            if expected.dtype != np.float32 or expected.shape != (1, 1000) or not np.isfinite(expected).all():
                raise AssertionError("Expected finite FP32 classifier logits [1,1000]")
            expected_outputs.append(expected)
            np.savez(allowed_path(output / f"image_{index}.npz"), image=image.numpy(), torch=expected)
        stage("torch_legacy_onnx_export")
        graph_path = allowed_path(output / "model.onnx")
        torch.onnx.export(model, (images[0],), str(graph_path), input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False, do_constant_folding=True)
        report["onnx_graph"] = file_identity(graph_path, allowed_path)
        stage("onnx_check_and_shape_inference")
        graph = onnx.load(str(graph_path))
        onnx.checker.check_model(graph, full_check=True)
        graph = onnx.shape_inference.infer_shapes(graph, check_type=True, strict_mode=True, data_prop=True)
        onnx.checker.check_model(graph, full_check=True)
        inferred_path = allowed_path(output / "inferred.onnx")
        onnx.save(graph, str(inferred_path))
        report["inferred_onnx_graph"] = file_identity(inferred_path, allowed_path)
        inventory = onnx_inventory(graph, onnx)
        write_json("onnx_inventory.json", inventory)
        report["onnx_operator_counts"] = inventory["operator_counts"]
        stage("relax_import_frozen_parameters")
        mod = from_onnx(graph, shape_dict={"image": [1, 3, 224, 224]}, keep_params_in_input=False)
        ir_path = allowed_path(output / "imported_relax.py")
        ir_path.write_text(mod.script(), encoding="utf-8")
        report["relax_ir"] = file_identity(ir_path, allowed_path)
        inventory = relax_inventory(mod, tvm)
        write_json("relax_inventory.json", inventory)
        report["relax_operator_counts"] = inventory["operator_counts"]
        stage("relax_build_llvm")
        executable = tvm.relax.build(mod, target="llvm")
        vm = tvm.relax.VirtualMachine(executable, tvm.cpu())
        for index, image in enumerate(images):
            stage(f"relax_execute_image_{index}")
            result = vm["main"](tvm.nd.array(image.numpy(), tvm.cpu()))
            actual = result.numpy() if hasattr(result, "numpy") else result[0].numpy()
            expected = expected_outputs[index]
            np.savez(allowed_path(output / f"image_{index}.npz"), image=image.numpy(), torch=expected, relax=actual)
            if not np.isfinite(actual).all():
                raise AssertionError("Relax output contains nonfinite values")
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise AssertionError(f"Output mismatch: {actual.shape}/{actual.dtype} versus {expected.shape}/{expected.dtype}")
            delta = np.abs(actual - expected)
            comparison = {"max_absolute_error": float(delta.max()), "max_relative_error": float((delta / np.maximum(np.abs(expected), ATOL)).max()), "passed": bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False))}
            report["images"].append({"index": index, "input_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(), "torch_output": helper.tensors([expected])[0], "relax_output": helper.tensors([actual])[0], "finite": True, "relax_vs_torch": comparison})
            write_json("results.json", report)
        if loader._state_dict_sha256(backbone) != state_hash:
            raise AssertionError("Model parameters or buffers changed during verification")
        if not all(record["relax_vs_torch"]["passed"] for record in report["images"]):
            raise AssertionError("Relax/PyTorch output mismatch at fixed thresholds")
        report["status"] = "passed"
        stage("complete")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_json("results.json", report)
        print(f"failed at {report['stage']}: {error}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
