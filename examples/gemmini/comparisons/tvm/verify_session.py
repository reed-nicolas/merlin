#!/usr/bin/env python3
"""Verify a supplied model2MLIR semantic session through ONNX and Relax LLVM on CPU.

The selected local factory owns model and workload choices. This verifier follows
the existing ExternalRuntimeSession protocol; it never selects a paper checkpoint,
casts weights, quantizes, downloads models, or qualifies device timing/model quality.
"""
import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from unittest import mock

from verify_resnet import file_identity, prepare_graph, relax_inventory, safe_path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_protocol(root):
    """Load the actual pure protocol without the unrelated MLIR capture initializer."""
    path = safe_path(root / "m2m/capture/external_runtime.py")
    name = "m2m.capture.external_runtime"
    if name in sys.modules:
        module = sys.modules[name]
        if safe_path(module.__file__) != path:
            raise ValueError("A different model2MLIR session protocol is already loaded")
        return module
    return load_module(name, path)


@contextmanager
def offline():
    settings = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    def reject(*args, **kwargs):
        raise RuntimeError("Session verification requires local artifacts; network connections are disabled")
    with mock.patch.dict(os.environ, settings), mock.patch.object(socket.socket, "connect", reject), \
            mock.patch.object(socket.socket, "connect_ex", reject), mock.patch.object(socket, "create_connection", reject):
        yield


def bfloat16_dtype():
    try:
        import ml_dtypes
    except ImportError as error:
        raise ValueError("Original BF16 verification requires the optional ml_dtypes dependency") from error
    return ml_dtypes.bfloat16


def array(value, np, torch, *, require_finite=True):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Session tensors must be dense CPU tensors")
        if value.dtype == torch.bfloat16:
            value = value.detach().contiguous().view(torch.uint16).numpy().view(bfloat16_dtype())
        else:
            try:
                value = value.detach().numpy()
            except TypeError as error:
                raise ValueError("Unsupported tensor dtype; implicit precision conversion is forbidden") from error
    elif hasattr(value, "numpy"):
        dtype = str(value.dtype)
        value = value.numpy()
        if dtype == "bfloat16":
            if value.dtype != np.uint16:
                raise ValueError("Expected raw uint16 storage from TVM's BF16 NumPy interface")
            value = value.view(bfloat16_dtype())
    value = np.asarray(value)
    if value.dtype.name not in ("bfloat16", "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"):
        raise ValueError("Unsupported session dtype: " + str(value.dtype))
    if any(extent <= 0 for extent in value.shape) or require_finite and not np.isfinite(value).all():
        raise ValueError("Session tensors require positive extents and finite values")
    return np.array(value, copy=True, order="C")


def torch_array(value, np, torch):
    if value.dtype.name == "bfloat16":
        return torch.from_numpy(value.copy().view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(value.copy())


def tensor_identity(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": int(value.nbytes),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def onnx_tensors(message, onnx):
    if isinstance(message, onnx.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            for child in value if field.label == field.LABEL_REPEATED else (value,):
                yield from onnx_tensors(child, onnx)


def load_onnx_graph(path, onnx):
    """Validate and bind external tensor files before ONNX opens them."""
    path = safe_path(path)
    identities = [file_identity(path, safe_path)]
    graph = onnx.load(str(path), load_external_data=False)
    external = {}
    tensors = list(onnx_tensors(graph, onnx))
    for tensor in tensors:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {entry.key: entry.value for entry in tensor.external_data}
        if len(fields) != len(tensor.external_data) or set(fields) - {"location", "offset", "length", "checksum"}:
            raise ValueError("Unsupported or duplicate ONNX external-data metadata")
        location = Path(fields.get("location", ""))
        if not fields.get("location") or location.is_absolute() or ".." in location.parts:
            raise ValueError("ONNX external weights must stay within their stage directory")
        artifact = safe_path(path.parent / location)
        if not artifact.is_relative_to(path.parent) or not artifact.is_file():
            raise ValueError("ONNX external weights must be local regular files")
        if artifact not in external:
            external[artifact] = file_identity(artifact, safe_path)
        offset, length = fields.get("offset", "0"), fields.get("length")
        if not offset.isdecimal() or length is not None and not length.isdecimal():
            raise ValueError("ONNX external-data ranges must be nonnegative integers")
        size = artifact.stat().st_size
        if int(offset) > size or length is not None and (int(length) < 1 or int(offset) + int(length) > size):
            raise ValueError("ONNX external-data range exceeds its weight file")
    # Path checking supports >2-GiB external-weight models without serializing them.
    onnx.checker.check_model(str(path), full_check=True)
    for tensor in tensors:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            onnx.external_data_helper.load_external_data_for_tensor(tensor, str(path.parent))
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
    identities.extend(external.values())
    for identity in identities:
        if file_identity(identity["path"], safe_path) != identity:
            raise AssertionError("ONNX model or external weights changed while loading")
    return graph, identities


def normalize(value, np, torch):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return tensor_identity(array(value, np, torch))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Session metadata keys must be strings")
        return {key: normalize(item, np, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(item, np, torch) for item in value]
    raise ValueError("Unsupported session metadata type: " + type(value).__name__)


def parameter_digest(module, torch):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Stage parameters must be dense CPU tensors")
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def endpoint(value):
    return value.program, value.input_index


def outputs(value, np, torch, *, require_finite=True):
    if isinstance(value, torch.Tensor) or hasattr(value, "numpy"):
        return [array(value, np, torch, require_finite=require_finite)]
    if isinstance(value, (tuple, list)) or hasattr(value, "__getitem__"):
        result = list(value)
        if result and all(isinstance(item, torch.Tensor) or hasattr(item, "numpy") for item in result):
            return [array(item, np, torch, require_finite=require_finite) for item in result]
    raise ValueError("Stages must return a tensor or flat nonempty tensor tuple")


class SessionComparisonError(AssertionError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def compare(actual, expected, np, rtol, atol, *, context=None):
    def summary(value):
        finite = bool(np.isfinite(value).all())
        return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": finite,
                "identity": tensor_identity(value) if finite else None}

    diagnostics = {**(context or {}), "actual": summary(actual), "reference": summary(expected),
                   "rtol": rtol if np.isfinite(rtol) else str(rtol), "atol": atol if np.isfinite(atol) else str(atol),
                   "max_absolute_error": None, "mismatch_count": None}
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not diagnostics["actual"]["finite"] or not diagnostics["reference"]["finite"]:
        diagnostics["reason"] = "shape/dtype/finite-value mismatch"
        raise SessionComparisonError("Session tensor shape/dtype/finite-value mismatch", diagnostics)
    floating = np.issubdtype(actual.dtype, np.floating) or actual.dtype.name == "bfloat16"
    if floating:
        with np.errstate(over="ignore", invalid="ignore"):
            left, right = actual.astype(np.float64), expected.astype(np.float64)
            matched = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=False)
            error = float(np.max(np.abs(left - right))) if actual.size else 0.0
        diagnostics["max_absolute_error"] = error if np.isfinite(error) else None
        if not np.isfinite(error):
            diagnostics["metric_unavailable"] = "absolute difference exceeds float64 range"
    else:
        matched = actual == expected
        # Python integers preserve differences beyond float64's exact integer range.
        diagnostics["max_absolute_error"] = max((abs(int(a) - int(b)) for a, b in zip(actual[~matched], expected[~matched])), default=0)
    diagnostics["mismatch_count"] = int(np.count_nonzero(~matched))
    if diagnostics["mismatch_count"]:
        diagnostics["reason"] = "numerical comparison failed"
        raise SessionComparisonError("Session numerical comparison failed", diagnostics)
    return {"passed": True, "max_absolute_error": diagnostics["max_absolute_error"], "actual": diagnostics["actual"]["identity"], "reference": diagnostics["reference"]["identity"]}


def describe_session(session, np, torch):
    phases = {"once_before_observations": 0, "per_observation": 1, "once_after_observations": 2}
    order = [phases.get(invocation.cadence, -1) for invocation in session.execution_schedule]
    if session.reset != "restore_initial_inputs" or -1 in order or order != sorted(order):
        raise ValueError("Unsupported session reset or interleaved/unknown cadence")
    names = [program.name for program in session.programs]
    scheduled = [invocation.program for invocation in session.execution_schedule]
    if len(set(scheduled)) != len(scheduled) or set(scheduled) != set(names):
        raise ValueError("Schedule must declare each program exactly once")
    initial = {endpoint(binding.target): array(binding.initial, np, torch) for binding in session.input_bindings}
    kinds = {endpoint(binding.target): binding.kind for binding in session.input_bindings}
    streams = {endpoint(stream.target) for stream in session.streams}
    if len(streams) != len(session.streams):
        raise ValueError("Duplicate session stream target")
    cadences = {item.program: item.cadence for item in session.execution_schedule}
    if any(cadences[stream.target.program] != "per_observation" for stream in session.streams):
        raise ValueError("Streams require a per-observation stage")
    assigned = set()
    for route in session.routes:
        key = route.source.program, endpoint(route.target)
        if route.update != "after_source" or key in assigned or kinds[endpoint(route.target)] != "state":
            raise ValueError("Unsupported or ambiguous state route")
        assigned.add(key)
    for binding in session.input_bindings:
        if binding.kind not in ("static", "stream", "state"):
            raise ValueError("Unsupported input binding kind")
        if (endpoint(binding.target) in streams) != (binding.kind == "stream"):
            raise ValueError("Stream binding kind differs from declared stream")
    for program in session.programs:
        if program.module.training or any(module.training for module in program.module.modules()):
            raise ValueError("Session stages must already be in evaluation mode")
        for index, value in enumerate(program.inputs):
            expected = array(value, np, torch)
            actual = initial[(program.name, index)]
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise ValueError("Initial binding differs from stage input ABI")
    record = {"version": session.version, "kind": session.kind, "observations": session.observations, "reset": session.reset,
              "programs": [{"name": program.name, "class": type(program.module).__module__ + "." + type(program.module).__qualname__,
                            "steps": program.steps} for program in session.programs],
              "metadata": normalize(session.metadata, np, torch), "schedule": [asdict(item) for item in session.execution_schedule],
              "routes": [asdict(item) for item in session.routes], "observation_output": asdict(session.observation_output),
              "final_output": asdict(session.final_output),
              "inputs": [{"target": asdict(item.target), "kind": item.kind, "value": tensor_identity(initial[endpoint(item.target)])}
                         for item in session.input_bindings],
              "streams": [{"target": asdict(item.target), "values": tensor_identity(array(item.values, np, torch))} for item in session.streams]}
    record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return record, initial


def verify_session(session, compile_stage, np, torch, *, repeats=2, rtol=1e-4, atol=1e-4, max_signatures=64):
    if not 2 <= repeats <= 10 or not 1 <= max_signatures <= 128:
        raise ValueError("Invalid session repeat/signature budget")
    description, initial = describe_session(session, np, torch)
    programs = {program.name: program for program in session.programs}
    parameter_hashes = {name: parameter_digest(program.module, torch) for name, program in programs.items()}
    compiled, checks, previous = {}, [], None
    high_water = 0
    for repeat in range(repeats):
        reference = {key: value.copy() for key, value in initial.items()}
        candidate = {key: value.copy() for key, value in initial.items()}
        last, fingerprints = {}, []

        def invoke(invocation, observation):
            nonlocal high_water
            program = programs[invocation.program]
            if observation is not None:
                for stream in session.streams:
                    if stream.target.program == program.name:
                        value = array(stream.values[observation], np, torch)
                        reference[endpoint(stream.target)] = value.copy()
                        candidate[endpoint(stream.target)] = value.copy()
            keys = [(program.name, index) for index in range(len(program.inputs))]
            expected_args, actual_args = [reference[key] for key in keys], [candidate[key] for key in keys]
            context = {"stage": invocation.cadence, "program": program.name, "repeat": repeat, "observation": observation}
            for index, (actual, expected) in enumerate(zip(actual_args, expected_args)):
                compare(actual, expected, np, rtol, atol, context={**context, "tensor": "input", "input_index": index})
            signature = tuple((str(value.dtype), tuple(value.shape)) for value in actual_args)
            key = program.name, signature
            tensors = [torch_array(value, np, torch) for value in expected_args]
            with torch.no_grad():
                expected = outputs(program.module(*tensors), np, torch)
            if any(not np.array_equal(array(value, np, torch), original) for value, original in zip(tensors, expected_args)):
                raise AssertionError("Reference stage mutated caller inputs")
            if key not in compiled:
                if len(compiled) >= max_signatures:
                    raise ValueError("Session exceeds the explicit compilation-signature budget")
                compiled[key] = compile_stage(program, tuple(actual_args), len(expected))
            copied = [value.copy() for value in actual_args]
            actual = outputs(compiled[key](tuple(copied)), np, torch, require_finite=False)
            if any(not np.array_equal(value, original) for value, original in zip(copied, actual_args)):
                raise AssertionError("Compiled stage mutated caller inputs")
            if len(actual) != len(expected):
                raise AssertionError("Stage output count differs from reference")
            values = [compare(a, b, np, rtol, atol, context={**context, "tensor": "output", "output_index": index})
                      for index, (a, b) in enumerate(zip(actual, expected))]
            last[program.name] = actual
            routes = []
            for route in session.routes:
                if route.source.program != program.name:
                    continue
                index, target = route.source.output_index, endpoint(route.target)
                if not 0 <= index < len(actual):
                    raise ValueError("State route selects an absent output")
                value = actual[index]
                if value.dtype != initial[target].dtype or value.ndim != initial[target].ndim:
                    raise ValueError("State route changes the declared dtype or rank")
                reference[target], candidate[target] = expected[index].copy(), value.copy()
                routes.append({"name": route.name, "target": asdict(route.target), "comparison": compare(candidate[target], reference[target], np, rtol, atol)})
            high_water = max(high_water, sum(value.nbytes for value in candidate.values()))
            fingerprints.append([tensor_identity(value) for value in actual])
            checks.append({"repeat": repeat, "observation": observation, "program": program.name, "outputs": values, "routed_states": routes})

        def phase(cadence, observation=None):
            for invocation in session.execution_schedule:
                if invocation.cadence == cadence:
                    count = 1 if cadence == "per_observation" else invocation.repeats
                    for _ in range(count):
                        invoke(invocation, observation)

        phase("once_before_observations")
        observations = []
        for observation in range(session.observations):
            phase("per_observation", observation)
            selector = session.observation_output
            if selector.program not in last or not 0 <= selector.output_index < len(last[selector.program]):
                raise ValueError("Observation selector is unavailable at its declared cadence")
            observations.append(tensor_identity(last[selector.program][selector.output_index]))
        phase("once_after_observations")
        selector = session.final_output
        if selector.program not in last or not 0 <= selector.output_index < len(last[selector.program]):
            raise ValueError("Final output selector is unavailable")
        fingerprints.append([tensor_identity(last[selector.program][selector.output_index]), observations])
        if previous is not None and fingerprints != previous:
            raise AssertionError("Session outputs differ after restoring initial inputs")
        previous = fingerprints
    for name, program in programs.items():
        if parameter_digest(program.module, torch) != parameter_hashes[name]:
            raise AssertionError("Stage parameters or registered buffers changed")
    return {"session": description, "checks": checks, "repeats": repeats, "reset_verified": True,
            "compilation_signatures": len(compiled), "parameter_sha256": parameter_hashes, "retained_stage_input_high_water_bytes": high_water}


class StageCompiler:
    def __init__(self, out, mode, np, torch, onnx, tvm, exporter="legacy", opset=None, bf16_fused_export=False):
        if exporter not in ("legacy", "dynamo"):
            raise ValueError("Unsupported ONNX exporter selection")
        if opset is not None and (type(opset) is not int or not 1 <= opset <= onnx.defs.onnx_opset_version()):
            raise ValueError("ONNX opset must be an integer supported by the selected ONNX dependency")
        self.out, self.mode, self.np, self.torch, self.onnx, self.tvm = out, mode, np, torch, onnx, tvm
        self.exporter = exporter
        self.opset = opset if opset is not None else (18 if exporter == "dynamo" else 17)
        if type(bf16_fused_export) is not bool or bf16_fused_export and (exporter != "dynamo" or self.opset < 18):
            raise ValueError("Fused BF16 export requires explicit dynamo selection and opset18 or newer")
        self.translation_source, self.export_options = None, {}
        if bf16_fused_export:
            path = safe_path(Path(__file__).with_name("bf16_export.py"))
            self.translation_source = file_identity(path, safe_path)
            helper = load_module("tvm_session_bf16_export", path)
            self.export_options["custom_translation_table"] = helper.translations(torch)
            # Preserve original initializer dtypes, including BF16 bias storage.
            self.export_options["optimize"] = False
        self.records = []

    def __call__(self, program, inputs, output_count):
        folder = self.out / ("stage_" + str(len(self.records)))
        folder.mkdir()
        graph_path = folder / "model.onnx"
        names = ["input_" + str(index) for index in range(len(inputs))]
        tensors = tuple(torch_array(value, self.np, self.torch) for value in inputs)
        # Tracing mutates plain Python cache attributes into FakeTensor objects.
        # Export an isolated stage so subsequent source calls and resets remain real.
        export_module = copy.deepcopy(program.module)
        exported = self.torch.onnx.export(export_module, tensors, str(graph_path), input_names=names,
                                         output_names=["output_" + str(index) for index in range(output_count)],
                                         opset_version=self.opset, dynamo=self.exporter == "dynamo", external_data=True, do_constant_folding=True, **self.export_options)
        del export_module
        graph, artifacts = load_onnx_graph(graph_path, self.onnx)
        from tvm.relax.frontend.onnx import from_onnx
        mod = from_onnx(graph, shape_dict={name: list(value.shape) for name, value in zip(names, inputs)}, keep_params_in_input=False)
        transformed, lowered = prepare_graph(mod, self.mode, self.tvm)
        record = {"program": program.name, "inputs": [tensor_identity(value) for value in inputs], "onnx": artifacts[0], "onnx_artifacts": artifacts,
                  "exporter": self.exporter, "opset": self.opset, "export_source_isolation": "deepcopy",
                  "pipelines": ["zero", "default_build"] if self.mode == "optimized" else ["default_build"]}
        if self.translation_source:
            if file_identity(self.translation_source["path"], safe_path) != self.translation_source:
                raise AssertionError("Selected BF16 export translations changed")
            record["translation_source"] = self.translation_source
            record["bf16_fused_arithmetic"] = "original BF16 operands/results; linear and SDPA FP32-math candidates, not native CPU kernel/layout reproductions; unchanged source reference gate"
            record["bf16_attention_profile"] = "fp32-math"
            record["onnx_export_optimization"] = False
            targets = self.export_options["custom_translation_table"]
            record["translated_aten_operations"] = {str(target): sum(node.op == "call_function" and node.target == target for node in exported.exported_program.graph.nodes) for target in targets}
        for name, value in (("imported", mod), ("graph", transformed), ("vm", lowered)):
            path = folder / (name + ".relax.py")
            path.write_text(value.script())
            record[name] = {"ir": file_identity(path, safe_path), "inventory": relax_inventory(value, self.tvm)}
        vm = self.tvm.relax.VirtualMachine(self.tvm.relax.build(lowered, "llvm", pipeline=None), self.tvm.cpu())
        self.records.append(record)

        def run(values):
            arguments = [self.tvm.nd.array(value.copy(), self.tvm.cpu()) for value in values]
            result = vm["main"](*arguments)
            if any(not self.np.array_equal(array(value, self.np, self.torch), before) for value, before in zip(arguments, values)):
                raise AssertionError("TVM stage mutated caller inputs")
            return result
        return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "tvm-source", "tvm-build", "factory", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--factory-name", default="get_model_and_inputs")
    parser.add_argument("--factory-arguments", type=Path, help="Local JSON keyword arguments passed unchanged to the selected factory")
    parser.add_argument("--artifact", type=Path, action="append", default=[], help="Explicit local checkpoint/input/config artifacts to hash")
    parser.add_argument("--graph-mode", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--onnx-exporter", choices=("legacy", "dynamo"), default="legacy", help="Explicit modern exporter for compatible source models; dynamo defaults to opset18 and needs onnxscript")
    parser.add_argument("--opset", type=int, help="Explicit ONNX schema version; defaults to17 legacy or18 dynamo; native BF16 Conv requires22")
    parser.add_argument("--bf16-fused-export", action="store_true", help="Opt-in dynamo BF16 linear/SDPA FP32-math candidates; original BF16 ABI, unchanged source comparison, no native CPU kernel/layout equivalence claim")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-signatures", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 10 or not 1 <= args.max_signatures <= 128 or not 0 <= args.rtol <= 0.1 or not 0 <= args.atol <= 0.1:
        parser.error("Invalid repeat/signature budget or numerical tolerances")
    sys.dont_write_bytecode = True
    out = safe_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "invocation": [sys.executable, *sys.argv], "host_session_verified": False,
              "model_quality_qualified": False, "paper_validation": False, "timing_qualified": False, "device_execution": False}
    try:
        report["verifier_source"] = file_identity(__file__, safe_path)
        report["graph_helper_source"] = file_identity(prepare_graph.__code__.co_filename, safe_path)
        source, build, model_root = safe_path(args.tvm_source), safe_path(args.tvm_build), safe_path(args.model2mlir_root)
        os.environ.update(TVM_LIBRARY_PATH=str(build), TVM_FFI="ctypes")
        sys.path[:0] = [str(safe_path(source / "python")), str(model_root)]
        import numpy as np
        import torch
        import onnx
        import tvm
        from tvm._ffi.base import _LIB
        protocol = load_protocol(model_root)
        if safe_path(protocol.__file__) != model_root / "m2m/capture/external_runtime.py":
            raise ValueError("Imported session protocol differs from selected model2MLIR source")
        if not safe_path(tvm.__file__).is_relative_to(source / "python") or safe_path(_LIB._name).parent != build:
            raise ValueError("Imported TVM source/library differs from the explicit selection")
        report["protocol_source"] = file_identity(protocol.__file__, safe_path)
        report["compiler_library"] = file_identity(_LIB._name, safe_path)
        report["versions"] = {"python": sys.version, "torch": torch.__version__, "onnx": onnx.__version__, "numpy": np.__version__, "tvm": tvm.__version__}
        report["build_info"] = dict(tvm.support.libinfo())
        report["graph_mode"], report["rtol"], report["atol"] = args.graph_mode, args.rtol, args.atol
        report["factory_source"] = file_identity(args.factory, safe_path)
        report["factory_name"] = args.factory_name
        report["factory_environment"] = {name: value for name, value in os.environ.items() if name.startswith("M2M_")}
        report["declared_artifacts"] = [file_identity(path, safe_path) for path in args.artifact]
        arguments = {}
        if args.factory_arguments:
            report["factory_arguments"] = file_identity(args.factory_arguments, safe_path)
            arguments = json.loads(safe_path(args.factory_arguments).read_text())
            if not isinstance(arguments, dict):
                raise ValueError("Factory arguments must be a JSON object")
        with offline():
            factory = load_module("tvm_session_factory", args.factory)
            model, inputs = getattr(factory, args.factory_name)(**arguments)
            artifacts = factory.get_verification_artifacts(model, inputs) if callable(getattr(factory, "get_verification_artifacts", None)) else ()
            report["factory_artifacts"] = [file_identity(path, safe_path) for path in artifacts]
            metadata = factory.get_session_spec(model, inputs) if callable(getattr(factory, "get_session_spec", None)) else None
            session = protocol.external_runtime_session(model, tuple(inputs), session=metadata)
            report["session"] = describe_session(session, np, torch)[0]
            report["onnx_exporter"] = args.onnx_exporter
            compiler = StageCompiler(out, args.graph_mode, np, torch, onnx, tvm, exporter=args.onnx_exporter, opset=args.opset, bf16_fused_export=args.bf16_fused_export)
            report["bf16_fused_export"] = args.bf16_fused_export
            if compiler.translation_source:
                report["translation_source"] = compiler.translation_source
            report["compiled_stages"] = compiler.records
            report.update(verify_session(session, compiler, np, torch, repeats=args.repeats, rtol=args.rtol, atol=args.atol, max_signatures=args.max_signatures))
        infos = [report["verifier_source"], report["graph_helper_source"], report["protocol_source"], report["factory_source"],
                 report["compiler_library"], *report["declared_artifacts"], *report["factory_artifacts"]]
        if "factory_arguments" in report:
            infos.append(report["factory_arguments"])
        if "translation_source" in report:
            infos.append(report["translation_source"])
        infos.extend(artifact for stage in report["compiled_stages"] for artifact in stage["onnx_artifacts"])
        for info in infos:
            if file_identity(info["path"], safe_path) != info:
                raise AssertionError("Selected source or declared artifact changed")
        report["limitations"] = ["Host frontend/session routing only; no selected paper checkpoint, application-quality or device/timing qualification",
                                 "Explicit artifact/source hashes are not a complete runtime source closure",
                                 "Python network connections and HuggingFace downloads disabled; this is not a hermetic process sandbox",
                                 "Input-state high-water describes diagnostic retained arrays, not deployed allocator peak or model fit"]
        report.update(status="passed", host_session_verified=True)
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if isinstance(error, SessionComparisonError):
            report["failure_diagnostics"] = error.diagnostics
    finally:
        (out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(out)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
