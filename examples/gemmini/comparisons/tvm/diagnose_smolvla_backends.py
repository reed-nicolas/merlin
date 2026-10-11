#!/usr/bin/env python3
"""Characterize complete native SmolVLA CPU attention sensitivity; no qualification."""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import smolvla_session as factory
from verify_resnet import ATOL, RTOL
from verify_session import offline


INPUT_NAMES = ("images", "image_masks", "language_tokens", "language_mask", "state", "noise")
SCOPES = ("smolvla.prefix_embed", "smolvla.prefix_cache", "smolvla.denoise")


def dense_flat_copy(value, torch):
    """Copy bytes without retaining an expanded singleton's zero last stride."""
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError("Diagnostic tensors must be dense CPU tensors")
    flat = torch.empty(value.numel(), dtype=value.dtype, device="cpu")
    flat.copy_(value.detach().reshape(-1))
    return flat


def tensor_bytes(value, torch):
    return dense_flat_copy(value, torch).view(torch.uint8).numpy().tobytes()


def tensor_identity(value, torch):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(tensor_bytes(value, torch)).hexdigest()}


def state_identity(model, torch):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(tensor_bytes(value, torch))
    return digest.hexdigest()


def backend_flags(torch):
    return {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "math_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
    }


def statistics(default, alternate, torch):
    if default.shape != alternate.shape or default.dtype != alternate.dtype:
        raise ValueError("Diagnostic comparison shape/dtype mismatch")
    if not torch.isfinite(default).all() or not torch.isfinite(alternate).all():
        raise ValueError("Diagnostic capture contains nonfinite values")
    left, right = default.double(), alternate.double()
    delta = (left - right).abs()
    scale = torch.maximum(left.abs(), right.abs())
    relative = delta[scale > 0] / scale[scale > 0]
    norm = torch.linalg.vector_norm(left)
    result = {
        "shape": list(default.shape), "dtype": str(default.dtype), "elements": default.numel(),
        "all_finite": True, "exactly_unequal_count": int((default != alternate).sum()),
        "max_absolute_error": float(delta.max()), "mean_absolute_error": float(delta.mean()),
        "rms_error": float(torch.sqrt((delta * delta).mean())),
        "relative_l2_error": float(torch.linalg.vector_norm(left - right) / norm) if norm > 0 else None,
        "symmetric_relative_error_max_nonzero": float(relative.max()) if relative.numel() else 0.0,
        "symmetric_relative_error_mean_nonzero": float(relative.mean()) if relative.numel() else 0.0,
        "relative_denominator": "max(abs(default),abs(math)); both-zero omitted",
        "outside_existing_pointwise_comparison_count": int((~torch.isclose(right, left, rtol=RTOL, atol=ATOL)).sum()),
    }
    if default.dtype == torch.bfloat16:
        def rank(value):
            bits = dense_flat_copy(value, torch).view(torch.int16).to(torch.int32).bitwise_and(65535)
            magnitude = bits.bitwise_and(32767)
            return torch.where(bits.bitwise_and(32768) != 0, 32768 - magnitude, 32768 + magnitude)
        distances = (rank(default) - rank(alternate)).abs().flatten().double()
        result["bf16_representable_step_distance"] = {
            "max": int(distances.max()), "mean": float(distances.mean()),
            "p50": float(torch.quantile(distances, 0.5)), "p95": float(torch.quantile(distances, 0.95)),
            "p99": float(torch.quantile(distances, 0.99)),
            "definition": "Monotone finite BF16 ranks; signed zeros unified. Sign crossings count steps through zero.",
        }
    return result


def cache_identity(cache, layers, torch):
    return {f"{index}:{name}": tensor_identity(cache[index][name], torch) for index in range(layers) for name in ("key_states", "value_states")}


def compare_identical_state_velocities(reference_velocity, candidate_velocity, captures, torch,
                                       *, expected_steps, candidate_prefix=None):
    """Replay every source state; separate local arithmetic from prefix conditioning.

    Callables receive the source denoise_step keyword ABI. Each invocation owns
    fresh copies, including the original BF16 cache. This is a diagnostic, not a
    replacement tolerance or a rollout of the candidate's own state trajectory.
    """
    indices = sorted(int(name.split("_")[1]) for name in captures
                     if name.startswith("step_") and name.endswith("_input_state"))
    if type(expected_steps) is not int or expected_steps <= 0 or indices != list(range(expected_steps)):
        raise ValueError("Velocity replay requires a complete contiguous source trajectory")
    prefix = (captures["prefix_pad_mask"], captures["prefix_key_cache"], captures["prefix_value_cache"])
    alternate = prefix if candidate_prefix is None else candidate_prefix
    if len(alternate) != 3 or any(left.shape != right.shape or left.dtype != right.dtype
                                  for left, right in zip(prefix, alternate, strict=True)):
        raise ValueError("Candidate prefix must preserve source mask/cache shapes and dtypes")
    if not torch.equal(prefix[0], alternate[0]):
        raise ValueError("Candidate prefix routing mask differs")
    if any(not torch.isfinite(value).all() for value in (*prefix, *alternate)):
        raise ValueError("Velocity replay prefix contains nonfinite values")
    bindings = {name: tensor_identity(value, torch) for name, value in captures.items()}
    alternate_bindings = [tensor_identity(value, torch) for value in alternate]
    original_rng = torch.random.get_rng_state().clone()
    outputs, rows = {}, []

    def invoke(callback, selected, state, timestep):
        mask, keys, values = (value.detach().clone() for value in selected)
        cache = {index: {"key_states": keys[index].clone(), "value_states": values[index].clone()}
                 for index in range(keys.shape[0])}
        state, timestep = state.detach().clone(), timestep.detach().clone()
        before = {"mask": tensor_identity(mask, torch), "state": tensor_identity(state, torch),
                  "timestep": tensor_identity(timestep, torch), "cache": cache_identity(cache, keys.shape[0], torch)}
        rng = torch.random.get_rng_state().clone()
        with torch.no_grad():
            result = callback(x_t=state, prefix_pad_masks=mask, past_key_values=cache, timestep=timestep)
        after = {"mask": tensor_identity(mask, torch), "state": tensor_identity(state, torch),
                 "timestep": tensor_identity(timestep, torch), "cache": cache_identity(cache, keys.shape[0], torch)}
        if before != after or not torch.equal(rng, torch.random.get_rng_state()):
            raise AssertionError("Velocity replay mutated state/mask/cache/timestep or consumed RNG")
        if result.shape != state.shape or result.dtype != state.dtype or not torch.isfinite(result).all():
            raise ValueError("Velocity replay output shape/dtype/finiteness differs")
        return result.detach().clone(), before

    try:
        for index in indices:
            stem = f"step_{index:02d}"
            state, timestep = captures[stem + "_input_state"], captures[stem + "_timestep"]
            reference, binding = invoke(reference_velocity, prefix, state, timestep)
            if not torch.equal(reference, captures[stem + "_velocity"]):
                raise AssertionError("Reference replay differs from the captured source velocity")
            candidate, candidate_binding = invoke(candidate_velocity, alternate, state, timestep)
            outputs[stem + "_reference_velocity"] = reference
            outputs[stem + "_candidate_velocity"] = candidate
            rows.append({"step": index, "reference_inputs": binding, "candidate_inputs": candidate_binding,
                         "comparison": statistics(reference, candidate, torch)})
        if bindings != {name: tensor_identity(value, torch) for name, value in captures.items()} or \
                alternate_bindings != [tensor_identity(value, torch) for value in alternate]:
            raise AssertionError("Velocity replay mutated supplied capture/prefix tensors")
    finally:
        torch.random.set_rng_state(original_rng)
    return outputs, {
        "diagnostic_only": True, "acceptance_contract_selected": False, "steps": len(indices),
        "identical_states": True, "identical_timesteps": True,
        "identical_prefix": all(torch.equal(left, right) for left, right in zip(prefix, alternate, strict=True)),
        "reference_replay_exact": True, "inputs_cache_rng_unchanged": True, "comparisons": rows,
        "interpretation": "Local velocity comparison at source states; independent rollout is reported separately.",
    }


def profiler_counts(profiler):
    operators = {event.key: event.count for event in profiler.key_averages() if "scaled_dot_product" in event.key}
    stages = dict.fromkeys(SCOPES, 0)
    for event in profiler.events():
        if event.name != "aten::scaled_dot_product_attention":
            continue
        parent = event.cpu_parent
        while parent is not None and parent.name not in stages:
            parent = parent.cpu_parent
        if parent is None:
            raise AssertionError("Native SDPA call outside captured policy stages")
        stages[parent.name] += 1
    return {"operators": operators, "native_sdpa_calls_by_stage": stages}


def run_source_policy(model, inputs, backend, seed, repeat, output_dir, np, torch, *, profile=False):
    """Observe the unchanged source call; restore temporary hooks and RNG on failure."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if backend not in ("default", "math"):
        raise ValueError("Unknown diagnostic attention backend")
    output_dir = factory.safe_path(output_dir)
    input_bindings = {name: tensor_identity(value, torch) for name, value in zip(INPUT_NAMES, inputs, strict=True)}
    original_flags = backend_flags(torch)
    original_rng = torch.random.get_rng_state()
    originals = ((model, "embed_prefix"), (model, "denoise_step"), (model.vlm_with_expert, "forward"))
    if any(name in owner.__dict__ for owner, name in originals):
        raise ValueError("Source methods already have instance overrides")
    embed_prefix, denoise_step, forward = model.embed_prefix, model.denoise_step, model.vlm_with_expert.forward
    captures, cache_checks = {}, []
    initial_cache = None
    step = 0

    def embed(*args, **kwargs):
        with torch.profiler.record_function(SCOPES[0]):
            value = embed_prefix(*args, **kwargs)
        for name, tensor in zip(("prefix_embeddings", "prefix_pad_mask", "prefix_attention_mask"), value, strict=True):
            captures[name] = tensor.detach().clone()
        return value

    def encode(*args, **kwargs):
        nonlocal initial_cache
        with torch.profiler.record_function(SCOPES[1]) if kwargs.get("fill_kv_cache") else nullcontext():
            value = forward(*args, **kwargs)
        if kwargs.get("fill_kv_cache"):
            cache = value[1]
            initial_cache = cache_identity(cache, model.config.num_vlm_layers, torch)
            for name in ("key_states", "value_states"):
                captures["prefix_" + name.replace("states", "cache")] = torch.stack([cache[index][name] for index in range(model.config.num_vlm_layers)]).detach().clone()
        return value

    def denoise(*args, **kwargs):
        nonlocal step
        before = cache_identity(kwargs["past_key_values"], model.config.num_vlm_layers, torch)
        if before != initial_cache:
            raise AssertionError("Source prefix cache changed between denoising calls")
        captures[f"step_{step:02d}_input_state"] = kwargs["x_t"].detach().clone()
        captures[f"step_{step:02d}_timestep"] = kwargs["timestep"].detach().clone()
        with torch.profiler.record_function(SCOPES[2]):
            value = denoise_step(*args, **kwargs)
        after = cache_identity(kwargs["past_key_values"], model.config.num_vlm_layers, torch)
        if before != after:
            raise AssertionError("Source denoising mutated the prefix cache")
        cache_checks.append({"step": step, "before": before, "after": after})
        captures[f"step_{step:02d}_velocity"] = value.detach().clone()
        step += 1
        return value

    started = time.monotonic()
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if profile else nullcontext()
    model.embed_prefix, model.denoise_step, model.vlm_with_expert.forward = embed, denoise, encode
    try:
        torch.manual_seed(seed)
        rng_before = torch.random.get_rng_state().clone()
        with sdpa_kernel(SDPBackend.MATH) if backend == "math" else nullcontext():
            active_flags = backend_flags(torch)
            with torch.no_grad(), profiler:
                cameras = range(inputs[0].shape[1])
                full = model.sample_actions([inputs[0][:, index] for index in cameras], [inputs[1][:, index] for index in cameras], *inputs[2:5], noise=inputs[5])
        if backend_flags(torch) != original_flags:
            raise AssertionError("SDPA context did not restore backend flags")
        if not torch.equal(rng_before, torch.random.get_rng_state()):
            raise AssertionError("Source inference consumed RNG despite explicit noise")
    finally:
        for owner, name in originals:
            delattr(owner, name)
        torch.random.set_rng_state(original_rng)
    if step != model.config.num_steps or full.shape != inputs[5].shape:
        raise AssertionError("Incomplete source denoising trajectory")
    captures["final_full_state"] = full.detach().clone()
    captures["final_actions"] = full[:, :, :model.config.action_feature.shape[0]].detach().clone()
    dt = -1.0 / model.config.num_steps
    for index in range(step):
        state = captures[f"step_{index + 1:02d}_input_state"] if index < step - 1 else full
        expected_time = torch.tensor(1.0 + index * dt, dtype=torch.float32).expand(inputs[5].shape[0])
        if not torch.equal(expected_time, captures[f"step_{index:02d}_timestep"]):
            raise AssertionError("Source timestep stream differs")
        if not torch.equal(state, captures[f"step_{index:02d}_input_state"] + dt * captures[f"step_{index:02d}_velocity"]):
            raise AssertionError("Source Euler update differs")
        captures[f"step_{index:02d}_output_state"] = state.detach().clone()
    if tensor_identity(captures["step_00_input_state"], torch) != input_bindings["noise"]:
        raise AssertionError("Source initial noise binding differs")
    if input_bindings != {name: tensor_identity(value, torch) for name, value in zip(INPUT_NAMES, inputs, strict=True)}:
        raise AssertionError("Source inference mutated fixture inputs")
    if any(not torch.isfinite(value).all() for value in captures.values()):
        raise AssertionError("Nonfinite source capture")
    path = factory.safe_path(output_dir / f"seed{seed}-{backend}-repeat{repeat}.npz")
    if path.exists():
        raise FileExistsError("Refusing to overwrite diagnostic captures")
    arrays = {name: dense_flat_copy(value, torch).view(torch.uint16).reshape(value.shape).numpy() if value.dtype == torch.bfloat16 else value.numpy() for name, value in captures.items()}
    np.savez_compressed(path, **arrays)
    metadata = {
        "seed": seed, "backend": backend, "repeat": repeat, "wall_s": time.monotonic() - started,
        "active_backend_flags": active_flags, "backend_flags_before": original_flags,
        "backend_flags_after": backend_flags(torch), "input_bindings": input_bindings,
        "captures": {name: tensor_identity(value, torch) for name, value in captures.items()},
        "output_file": factory.file_identity(path, factory.safe_path),
        "storage": "BF16 as exact uint16 bits; other tensors in native NumPy dtype",
        "rng_unchanged": True, "inputs_unchanged": True, "noise_binding_exact": True,
        "euler_update_exact_all_steps": True,
        "temporary_wrappers_restored": all(name not in owner.__dict__ for owner, name in originals),
        "prefix_cache_checks": cache_checks,
    }
    if profile:
        metadata["attention_profile"] = profiler_counts(profiler)
    return captures, metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--backbone-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", required=True, type=int, nargs="+")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--fixture-kind", choices=("synthetic", "dataset"))
    parser.add_argument("--input-source")
    parser.add_argument("--input-manifest", type=Path,
                        help="Source-bound manifest from prepare_smolvla_observation.py")
    parser.add_argument("--velocity-replay", action="store_true",
                        help="Replay all source states with shared and alternate prefix caches; diagnostic only")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds):
        parser.error("Seeds must be distinct integers in [0, 2**63)")
    if not 1 <= args.cpu_threads <= 4:
        parser.error("CPU thread budget must be between 1 and 4")
    if args.input_npz is not None and (args.fixture_kind is None or not args.input_source or not args.input_source.strip()):
        parser.error("Supplied fixtures require explicit --fixture-kind and attributed --input-source")
    args.fixture_kind = args.fixture_kind or "synthetic"
    if args.fixture_kind == "dataset" and (args.input_npz is None or not args.input_source or not args.input_source.strip()):
        parser.error("Dataset fixtures require --input-npz and attributed --input-source")
    if args.input_npz is not None and len(args.seeds) != 1:
        parser.error("An explicit fixture has fixed noise; select one seed for its RNG audit")
    if args.input_manifest is not None and (args.input_npz is None or args.fixture_kind != "dataset"):
        parser.error("Recorded observation manifests require an explicit dataset NPZ")
    for name in ("checkpoint_dir", "backbone_dir", "output_dir", "input_npz", "input_manifest"):
        if getattr(args, name) is not None:
            setattr(args, name, factory.safe_path(getattr(args, name)))
    from merlin.common.paths import artifacts_dir
    if not args.output_dir.is_relative_to(factory.safe_path(artifacts_dir())):
        parser.error("Diagnostic output must stay inside the configured artifacts directory")
    if args.output_dir.exists():
        raise FileExistsError("Refusing an existing diagnostic output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    fixture_manifest = factory.observation_manifest(args.input_npz, args.input_manifest)
    config_json, checkpoint_artifacts = factory.checkpoint_artifacts(args.checkpoint_dir)
    backbone_artifacts = factory.directory_artifacts(args.backbone_dir)
    if not factory.safe_path(args.backbone_dir / "config.json").is_file() or not any(Path(item["path"]).suffix == ".safetensors" for item in backbone_artifacts):
        raise FileNotFoundError("Complete local backbone configuration and weights are required")
    if importlib.metadata.version("lerobot") != "0.5.1":
        raise ValueError("This source policy requires lerobot==0.5.1")
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from transformers.integrations import sdpa_attention

    original_threads, original_rng = torch.get_num_threads(), torch.random.get_rng_state()
    original_flags = backend_flags(torch)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = factory.safe_path(args.output_dir / "results.json")
    report = {
        "status": "running", "development_diagnostic_only": True, "model_quality_qualified": False,
        "device_execution": False, "host_session_verified": False, "active_gate_changed": False,
        "source_closure_verified": False, "backbone_revision_verified": False,
        "fixture_kind": args.fixture_kind, "synthetic_inputs": args.fixture_kind == "synthetic",
        "fixture_origin": "supplied_npz" if args.input_npz is not None else "generated_seeded",
        "input_source": args.input_source or "Seeded synthetic preprocessed tensors",
        "observation_manifest": fixture_manifest, "observation_schema_verified": fixture_manifest is not None,
        "reference": "Unchanged LeRobot sample_actions, original dtype, normalized physical action chunk",
        "intervention": "Public sdpa_kernel(SDPBackend.MATH) context",
        "existing_pointwise_comparison": {"rtol": RTOL, "atol": ATOL, "reference": "default native source policy", "acceptance_contract_selected": False},
        "geometry": factory.GEOMETRY, "checkpoint_revision": factory.MODEL_REVISION,
        "checkpoint_bindings": checkpoint_artifacts + backbone_artifacts,
        "cpu_threads": args.cpu_threads, "seeds": args.seeds, "cases": [],
    }
    started = time.monotonic()

    def persist():
        output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        torch.set_num_threads(args.cpu_threads)
        report["runtime_context"] = {"cpu_capability": torch.backends.cpu.get_cpu_capability(), "torch_config": torch.__config__.show(), "torch_parallel_info": torch.__config__.parallel_info()}
        with offline():
            config = PreTrainedConfig.from_pretrained(str(args.checkpoint_dir), local_files_only=True)
            config.device, config.vlm_model_name = "cpu", str(args.backbone_dir)
            policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint_dir), config=config, local_files_only=True, strict=True).eval().requires_grad_(False)
        model = policy.model
        if model._rtc_enabled() or config.compile_model:
            raise ValueError("Diagnostic requires the original uncompiled source policy")
        sources = {factory.safe_path(__file__), factory.safe_path(factory.__file__), factory.safe_path(torch.nn.functional.__file__), factory.safe_path(sdpa_attention.__file__)}
        sources.update(factory.safe_path(function.__code__.co_filename) for function in (factory.file_identity, factory.safe_path, offline.__wrapped__))
        sources.update(factory.safe_path(sys.modules[module.__class__.__module__].__file__) for module in model.modules() if getattr(sys.modules.get(module.__class__.__module__), "__file__", None))
        sources.update(factory.safe_path(sys.modules[name].__file__) for name in (PreTrainedConfig.__module__, config.__class__.__module__, "torch.nn.attention"))
        report["source_bindings"] = [factory.file_identity(path, factory.safe_path) for path in sorted(sources)]
        report["dependencies"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "lerobot", "numpy", "onnx", "safetensors")}
        if args.input_npz is not None:
            report["fixture_binding"] = factory.file_identity(args.input_npz, factory.safe_path)
        if args.input_manifest is not None:
            report["fixture_manifest_binding"] = factory.file_identity(args.input_manifest, factory.safe_path)
        cameras = fixture_manifest["selected_cameras"] if fixture_manifest else [
            name for name in config_json["input_features"] if name.startswith("observation.images.")]
        vision = model.vlm_with_expert.get_vlm_model().vision_model
        report["attention_attribution"] = {
            "camera_names": cameras, "vision_layers": len(vision.encoder.layers),
            "vision_attention_implementation": vision.config._attn_implementation,
            "expected_native_sdpa_calls": len(cameras) * len(vision.encoder.layers),
            "native_sdpa_stage": "prefix image embedding",
            "vlm_expert_attention": "LeRobot get_attention_interface returns manual eager_attention_forward; no native SDPA dispatch",
        }
        parameter_bytes = {}
        for value in model.state_dict().values():
            dtype = str(value.dtype)
            parameter_bytes[dtype] = parameter_bytes.get(dtype, 0) + value.numel() * value.element_size()
        report["model_state_bytes_by_dtype"] = parameter_bytes
        state_before = state_identity(model, torch)
        modes_before = {name: module.training for name, module in model.named_modules()}
        persist()
        for case_index, seed in enumerate(args.seeds):
            inputs = factory.fixture_inputs(config, len(cameras), input_npz=args.input_npz, seed=seed)
            if torch.any(inputs[2] >= model.vlm_with_expert.config.text_config.vocab_size):
                raise ValueError("Fixture token ID is outside the backbone vocabulary")
            default, left = run_source_policy(model, inputs, "default", seed, 0, args.output_dir, np, torch, profile=case_index == 0)
            math, right = run_source_policy(model, inputs, "math", seed, 0, args.output_dir, np, torch, profile=case_index == 0)
            if left["input_bindings"] != right["input_bindings"]:
                raise AssertionError("Compared policies received different observations/noise")
            case = {
                "seed": seed, "runs": [left, right],
                "comparisons": {name: statistics(default[name], math[name], torch) for name in default},
                "action_dimensions": [statistics(default["final_actions"][:, :, index], math["final_actions"][:, :, index], torch) for index in range(config.action_feature.shape[0])],
                "action_timesteps": [statistics(default["final_actions"][:, index, :], math["final_actions"][:, index, :], torch) for index in range(config.chunk_size)],
            }
            if args.velocity_replay:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                def math_velocity(**kwargs):
                    with sdpa_kernel(SDPBackend.MATH):
                        return model.denoise_step(**kwargs)

                replays = {}
                for name, candidate_prefix in (("shared_prefix", None), ("alternate_prefix", (
                        math["prefix_pad_mask"], math["prefix_key_cache"], math["prefix_value_cache"]))):
                    tensors, metadata = compare_identical_state_velocities(
                        model.denoise_step, math_velocity, default, torch,
                        expected_steps=config.num_steps, candidate_prefix=candidate_prefix)
                    path = factory.safe_path(args.output_dir / f"seed{seed}-velocity-{name}.npz")
                    if path.exists():
                        raise FileExistsError("Refusing to overwrite velocity replay captures")
                    np.savez_compressed(path, **{key: value.numpy() for key, value in tensors.items()})
                    metadata["output_file"] = factory.file_identity(path, factory.safe_path)
                    replays[name] = metadata
                case["identical_state_velocity_replays"] = replays
            if case_index == 0:
                repeats = {}
                for backend, first in (("default", default), ("math", math)):
                    second, metadata = run_source_policy(model, inputs, backend, seed, 1, args.output_dir, np, torch)
                    if any(not torch.equal(first[name], second[name]) for name in first):
                        raise AssertionError("Same-backend source inference is not repeatable")
                    repeats[backend] = {"all_captures_exact": True, "run": metadata}
                case["same_backend_repeatability"] = repeats
                left_ops, right_ops = left["attention_profile"]["operators"], right["attention_profile"]["operators"]
                if not any("flash_attention_for_cpu" in name for name in left_ops) or not any("attention_math" in name for name in right_ops) or any("flash_attention" in name for name in right_ops):
                    raise AssertionError("Profiler did not establish CPU flash versus MATH dispatch")
                expected = report["attention_attribution"]["expected_native_sdpa_calls"]
                for metadata in (left, right):
                    if metadata["attention_profile"]["native_sdpa_calls_by_stage"] != {SCOPES[0]: expected, SCOPES[1]: 0, SCOPES[2]: 0}:
                        raise AssertionError("Native SDPA stage attribution differs")
            report["cases"].append(case)
            persist()
            print(json.dumps({"seed": seed, "final_actions": case["comparisons"]["final_actions"]}), flush=True)
        if state_identity(model, torch) != state_before or {name: module.training for name, module in model.named_modules()} != modes_before:
            raise AssertionError("Source model state or training modes changed")
        identities = report["source_bindings"] + report["checkpoint_bindings"] + ([report["fixture_binding"]] if "fixture_binding" in report else [])
        if fixture_manifest:
            identities += fixture_manifest["bindings"] + [report["fixture_manifest_binding"]]
        if any(factory.file_identity(item["path"], factory.safe_path) != item for item in identities):
            raise AssertionError("Source/checkpoint/fixture bindings changed")
        if backend_flags(torch) != original_flags:
            raise AssertionError("Backend state changed")
        report.update(status="characterized", model_state_sha256_before=state_before, model_state_unchanged=True, training_modes_unchanged=True, source_checkpoint_bindings_unchanged=True, backend_restored=True, elapsed_wall_s=time.monotonic() - started)
        persist()
    except BaseException as error:
        report.update(status="failed", error=repr(error), backend_after=backend_flags(torch))
        persist()
        raise
    finally:
        torch.set_num_threads(original_threads)
        torch.random.set_rng_state(original_rng)
    return report


if __name__ == "__main__":
    main()
