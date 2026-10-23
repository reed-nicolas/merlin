"""Capture fresh-vision SmolVLA policy operators and close the native reference.

This is source-native capture, not Atlas execution or a quality qualification.
Every Linear and Conv2d invocation records its actual incoming tensor. Parameters
are stored once per module, while ten denoising steps retain distinct inputs.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from capture_smolvla import ARCHIVE_SHA, WEIGHTS_SHA
from capture_tinyllama_native import identity, safe_path, verify_declared_sources
from evaluate_smolvla_retained_prefix import INPUT_NAMES, INPUT_SHA, STATE_SHA, state_sha, tensor_id, tensor_words


def require_equal(actual, expected, name):
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.tobytes() != expected.tobytes()):
        raise ValueError("Native reference changed: " + name)


def validate_trajectory(captures, reference, steps):
    if steps != list(range(10)):
        raise ValueError("Incomplete or reordered denoising trajectory")
    if set(captures) != set(reference):
        raise ValueError("Native trajectory field set changed")
    for name, actual in captures.items():
        require_equal(actual, reference[name], name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=safe_path, required=True)
    parser.add_argument("--observation-input", type=safe_path, required=True)
    parser.add_argument("--out", type=safe_path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "failed", "command": [sys.executable, *sys.argv], "cwd": os.getcwd(),
              "fresh_vision_executed": False, "atlas_executed": False, "timing_qualified": False,
              "policy_quality_qualified": False, "cpu_threads": 1}
    pins, handles = [], []

    def pin(path, expected=None):
        row = identity(safe_path(path), expected)
        pins.append(row)
        return row

    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        pin(__file__)
        for name in ("capture_smolvla", "capture_tinyllama_native", "evaluate_smolvla_retained_prefix",
                     "capture_smolvla_full_query", "evaluate_smolvla_attention", "source_relocation"):
            pin(sys.modules[name].__file__)
        pin(args.source_receipt)
        parent = json.loads(args.source_receipt.read_text())
        if parent["status"] != "passed_dynamic_block_model_and_retained_policy_continuation":
            raise ValueError("Require retained authenticated selected-block source")
        source = parent["source"]
        verify_declared_sources(source)
        for item in parent["source_bookends"]:
            pin(item["path"], item["sha256"])
        versions = {key: importlib.metadata.version(key) for key in ("torch", "transformers", "lerobot", "numpy")}
        if versions != source["versions"] or sys.version != source["python"]:
            raise ValueError("Native source environment changed")
        pin(source["weights"]["path"], WEIGHTS_SHA)
        pin(source["historical_original_archive"]["path"], ARCHIVE_SHA)
        pin(args.observation_input, INPUT_SHA)
        with np.load(args.observation_input, allow_pickle=False) as archive:
            if set(archive.files) != set(INPUT_NAMES):
                raise ValueError("Observation fixture fields changed")
            inputs = {name: torch.from_numpy(archive[name].copy()) for name in INPUT_NAMES}
        with np.load(source["historical_original_archive"]["path"], allow_pickle=False) as archive:
            reference = {name: archive[name].copy() for name in archive.files}
        policy_dir = safe_path(source["weights"]["path"]).parent
        config = PreTrainedConfig.from_pretrained(str(policy_dir), local_files_only=True)
        if not isinstance(config, SmolVLAConfig):
            raise ValueError("Wrong policy configuration")
        config.device = "cpu"
        config.vlm_model_name = str(safe_path(source["backbone_config"]["path"]).parent)
        policy = SmolVLAPolicy.from_pretrained(str(policy_dir), config=config,
                                              local_files_only=True, strict=True).eval().requires_grad_(False)
        model = policy.model
        if model._rtc_enabled() or config.compile_model or state_sha(model, torch) != STATE_SHA:
            raise ValueError("Native state or execution mode changed")
        before = {name: tensor_id(value, torch) for name, value in inputs.items()}
        rng = torch.random.get_rng_state().clone()
        parameters, calls, captures = {}, [], {}
        (args.out / "parameters").mkdir()
        (args.out / "calls").mkdir()

        def save_tensor(path, value):
            raw = tensor_words(value, torch)
            path.write_bytes(raw)
            return {**tensor_id(value, torch), "file": str(path.relative_to(args.out)), "bytes": len(raw)}

        def save_array(value):
            flat = torch.empty(value.numel(), dtype=value.dtype).copy_(value.detach().reshape(-1))
            if value.dtype == torch.bfloat16:
                flat = flat.view(torch.uint16)
            return flat.reshape(value.shape).numpy().copy()

        def hook(name):
            def capture(module, incoming, output):
                if len(incoming) != 1 or not isinstance(output, torch.Tensor):
                    raise ValueError("Unexpected matrix operator ABI: " + name)
                index = len(calls)
                if name not in parameters:
                    stem = f"{len(parameters):04d}"
                    item = {"name": name, "kind": type(module).__name__,
                            "weight": save_tensor(args.out / "parameters" / (stem + "-weight.bin"), module.weight)}
                    if module.bias is not None:
                        item["bias"] = save_tensor(args.out / "parameters" / (stem + "-bias.bin"), module.bias)
                    if isinstance(module, torch.nn.Conv2d):
                        item.update(stride=list(module.stride), padding=list(module.padding),
                                    dilation=list(module.dilation), groups=module.groups)
                    parameters[name] = item
                stem = f"{index:05d}"
                calls.append({"index": index, "module": name,
                              "input": save_tensor(args.out / "calls" / (stem + "-input.bin"), incoming[0]),
                              "output": save_tensor(args.out / "calls" / (stem + "-output.bin"), output)})
                if index % 100 == 0:
                    print(json.dumps({"milestone": "native_operator", "index": index, "module": name}), flush=True)
            return capture

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                handles.append(module.register_forward_hook(hook(name)))
        original_embed, original_forward, original_denoise = model.embed_prefix, model.vlm_with_expert.forward, model.denoise_step
        steps = []

        def embed(*arguments, **kwargs):
            result = original_embed(*arguments, **kwargs)
            for name, value in zip(("prefix_embeddings", "prefix_pad_mask", "prefix_attention_mask"), result):
                captures[name] = save_array(value)
                require_equal(captures[name], reference[name], name)
            report["fresh_vision_executed"] = True
            print(json.dumps({"milestone": "fresh_vision_prefix_raw_exact", "operators": len(calls)}), flush=True)
            return result

        def forward(*arguments, **kwargs):
            result = original_forward(*arguments, **kwargs)
            if kwargs.get("fill_kv_cache"):
                for field in ("key_states", "value_states"):
                    name = "prefix_" + field.replace("states", "cache")
                    captures[name] = save_array(torch.stack([result[1][i][field] for i in range(16)]))
                    require_equal(captures[name], reference[name], name)
            return result

        def denoise(*arguments, **kwargs):
            index = len(steps)
            stem = f"step_{index:02d}"
            captures[stem + "_input_state"] = save_array(kwargs["x_t"])
            captures[stem + "_timestep"] = save_array(kwargs["timestep"])
            result = original_denoise(*arguments, **kwargs)
            captures[stem + "_velocity"] = save_array(result)
            steps.append(index)
            print(json.dumps({"milestone": "native_denoising_step", "step": index, "operators": len(calls)}), flush=True)
            return result

        model.embed_prefix, model.vlm_with_expert.forward, model.denoise_step = embed, forward, denoise
        try:
            with torch.inference_mode():
                final = model.sample_actions([inputs["images"][:, i] for i in range(2)],
                                             [inputs["image_masks"][:, i] for i in range(2)],
                                             inputs["language_tokens"], inputs["language_mask"],
                                             inputs["state"], noise=inputs["noise"])
        finally:
            del model.embed_prefix, model.vlm_with_expert.forward, model.denoise_step
        captures["final_full_state"] = save_array(final)
        captures["final_actions"] = save_array(final[:, :, :6])
        if len(steps) != 10:
            raise ValueError("Incomplete denoising trajectory")
        for index in steps:
            captures[f"step_{index:02d}_output_state"] = captures[f"step_{index + 1:02d}_input_state"].copy() if index < 9 else captures["final_full_state"].copy()
        validate_trajectory(captures, reference, steps)
        if (before != {name: tensor_id(value, torch) for name, value in inputs.items()}
                or not torch.equal(rng, torch.random.get_rng_state()) or state_sha(model, torch) != STATE_SHA):
            raise ValueError("Native input, RNG or state mutated")
        for item in tuple(pins):
            identity(item["path"], item["sha256"])
        np.savez(args.out / "trajectory.npz", **captures)
        report.update(status="PASS_FRESH_VISION_NATIVE_POLICY_CAPTURE", source=source,
                      source_bookends=pins, versions=versions, parameters=parameters, calls=calls,
                      trajectory=identity(args.out / "trajectory.npz"),
                      all_archived_fields_raw_exact=True, input_state_rng_bookends_equal=True,
                      parameter_bytes=sum(item["weight"]["bytes"] + item.get("bias", {}).get("bytes", 0) for item in parameters.values()),
                      invocation_bytes=sum(item["input"]["bytes"] + item["output"]["bytes"] for item in calls))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        while handles:
            handles.pop().remove()
        report["wall_seconds"] = time.monotonic() - started
        (args.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "wall_seconds", "fresh_vision_executed")}), flush=True)


if __name__ == "__main__":
    main()
