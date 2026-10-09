"""Local, complete pretrained SmolVLA action-chunk factory for verify_session.py.

The default observation supplies every configured camera. Inputs are explicitly
preprocessed tensors; image/token preprocessing, robot execution and application
quality are outside this neural policy session. Original LeRobot precision is
preserved, including its BF16 backbone and prefix cache. A runtime unable to
represent those tensors must refuse this session rather than cast them.
"""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

from verify_resnet import file_identity, safe_path


MODEL_ID = "lerobot/smolvla_base"
MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
# Official pinned model.safetensors page, SHA256 field (not its Xet hash).
WEIGHT_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
MODEL_SOURCE = f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/model.safetensors"
BACKBONE_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GEOMETRY = {
    "num_vlm_layers": 16, "num_expert_layers": 0, "expert_width_multiplier": 0.75,
    "num_steps": 10, "chunk_size": 50, "n_action_steps": 50,
    "max_state_dim": 32, "max_action_dim": 32, "tokenizer_max_length": 48,
    "resize_imgs_with_padding": [512, 512], "attention_mode": "cross_attn",
    "self_attn_every_n_layers": 2, "use_cache": True, "load_vlm_weights": True,
    "vlm_model_name": BACKBONE_ID, "adapt_to_pi_aloha": False,
    "use_amp": False, "n_obs_steps": 1, "empty_cameras": 0,
    "add_image_special_tokens": False, "prefix_length": 0, "pad_language_to": "max_length",
    "min_period": 0.004, "max_period": 4.0,
}


def directory_artifacts(directory):
    """Hash a supplied model directory, pruning prohibited names before traversal."""
    root = safe_path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Local model directory is absent: {root}")
    result = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories
                                if not any(word in name.lower() for word in ("hammer", "vlsi")))
        for name in sorted(files):
            if any(word in name.lower() for word in ("hammer", "vlsi")):
                continue
            result.append(file_identity(Path(current) / name, safe_path))
    return result


def checkpoint_artifacts(directory):
    root = safe_path(directory)
    for name in ("config.json", "model.safetensors"):
        if not safe_path(root / name).is_file():
            raise FileNotFoundError(f"Pinned complete SmolVLA checkpoint requires {root / name}")
    weight = file_identity(root / "model.safetensors", safe_path)
    if weight["sha256"] != WEIGHT_SHA256:
        raise ValueError("Weights are not the pinned official full SmolVLA checkpoint")
    config = json.loads(safe_path(root / "config.json").read_text())
    if config.get("type") != "smolvla" or any(config.get(key) != value for key, value in GEOMETRY.items()):
        raise ValueError("Configuration differs from the complete pinned SmolVLA geometry/precision")
    camera_names = [f"observation.images.camera{index}" for index in range(1, 4)]
    features = config.get("input_features", {})
    if any(features.get(name) != {"type": "VISUAL", "shape": [3, 256, 256]} for name in camera_names):
        raise ValueError("Pinned SmolVLA requires its three configured camera features")
    if features.get("observation.state") != {"type": "STATE", "shape": [6]} or \
            config.get("output_features", {}).get("action") != {"type": "ACTION", "shape": [6]}:
        raise ValueError("Pinned SmolVLA physical state/action feature extent differs")
    return config, directory_artifacts(root)


def fixture_inputs(config, camera_count, *, input_npz=None, seed=0):
    """Validate exact tensor ABI; never silently coerce a supplied fixture's dtype."""
    import numpy as np
    import torch

    height, width = config.resize_imgs_with_padding
    specs = {
        "images": ((1, camera_count, 3, height, width), np.dtype("float32")),
        "image_masks": ((1, camera_count), np.dtype("bool")),
        "language_tokens": ((1, config.tokenizer_max_length), np.dtype("int64")),
        "language_mask": ((1, config.tokenizer_max_length), np.dtype("bool")),
        "state": ((1, config.max_state_dim), np.dtype("float32")),
        "noise": ((1, config.chunk_size, config.max_action_dim), np.dtype("float32")),
    }
    if input_npz is not None:
        path = safe_path(input_npz)
        with np.load(path, allow_pickle=False) as fixture:
            if set(fixture.files) != set(specs):
                raise ValueError("SmolVLA fixture requires exactly images, image_masks, language_tokens, language_mask, state, noise")
            arrays = [np.array(fixture[name], copy=True, order="C") for name in specs]
        for (name, (shape, dtype)), value in zip(specs.items(), arrays):
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"SmolVLA fixture shape/dtype/finite-value mismatch: {name}")
        inputs = tuple(torch.from_numpy(value) for value in arrays)
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        state = torch.zeros(1, config.max_state_dim)
        state[:, :config.robot_state_feature.shape[0]] = torch.randn(
            1, config.robot_state_feature.shape[0], generator=generator)
        inputs = (
            torch.rand(specs["images"][0], generator=generator) * 2 - 1,
            torch.ones(specs["image_masks"][0], dtype=torch.bool),
            torch.randint(0, 256, specs["language_tokens"][0], generator=generator),
            torch.ones(specs["language_mask"][0], dtype=torch.bool), state,
            torch.randn(specs["noise"][0], generator=generator),
        )
    if not torch.all((inputs[0] >= -1) & (inputs[0] <= 1)):
        raise ValueError("Preprocessed SmolVLA images must be in [-1,1]")
    if torch.any(inputs[2] < 0):
        raise ValueError("Language token IDs must be nonnegative")
    if torch.count_nonzero(inputs[4][:, config.robot_state_feature.shape[0]:]):
        raise ValueError("SmolVLA physical state must be zero padded to max_state_dim")
    return inputs


class SessionProvider:
    def __init__(self, session, artifacts):
        self.session, self.artifacts = session, tuple(artifacts)

    def external_runtime_session(self):
        return self.session


def make_policy_session(model, inputs, loader, provenance):
    """Split upstream inference, preserving exact timesteps and physical action decode."""
    import torch
    from torch import nn
    from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session

    class PrefixStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, images, image_masks, tokens, language_mask, state):
            embeds, masks, attention_masks = self.model.embed_prefix(
                [images[:, index] for index in range(images.shape[1])],
                [image_masks[:, index] for index in range(images.shape[1])],
                tokens, language_mask, state=state)
            _, cache = self.model.vlm_with_expert.forward(
                attention_mask=loader.make_att_2d_masks(masks, attention_masks),
                position_ids=torch.cumsum(masks, dim=1) - 1, past_key_values=None,
                inputs_embeds=[embeds, None], use_cache=True, fill_kv_cache=True)
            layers = range(int(self.model.config.num_vlm_layers))
            keys = torch.stack([cache[index]["key_states"] for index in layers])
            values = torch.stack([cache[index]["value_states"] for index in layers])
            return masks, torch.stack((keys, values))

    prefix = PrefixStage().eval().requires_grad_(False)
    with torch.no_grad():
        masks, cache = prefix(*inputs[:5])
    steps = int(model.config.num_steps)
    dt = -1.0 / steps
    # Upstream uses 1 + step*dt, not repeated FP32 addition of dt.
    times = torch.stack([torch.tensor(1.0 + step * dt, dtype=torch.float32).expand(1)
                         for step in range(steps)])
    flow = loader.SmolVLAFlowSession(model).eval()
    flow_metadata = {
        "kind": "action_chunk", "steps": steps,
        "states": [{"name": "prefix_kv_cache", "input_index": 1, "output_index": 1},
                   {"name": "flow_state", "input_index": 2, "output_index": 0}],
        "streams": [{"name": "upstream_timestep", "input_index": 3, "values": times}],
        "quality": {"key": "actions", "output_index": 0},
    }
    action_dim = int(model.config.action_feature.shape[0])
    action = loader.SmolVLAActionDecodeStage(model.config.chunk_size, action_dim).eval()
    metadata = {
        "kind": "action_chunk", "paper_ready": False,
        "stages": ["prefix_encode", "flow_denoise", "action_decode"],
        "stage_schedule": [{"name": name, "steps": count, "execution": execution, "timed": True}
                           for name, count, execution in [("prefix_encode", 1, "compiled"),
                               ("flow_denoise", steps, "compiled_recurrent"),
                               ("action_decode", 1, "compiled")]],
        "bindings": [
            {"name": "prefix_pad_masks", "from": {"program": "prefix_encode", "output_index": 0},
             "to": {"program": "flow_denoise", "input_index": 0}},
            {"name": "prefix_kv_cache", "from": {"program": "prefix_encode", "output_index": 1},
             "to": {"program": "flow_denoise", "input_index": 1}},
            {"name": "decoded_flow_state", "from": {"program": "flow_denoise", "output_index": 0},
             "to": {"program": "action_decode", "input_index": 0}}],
        "quality_program": "flow_denoise",
        "parameters": {"denoise_steps": steps, "action_horizon": int(model.config.chunk_size),
                       "physical_action_dim": action_dim, "camera_count": inputs[0].shape[1],
                       "batch": 1, "semantic_observations": "denoising iterations for one action chunk",
                       "input_boundary": "preprocessed normalized images/tokens/zero-padded state",
                       "output_boundary": "normalized physical action chunk; no robot/unnormalizer"},
        "provenance": provenance,
    }
    session = make_external_runtime_session(version=2, programs=(
        ExternalRuntimeProgram("prefix_encode", prefix, inputs[:5], 1),
        ExternalRuntimeProgram("flow_denoise", flow, (masks, cache, inputs[5], times[0]), steps, flow_metadata),
        ExternalRuntimeProgram("action_decode", action, (inputs[5],), 1)), metadata=metadata)
    # An independent upstream full inference is the splitting correctness gate.
    with torch.no_grad():
        expected = model.sample_actions(
            [inputs[0][:, index] for index in range(inputs[0].shape[1])],
            [inputs[1][:, index] for index in range(inputs[1].shape[1])],
            inputs[2], inputs[3], inputs[4], noise=inputs[5])[:, :, :action_dim]
        state = inputs[5]
        for timestep in times:
            state, cache, _ = flow(masks, cache, state, timestep)
        actual = action(state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    provenance["upstream_sample_actions_equivalence"] = {"passed": True, "exact": True,
                                                        "shape": list(actual.shape)}
    return session


def get_model_and_inputs(*, checkpoint_dir, backbone_dir, model2mlir_root,
                         input_npz=None, fixture_kind="synthetic", input_source=None,
                         selected_cameras=None, seed=0, cpu_threads=1, precision="original"):
    if precision != "original":
        raise ValueError("Only original LeRobot precision is supported; implicit BF16 conversion is forbidden")
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 4 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid explicit CPU thread/fixture seed budget")
    if fixture_kind not in ("synthetic", "dataset") or (fixture_kind == "dataset" and (not input_npz or not input_source)):
        raise ValueError("Dataset fixtures require both a local NPZ and attributed input_source")
    config_json, checkpoint_files = checkpoint_artifacts(checkpoint_dir)
    backbone = safe_path(backbone_dir)
    backbone_files = directory_artifacts(backbone)
    if not safe_path(backbone / "config.json").is_file() or not any(
            Path(item["path"]).name.endswith(".safetensors") for item in backbone_files):
        raise FileNotFoundError("Original-precision SmolVLA requires local backbone config, processor and weights")
    loader_path = safe_path(Path(model2mlir_root) / "workloads/smolvla/loader.py")
    if selected_cameras is None:
        selected_cameras = [name for name in config_json["input_features"] if name.startswith("observation.images.")]
    configured = {name for name in config_json["input_features"] if name.startswith("observation.images.")}
    if not isinstance(selected_cameras, list) or not selected_cameras or \
            len(set(selected_cameras)) != len(selected_cameras) or any(name not in configured for name in selected_cameras):
        raise ValueError("Selected cameras must be an explicit nonempty unique configured subset")
    if importlib.metadata.version("lerobot") != "0.5.1":
        raise ValueError("This original policy recipe requires lerobot==0.5.1")
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla import smolvlm_with_expert

    torch.set_num_threads(cpu_threads)
    config = PreTrainedConfig.from_pretrained(str(safe_path(checkpoint_dir)), local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError("Pinned checkpoint did not resolve to SmolVLAConfig")
    config.device = "cpu"
    config.vlm_model_name = str(backbone)
    policy = SmolVLAPolicy.from_pretrained(str(safe_path(checkpoint_dir)), config=config,
                                         local_files_only=True, strict=True).eval().requires_grad_(False)
    model = policy.model
    spec = importlib.util.spec_from_file_location("tvm_smolvla_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    inputs = fixture_inputs(config, len(selected_cameras), input_npz=input_npz, seed=seed)
    if torch.any(inputs[2] >= model.vlm_with_expert.config.text_config.vocab_size):
        raise ValueError("Fixture token ID is outside the pinned backbone vocabulary")
    sources = [file_identity(path, safe_path) for path in (
        __file__, loader_path, sys.modules[SmolVLAPolicy.__module__].__file__,
        sys.modules[SmolVLAConfig.__module__].__file__, smolvlm_with_expert.__file__,
        sys.modules[PreTrainedConfig.__module__].__file__)]
    files = checkpoint_files + backbone_files + sources
    if input_npz:
        files.append(file_identity(input_npz, safe_path))
    dtype_bytes = {}
    for value in model.state_dict().values():
        dtype = str(value.dtype)
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + value.numel() * value.element_size()
    provenance = {"checkpoint": MODEL_ID, "checkpoint_revision": MODEL_REVISION,
                  "official_weight_sha256": WEIGHT_SHA256, "official_weight_source": MODEL_SOURCE,
                  "full_checkpoint": True, "precision": "original_lerobot",
                  "synthetic_inputs": fixture_kind == "synthetic", "fixture_seed": seed if not input_npz else None,
                  "input_source": input_source or "seeded synthetic preprocessed tensors",
                  "selected_cameras": selected_cameras, "all_configured_cameras": set(selected_cameras) == configured,
                  "backbone": BACKBONE_ID, "local_backbone": str(backbone),
                  "backbone_revision_verified": False, "declared_artifacts": files,
                  "dependencies": {name: importlib.metadata.version(name) for name in
                                   ("lerobot", "torch", "transformers", "safetensors", "numpy", "onnx")},
                  "model_state_bytes_by_dtype": dtype_bytes, "model_quality_qualified": False,
                  "source_closure_verified": False}
    session = make_policy_session(model, inputs, loader, provenance)
    for identity in files:
        if file_identity(identity["path"], safe_path) != identity:
            raise ValueError("Selected SmolVLA artifact/source changed while loading the policy session")
    return SessionProvider(session, [Path(item["path"]) for item in files]), ()


def get_verification_artifacts(model, _inputs):
    return model.artifacts
