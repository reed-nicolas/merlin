"""SmolVLA flow-loss loader for the external adapter capture worker.

Select ATLAS_EXPORT_DIR (the atlas-torchao export with manifest.json, policy.json,
and model.safetensors), ATLAS_FX_BUNDLE (its verified fx/ bundle, which supplies
only the recorded boundary input tensors), and the frozen
ATLAS_EXPECTED_CONTRACT_SHA256 and ATLAS_EXPECTED_POLICY_SHA256. The policy
architecture is read from the export's lerobot model id in the local Hugging
Face cache. ATLAS_FIXTURE_CASE selects a recorded input case (default 0).

The policy is rebuilt from the raw export exactly as atlas-torchao's
``fx_export.py --boundary smolvla_flow_loss`` does, and the adapter re-exports
the flow-loss forward live; no saved graph is loaded. Invoke the worker with
--already-quantized --quant-adapter atlas_fp8 and the export's manifest.json and
policy.json as --contract and --policy.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

from atlas_mlir_native.capture import normalize_capture, prepare_model  # noqa: F401


def _record(value):
    import torch

    payload = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(dtype=str(value.dtype), shape=list(value.shape), bytes=len(payload),
                sha256=sha256(payload).hexdigest())


def boundary_inputs(bundle, case):
    """Recorded flow-loss inputs of one fixture case, authenticated by the bundle manifest."""
    from safetensors.torch import load_file

    manifest = json.loads((bundle / "fx_manifest.json").read_text())
    record = manifest["files"]["fixtures.safetensors"]
    if sha256((bundle / "fixtures.safetensors").read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("fixture archive differs from its bundle manifest")
    if manifest.get("provenance", {}).get("boundary", "smolvla_flow_loss") != "smolvla_flow_loss":
        raise ValueError("bundle boundary is not the SmolVLA flow-loss forward")
    parity = json.loads((bundle / "parity.json").read_text())
    names = parity["cases"][case]["inputs"]
    fixtures = load_file(str(bundle / "fixtures.safetensors"))
    return tuple(fixtures[name] for name in names), names


def drop_decoded_weights(policy, export_policy):
    """Replace decoded quantized weights by payload-free placeholders, as raw_capture.raw_linear does.

    load_quantized decodes each site's codes into ``weight`` and checks that the module
    re-derives the exported codes bit for bit. Native capture reads only those codes, so
    the decoded copy is not source state; only its dtype and shape remain.
    """
    import torch
    from torch import nn

    for name, site in export_policy["sites"].items():
        if site["status"] != "quantized" or site["kind"] not in {"linear", "conv2d"}:
            continue
        module = policy.get_submodule(name)
        weight = module.weight
        if list(weight.shape) != site["weight_shape"] or str(weight.dtype).removeprefix("torch.") != site["weight_dtype"]:
            raise ValueError(f"decoded weight dtype or shape differs from policy: {name}")
        module.weight = nn.Parameter(torch.empty(tuple(weight.shape), dtype=weight.dtype, device="meta"),
                                     requires_grad=False)


def drop_derived_buffers(policy, tensors, prefix="policy."):
    """Remove non-persistent derived buffers the export does not carry (strict re-export fails if one is read)."""
    removed = []
    for path, module in policy.named_modules():
        for name in sorted(module._non_persistent_buffers_set):
            full = f"{path}.{name}" if path else name
            if full not in tensors:
                del module._buffers[name]
                module._non_persistent_buffers_set.discard(name)
                removed.append(prefix + full)
    return removed


def drop_uncalled_sites(policy, bundle, export_policy, prefix="policy."):
    """Detach quantized modules the captured boundary never calls, as the bundle declares.

    The bundle's graph_inspection.json (hash-bound by fx_manifest.json) lists them from the
    capture's eager forward hooks; strict re-export fails if one is reached after removal.
    """
    manifest = json.loads((bundle / "fx_manifest.json").read_text())
    path = bundle / "graph_inspection.json"
    if sha256(path.read_bytes()).hexdigest() != manifest["files"]["graph_inspection.json"]["sha256"]:
        raise ValueError("graph inspection differs from its bundle manifest")
    removed = []
    for row in json.loads(path.read_text()).get("unused_quantized_modules", []):
        site = row["site"]
        if row["module"] != prefix + site or export_policy["sites"].get(site, {}).get("status") != "quantized":
            raise ValueError(f"declared uncalled module is not a quantized policy site: {row}")
        parent, _, attr = site.rpartition(".")
        delattr(policy.get_submodule(parent), attr)
        removed.append(site)
    return removed


def fold_vision_position_ids(policy, image):
    """Bind SmolVLM's input-independent patch position ids as one exported constant.

    The vision tower always builds an all-true patch mask for its fixed image size, so the
    position ids (an arange, bucketize, and boolean-mask scatter) depend on no input. The ids
    are computed once by the unchanged module and the replacement forward is checked bit for
    bit against the original on the boundary image; the patch embedding and position table run
    unchanged in the captured graph.
    """
    import torch

    embeddings = [module for module in policy.modules() if type(module).__name__ == "SmolVLMVisionEmbeddings"]
    if len(embeddings) != 1:
        raise ValueError("expected one SmolVLM vision embedding")
    module = embeddings[0]
    size = module.patch_size
    mask = torch.ones((image.shape[0], image.shape[2] // size, image.shape[3] // size), dtype=torch.bool)
    captured = []
    handle = module.position_embedding.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
    with torch.no_grad():
        expected = module(image, mask)
    handle.remove()
    if len(captured) != 1 or captured[0].dtype != torch.int64:
        raise ValueError("vision position ids were not observed exactly once")
    ids = captured[0]

    def forward(pixel_values, patch_attention_mask):
        patches = module.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return patches + module.position_embedding(ids)

    with torch.no_grad():
        if not torch.equal(forward(image, mask), expected):
            raise ValueError("constant vision position ids change the embedding output")
    module.forward = forward
    return {"shape": list(ids.shape), "sha256": _record(ids)["sha256"],
            "is_arange": bool(torch.equal(ids, torch.arange(ids.numel()).reshape(ids.shape)))}


def get_model_and_inputs():
    import torch

    import atlas_fx_ops  # noqa: F401  (register the source custom operators)
    import eval_policy
    import fx_export
    import load_quantized

    export = Path(os.environ["ATLAS_EXPORT_DIR"])
    bundle = Path(os.environ["ATLAS_FX_BUNDLE"])
    case = int(os.environ.get("ATLAS_FIXTURE_CASE", "0"))
    expected_contract = os.environ["ATLAS_EXPECTED_CONTRACT_SHA256"]
    expected_policy = os.environ["ATLAS_EXPECTED_POLICY_SHA256"]
    for path, expected in ((export / "manifest.json", expected_contract), (export / "policy.json", expected_policy)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"selected source receipt digest differs: {path.name}")
    tensors, _, manifest = load_quantized.read_export(export, verify=True)
    if manifest.get("model", {}).get("kind") == "causal_lm":
        raise ValueError("export is not a lerobot policy")
    policy, _ = eval_policy.load_policy(manifest["model"]["id"], "cpu", torch.bfloat16)
    load_quantized.load_quantized(manifest["model"]["id"], export, "cpu", build=lambda: policy,
                                  mode="ref_conv", options={})
    export_policy = json.loads((export / "policy.json").read_text())
    drop_decoded_weights(policy, export_policy)
    uncalled = drop_uncalled_sites(policy, bundle, export_policy)
    removed = drop_derived_buffers(policy, tensors)
    boundary = fx_export.SmolVLAFlowLoss(policy, 1).eval()
    arguments, names = boundary_inputs(bundle, case)
    with torch.no_grad():
        before = boundary(*arguments)
    positions = fold_vision_position_ids(policy, arguments[0])
    with torch.no_grad():
        if not torch.equal(boundary(*arguments), before):
            raise ValueError("constant vision position ids change the boundary output")
    out = os.environ.get("ATLAS_SOURCE_OUT")
    if out:
        from safetensors.torch import save_file

        root = Path(out)
        root.mkdir(parents=True, exist_ok=True)
        tensors = {f"input_{index}": value for index, value in enumerate(arguments)}
        save_file(tensors, str(root / "inputs.safetensors"))
        source = dict(kind="smolvla_flow_loss", fixture_case=case, fixture_names=names,
                      input_keys=list(tensors), tensors={key: _record(value) for key, value in tensors.items()},
                      contract_sha256=expected_contract, policy_sha256=expected_policy,
                      removed_derived_buffers=removed, removed_uncalled_sites=uncalled,
                      vision_position_ids=positions)
        (root / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    return prepare_model(boundary, arguments, export / "manifest.json", export / "policy.json",
                         export_prefix="policy.", expected_contract_sha256=expected_contract,
                         expected_policy_sha256=expected_policy)
