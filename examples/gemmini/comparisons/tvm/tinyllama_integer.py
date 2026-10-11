"""Explicit TinyLlama W8A8 projection reference and development quality contract.

This module owns a numerical recipe, not a device runtime. Learned projections
use signed int8 operands and an actual int32 contraction; attention and surrounding
model operations retain the factory's explicit CPU FP32 recipe. No calibration,
evaluation-selected scales, checkpoint clipping or source-quality claim is hidden.
"""
import hashlib
import json
import math
import re

import numpy as np
import torch
from torch import nn


RECIPE = {
    "name": "tinyllama_w8a8_dynamic_token_v1",
    "weight": "per_output_channel_maxabs_div127",
    "activation": "dynamic_per_token_maxabs_div127",
    "rounding": "nearest_ties_even",
    "range": [-127, 127], "zero_point": 0,
    "zero_scale": 1.0, "minimum_nonzero_scale": torch.finfo(torch.float32).tiny,
    "accumulator": "int32", "maximum_k": 131071,
    "dequantization": "float32(accumulator)*activation_scale*weight_scale",
    "bias": "none", "attention_and_state": "float32_cpu",
}


def recipe(activation_scale="maxabs"):
    if activation_scale not in ("maxabs", "power_of_two"):
        raise ValueError("Unknown integer activation scale policy")
    if activation_scale == "maxabs":
        return dict(RECIPE)
    return {**RECIPE, "name": "tinyllama_w8a8_dynamic_token_pot_v1",
            "activation": "dynamic_per_token_ceil_power_of_two_maxabs_div127",
            "scale_policy": "smallest_float32_normal_power_of_two_at_least_maxabs_div127"}
QUALITY_CONTRACT = {
    "name": "tinyllama_language_development_v1",
    "reference": "unchanged_checkpoint_explicit_bfloat16_to_float32_cpu_eager",
    "minimum_documents": 8, "tokens_per_document": 128,
    "maximum_mean_nll_increase_nats": 0.20,
    "minimum_next_token_top1_agreement": 0.80,
    "purpose": "development_quantization_loss_check_not_paper_language_benchmark",
    "selection": "first128_token_ids_each_document_in_declared_order_no_filtering",
    "session": {"batch": 1, "prefill_tokens": 128, "decode_tokens": 32,
                "kv_capacity": 160, "repeats": 2, "reset": "between_repeats"},
}
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/validation-00000-of-00001.parquet"
WIKITEXT_SHA256 = "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"
WIKITEXT_SOURCE = f"https://huggingface.co/datasets/Salesforce/wikitext/blob/{WIKITEXT_REVISION}/{WIKITEXT_FILE}"


def prepare_wikitext_corpus(parquet_path, checkpoint_dir, output_dir):
    """Prepare the frozen development corpus from one user-supplied pinned artifact.

    Selection depends only on dataset order and pinned tokenizer length: first eight
    Wikipedia articles with at least160 IDs. Quality scores consume first128 IDs per
    article; the first article also supplies the128+32teacher-forced session text.
    No model logits or quantization feedback participate in input selection.
    """
    from verify_resnet import safe_path, file_identity
    source = safe_path(parquet_path)
    artifact = file_identity(source, safe_path)
    if artifact["sha256"] != WIKITEXT_SHA256:
        raise ValueError("WikiText validation artifact differs from the pinned source bytes")
    from transformers import AutoTokenizer
    import pyarrow.parquet as parquet
    checkpoint = safe_path(checkpoint_dir)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        safe_path(checkpoint / name)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    table = parquet.read_table(source, columns=["text"])
    articles, current, title, start = [], [], None, 0
    for row, text in enumerate(table.column("text").to_pylist()):
        if not isinstance(text, str):
            raise ValueError("Pinned WikiText text column contains a non-string value")
        if re.fullmatch(r"\s*=\s+[^=].*[^=]\s+=\s*", text):
            if title is not None:
                articles.append((start, title, "".join(current)))
            title, start, current = text.strip(), row, [text]
        elif title is not None:
            current.append(text)
    if title is not None:
        articles.append((start, title, "".join(current)))
    documents = []
    for row, title, text in articles:
        if len(tokenizer(text, add_special_tokens=True)["input_ids"]) >= 160:
            documents.append({"id": f"validation_article_row_{row}", "text": text,
                              "source": WIKITEXT_SOURCE + f"#article_start_row={row}", "title": title})
            if len(documents) == 8:
                break
    if len(documents) != 8:
        raise ValueError("Pinned validation corpus lacks eight eligible articles")
    payload = {"source": WIKITEXT_SOURCE,
               "selection": "first8_top_level_articles_in_parquet_order_with_at_least160_pinned_tokenizer_ids;first128_ids_each_for_quality",
               "dataset_artifact": artifact, "dataset_revision": WIKITEXT_REVISION,
               "dataset_config": "wikitext-2-raw-v1", "split": "validation",
               "documents": documents}
    destination = safe_path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    quality = safe_path(destination / "language-quality.json")
    session = safe_path(destination / "session.txt")
    quality.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    session.write_text(documents[0]["text"], encoding="utf-8")
    return {"quality_corpus_path": str(quality), "corpus_path": str(session),
            "workload": "development_language", "prefill_tokens": 128,
            "decode_tokens": 32, "quantized": True}


def identity(value):
    value = value.detach().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.numel() * value.element_size(),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def independent_quantize_rows(value, activation_scale="maxabs"):
    """Independent encoder: frexp scale selection and explicit floor/parity RNE.

    Division retains the recipe's FP32 rounding before integer rounding. Neither
    Torch round/log2 nor NumPy rint supplies the integer rounding oracle.
    """
    value = np.asarray(value)
    if value.dtype != np.float32 or value.ndim < 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("Encoding oracle requires a nonempty finite FP32 tensor")
    maximum = np.max(np.abs(value), axis=-1, keepdims=True)
    scale = np.where(maximum == 0, np.float32(1),
                     np.maximum(maximum / np.float32(127), np.finfo(np.float32).tiny))
    if activation_scale == "power_of_two":
        # Frexp is independent of the exported log2/pow expression and is exact for
        # finite float32 input, including immediately adjacent dyadic boundaries.
        mantissa, exponent = np.frexp(scale)
        scale = np.ldexp(np.ones_like(scale), exponent - (mantissa == np.float32(.5)))
    elif activation_scale != "maxabs":
        raise ValueError("Unknown integer activation scale policy")
    ratio = (value / scale).astype(np.float64)
    lower = np.floor(ratio)
    fraction = ratio - lower
    rounded = lower + ((fraction > .5) | ((fraction == .5) & (np.remainder(lower, 2) != 0)))
    return np.clip(rounded, -127, 127).astype(np.int8), scale


def check_operand_encoding(value, codes, scales, activation_scale="maxabs"):
    """Check emitted codes and scales against actual inputs, before any dot check."""
    value, codes, scales = np.asarray(value), np.asarray(codes), np.asarray(scales)
    expected_codes, expected_scales = independent_quantize_rows(value, activation_scale)
    if codes.dtype != np.int8 or codes.shape != value.shape:
        raise ValueError("Emitted activation codes must have exact INT8 shape")
    if scales.dtype != np.float32 or scales.shape != expected_scales.shape or not np.isfinite(scales).all():
        raise ValueError("Emitted activation scales must have exact finite FP32 shape")
    if not np.array_equal(scales, expected_scales):
        raise AssertionError("Activation scale encoding disagrees with independent oracle")
    if not np.array_equal(codes, expected_codes):
        raise AssertionError("Activation code rounding/clipping disagrees with independent oracle")
    return {"passed": True, "oracle": "numpy_fp32_division_frexp_floor_parity_rne",
            "checked_codes": int(codes.size), "checked_scales": int(scales.size),
            "rounding": "nearest_ties_even", "range": [-127, 127], "scale_policy": activation_scale}


def diagnose_activation_pair(source, actual, activation_scale="power_of_two", *, actual_codes=None, actual_scales=None, maximum_elements=160 * 5632):
    """Bounded local encoder checks and propagated input/threshold diagnostics.

    This reports a changed quantized program, never a model acceptance verdict.
    Supplying emitted codes/scales checks an implementation on its actual inputs.
    """
    source, actual = np.asarray(source), np.asarray(actual)
    if type(maximum_elements) is not int or maximum_elements < 1 or source.shape != actual.shape or source.size > maximum_elements:
        raise ValueError("Activation diagnostic requires matching shapes within its element budget")
    source_codes, source_scales = independent_quantize_rows(source, activation_scale)
    expected_codes, expected_scales = independent_quantize_rows(actual, activation_scale)
    if (actual_codes is None) != (actual_scales is None):
        raise ValueError("Actual emitted codes and scales must be supplied together")
    actual_codes = expected_codes if actual_codes is None else np.asarray(actual_codes)
    actual_scales = expected_scales if actual_scales is None else np.asarray(actual_scales)
    local = check_operand_encoding(actual, actual_codes, actual_scales, activation_scale)
    differences = actual_codes != source_codes
    result = {"scope": "local_actual_input_encoding_and_propagated_source_input",
              "source_input": identity(torch.from_numpy(source.copy())),
              "actual_input": identity(torch.from_numpy(actual.copy())),
              "local_actual_encoding": local,
              "input_mismatches": int(np.count_nonzero(source != actual)),
              "max_absolute_input_error": float(np.max(np.abs(source.astype(np.float64) - actual.astype(np.float64)))),
              "scale_mismatches": int(np.count_nonzero(source_scales != actual_scales)),
              "code_mismatches": int(np.count_nonzero(differences)), "first_code_difference": None}
    if np.any(differences):
        index = tuple(int(v) for v in np.argwhere(differences)[0])
        scale_index = (*index[:-1], 0)
        source_ratio = float(np.float32(source[index] / source_scales[scale_index]))
        actual_ratio = float(np.float32(actual[index] / actual_scales[scale_index]))
        boundary = math.floor(min(source_ratio, actual_ratio)) + .5
        result["first_code_difference"] = {"index": list(index), "source_code": int(source_codes[index]),
                                          "actual_code": int(actual_codes[index]), "source_input": float(source[index]),
                                          "actual_input": float(actual[index]), "source_scale": float(source_scales[scale_index]),
                                          "actual_scale": float(actual_scales[scale_index]), "source_ratio": source_ratio,
                                          "actual_ratio": actual_ratio, "nearest_lower_half_integer": boundary,
                                          "same_scale_crosses_rounding_boundary": bool(source_scales[scale_index] == actual_scales[scale_index]
                                                                                       and min(source_ratio, actual_ratio) <= boundary <= max(source_ratio, actual_ratio))}
    return result


def quantize_rows(value, activation_scale="maxabs"):
    """Symmetric row-wise quantization; finite zero/subnormal rows are well defined."""
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum),
                        (maximum / 127.0).clamp(min=torch.finfo(torch.float32).tiny))
    if activation_scale == "power_of_two":
        power = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        # Float32 log2 can round a just-above-dyadic input down to an integer.
        # Correct that boundary using the original scale, with no model-specific rule.
        scale = torch.where(power < scale, power * 2.0, power)
    elif activation_scale != "maxabs":
        raise ValueError("Unknown integer activation scale policy")
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


class IntegerProjection(nn.Module):
    """OI FP32 learned weight becomes one contiguous KN int8 constant and N scales."""
    def __init__(self, weight, activation_scale="maxabs"):
        super().__init__()
        recipe(activation_scale)
        self.activation_scale = activation_scale
        if (weight.ndim != 2 or weight.dtype != torch.float32 or weight.device.type != "cpu"
                or not bool(torch.isfinite(weight).all())):
            raise ValueError("Integer projection requires a finite CPU FP32 weight matrix")
        self.out_features, self.in_features = weight.shape
        if not 1 <= self.in_features <= RECIPE["maximum_k"]:
            raise ValueError("Integer projection reduction exceeds the signed-int32 bound")
        packed, scales = quantize_rows(weight.detach())
        self.register_buffer("weight_kn", packed.t().contiguous())
        self.register_buffer("weight_scale", scales.flatten().contiguous())

    def integer_inputs(self, value):
        if not torch.compiler.is_compiling():
            if (value.dtype != torch.float32 or value.device.type != "cpu"
                    or value.ndim < 1 or value.shape[-1] != self.in_features
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Integer projection requires finite CPU FP32 activations with matching K")
        quantized, scale = quantize_rows(value, self.activation_scale)
        return quantized.reshape(-1, self.in_features).contiguous(), scale

    def forward(self, value):
        quantized, scale = self.integer_inputs(value)
        accumulator = torch._int_mm(quantized, self.weight_kn)
        result = accumulator.reshape(*value.shape[:-1], self.out_features).to(torch.float32)
        return result * scale * self.weight_scale


def export_options(torch_module):
    """Modern exporter preserves signed-int8 operands with standard MatMulInteger."""
    from onnxscript import opset18 as op
    from onnxscript.onnx_types import INT8, INT32

    def integer_mm(a: INT8, b: INT8) -> INT32:
        return op.MatMulInteger(a, b)

    return {"custom_translation_table": {torch_module.ops.aten._int_mm.default: integer_mm}}


def projection_names():
    return [f"model.layers.{layer}.{group}.{name}_proj"
            for layer in range(22)
            for group, names in (("self_attn", ("q", "k", "v", "o")),
                                 ("mlp", ("gate", "up", "down")))
            for name in names] + ["lm_head"]


def projection_shape(name):
    if name == "lm_head":
        return 32000, 2048
    leaf = name.rsplit(".", 1)[-1]
    return {"q_proj": (2048, 2048), "k_proj": (256, 2048), "v_proj": (256, 2048),
            "o_proj": (2048, 2048), "gate_proj": (5632, 2048),
            "up_proj": (5632, 2048), "down_proj": (2048, 5632)}[leaf]


def install_projections(lm, activation_scale="maxabs"):
    """Replace exactly the complete 155 learned contractions, never an attention op."""
    linears = {name: module for name, module in lm.named_modules() if isinstance(module, nn.Linear)}
    expected = projection_names()
    if (set(linears) != set(expected)
            or any(module.bias is not None or tuple(module.weight.shape) != projection_shape(name)
                   for name, module in linears.items())):
        raise ValueError("TinyLlama must expose exactly 155 complete bias-free learned projections")
    records = []
    for name in expected:
        original = linears[name]
        projection = IntegerProjection(original.weight, activation_scale=activation_scale).eval()
        channels = np.unique(np.linspace(0, projection.out_features - 1,
                                         min(7, projection.out_features), dtype=int))
        q_expected, scale_expected = independent_quantize_rows(original.weight.detach().numpy()[channels])
        if (not np.array_equal(projection.weight_kn.numpy()[:, channels].T, q_expected)
                or not np.array_equal(projection.weight_scale.numpy()[channels], scale_expected.flatten())):
            raise AssertionError("Trained weight quantization disagrees with independent NumPy oracle")
        parent, _, leaf = name.rpartition(".")
        setattr(lm.get_submodule(parent) if parent else lm, leaf, projection)
        records.append({"name": name, "source_weight": identity(original.weight),
                        "packed_weight": identity(projection.weight_kn),
                        "weight_scale": identity(projection.weight_scale),
                        "input_k": projection.in_features, "output_n": projection.out_features,
                        "weight_quantization_reference": {"passed": True, "oracle": "numpy_fp32_division_floor_parity_rne",
                                                          "channels": channels.tolist(), "full_k": projection.in_features},
                        "placement": "gemmini_int8_int32_candidate"})
        # Release original FP32 projection promptly rather than retaining a second checkpoint.
        linears[name] = None
    return records


def check_projection(projection, value, *, rows=3, columns=7):
    """Independent NumPy int64 dot checks sampled outputs over the complete K axis."""
    if type(rows) is not int or type(columns) is not int or rows < 1 or columns < 1:
        raise ValueError("Projection reference requires positive integer row/column budgets")
    quantized, _scale = projection.integer_inputs(value)
    row_indices = np.unique(np.linspace(0, quantized.shape[0] - 1,
                                        min(rows, quantized.shape[0]), dtype=int))
    column_indices = np.unique(np.linspace(0, projection.out_features - 1,
                                           min(columns, projection.out_features), dtype=int))
    a = quantized.numpy()[row_indices].astype(np.int64)
    encoding = check_operand_encoding(value.detach().reshape(-1, projection.in_features).numpy()[row_indices],
                                      quantized.numpy()[row_indices], _scale.detach().reshape(-1, 1).numpy()[row_indices], projection.activation_scale)
    b = projection.weight_kn.numpy()[:, column_indices].astype(np.int64)
    expected = a @ b
    actual = torch._int_mm(quantized, projection.weight_kn).numpy()[np.ix_(row_indices, column_indices)]
    if not np.array_equal(actual.astype(np.int64), expected):
        raise AssertionError("Native integer contraction disagrees with independent int64 reference")
    return {"passed": True, "oracle": "numpy_int64_complete_k_sampled_rows_columns",
            "k": projection.in_features, "rows": row_indices.tolist(),
            "columns": column_indices.tolist(), "checked_elements": int(expected.size),
            "max_absolute_error": 0, "activation_encoding": encoding}


def language_inputs(path, tokenizer, safe_path, file_identity, activation_scale="maxabs"):
    """Freeze an attributed local corpus before any quality inference or quantization.

    JSON owns source/selection and documents [{id, text, source}]. This declaration
    and its content hash are recorded; attribution truth is not automatically proven.
    The corpus cannot silently collapse into the short frontend diagnostic.
    """
    path = safe_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if (not isinstance(payload.get("source"), str) or not payload["source"].strip()
            or not isinstance(payload.get("selection"), str) or not payload["selection"].strip()
            or not isinstance(documents, list) or len(documents) < QUALITY_CONTRACT["minimum_documents"]):
        raise ValueError("Language quality requires an attributed corpus with at least eight documents")
    ids, digests, tokens, records = set(), set(), [], []
    count = QUALITY_CONTRACT["tokens_per_document"]
    for document in documents:
        if (not isinstance(document, dict) or not isinstance(document.get("id"), str)
                or not document["id"].strip() or document["id"] in ids
                or not isinstance(document.get("source"), str) or not document["source"].strip()
                or not isinstance(document.get("text"), str) or not document["text"].strip()):
            raise ValueError("Language quality documents require unique IDs, text and attribution")
        encoded = tokenizer(document["text"], return_tensors="pt", add_special_tokens=True)["input_ids"]
        if (encoded.ndim != 2 or encoded.shape[0] != 1 or encoded.shape[1] < count
                or encoded.dtype != torch.int64 or bool((encoded < 0).any())
                or bool((encoded >= 32000).any())):
            raise ValueError("Each language quality document must provide at least 128 valid token IDs")
        selected = encoded[:, :count].contiguous()
        selected_identity = identity(selected)
        if selected_identity["sha256"] in digests:
            raise ValueError("Language quality documents must contain distinct selected token sequences")
        ids.add(document["id"])
        digests.add(selected_identity["sha256"])
        tokens.append(selected)
        records.append({"id": document["id"], "source": document["source"],
                        "text_sha256": hashlib.sha256(document["text"].encode()).hexdigest(),
                        "token_ids": selected.tolist(), "token_identity": selected_identity})
    contract = {**QUALITY_CONTRACT, "corpus": file_identity(path, safe_path),
                "corpus_source": payload["source"], "selection_declaration": payload["selection"],
                "documents": records, "calibration": "none_dynamic_activation_recipe",
                "quantization_recipe": recipe(activation_scale)}
    if payload.get("dataset_artifact") is not None:
        selected = payload["dataset_artifact"]
        if (not isinstance(selected, dict) or set(selected) != {"path", "sha256"}
                or file_identity(safe_path(selected["path"]), safe_path) != selected):
            raise ValueError("Language corpus source dataset identity differs from its preparation")
        contract["dataset_artifact"] = selected
    contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return tokens, contract


def language_reference(lm, token_documents):
    """Retain only next-token NLL/top1 labels, avoiding full vocabulary trajectories."""
    result = []
    with torch.no_grad():
        for tokens in token_documents:
            logits = lm(input_ids=tokens, use_cache=False).logits[:, :-1, :].float()
            labels = tokens[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.flatten(), reduction="none")
            result.append({"nll": nll.detach().clone(), "top1": logits.argmax(-1).detach().clone()})
    return result


def language_quality(lm, token_documents, reference, contract):
    closure = {key: value for key, value in contract.items() if key != "sha256"}
    if hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest() != contract["sha256"]:
        raise ValueError("Frozen language quality contract changed before evaluation")
    if len(token_documents) != len(reference) or len(reference) != len(contract["documents"]):
        raise ValueError("Language quality reference must cover the complete frozen document list")
    candidate = language_reference(lm, token_documents)
    differences, agreements, records = [], [], []
    for document, expected, actual in zip(contract["documents"], reference, candidate):
        difference = actual["nll"] - expected["nll"]
        agreement = actual["top1"] == expected["top1"]
        if not bool(torch.isfinite(difference).all()):
            raise AssertionError("Language quality produced nonfinite likelihoods")
        differences.append(difference)
        agreements.append(agreement.flatten())
        records.append({"id": document["id"], "predicted_tokens": difference.numel(),
                        "reference_mean_nll": float(expected["nll"].mean()),
                        "integer_mean_nll": float(actual["nll"].mean()),
                        "next_token_top1_agreement": float(agreement.float().mean())})
    increase = float(torch.cat(differences).mean())
    agreement = float(torch.cat(agreements).float().mean())
    return {"evaluated": True, "contract_sha256": contract["sha256"],
            "predicted_tokens": sum(item.numel() for item in differences),
            "mean_nll_increase_nats": increase, "perplexity_ratio": math.exp(increase),
            "next_token_top1_agreement": agreement, "documents": records,
            "passed": increase <= QUALITY_CONTRACT["maximum_mean_nll_increase_nats"]
            and agreement >= QUALITY_CONTRACT["minimum_next_token_top1_agreement"],
            "paper_language_quality": False}


def operation_placement():
    return {"learned_projections": {"count": 155, "arithmetic": "int8_int8_int32",
                                    "placement": "gemmini_candidate"},
            **{name: {"placement": "cpu", "arithmetic": arithmetic}
               for name, arithmetic in (
                   ("token_embedding", "int64_gather_float32_weights"),
                   ("rms_norm", "float32"), ("rope", "float32_sin_cos_and_matmul"),
                   ("qk_contractions", "float32_batched_matmul"),
                   ("attention_mask", "boolean_int64_float32"),
                   ("softmax", "float32"), ("pv_contractions", "float32_batched_matmul"),
                   ("silu_gate_residual", "float32"),
                   ("activation_quantization_dequantization", "float32_round_clip_int8_cast"),
                   ("kv_state_update", "float32_functional_scatter"),
                   ("positions_and_token_selection", "int64"))}}


def memory_plan(lm, records, prefill_tokens, decode_tokens):
    """Actual constant bytes plus explicit deployment scenarios; never a peak/fit claim."""
    constants = sum(value.numel() * value.element_size() for value in lm.state_dict().values())
    packed = sum(item["packed_weight"]["bytes"] for item in records)
    scales = sum(item["weight_scale"]["bytes"] for item in records)
    capacity = prefill_tokens + decode_tokens
    cache = 2 * 22 * 4 * capacity * 64 * 4
    positions = 22 * 8
    io = {"prefill_input_tokens": prefill_tokens * 8, "decode_input_token": 8,
          "returned_logits": 32000 * 4, "simultaneous_kv_input_output": 2 * cache,
          "simultaneous_position_input_output": 2 * positions}
    return {"kind": "constant_and_session_io_lower_bound_not_target_peak",
            "constants_shared_once_bytes": constants,
            "constants_duplicated_two_stage_bytes": 2 * constants,
            "int8_packed_weights_bytes": packed, "float32_weight_scales_bytes": scales,
            "unquantized_constants_bytes": constants - packed - scales,
            "packing": "KN_contiguous_one_byte_no_extra_device_copy_in_this_plan",
            "one_kv_state_bytes": cache, "kv_capacity": capacity, "io_bytes": io,
            "shared_constants_plus_io_lower_bound_bytes": constants + sum(io.values()),
            "duplicated_constants_plus_io_lower_bound_bytes": 2 * constants + sum(io.values()),
            "missing_peak_terms": ["graph_workspace", "packing_copies_if_required",
                                   "linked_image", "stack", "platform_reservations", "fixtures"],
            "stage_constant_sharing_implemented": False, "actual_platform_fit": "unqualified",
            "whole_model_spike": "pending_runtime_and_static_export_feasibility"}
