"""Full, locally supplied TinyLlama prefill/decode session for verify_session.py.

The checkpoint is the official unmodified BF16 safetensors artifact at the pinned
revision below. ``precision="float32"`` explicitly converts those trained weights
for a CPU frontend diagnostic. It is not original-precision or model-quality
qualification. No model artifacts are downloaded and no layer count is changed.
"""

import hashlib
import json
import sys
from verify_resnet import file_identity, safe_path


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
WEIGHT_SHA256 = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
MODEL_SOURCE = f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}"
GEOMETRY = {"model_type": "llama", "num_hidden_layers": 22,
            "hidden_size": 2048, "intermediate_size": 5632,
            "num_attention_heads": 32, "num_key_value_heads": 4,
            "vocab_size": 32000, "tie_word_embeddings": False}
MODEL_SEMANTICS = {**GEOMETRY, "hidden_act": "silu", "attention_bias": False,
                   "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
                   "rope_theta": 10000.0, "rope_scaling": None,
                   "bos_token_id": 1, "eos_token_id": 2, "pretraining_tp": 1}


def checkpoint_artifacts(checkpoint_dir):
    """Validate only the explicit local artifact set before Transformers can read it."""
    root = safe_path(checkpoint_dir)
    identities = {}
    for name in ("config.json", "model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json"):
        path = safe_path(root / name)
        if not path.is_file():
            raise FileNotFoundError(f"Required full TinyLlama artifact is absent: {path}")
        identities[name] = {**file_identity(path, safe_path), "bytes": path.stat().st_size}
    if identities["model.safetensors"]["sha256"] != WEIGHT_SHA256:
        raise ValueError("TinyLlama weights differ from the pinned official full checkpoint")
    config = json.loads(safe_path(root / "config.json").read_text())
    if any(config.get(name) != value for name, value in MODEL_SEMANTICS.items()):
        raise ValueError("TinyLlama config differs from the complete pinned model geometry")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise ValueError("Pinned TinyLlama checkpoint configuration must declare bfloat16")
    # Transformers can optionally read these; record them when present and validate
    # their paths before the library sees the enclosing directory.
    for name in ("generation_config.json", "tokenizer.model", "added_tokens.json"):
        path = safe_path(root / name)
        if path.is_file():
            identities[name] = {**file_identity(path, safe_path), "bytes": path.stat().st_size}
    return root, identities


def validate_reference_stages(capture, causal, torch):
    """Check functional cache wrappers against the original full-sequence HF model."""
    capacity = capture.prefill_tokens + capture.decode_tokens
    prefill = causal.FixedCachePrefillStage(capture.lm, capacity).eval()
    decode = causal.FixedCacheDecodeStep(capture.lm, capacity).eval()
    errors = []
    # BF16 is a separate explicit reference option; this CPU ONNX verifier uses FP32.
    rtol = atol = 1e-4 if next(capture.lm.parameters()).dtype == torch.float32 else 0.05
    with torch.no_grad():
        logits, cache, position = prefill(capture.tokens[:, :capture.prefill_tokens])
        for length in range(capture.prefill_tokens, capacity + 1):
            full_logits = capture.lm(
                input_ids=capture.tokens[:, :length], use_cache=False).logits[:, -1, :]
            torch.testing.assert_close(logits, full_logits, rtol=rtol, atol=atol)
            expected_position = torch.full_like(position, length)
            if not torch.equal(position, expected_position):
                raise AssertionError("Functional TinyLlama cache used lengths are incorrect")
            if bool(cache[..., length:, :].count_nonzero()):
                raise AssertionError("Functional TinyLlama cache padding is not zero")
            errors.append({"tokens": length, "max_absolute_logit_error":
                           float((logits - full_logits).abs().max()),
                           "position": position.tolist(), "unused_cache_slots_zero": True})
            if length < capacity:
                before_cache, before_position = cache.clone(), position.clone()
                logits, next_cache, next_position = decode(
                    capture.tokens[:, length:length + 1], cache, position)
                if not torch.equal(cache, before_cache) or not torch.equal(position, before_position):
                    raise AssertionError("Functional TinyLlama decode mutated caller state")
                torch.testing.assert_close(next_cache[..., :length, :], cache[..., :length, :],
                                           rtol=0, atol=0)
                cache, position = next_cache, next_position
    return {"reference": "original_full_sequence_hf_use_cache_false", "passed": True,
            "rtol": rtol, "atol": atol, "checks": errors}


def get_model_and_inputs(*, checkpoint_dir, model2mlir_root, corpus_path,
                         prefill_tokens=8, decode_tokens=3, precision="float32",
                         cpu_threads=2):
    """Build compiled prefill plus teacher-forced recurrent decode on actual text.

    ``corpus_path`` is a local UTF-8 text file tokenized with this checkpoint's
    tokenizer. All selected IDs are recorded. Short inputs establish session and
    frontend semantics only; ``paper_ready`` remains false.
    """
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Precision must be explicitly float32 or bfloat16")
    if (type(prefill_tokens) is not int or type(decode_tokens) is not int
            or prefill_tokens < 1 or decode_tokens < 2):
        raise ValueError("Require a positive prefill and at least two recurrent decode steps")
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
        raise ValueError("cpu_threads must be between 1 and 8")
    if prefill_tokens + decode_tokens > 2048:
        raise ValueError("Session exceeds the checkpoint context capacity")
    checkpoint, artifacts = checkpoint_artifacts(checkpoint_dir)
    corpus = safe_path(corpus_path)
    text = corpus.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Token corpus must contain actual text")
    corpus_identity = file_identity(corpus, safe_path)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import cache_utils
    from verify_session import load_module, load_protocol

    if not isinstance(getattr(cache_utils.StaticLayer(prefill_tokens + decode_tokens),
                              "cumulative_length", None), torch.Tensor):
        raise RuntimeError("The selected Transformers cache API is incompatible with m2m; use the prepared Transformers 5.4.0 environment")
    torch.set_num_threads(cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False, use_fast=True)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    capacity = prefill_tokens + decode_tokens
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < capacity:
        raise ValueError(f"Actual token corpus must contain at least {capacity} token IDs")
    tokens = tokens[:, :capacity].contiguous()
    if tokens.dtype != torch.int64 or bool((tokens < 0).any()) or bool((tokens >= 32000).any()):
        raise ValueError("Tokenizer emitted an invalid TinyLlama token ABI")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[precision]
    lm, loading = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=dtype, attn_implementation="eager",
        output_loading_info=True)
    if any(loading.get(name) for name in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Full TinyLlama checkpoint did not load exactly")
    if any(getattr(lm.config, name, None) != value for name, value in MODEL_SEMANTICS.items()
           if name not in ("rope_theta", "rope_scaling")):
        raise ValueError("Loaded TinyLlama model geometry differs from the pinned checkpoint")
    rope = getattr(lm.config, "rope_parameters", None)
    if rope is None:
        valid_rope = lm.config.rope_theta == 10000.0 and lm.config.rope_scaling is None
    else:
        valid_rope = (isinstance(rope, dict) and rope.get("rope_type") == "default"
                      and rope.get("rope_theta") == 10000.0
                      and rope.get("partial_rotary_factor", 1.0) == 1.0)
    if not valid_rope:
        raise ValueError("Loaded TinyLlama RoPE semantics differ from the pinned checkpoint")
    if any(value.dtype != dtype or value.device.type != "cpu" for value in lm.parameters()):
        raise ValueError("Loaded model violates explicit CPU precision selection")
    lm.eval().requires_grad_(False)
    model_root = safe_path(model2mlir_root)
    load_protocol(model_root)
    causal_path = safe_path(model_root / "m2m/capture/causal_session.py")
    causal = load_module("m2m.capture.causal_session", causal_path)
    provenance = {
        "checkpoint": MODEL_ID, "checkpoint_revision": REVISION,
        "checkpoint_source": MODEL_SOURCE, "full_checkpoint": True,
        "checkpoint_artifacts": artifacts, "checkpoint_tensor_dtype": "bfloat16",
        "execution_precision": precision,
        "weight_conversion": "bfloat16_to_float32" if precision == "float32" else "none",
        "reference_precision": precision, "reference_role": "compiler_correctness",
        "quality_reference": "not_evaluated", "synthetic_tokens": False,
        "token_source": "local_utf8_corpus_with_pinned_checkpoint_tokenizer",
        "corpus": corpus_identity, "token_ids": tokens.tolist(),
        "token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "causal_session_source": file_identity(causal_path, safe_path),
        "transformers_version": transformers.__version__,
        "transformers_cache_source": file_identity(cache_utils.__file__, safe_path),
        "transformers_model_source": file_identity(sys.modules[type(lm).__module__].__file__, safe_path),
        "tokenizer_version": tokenizer.__class__.__name__,
        "decode_policy": "teacher_forced_actual_tokens",
        "parameter_count": sum(value.numel() for value in lm.parameters()),
        "parameter_bytes": sum(value.numel() * value.element_size() for value in lm.parameters()),
        "kv_cache_bytes": 2 * 22 * 4 * capacity * 64 * torch.empty((), dtype=dtype).element_size(),
        "position_bytes": 22 * 8, "cpu_threads": cpu_threads,
    }
    capture = causal.CausalMultiProgramCapture(
        lm, tokens, checkpoint=MODEL_ID, full_checkpoint=True, paper_ready=False,
        provenance=provenance, prefill_tokens=prefill_tokens, decode_tokens=decode_tokens)
    capture.provenance["functional_cache_reference"] = validate_reference_stages(
        capture, causal, torch)
    return capture, ()


def get_verification_artifacts(model, _inputs):
    """Selected checkpoint/corpus/source closure for verifier pre/post file hashes."""
    provenance = model.provenance
    identities = [*provenance["checkpoint_artifacts"].values(),
                  provenance["corpus"], provenance["causal_session_source"],
                  provenance["transformers_cache_source"], provenance["transformers_model_source"]]
    return tuple(safe_path(identity["path"]) for identity in identities)
