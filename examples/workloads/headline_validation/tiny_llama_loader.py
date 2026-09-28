"""Provenance-bearing TinyLlama prefill loader for headline compiler checks.

The model and inputs still come from model2MLIR's workload loader.  This adapter
only records what that loader's ordinary (non-session) path otherwise omits:
the selected cached checkpoint, its complete architecture, and the synthetic
input scope.  It must run in the selected model2MLIR interpreter with that
checkout on ``sys.path`` (the Merlin capture worker supplies ``--m2m-dir``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from workloads.tiny_llama import loader as model2mlir_loader


def get_model_and_inputs():
    model, inputs = model2mlir_loader.get_model_and_inputs()
    if callable(getattr(model, "external_runtime_session", None)):
        # The explicit prefill/decode session already declares its own provenance.
        return model, inputs

    truncated = bool(os.environ.get("M2M_LLAMA_LAYERS"))
    provenance = {
        "capture_scope": "single_prefill_forward",
        "input_source": "seeded_synthetic_token_ids",
        "synthetic_inputs": True,
        "paper_ready": False,
    }
    if truncated:
        provenance.update(checkpoint="random_init", full_checkpoint=False)
    else:
        from transformers.utils.hub import cached_file

        checkpoint = model2mlir_loader._MODEL_ID
        config_file = Path(cached_file(checkpoint, "config.json", local_files_only=True))
        weights_file = Path(cached_file(checkpoint, "model.safetensors", local_files_only=True))
        if config_file.parent != weights_file.parent or config_file.parent.parent.name != "snapshots":
            raise RuntimeError("TinyLlama config and weights do not resolve to one cached snapshot")
        revision = config_file.parent.name
        config = json.loads(config_file.read_text(encoding="utf-8"))
        loaded = model.lm.config
        if (
            getattr(loaded, "_commit_hash", None) != revision
            or getattr(loaded, "num_hidden_layers", None) != config.get("num_hidden_layers")
        ):
            raise RuntimeError("TinyLlama loaded model does not match the selected cached snapshot")
        provenance.update(
            checkpoint=checkpoint,
            checkpoint_revision=revision,
            full_checkpoint=True,
        )
    model.session_provenance = provenance
    return model, inputs


def get_session_spec(model, inputs):
    return model2mlir_loader.get_session_spec(model, inputs)
