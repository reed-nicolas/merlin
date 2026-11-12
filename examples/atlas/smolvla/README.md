# SmolVLA flow-loss forward

This example takes the SmolVLA flow-matching loss forward (`lerobot/smolvla_base`, `atlas_v1_awq_tile_static`, one camera) from [atlas-torchao](https://github.com/ucb-ee290c/atlas-torchao) through the same route as the TinyLlama whole-model example: a live capture through model2MLIR's `atlas_fp8` adapter, `stage-capture`, and `native-compile --application`. Weights, codes, and exponents come unchanged from the export.

Create the bundle with atlas-torchao's own commands (`docs/algos/howto-model-to-fx-graph.md`, section 8): `eval_policy.py --export-only --algo atlas_v1_awq_tile_static`, then `fx_export.py --boundary smolvla_flow_loss --n-images 1`, then `scripts/tile_flow/verify_bundle.py`. The loader reads the export for the weights and the bundle's `fx/` only for its recorded fixture inputs; it never loads the saved graph.

`adapter_loader.py` rebuilds the policy from the raw export as `fx_export.py` does and hands `fx_export.SmolVLAFlowLoss` to `atlas_mlir_native.capture.prepare_model`, which re-exports it under the adapter. Before that it keeps only source state: decoded weights become payload-free placeholders, the four unread `rotary_emb` buffers that the export does not carry are removed, and the quantized `lm_head` that the bundle declares uncalled is detached. A strict re-export fails if any removed object is read.

SmolVLM's vision tower builds an all-true patch mask for its fixed image size, so its position ids depend on no input; exported directly they have a data-dependent extent. The loader computes them once with the unchanged module, binds them as one constant, and checks the embedding and the whole boundary output bit for bit before capture.

Set `ATLAS_EXPORT_DIR`, `ATLAS_FX_BUNDLE`, `ATLAS_EXPECTED_CONTRACT_SHA256`, and `ATLAS_EXPECTED_POLICY_SHA256`, optionally `ATLAS_FIXTURE_CASE` and `ATLAS_SOURCE_OUT` (which receives `inputs.safetensors` and `source.json`), then run the capture worker from the whole-model README with `--loader examples/atlas/smolvla/adapter_loader.py`. The interpreter needs lerobot 0.6.1 with torchvision and the `atlas_fp8` adapter installed.
