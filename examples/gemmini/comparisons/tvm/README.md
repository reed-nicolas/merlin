# Gemmini: TVM comparison plan

The [isolated host setup](HOST_SETUP.md) now builds LLVM 18.1.8 and the pinned TVM checkout with an explicit guard for unavailable automatic-copy lowering. The synthetic Relax CPU matmul and guard-rejection checks passed. Host requirements and CMake configuration are provided here; model frontend and Gemmini execution work remain pending.

The YAML files and model selections below are planning inputs, with no executable schema or runner. They do not establish compilation, numerical correctness, hardware execution or timing. Paths in YAML use the Merlin checkout as the base unless an explicit environment reference is present; environment references are documentation, not automatically expanded inputs.

[target.yaml](target.yaml) records the proposed TVM source and unresolved implementation choices. The planned validation models are listed below. The existing [target descriptor](../../target/descriptor.yaml) remains the target setup reference; its selections are not automatically accepted for this comparison.

| Model | Validation scope | Required stages |
| --- | --- | --- |
| ResNet-50 (`resnet50`) | Full model | — |
| TinyLlama (`tiny_llama`) | Full application | Prefill, decode |
| SmolVLA (`smolvla`) | Full application | Prefix, denoising, action decode |

For every model, source, checkpoint, input dataset, capture, quantization, calibration, host partition, numerical acceptance and measurement boundary remain unselected.

The repository starting point is Apache TVM v0.19.0 plus the five UCB frontend fixes retained on `ucb-bar/tvm` branch `merlin/relax-onnx-fixes`. The independent fork checkout is `third_party/baselines/tvm-gemmini`, with `gemmini/bringup` for new work. See the [dependency setup](../../../../third_party/baselines/README.md) for remotes and pins. Stock int8 Gemmini with a Linux host and Relax VM is a recommendation pending platform verification and paper-scope choices. Stock versus MX hardware, hardware revision, loop policy, and C-library versus handwritten xDSL lowering remain unresolved.

Keep compiler lowering, hardware configuration, runtime, quantization/calibration, model inputs and measurement boundaries explicit for each compared implementation. Full checkpoint and input provenance, host fallback coverage and numerical thresholds must be selected before execution. Full validation models stay separate from compiler development and tuning; partial models or isolated stages are diagnostics.

A future thin TVM adapter belongs at `packages/merlin-analysis/src/merlin/baselines/tvm_gemmini.py`, with target choices supplied by configuration and substantive lowering/runtime integration owned by the TVM fork. No placeholder module is supplied. Captures, weights, binaries and receipts belong beneath the configured `out/` roots, not here. Do not infer compatibility or qualification from repository checkout or branch setup.
