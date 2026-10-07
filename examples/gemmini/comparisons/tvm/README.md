# Gemmini: TVM comparison plan

The [isolated host setup](HOST_SETUP.md) builds LLVM 18.1.8 and the pinned TVM checkout with an explicit guard for unavailable automatic-copy lowering. The host/guard checks and ten synthetic PyTorch → ONNX → Relax CPU cases passed. The importer ablation confirms the inherited opset-18 reduction-axis fix is needed for RMSNorm. Full-model and Gemmini execution work remain pending.

The YAML files and model selections below are planning inputs, with no executable schema or runner. They do not establish compilation, numerical correctness, hardware execution or timing. Paths in YAML use the Merlin checkout as the base unless an explicit environment reference is present; environment references are documentation, not automatically expanded inputs.

[target.yaml](target.yaml) records the proposed TVM source and unresolved implementation choices. The planned validation models are listed below. The existing [target descriptor](../../target/descriptor.yaml) remains the target setup reference; its selections are not automatically accepted for this comparison.

| Model | Validation scope | Required stages |
| --- | --- | --- |
| Canonical ResNet-50 (`resnet50`), first | Full model; exact variant/checkpoint pending | — |
| TinyLlama 1B (`tiny_llama`), second | Full application; exact checkpoint pending | Prefill, decode |
| SmolVLA (`smolvla`) | Full application | Prefix, denoising, action decode |

For every model, source, checkpoint, input dataset, capture, quantization, calibration, host partition, numerical acceptance and measurement boundary remain unselected.

The repository starting point is Apache TVM v0.19.0 plus the five UCB frontend fixes retained on `ucb-bar/tvm` branch `merlin/relax-onnx-fixes`. The independent fork checkout is `third_party/baselines/tvm-gemmini`, with `gemmini/bringup` for new work. See the [dependency setup](../../../../third_party/baselines/README.md) for remotes and pins. Jack selected Gemmini's C operator library for baseline fairness. Stock int8 Gemmini with a Linux host and Relax VM remains a proposal pending platform verification. Hardware revision/configuration, generated headers, host runtime and instruction-loop policy remain explicit inputs.

The model order is canonical ResNet50, TinyLlama 1B, then SmolVLA. Final timing must come from FireSim; Agustin/Jack will coordinate lab access and timing capture when deployment is ready. Functional checks and host CPU results do not supply these final numbers.

The [older microTVM port](https://github.com/apache/tvm/pull/13770) was closed without merging. Its [pinned implementation](https://github.com/fzi-peccia/tvm/tree/3b07a14bb936be64115403d839f5ef002ca8102d) uses TFLite/Relay, CRT/AOT and Gemmini C calls. Reuse its call construction, layout/quantization handling and deployment examples selectively after checking current contracts. It is not a Relax backend or a verified implementation of these three workloads; no merge or revision change is planned.

Keep compiler lowering, hardware configuration, runtime, quantization/calibration, model inputs and measurement boundaries explicit for each compared implementation. Full checkpoint and input provenance, host fallback coverage and numerical thresholds must be selected before execution. Full validation models stay separate from compiler development and tuning; partial models or isolated stages are diagnostics.

A future thin TVM adapter belongs at `packages/merlin-analysis/src/merlin/baselines/tvm_gemmini.py`, with target choices supplied by configuration and substantive lowering/runtime integration owned by the TVM fork. No placeholder module is supplied. Captures, weights, binaries and receipts belong beneath the configured `out/` roots, not here. Do not infer compatibility or qualification from repository checkout or branch setup.
