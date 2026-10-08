# Gemmini: TVM comparison plan

The [isolated host setup](HOST_SETUP.md) builds LLVM 18.1.8 and the pinned TVM checkout with an explicit guard for unavailable automatic-copy lowering. The host/guard checks, twenty synthetic frontend cases and the full ResNet50 v1.5 host structural diagnostic passed. Importer ablations establish several inherited fixes and exposed an int64-narrowing bug now corrected locally. Pretrained-model qualification and Gemmini execution remain pending.

The YAML files and model selections below are planning inputs, with no executable schema or runner. They do not establish compilation, numerical correctness, hardware execution or timing. Paths in YAML use the Merlin checkout as the base unless an explicit environment reference is present; environment references are documentation, not automatically expanded inputs.

[target.yaml](target.yaml) records the proposed TVM source and unresolved implementation choices. The planned validation models are listed below. The existing [target descriptor](../../target/descriptor.yaml) remains the target setup reference; its selections are not automatically accepted for this comparison.

| Model | Validation scope | Required stages |
| --- | --- | --- |
| Canonical ResNet-50 (`resnet50`), first | Full model; exact variant/checkpoint pending | — |
| TinyLlama 1B (`tiny_llama`), second | Full application; exact checkpoint pending | Prefill, decode |
| SmolVLA (`smolvla`) | Full application | Prefix, denoising, action decode |

For every model, source, checkpoint, input dataset, capture, quantization, calibration, host partition, numerical acceptance and measurement boundary remain unselected.

The existing ResNet loader is `model2MLIR/workloads/resnet50_v1_5/loader.py`, inspected at model2MLIR revision `7915e23475c6db446a3c404847b11e8bc72c8a27`. It constructs torchvision ResNet50 with `IMAGENET1K_V2` weights and consumes float32 NCHW `[1,3,224,224]` images. This identifies an available implementation; “canonical ResNet50” still needs an explicit variant before the paper workload is frozen.

The loader distinguishes attributed preprocessed image streams from seeded Gaussian diagnostic inputs, and records checkpoint/input hashes when executed. It defaults to a 256-image session; the current TVM driver exports a single forward. Neither a stored `_full` directory name nor that forward alone establishes the desired benchmark session. The old TVM/K1 driver also perturbs all-zero parameter tensors; reusing that behavior would change checkpoint identity. A new Gemmini adapter must preserve the selected checkpoint and record actual inputs, preprocessing and any quantization.

[verify_resnet.py](verify_resnet.py) now exports and inventories that full architecture with explicitly random-initialized weights and two synthetic images. Strict ONNX checking, Relax import/build and both CPU comparisons passed, with maximum absolute error `9.16e-5` at fixed `rtol=atol=1e-4`. The graph contains 53 convolutions, 16 residual adds, 49 ReLUs, pooling and one classifier GEMM; all inferred shapes/dtypes are concrete. Follow the [host recipe](HOST_SETUP.md#resnet50-host-structural-diagnostic) for reproduction and artifact details. This diagnostic does not select the paper variant or provide trained accuracy, quantized/device correctness or timing.

The repository starting point is Apache TVM v0.19.0 plus the five UCB frontend fixes retained on `ucb-bar/tvm` branch `merlin/relax-onnx-fixes`. The independent fork checkout is `third_party/baselines/tvm-gemmini`, with `gemmini/bringup` for new work. See the [dependency setup](../../../../third_party/baselines/README.md) for remotes and pins. Jack selected Gemmini's C operator library for baseline fairness. Stock int8 Gemmini with a Linux host and Relax VM remains a proposal pending platform verification. Hardware revision/configuration, generated headers, host runtime and instruction-loop policy remain explicit inputs.

The model order is canonical ResNet50, TinyLlama 1B, then SmolVLA. Final timing must come from FireSim; Agustin/Jack will coordinate lab access and timing capture when deployment is ready. Functional checks and host CPU results do not supply these final numbers.

The [older microTVM port](https://github.com/apache/tvm/pull/13770) was closed without merging. Its [pinned implementation](https://github.com/fzi-peccia/tvm/tree/3b07a14bb936be64115403d839f5ef002ca8102d) uses TFLite/Relay, CRT/AOT and Gemmini C calls. Reuse its call construction, layout/quantization handling and deployment examples selectively after checking current contracts. It is not a Relax backend or a verified implementation of these three workloads; no merge or revision change is planned.

## First C-library integration contract

Start with a dense row-major `A[M,K]: int8`, `B[K,N]: int8` → `C[M,N]: int32` matmul, with no bias, activation, transpose or requantization. This is a proposed stock-integer kernel contract, not yet an implemented backend. Verify signed 8-bit `elem_t`, signed 32-bit `acc_t`, integer parameter flags and matching hardware before compiling it.

The inspected `tiled_matmul_auto` interface accepts M/N/K, A/B, null bias and C; element strides K/N/N/N; generated-header identity scales; `NO_ACTIVATION`; `full_C=true`; `low_D=false`; and the selected dataflow. The WS path calls `gemmini_loop_ws`, so it requires an explicit decision on that instruction path. Keep bias, output scaling and convolution fusion out of this first numerical case, then validate them separately.

The local library inspection used Gemmini revision `69a1c0383283d6ac95c99d2e53136fef4b0bb67e` and gemmini-rocc-tests revision `c4ed2e722e10ce33fadbcc44f4ffa6d6bba840fc`. Its clean `include/gemmini_params.h` declares unsigned 8-bit elements, low-precision floating-point flags and unsigned 64-bit accumulators. That header is incompatible with the proposed signed-int8/int32 buffers; do not substitute integer typedefs or assume it matches the intended FireSim build. Obtain the actual hardware configuration and generated headers together.

The library's CPU branch does not support `full_C`: it warns and continues through an element-width output cast. Use an independent bounded int64 reference for expected integer results. Cover negative operands, rectangular shapes, padded strides, partial array tiles and K accumulation across tiles. The accelerator path emits a final fence, but completion, compiler ordering and memory visibility still need validation on the chosen runtime. Source inspection does not establish those execution properties.

Next: bind the matching library/header/configuration and guest ABI, implement the checked wrapper and Relax call binding, cross-compile, verify generated accelerator instructions, then run numerical cases on the functional platform and FireSim. Model quantization and canonical ResNet50 checkpoint selection can proceed independently.

Keep compiler lowering, hardware configuration, runtime, quantization/calibration, model inputs and measurement boundaries explicit for each compared implementation. Full checkpoint and input provenance, host fallback coverage and numerical thresholds must be selected before execution. Full validation models stay separate from compiler development and tuning; partial models or isolated stages are diagnostics.

A future thin TVM adapter belongs at `packages/merlin-analysis/src/merlin/baselines/tvm_gemmini.py`, with target choices supplied by configuration and substantive lowering/runtime integration owned by the TVM fork. No placeholder module is supplied. Captures, weights, binaries and receipts belong beneath the configured `out/` roots, not here. Do not infer compatibility or qualification from repository checkout or branch setup.
