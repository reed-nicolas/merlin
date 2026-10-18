---
title: Guarded egraph candidates and measured selection
kind: design
status: current
owner: core
last_verified: 2026-10-18
related: [architecture, compiler_plane]
code_refs: [src/merlin/semantic_compiler/egraph_terms.py, src/merlin/semantic_compiler/rewrite_selection.py, examples/k1_cpu/egglog_relu.py, merlin/tests/ir/test_egraph_terms.py, merlin/tests/ir/test_rewrite_selection.py, merlin/tests/ir/test_k1_egglog_relu.py]
---

# Guarded egraph candidates and measured selection

The optional Python egglog path shares immutable expression construction, bounded saturation, joint extraction, candidate costs and qualified measurement selection. Providers own their semantics, arithmetic equalities, guard evidence, decoding and executable checking. The existing Rust `egg` bridge is a separate dependency boundary; Python egglog does not replace it or inherit its certificates.

| Layer | Shared mechanism | Provider obligation |
| --- | --- | --- |
| Encoding | `Value.anchor`, `Value.apply`, ordered `Inputs` forests | Typed source validation, immutable anchor identities and explicit effect boundary |
| Rules | Rule registration and bounded fixed-point iteration | Exact equalities, numeric policy, guards and accepted caller contracts |
| Extraction | Joint greedy DAG extraction with a marginal-cost callback | Operation costs, materialization and reuse policy, decoder validation |
| Shortlist | DAG operation costs, pressure estimates and required controls | Declared resource demands, cost-fact provenance, actual allocation checking |
| Selection | Binary-bound receipts, complete target grids, regret comparison | Compilation, execution, independent oracle and guard checks |

State tokens, accumulator/weight handles, memory effects and mutable resources stay outside value eclasses. An anchor names a value across a provider's fixed boundary; it is not evidence that two loads or stateful computations are interchangeable. Providers must validate extracted anchor identities and dominance, reverify decoded IR, compile it and reject unavailable candidates. Shared extraction does not allocate registers or produce a legal schedule.

`egraph_terms.saturate` requires nonempty roots and positive work bounds. It stops only at a fixed point; an exhausted round budget or observed function-row budget raises instead of silently returning an incomplete result. The row limit is checked after each round and is not a hard allocation limit inside egglog. Providers also bound source size. `extract` processes the ordered root forest together so a shared producer is charged once. A supplied marginal cost is an extraction heuristic, not measured runtime.

`rewrite_selection.estimate_dag` consumes a topological materialization, charges named producers once and estimates overlapping live resources. Pressure is a soft shortlist penalty: actual target allocation and scheduling remain authoritative. The shortlist retains the native control and explicitly required hand controls before considering estimated scores. Duplicate sources normally collapse; an explicitly required comparison control may retain identical bytes, which should be reported as a tie rather than additional candidate diversity.

`select_measured` checks the provider's admission certificate before compilation and requires a measurement bound to the actual emitted program SHA-256. Output, independent-oracle and guard checks must each be literal `True`. Target, metric and evidence tier must match the caller's declaration. A numerical difference aborts the experiment. An explicit compile/infrastructure refusal is reported separately, and a candidate with an incomplete target grid cannot win. Selection uses the first recorded invocation as its score; callers must declare and justify their invocation policy and retain the other values.

The reusable modules contain no target names, arithmetic rules, numerical assumptions, instruction encodings, target latencies or regex parsers. Atlas's virtual-IR provider can supply raw-BF16 guarded rules, fixed state/resource reconstruction and external compiler/model/RTL adapters at this boundary. Its arithmetic and physical qualification remain with that provider. A successful generic extraction is not an Atlas numerical or hardware verdict.

## Second-provider diagnostic

`examples/k1_cpu/egglog_relu.py` is a handwritten example edge for the canonical CPU contract's scalar signed-int8 elementwise/ReLU capability. It is not a generated backend, K1 board runner, RVV implementation or scored device fallback. The example accepts an explicitly materialized JSON snapshot of the canonical YAML plus the raw YAML SHA-256; it validates the selected capability before building an expression.

For signed i8, define `R(x)=max(x,0)`. Its range is `[0,127]`, on which `R` is the identity, so `R(R(x))=R(x)` and induction gives `R^n(x)=R(x)` for every integer `n>=1`. The example exhaustively checks all 256 raw encodings and several repetition depths. It uses no floating-point identities, overflow behavior, sampled range inference, VLEN assumption or caller precondition. The egglog rule is restricted to the validated `i8` constructor vocabulary; unknown operations, anchors and types fail closed.

The native repeated-ReLU expression, a hand single-ReLU control and the actual shared-egglog extraction all pass through the same shared selector. Each materializes a scalar Python function and executes on the current host against an independent raw-sign-bit oracle, over every i8 encoding and additional tail lengths, with input/output guards and repeated invocations. The receipt metric counts actual calls to the instrumented scalar primitive. Its tier is `host_functional_cost`: this is an abstract operation-count comparison with host functional execution, not K1 wall time, device cycles, simulator timing or an RVV throughput estimate. A hand/egglog byte tie is retained explicitly. Point-in-time receipts and measurements belong outside durable docs.

## Survey of alternative target edges

The survey follows each target's canonical authored contract and ownership notes. It is static inspection of those inputs; historical measurements mentioned in a contract were not rerun by this design exercise.

| Target | Lowering and numerical surface | Executable readiness and qualification boundary |
| --- | --- | --- |
| Gemmini Universal | Weight-stationary int8 mesh, int32 accumulation with narrow int8 readout and target-defined scaling; full-width readout is forbidden | Canonical contract records hand-run functional/RTL evidence, but no reference compiler or registered runtime backend. The functional extension is blind to the defining readout restriction, so it cannot qualify that boundary. |
| K1 CPU | Scalar/RVV elementwise and reductions; integer and declared floating types; scalable vector length, with fixed VLEN requiring a same-environment runtime probe | Linux silicon wall-time authority is declared, with affinity/governor/temperature/warmup controls and unavailable userspace cycle counters. No board or probe is supplied here, so the example remains a host scalar diagnostic. |
| Muon | SIMT float32 plus matrix-unit block-scaled formats and int8-to-int32 contraction; target-defined requantization | Executable backend, introspection and Cyclotron support live in an explicitly selected OOT provider. Reference metadata supplies no executable plugins or new facts/IRDL qualification; cycle-model overcount caveats remain explicit. |
| Saturn | RVV int8/fp16/fp32 and revision-dependent int8 OPU command buffers; widening accumulation, residency and deterministic partition obligations | Executable support lives in a separate selected OOT provider. Reference discovery does not load it. OPU claims require the concrete hardware pin; model/vector metadata and pure import tests do not establish RTL execution. |
| Voyager accelerator | Int8 matrix contraction into int24, with separate BF16 vector egress and parameter-queue command buffers | Only a draft residual exists. No complete executable provider, oracle adapter or declared measurement authority; vector instruction semantics, memory model and several build/protocol facts remain unknown. A matmul cannot publish output without the vector path. |

K1's exact integer diagnostic avoids assuming missing device facts. Promoting it to K1 performance evidence requires the configured cross-toolchain, board probe, actual requested scalar/RVV mode, guarded tail execution and the canonical silicon measurement controls. Other target providers must supply their own similarly bounded semantic and executable evidence; shared infrastructure does not fill their unknowns.
