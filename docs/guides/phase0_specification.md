---
title: Defining and inspecting Phase 0 inputs
kind: guide
status: current
owner: targetgen
last_verified: 2026-09-27
related: [generating_capsules, adding_a_target, integrations]
code_refs:
  - src/merlin/targetgen/software_spec.py
  - src/merlin/targetgen/rtl/circt_introspect.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/evidence.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/generation.py
---

# Define the software contract before generating tests

Phase 0 combines selected hardware evidence, a software specification, and workload/test
policy. See the matching [Atlas](../../examples/atlas/phase0/README.md) and
[Gemmini](../../examples/gemmini/phase0/README.md) examples. The installed generator belongs
to `merlin-experiments`; target-specific inputs belong to examples or selected OOT support.

## Start with five decisions

Write what software may rely on, not everything you know about the accelerator:

1. Which operations run on the accelerator, which are fused, and which need the host?
2. What inputs are legal: precision, rank, layout, tails, broadcasting and aliasing?
3. What arithmetic must results obey: accumulation, rounding, overflow and exceptional values?
4. Which quantization formats apply to which operations, and with what scale/zero-point rules?
5. What crosses between host and accelerator, and is it copied or converted?

Start with one operation instead of promising an entire framework. For example, this is
one Gemmini contraction declaration inside `operations`:

```yaml
operations:
  contraction:
    placement: accelerator
    operand_dtypes: [int8]
    accumulator_dtype: i32
    ranks: [2, 3, 4]
    layouts: [row_major_contiguous]
    tails: [zero_pad_valid_window]
    broadcasting: [none, independent_batches]
    aliasing: [disjoint_inputs_outputs]
```

This is a partial starter, not a standalone SW spec or a certificate. Add the schema,
target and explicit numerical semantics; define quantization and transfers
before claiming those paths. Missing declarations are not inherited from this snippet or
guessed from an instruction name. For Gemmini, i32 readout does not make the internal i20
partial sum an i32 computation. Copy the matching full [Gemmini spec](../../examples/gemmini/target/software-spec.yaml)
or [Atlas spec](../../examples/atlas/target/software-spec.yaml) as a worked example, not as
defaults for another target. Omitted review status means **unreviewed**, never admitted.
Add reviewed declarations only when independent evidence supports exactly what you wrote.

Use the operation's name once. A shared family name such as `contraction` or `movement`
selects that family; a custom name must explicitly name its `ops` or `families`. Named
groups are normalized to the same internal operation rows as the legacy list syntax.
Transfer names work the same way:

```yaml
transfer_contracts:
  operand_load:
    from: host
    to: accelerator
    copy: {dtype: int8, layout: row_major_contiguous}
```

`copy` explicitly means a bit-preserving transfer with the same dtype and layout
at both endpoints; it does not quantize, cast, transpose or dispatch. Add
`valid_window: required` when that is part of the software contract. Type-changing
transfers retain expanded `signature`/`semantics` declarations with explicit
source/result types and conversion rules. Do not mix compact and expanded fields
in one declaration. Both forms produce the same canonical consumer representation
in generated `software/software-spec.json`; the exact authored bytes are also saved.

No placeholder review/evidence fields are needed to author this unreviewed transfer.
The backend owns its physical storage and transport. Software-visible conversion,
layout and valid-window requirements remain explicit. Reviewed transfer claims still
require explicit evidence; missing or explicitly unqualified evidence cannot grant
admission. Quantization format IDs may also be omitted: a stable dtype-pair identity
is generated. Author eligible operations and residual scale/zero-point rules that
the selected readout cannot establish. Do not copy derived tensor mode,
granularity or symmetric zero-point values into this file. Inspect their basis
and remaining unknowns in the generated quantization contract: omission never
proves that an unknown format became supported.

## What is the SW spec?

`target/software-spec.yaml` is a versioned `merlin.software_spec.v1` declaration of
software-visible behavior. It is not a second handwritten hardware geometry table.

| Input | Define explicitly | Do not treat it as |
| --- | --- | --- |
| SW spec | Operation/signature constraints; layouts, tails, broadcasting, aliasing, placement, selected numeric behavior, quantization eligibility and transfer constraints | Automatically proven by an instruction name or storage width |
| Selected OOT provider/backend config | Runtime implementation, ISA vocabulary/protocol ownership, extraction anchors and callable references | A second mandatory software spec to hand-maintain |
| Hardware selection | Which evidence is required and which source/configuration is selected | A capability declaration or certificate |
| Extracted RTL facts | Array and memory geometry, interfaces, observed decoder fields, datatype evidence and structural timing where established | Complete operation latency, numerical behavior, endpoint kind or software legality |
| Recipe/workload policy | Application roster, semantic seeds, tolerances, oracle tiers, holdouts and performance objectives | Hardware facts or generated capsules |

The authored spec holds `operations`, `numerical_semantics`, `quantization` and
`transfer_contracts`. Review status defaults safely to unreviewed. Operation rows require placement
and executable typed constraints. Use semantic `families` for a shared class, or exact
`ops` selectors when behavior must be operation-specific; do not duplicate both lists.
Numerical semantics select an independent model and its rounding/reduction policy,
never a target-name default. Generated source audits, test counts, qualification hashes
and long diagnostic reports belong in artifacts, not this YAML.

The Atlas and Gemmini examples use directly named operations and transfers; authors
do not need IDs, `signature` wrappers or duplicate copy endpoint constraints.
Expanded v1 list declarations remain readable for compatibility and generated
inspection, but are not the starter format. The examples do not
repeat mesh/memory geometry, opcode maps, calibration coefficients, historical runs or
backend code. Gemmini explicitly constrains the internal i20 partial-sum domain even
though its software readout is i32. Atlas explicitly constrains the finite-normal FP8
and BF16 domain; unknown block-size/scale parameters remain unknown. Host capabilities
are a separate manifest bound to the selected host compiler, not accelerator facts.

Version 1 remains backwards compatible: an old inline `capability_contract` and
`evidence` mapping are optional legacy fields. New minimal specs require a separately
selected same-target OOT provider or explicit backend-config contract. The pure
`capability_contract(spec, base_contract=selected_backend)` projection preserves that
selected policy without filesystem lookup or guessed protocol. Inline legacy declarations
retain their recorded precedence. The exact backend/config bytes are independently
frozen, and its broader capability list cannot bypass typed SW admission.

When reviewing a backend declaration separately from its implementation,
`MERLIN_TARGET_CONTRACT=/selected/target_contract.yaml` explicitly selects that
declaration while `MERLIN_TARGET_PATH` still selects the OOT support code. Phase 0
records both source selections and refuses a different target identity. Frozen
execution consumes the saved contract bytes, not the live override. Selecting a
new declaration does not prove that the older implementation satisfies it.

Unknowns remain explicit. Review operation legality, exact numeric behavior, quantization
granularity/zero points and ABI ordering before changing a spec's status. A width alone
does not identify a floating-point format. “Not extracted by the current reader” does not
mean “impossible to infer from RTL”: first inspect the available elaboration and extraction
coverage, then document the residual declaration and its independent evidence.

The generated `funct_decode_table` is a historical field name, not a promise of a full
instruction set. Its `scope: observed_decode_field` and `complete_isa: false` mean that
the listed equalities apply only to the field the RTL compared. That field may concatenate
nonadjacent instruction bits. Do not use its width or values to select RoCC versus a
self-hosted endpoint, or to accept/reject complete instruction words. Keep the observation
in `facts.json` for audit; use a separately established executable interface and a
complete-word ISA definition for those decisions. Missing evidence leaves the generated
endpoint unresolved rather than promoting a plausible architecture guess.

TorchAO uses public extension points, not source patches. A selected accelerator format
must also be legal for the operation receiving it. An int8 contraction does not establish
int8 LSTM, general normalization, or support for another hardware configuration.

Do not handwrite a second TorchAO recipe. The evidence export writes
`software/quantization-recipes.json`, indexing generated, content-addressed recipes
under `software/quantization-recipes/`. Each recipe intersects the selected hardware
format with the SW spec's eligible operations and typed constraints. Host-only
operations are excluded; explicit signature refusals remain unquantized. Missing
hardware scale parameters do not acquire a default recipe. These are diagnostic
transformation inputs, not accelerator-admission certificates.
An unknown block size does not obstruct a selected per-tensor or per-channel
recipe, which has no block axis; the generated contract records it as not applicable.

Select a recipe path from that index and use the existing capture worker:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/coverage_mlp/loader.py \
  --dtype int8 --seed 0 --recipe "$GENERATED_RECIPE" \
  --materialize-bundle --out "$QUANTIZED_CAPTURE"
```

Inspect `meta.json`'s `quantization_stats` for the observers, epsilon, actual
calibration source/count and per-operation SW decisions. Observer defaults are
framework transformation policy, not RTL semantics, and need not clutter the
SW spec. The adapter defaults to min/max observation for FP8 because TorchAO PT2E's
histogram observer requires an integer dtype; an explicit FP8 histogram request is
refused. `integerization_receipt` records true integer contractions. When the SW
spec selects `integer_reference`, its `golden_agreement` compares the rewritten
graph exactly against an independent PT2E integer interpreter; the
`integer-reference.json` outputs and reference source identity are byte-bound by
the capture receipt. `portable_agreement` separately compares with TorchAO's
portable Q/DQ graph and may differ because its arithmetic is not integerized.
`recipe_agreement` compares against the original floating model. A single example-input calibration does not
establish model accuracy. Preserve the original FP32 capture as reference lineage,
and derive a fresh corpus from the actual quantized capture before claiming its
precision coverage. Scoped dynamic module transforms are not currently supported
by this signature-screened recipe route.

Reviewed operation declarations can carry `numerical_contract` with `status`,
structured `semantics` and review `evidence`. The corresponding declaration on
the pinned host capability spec is independent of accelerator arithmetic.
Selected contraction-level `numerical_semantics` apply only to accelerator
contractions, not automatically to host operations or other operation families.
Unknown overflow, rounding or reduction behavior blocks numerical declaration
completeness even when the storage and accumulator widths are known.

Framework realization and integer lowering are also separate. The trace-capable
model2MLIR PT2E integerizer supports symmetric per-tensor operands and per-output-
channel weights for Linear, Conv2d and batched matmul. Reduction-axis scales and
nonzero zero points are refused. Remaining float contractions still block an
integer-accelerator claim; native FP8 tensors projected to FP32 are explicitly
reported as a precision-realization blocker. A framework lowering receipt does
not qualify the target's numerical semantics or compiler implementation.

## Derive rather than hand-list the default corpus

Atlas and Gemmini use `capsule_policy: derived_only` recipes with no authored
capsules. Historical seeds remain separately selectable compatibility inputs.
Capture the four independent iteration workloads with `--materialize-bundle`,
then use installed `merlin experiment corpus derive` with all four explicit
`--application-capture LABEL=PATH` selections and exact `--rtl-facts` bytes.
See each target's Phase 0 walkthrough for the complete command.

Derivation writes a complete demand census, requirements, a candidate synthesis
plan/profile and an exact evidence export. It neither constructs an agent nor
silently selects historical corpora or headline workloads. The generated plan
remains diagnostic until actual writers, oracles, placement and independent
numerical checks satisfy the obligations. Repeating the same selection into its
immutable output root must preserve bytes; changed inputs need a new root.

## Inspect exactly what the generator used

The experiment selects `software_spec`, `hardware_spec` and `evidence_mode`; the recipe
also references the same software spec. Select exact extracted bytes with
`--phase0-rtl-facts /absolute/generated/facts.json`. Extraction and Phase 0 selection are
separate: selection does not rebuild hardware or silently regenerate missing facts.

Each new run writes the following beneath `<run>/phase0/`:

| Artifact | What to inspect |
| --- | --- |
| `hardware/circt/facts.json` | Byte-for-byte copy of the selected extraction; absence remains a diagnostic gap |
| `hardware/effective-views/` | Actual resolved facts, target/performance profiles, execution capabilities, readout inputs, quantization and toolchain observations |
| `software/software-spec.json`, `software/contract.json`, `software/datapath.json` | Selected software declaration and the views passed to generation |
| `software/source-snapshots/` | Observed source bytes, indexed by role and hash; not candidate grants |
| `evidence-manifest.json` | Artifact hashes, source identities, consumer-to-artifact mapping and qualification blockers |
| `software/framework/pytorch-opset.json` | Versioned operator catalog observed in the selected capture interpreter: registered ATen overloads, Core ATen and decomposition sets, each with its own scope |
| `coverage/application-inventory.json` | Exact selected detailed inventory bytes, when an adjacent conformance sidecar is selected |
| `software/frontend/index.json` | Application-to-trace, typed graph and workload-specific framework catalog paths |
| `software/frontend/`, `coverage/application-graphs/` | Original/quantized/prepared PyTorch graphs, exact lowering lineage and typed MLIR producer–consumer edges when available |
| `software/host-capabilities.json` | Selected host compiler identities and explicit operation/precision declarations; a dtype profile alone is not operation support |
| `coverage/operation-accounting.json` | Per-application and combined operation partitions, provenance groups, signature/ordinal traceability and declared-versus-observed support |
| `coverage/phase1-capsule-coverage.json` → `phase1_witness_basis` | Finite source-operation and typed-edge witness universe, a compact inventoried selection from the selected cohort, uncovered obligations and the selection's minimum-proof status |
| `coverage/README.md` | Automatically rendered summary of those same operation and quantization views |
| `software/quantization-contract.json` | All authored formats, matching hardware recipes, parameter unknowns and operation-scoped quantization decisions |
| `coverage/generation.json` | Written/omitted capsules, failures, synthesis input identity and diagnostic status |
| `capsules/MANIFEST.yaml` | Actual members and distinct functional/performance/diagnostic selections |

Raw extraction and resolved consumer views have separate hashes. Current tool observations
do not retroactively identify the tools that produced unstamped old facts. Matching target
names, configuration names or dates does not prove that separate HW-MLIR, FIRRTL and hierarchy
files came from one elaboration.

The examples select `diagnostic` until their required semantics, source consistency and
coverage are qualified. Diagnostic output cannot become a verified release by relabeling it
or passing `--phase0-evidence-mode verified`: unresolved evidence must be refused.
The current selector conservatively emits diagnostic snapshots; editing the spec's
review-status field alone does not establish a qualified production-evidence path.

## Inspect the operator split and quantization contract

Accounting starts at the frontend, not by guessing PyTorch calls from MLIR tags.
The trace-capable model2MLIR conversion records the original captured graph, the
quantized graph actually lowered, the prepared/decomposed graph and the final MLIR.
These stages have separate counts and identities: one source operation may expand
into several lowered operations, or several source operations may fuse together.
Counts describe static captured call sites, not execution frequency inside loops.

The framework catalog inventories the selected PyTorch build's registered **ATen
overloads**, not every Python API or custom operator. Each workload retains its own
capture-process catalog when available; catalogs from another version are not silently
substituted. Missing frontend receipts remain unknown, including for older captures
and loaders that supply an already-quantized model without its original graph. Frozen
replay uses saved bytes and never queries another installed torch.

`coverage/operation-accounting.json` provides both per-application splits and their
conglomeration under `overall`. Its partitions distinguish accelerator candidates,
host work, unsupported/unresolved compute, support-lowering obligations, nested bodies
and structural operations. Candidate admission is screened against the selected hardware
contract and SW signature constraints; it is not proof that a compiler lowered the operation.
The declared support universe also retains operations absent from the selected workloads.
Do not use the observed subset to claim support for an entire untested ATen overload.

The exact normalized graph keeps separate obligations for independent computation and
support lowering. A `tensor.empty`, reshape, slice, constant or copy-like node remains
in the node/SSA denominator, but is not an independent accelerator arithmetic demand:
it needs a compiler-owned typed lowering and shape/value-preservation receipt. Until
that receipt is verified, `support_lowering` and support-mediated SSA dependencies stay
open. The operation-accounting report marks their independent
`accelerator_admission` as `not_applicable` while retaining every
`support_lowering_required` obligation; this is not a successful lowering.
They are not assigned a host/device lane or counted as direct transfers merely
because their source signature appears in a capsule. This avoids adding every lowered
helper operation to the target's SW spec or claiming a nonexistent host fallback.
Historical v1 coverage reports remain inspectable, but verified admission requires a
newly frozen v2 report with these role-specific checks.

Host and accelerator admission are independent: both may accept an operation, or neither
may have sufficient evidence. Requested placement and actual execution are separate
records. `host_required` means a host obligation, not demonstrated host compilation.
Inspect operand/weight storage and compute formats, accumulator/output precision and
shape/layout restrictions. Unknown host precision must not select a different default
profile. Host capability declarations belong beside the selected external compiler;
accelerator RTL does not establish host compiler support.

The examples provide `target/host-capabilities.yaml`, selected by the descriptor's
`host_lane.capability_spec`. It binds the observed example compiler package digest
and precision strategy without changing the payload. Its operation list is deliberately
unreviewed: the examples list only typed matmul/batch-matmul candidates observed in
their exact schedule files. This is not execution qualification and does not establish
normalization, activation or general reduction support. Never populate it from a package
name or blanket dtype claim. When selecting another host package, update this separate
input with that package's exact identity and independently reviewed operation constraints.

Typed producer–consumer edges identify casts, quantization conversions and possible
host/accelerator transfers. Their semantics and capsule witnesses must be explicit;
different dtype labels alone do not prove a conversion or an executed transfer.

Select a generated detailed sidecar through the existing conformance requirement.
Its byte copy, parsed accounting, content digest and declared-roster comparison travel
together. A missing sidecar is reported as `not_available`, never as zero uncovered work.
Held-out claim models are not added to this derivation inventory.
When a detailed inventory is selected, synthesis refuses a capture whose name
matches a held-out claim model after case and punctuation normalization; use
separate iteration workloads rather than renaming a validation capture.
Public synthesized capsule descriptions report counts and signatures, not source
model names. An exact integer-operation capsule uses a deterministic source ordinal
within the digest-bound detailed inventory together with the capture hash, so the
group matcher can still account for every occurrence. Keep the full name-to-ordinal
mapping in that inventory; publishing the inventory itself still exposes its
source names. An older frozen corpus that exposed names must be regenerated
rather than relabeled after the fact.

You can inspect a diagnostic subset before a complete requirement is ready, using
the existing inventory producer. It automatically exports the evidence bundle and
its generated coverage README beside the inventory:

```sh
python build_tools/scripts/check_conformance_coverage.py --target gemmini \
  --application-capture example_mlp=/absolute/capture/model.mlir \
  --software-spec examples/gemmini/target/software-spec.yaml \
  --rtl-facts /configured/out/artifacts/audits/gemmini/facts.json \
  --inventory-out /configured/out/artifacts/verification/gemmini/diagnostic.application-demands.json
```

Open `diagnostic.application-demands.evidence/coverage/README.md` beside that output.
The same producer accepts the Atlas example's target/spec/facts. A subset is explicitly
not the full declared roster, and an incomplete inventory still exits nonzero after
writing its diagnostic reports. Changed outputs require a new versioned destination.

Each `quantization.formats` entry references SW operation IDs through
`eligible_operations`. Use a distinct `id` for each format or variant; same-width alternatives
must not borrow each other's scale/readout evidence. Values inferable from the selected
readout come from existing `quant_recipe` derivation. Residual authored parameters must be
explicit, with unknowns retained rather than filled by framework defaults.

The generated quantization contract keeps authored formats, matching hardware candidates,
parameter conflicts and observed operation decisions together. A host-only operation or
an unsupported signature cannot become eligible simply because another operation uses
that format. Unknown legality, numerics or format-specific evidence blocks eligibility.
Framework and compiler realization are separate, unevaluated routes until executed.

This is input for the existing `quant_layer_plan` and `_recipe_quantizer` adapters, which
use TorchAO's public `AOBaseConfig`/PT2E extension points. It does not patch TorchAO, select
a whole-model format automatically, or certify that a custom format is implemented there.

## Regeneration, coverage and phase handoffs

Keep generated facts, application inventories, conformance requirements, synthesis plans,
capsules, goldens and weights under the configured output root, never committed in examples.
The example's `artifacts/` folder is a navigation guide, not an output destination.
Capsule MLIR references its copied `capsule.weights.safetensors` relatively. Its
frontend relocation receipt records a hash of the capture's original weight-reference
spelling, not the ephemeral absolute capture-cache path.

Changing the software spec or recipe changes synthesis identity. Generate a new requirement
and synthesis pair with the newly selected inputs. Legacy synthesis references without
these commitments remain diagnostic; a digest-bound mismatch refuses generation even in
diagnostic mode. Never edit old synthesis YAML, archived capsules or run receipts to repair
a mismatch. Select new artifacts explicitly through `--phase0-conformance-spec` and
`--phase0-synth-profile` on a fresh run.

Account for every captured application operation as accelerator work, explicitly allowed
host work, or an unresolved/unsupported obligation. A capsule count is not an operation
coverage proof. Inspect omissions, independent goldens and placement checks, then follow
the [generation and reviewed-release guide](generating_capsules.md).

Verified whole-workload admission checks exact source lineage, reviewed independent
compute placement/numerics, support-lowering and shape receipts, typed transfer
obligations, and coverage in the **actual admitted cohort**. A larger generated source
pool is not proof that a selected cohort covers them.
The Phase 0 `phase1_witness_basis` selects a small set of capsule witnesses for the
finite source-operation signatures and conditional typed edges recorded in that
report. It lists the complete scoped universe and obligations without a witness.
When a distinct obligation uniquely requires each selected capsule and those
capsules cover the witnessable universe, the report proves an exact minimum;
otherwise its selected size is only an upper bound. The generated corpus is not
pruned. For the Gemmini int8 r18 diagnostic, four source capsules are an exact
minimum for the 1,106 source-operation and typed-edge witness records; all 43
Phase 1 capsules remain in the selected cohort. This four-capsule calculation
does not cover numerical, ISA, precision, tail, shape or other conformance axes,
or establish that a compiler can execute even the selected four.

This witness set is a plan for Phase 1 verification, not a correctness guarantee.
Phase 0 freezes the finite support domain, selected input identities and open proof
obligations. Phase 1 must discharge those obligations against emitted compiler
artifacts and bound target execution, including operation placement, typed support
routes, transfers and numerical behavior. Whole-module formal-proof eligibility is
reported separately; witness coverage does not establish it. A universal claim
also requires sound compositional proofs for the supported domain and hardware
conformance, beyond any finite capsule selection.
The completeness record establishes test obligations, not a working target compiler:
Phase 1 still has to lower, execute and numerically qualify its generated implementation.
Existing hardware-source and independent-reference qualification requirements remain
separate, and a complete coverage report cannot upgrade diagnostic evidence.

Phase 1 consumes the functional selection to establish correctness and freeze a compiler.
Phase 2 consumes its own performance selection plus that exact functional compiler/evidence.
Holdout claim models stay owner-only until the prescribed evaluation point. Full-model
compilation, executable numerical agreement, dispatch evidence and hardware qualification
remain separate requirements; none follows merely from producing these files.
