# MX Gemmini Phase 0

The [`mx-gemmini-functional`](../experiment.yaml) experiment derives capsules from the
selected RTL facts, the selected out-of-tree provider's
`mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml`, and
observed application captures. The [recipe](recipe.yaml) uses `derived_only` membership.
The former hand-authored recipe is retained verbatim as
[`recipe-legacy.yaml`](recipe-legacy.yaml) for reproduction; it is not a current
qualification claim.

The selected candidate specifies `GemminiMxFPConfigs.standaloneMxFPConfig` at
Gemmini `2029218197f771ce71416f859d975bea47b7aabc` and MxGen
`56ef1c6810924e1cb0af07add09156b0e2f53576`. It declares separate MXFP8,
MXFP6, and MXFP4 contraction contracts. The OOT source checker covers the
selected files and command fields. Isolated elaboration and narrow scale-load
simulator checks exist for this candidate. The broader format, two-wave, and
Spike results in the guide used the older `f016...` revision; they do not
transfer to this pin. The spec remains **unreviewed**: full protocol and
numerical checks, a complete toolchain-closure receipt, Phase 0 L0–L3
capsules, and whole-model/host semantics still need admission.
The independent MX numerical model is loaded through `MERLIN_MLC_DIR`; its frozen
source identity must be checked against the selected RTL before it can serve as
an oracle. A TorchAO fake-quant
capture is only an operand-conversion and coverage diagnostic, not an L2/L3
oracle. The source-level config audit is recorded in the guide; the retained
Phase 1 `hwbringup_mx_v0` inputs still describe an older default/GPU-local
mapping and do not qualify the selected standalone config.

The selected `mx_gemmini.synth.yaml` is a retained, unverified legacy sidecar.
Phase 0 preflight refuses it; its older BF16/int8 entries are not an MX corpus.
Fresh synthesis currently also needs an explicit same-target backend capability
contract and derived tile geometry bound to the selected RTL. The retained
conformance requirement also admits BF16/int8 accelerator cells that conflict
with this three-format software spec; synthesis rejects those cells. Select
verified MX application captures, derive a new requirement, and review a new
digest-bound sidecar under the artifact root before running Phase 0. Preserve
the historical inputs.

The OOT package's `docs/iteration_roster_candidate.md` gives the current
model/input materializer, RTL source check, derived site inventory, and
per-model policy selection workflow. The model/operator chooses the exact
FP8, FP6, FP4, or host assignment after inventory derivation. Its frozen
selection is a capture diagnostic; it is not yet a reviewed Merlin Phase 0
application-demand sidecar or admitted capsule corpus.

[The MX Phase 0 guide](../../../docs/guides/mx_gemmini_phase0.md) records the
format rules, supported operation boundary, capture policy, and evidence gates.

Provision the selected OOT support and numerical model, replace the legacy
sidecar, then inspect and run with fresh paths under the configured output root:

```sh
merlin experiment inspect mx-gemmini-functional --phase 0
merlin experiment preflight mx-gemmini-functional --phase 0
merlin experiment run mx-gemmini-functional --phase 0 --run-dir /configured/out/runs/mx_gemmini/phase0/example-1
merlin experiment corpus prepare /configured/out/runs/mx_gemmini/phase0/example-1 --output /configured/out/artifacts/protocols/mx_gemmini-review-1
merlin experiment corpus inspect /configured/out/artifacts/protocols/mx_gemmini-review-1
```

A prepared corpus needs operator review before sealing. A seal acknowledges
reviewed inputs; it does not certify numerical behavior, source closure, or a
functional compiler. Keep generated runs, holdouts, weights, and goldens outside
source examples.
