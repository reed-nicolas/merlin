# Gemmini target inputs

This directory contains **inputs**, not generated hardware facts, capsules, compiler
candidates, or passing results. Start with the [Phase 0 walkthrough](../phase0/README.md)
to see how these inputs become a frozen corpus. The experiment's single entry point
is [experiment.yaml](../experiment.yaml).

| File | Owner | Purpose |
| --- | --- | --- |
| [software-spec.yaml](software-spec.yaml) | User, with accelerator/software reviewers | Software-visible arithmetic, typed operation placement, quantization and host/device transfers that RTL cannot establish alone. `unknown`/unreviewed is not support. |
| [hardware.yaml](hardware.yaml) | User/tool integrator | Selects exact RTL source production and questions the extractor must audit. Its expected port types are assertions to check, not generated facts. |
| [descriptor.yaml](descriptor.yaml) | Experiment operator | Workloads, tool resources, grading and host-lane selection. It is not the SW spec or an RTL fact file. |
| [host-capabilities.yaml](host-capabilities.yaml) | Host-compiler owner/reviewer | Candidate host operation signatures bound to one compiler package. Still unreviewed; never substitutes for Gemmini support. |
| [contracts/residual.yaml](contracts/residual.yaml) | Target author | Prototype capability intent and source-location anchors that the generic deriver cannot infer from RTL. |
| [contracts/target_contract.yaml](contracts/target_contract.yaml) | Target author/reviewer | Selected prototype contract for this example's current Phase 0 experiment; not a certified generated contract. Its parsed content currently matches the residual, so changes must keep them consistent. |
| [evidence_concepts.yaml](evidence_concepts.yaml) | Target author | Vocabulary for classifying discovered evidence; it is not evidence itself. |

The [kernel-mining inputs](../kernel-mining/README.md) are a separate, optional
authoring concern. The [verification walkthrough](../verification/README.md)
contains runnable probes, an authored arithmetic reference, and instructions
for inspecting their receipts. Those tools produce evidence under ignored output
roots; do not edit a generated artifact to make a result pass.
Evidence concepts are read from the **selected** target directory: if an OOT
provider is selected, its vocabulary—not this example's file—is in effect.

The ownership chain is:

`user-authored target inputs + selected external RTL → deterministic CIRCT facts → derived Phase 0 capsules and coverage → Phase 1 compiler candidate → Phase 2 optimized candidate`

Facts, capsules, coverage and execution receipts are generated outputs, stored in
the selected run/artifact directory with content identities. The agent authors
compiler implementations only in the declared Phase 1/2 workspace; it does not
author the SW spec or overwrite the Phase 0 evidence. A published target backend
remains out of tree. Historical receipts retain their original paths and hashes.
