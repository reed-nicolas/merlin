# Phase 1: functional compiler

Select the reviewed Phase 0 manifest's `phase_corpora.phase1` functional members;
retain its evidence-manifest identity and selected software/hardware declarations.
The example's current diagnostic derivation is not that admission release.
Resolve required semantics and source consistency, regenerate a fresh run, and
complete independent coverage/numerical review before verified execution.
The Phase 2 selection is distinct and must not replace the functional population.

The [catalog definition](../experiment.yaml) uses the installed RTLchecks
treatment with an explicit bundle identity, manifest and timing path. These
retained inputs must be regenerated and reviewed for the selected corpus before
verified execution. See [execution prerequisites](../../../experiments/README.md#definitions-and-execution)
for missing artifacts, tool provisioning and historical-resume limitations.

[The target descriptor](../target/descriptor.yaml) selects the public inputs in
`contracts/hwbringup_atlas_v0/`:

- Architecture overview and ISA green card.
- ISA definition requiring the external `npu_model` dependency.
- Curated RTL source evidence, with a note identifying the full external tree.
- Worked assembly examples.
- Preflight assembly fixtures and their target-owned assembler/runner adapter.

These are supplied experiment inputs, not newly generated compilers or certification
results. Their bytes are preserved from the former experiment directory. The curated
RTL subset is not a replacement for the complete external hardware repository.
Merlin's shared preflight protocol consumes the declared adapter; it contains no Atlas
instruction encoding. Path validation does not imply that a hardware probe passed.

Start from [the experiment definition](../experiment.yaml) and
[local tooling setup](../target/README.md). The full-mode task prompt is generated
by the shared renderer. Prepare and review a fresh release through the
[Phase 0 handoff](../../../experiments/README.md#reviewed-phase-0-handoff) before
verified execution. Generate new bundles for these paths; historical bundles and
receipts retain their original bytes. Other harness resources still use the
descriptor's retained `resources_root`.

An optional target-owned check can consume the explicitly selected
`scheduling-evidence.json` produced by
[`rtlgraph_evidence.py`](../target/rtlgraph_evidence.py). Run it after emitting an
assigned assembly program, independently of the functional certificate:

```sh
python examples/atlas/target/rtlgraph_check.py \
  --source "$PROGRAM" --assembly atlas-opt-native \
  --selection "$SCHEDULING_SELECTION" --atlas-opt "$ATLAS_OPT" \
  --output "$NEW_CHECK_DIR" --validation static
```

The selected manifest must retain its native contract and every hashed member.
The checker verifies profile/evidence agreement and the exact selected compiler
bytes, reproduces the saved footprints, and invokes `atlas-opt --check` with the
same profiles and robust DMA timing. `NEW_CHECK_DIR` must be a new directory
under the configured artifact root. It contains the input and native source,
actual commands, stdout/stderr logs, reproduced footprints, and `report.json`.
A rejected schedule or failed preflight returns a nonzero exit status and keeps
the report. Provision the compiler's runtime library search path in the calling
environment when necessary.

Assembly dialect selection is required at the CLI. `atlas-opt-native` means
compiler syntax such as `dma.load.ch3 x6, x18, x12`, `vload m8, 0(x6)`, and
`vmatpush.weight.mxu0 w0, m5`. The public Merlin examples use different syntax:
`DMA.LOAD x6, x18, x12, 3`, `VLOAD 8, x6, 0`, and
`VMATPUSH.W.MXU0 0, 5`. Select `--assembly merlin-atlas --assembler
"$DECLARED_ASSEMBLER"` to use the bounded target-owned
[`assembly bridge`](../target/rtlgraph_assembly.py). The assembler is the
explicitly chosen `baremetal/assembler.py` from the descriptor's external
`atlas_npu` setup. Before executing the compiler, the bridge requires exact
encoded-word agreement between the original and round-trip assembly. Unsupported
forms fail closed. The compiler does not emit instruction words; this check
establishes translation agreement with the declared assembler.

A passing report establishes same-model schedule consistency. Static validation
checks CFG hazards and reports no execution cost. `--validation dynamic` follows
one modeled execution and retains its modeled cycle estimate; DMA completion
still requires matching waits. Neither mode evaluates tensor arithmetic, replays
RTL, measures hardware timing, or qualifies Phase 1 numerical correctness. The
shared Phase 1 grading protocol does not automatically run this optional tool.
