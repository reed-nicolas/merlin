# Atlas: Phase 2 handoff

Select `phase_corpora.phase2` from the reviewed derivation manifest, separately
from Phase 1 functional conformance. Preserve the SW-spec/hardware-evidence
identity and exact frozen functional compiler; new formats, operations or source
selection require renewed functional qualification, not only a performance run.
Phase 0 diagnostic members and unresolved source-consistency claims cannot be
promoted to measured optimization inputs by renaming their selection.

Start with the [functional workflow](../phase1/README.md). Retain the exact
frozen submission, functional run identity, descriptor and grading evidence;
an authoring exit code or a changed compiler with a freshly computed hash is
not that handoff.

The [catalog](../../../experiments/catalog.yaml) provides two shared templates:
[model portfolio](../../../experiments/definitions/model-portfolio-template.yaml)
for bounded compiler authoring and structural analysis, and
[measured claims](../../../experiments/definitions/measured-claims-template.yaml)
for its separately qualified measurement protocol. They are different evidence
schemas. Neither is a ready-to-run Atlas performance experiment.

Copy a template into an authored input directory, keep `kind: template` until
every declared input and budget is supplied, and set `target: atlas`.
Use the descriptor and compiler identity from the frozen functional run.
The measured protocol additionally requires its exact engine certificates,
performance inputs and provisioned managed execution resources; a target name
does not establish that this target supports that protocol.

Follow the [installed deployment and execution guide](../../../experiments/README.md#definitions-and-execution)
for required package/source roots, campaign inputs and supervisor selection.
After completing the definition, inspect and preflight its path with
`merlin experiment inspect /absolute/inputs/performance.yaml --phase 2` and
`merlin experiment preflight /absolute/inputs/performance.yaml --phase 2`.
Preflight is not hardware or simulator qualification.

Generated candidates, checkpoints and measurements belong beneath the configured
output root, not this directory. Preserve original compiler bytes and retain
certification/publication records separately. Resume only the recorded run with
its original inputs; changed sources or evidence require new qualification.

For model captures and intermediate IR, see [whole-model inspection](../whole-model/README.md).
No target-specific end-to-end optimization or accelerator performance result is
established by this guide.

## Optional paired schedule feedback

The target-owned [paired checker](../target/rtlgraph_compare.py) can compare one
assembly fixture before and after Atlas scheduling using an imported
`merlin.scheduling_evidence.v1` selection and the exact selected `atlas-opt`
binary. This is a separate opt-in diagnostic, not a required stage or an Atlas
performance experiment. Supply paths to your selected inputs and a fresh output
directory beneath the configured artifact root:

```sh
python examples/atlas/target/rtlgraph_compare.py \
  --kernel /absolute/inputs/kernel.S \
  --selection /absolute/inputs/selection/scheduling-evidence.json \
  --compiler /absolute/tools/atlas-opt \
  --output /absolute/artifacts/paired-schedule \
  --validation static
```

The baseline must pass before scheduling begins, and the candidate must pass a
separate check with the same compiler and selected component projections.
Both assembly streams, commands, diagnostics and their identities remain in
the output directory. A rejected baseline or candidate produces no cost delta.
The default assembly is `atlas-opt-native`; the Phase 1 checker documents
explicit conversion prerequisites for Merlin native assembly.

Static validation supplies no cycles. `--validation dynamic` reports the
compiler scheduling simulator's before/after cycle cost when both checks pass.
These engine-relative costs include approximate DMA estimates; they are not
measured hardware cycles, whole-model latency or a roofline. The report keeps
hardware timing `UNMEASURED` and functional qualification `NOT_ESTABLISHED`.
Neither mode changes the Phase 0 admission or frozen Phase 1 handoff requirements.

To also check arithmetic, supply `--model-root /absolute/tools/npu-model`,
`--model-python /absolute/tools/model-venv/bin/python`, and
`--model-fixture /absolute/inputs/fixture.json`. The independent
[numerical observer](../target/rtlgraph_model.py) runs both native assembly
streams in separate processes with identical initial inputs. Every selected
output must match the supplied reference bytes and the other arm. With
`--assembler`, model encodings must also match the selected baremetal assembler.
A model with incorrect `DMA.CONFIG` or other encodings is rejected.
Encoding agreement does not validate execution semantics: also check the selected
model's VMEM address units and operand capture behavior against the hardware.
The observer does not patch external tools.

The fixture uses schema `merlin.atlas_model_fixture.v1`, with `dram_size`,
`max_cycles`, `reference_provenance`, and `inputs`/`outputs` arrays. Inputs select
a DRAM `base` and a relative raw-byte `path` with `sha256`. Outputs select a
unique identifier `name`, either `space: dram` with `base`/`length` or
`space: mrf_bf16` with `registers`, and a `reference` path/hash. MRF references
concatenate each register's row-major BF16 bytes in the declared register order.
Use independently computed references; the observer cannot establish their
correctness from a hash. Fixture members, model sources, runtime, assembly,
logs and outputs retain identities in the numerical report.

Without a fixture, numerical equivalence remains `UNKNOWN`. With one, a
mismatch, encoding disagreement, timeout or incomplete execution rejects the
pair. Success establishes only that supplied numerical witness. Python-model
ticks are reported separately from the compiler's estimated cycles and are
not hardware measurements.
