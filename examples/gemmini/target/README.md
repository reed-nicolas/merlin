# Target inputs and source qualification

`software-spec.yaml` is authored software-visible semantics, not generated tests.
`hardware.yaml` selects the deterministic production policy and direct audit
questions. `host-capabilities.yaml` is a separate declaration bound to immutable
host compiler bytes; its precision strategy is not an operation support claim.

The software spec is a small authored input: selected arithmetic/overflow and
readout semantics, typed operation placement constraints, quantization parameters,
and explicit transfer candidates. It contains no backend, ISA table, mesh size,
calibration history or generated qualification records. Those come from the
explicitly selected OOT provider and generated RTL/characterization artifacts.
The twenty-bit partial-sum bound is a numerical contract, not thirty-two-bit SRAM
geometry. Review/unknown states prevent those choices being mistaken for working
compiler support. Missing host support is not silently filled in.

Name an operation once and write its typed constraints directly underneath it.
For unchanged lane crossings, `copy: {dtype: int8, layout: row_major_contiguous}`
states both endpoint types/layouts and bit-preserving semantics once. The generator
expands these into the full consumer contract in artifacts; no target-specific
defaults or review claims are added. See the [authoring guide](../../../docs/guides/phase0_specification.md).

Merlin's v1 reader also accepts old inline `capability_contract` documents, but
new minimal specs require a separately selected same-target backend contract.
Loading the YAML never discovers one or imports its runtime. A selected provider's
broader legacy operation list cannot override the spec's typed SW admission screen.

[`descriptor.yaml`](descriptor.yaml) owns experiment resources;
[`evidence_concepts.yaml`](evidence_concepts.yaml) names evidence concepts.
Runtime backends, reference-program helpers and calibration live in the selected
OOT support package, not these example inputs. Select it and the extraction tools
explicitly before the commands below:

```sh
export MERLIN_TARGET_PATH=/path/to/gemmini-mlir/merlin-support
export MERLIN_MLC_DIR=/path/to/ModelIR
```

Use the companion revision recorded in
[`target_support.json`](../../../build_tools/upstreams/target_support.json).
Calibration coefficients are screening estimates, not new measurements.

The selected integer RTL has 8-bit signed operands, a 20-bit MAC result and
32-bit accumulator storage. Those are different quantities. A mathematical
int32 golden needs proof that internal partial sums cannot overflow, or an
independent width-aware model. The software requantization rounding parameter
also must not be confused with the RTL's integer rounding shift.
Compute datapath facts also preserve `declared_carrier`: exact FIRRTL port types,
directions, signedness and element widths, including row-vector declarations.
This keeps an explicit `SInt<20>` observation inspectable even when no tensor
quantization format exists for twenty bits. Carrier signedness alone does not
establish a quantizer scale/zero-point or whole-operation overflow policy.

Produce a new source bundle from the exact selected FIRRTL:

```sh
python -m merlin.targetgen.rtl.source_selection \
  --target gemmini --generator gemmini --config GemminiRocketConfig \
  --core-root Gemmini --firrtl /selected/elaboration/design.fir \
  --hierarchy /selected/elaboration/top_module_hierarchy.json \
  --firtool /selected/circt/bin/firtool --output /generated/gemmini/source-1 \
  --drop-annotation-class 'chisel3.experimental.EnumAnnotations$EnumComponentAnnotation' \
  --drop-annotation-class 'chisel3.experimental.EnumAnnotations$EnumDefAnnotation' \
  --drop-annotation-class 'chisel3.experimental.EnumAnnotations$EnumVecAnnotation'
python -m merlin.targetgen.rtl.circt_introspect --target gemmini \
  --source-bundle /generated/gemmini/source-1/source-selection.json \
  --out /generated/gemmini/source-1/facts.json
merlin-target-tools rtl-source-audit \
  --source-bundle /generated/gemmini/source-1/source-selection.json \
  --facts /generated/gemmini/source-1/facts.json \
  --hardware-spec examples/gemmini/target/hardware.yaml \
  --output /generated/gemmini/source-1/validation.json
```

The producer preserves original FIRRTL, records a separate generation input
when explicitly removing non-functional Chisel enum metadata, runs the selected
`firtool`, extracts an exact HW module closure, and derives hierarchy from actual
FIRRTL instances. It never substitutes watchdog or datapath behavior. Old
hierarchy files may precede specialization/deduplication; discrepancies remain
diagnostics, not guessed aliases. `validation.json` compares direct source slices
with extracted values and records the storage-versus-compute precision gap.

Source consistency verifies provenance, not compiler or numerical conformance.
Operation, quantization, transfer and host support stay unreviewed until their
own execution receipts exist. Use fresh facts for a fresh frozen Phase 0 run;
preserve old artifacts unchanged. See the [Phase 0 walkthrough](../phase0/README.md).

### Independently characterize the arithmetic cell

The example provides deterministic vectors for the exact selected `MacUnit` and
an independent SpecIR signed multiply-add reference. All 65,536 input pairs are
checked at zero addend, then 11,520 accumulator-boundary cases check truncation,
overflow and signed interpretation. This checks the cell's 20-bit wrapping result;
it does not qualify a complete mesh, DMA transaction, readout or compiler.

```sh
python examples/gemmini/target/characterize_cell.py \
  --hw-source /generated/gemmini/source-1/core.hw.mlir \
  --specir-root /selected/SpecIR \
  --circt-opt /selected/circt/bin/circt-opt \
  --verilator /selected/bin/verilator \
  --output /generated/gemmini/cell-1
```

Inspect `characterization.json`, `cases.json`, `vectors.txt`, `cell.hw.mlir`,
`verilog/`, `harness.cpp`, and `execute.stdout.log`. The harness is generated by
the shared producer, never hand-authored in the output folder. The receipt binds
the original HW source, independent reference sources, vectors, tool binaries,
emitted files and executed native binary. `verify_characterization` from
`merlin_experiments.phase0.cell_probe` checks a saved or moved artifact's hashes
and zero failed outputs without reopening live RTL. Selecting the receipt for a
frozen run still requires separately pinning its bytes; it is not a compiler
certificate or proof that the finite vector set covers every possible addend.

Mathematical integer capsule goldens are valid only in a bounded domain. The
shared `integer_partial_sum_bound` requires actual operand/initial-addend values
and the reduction extent. For unrestricted signed int8 inputs, K=31 passes its
conservative i20 bound and K=32 does not. Wider accumulator storage or a small
final result does not prove that intermediate partial sums avoided wrapping.

The host manifest now lists only typed matmul/batch-matmul candidates found in
its pinned schedule. No standalone normalization or activation is claimed.
Exact `ops` selectors do not widen to a shared semantic family: a declaration
for `linalg.matmul` does not cover a captured `linalg.generic` contraction.
Inspect `coverage/operation-accounting.json` against the selected schedule
before reviewing the host declaration. A transform-interpreter success alone
does not establish that a selector matched. Even a native whole-program result
matching its saved golden is finite host-execution evidence, not per-operation
RVV support, accelerator execution, or a reviewed host numerical contract.
Typed load/readout candidates describe bit-preserving crossings; they remain
unreviewed and never imply that FP32-to-int8 quantization, dispatch, or DMA
execution was already implemented. The selected hardware readout recipe supplies
symmetric, per-tensor W8A8 to the generated quantization contract; the authored
spec limits that format to contractions. Per-channel software epilogues need a
separate route and qualification rather than being inferred from a format name.
The spec explicitly declares an FP32 scale encoding: the selected Gemmini
configuration's `gemmini_params.h` defines `acc_scale_t` as `float`, while CIRCT
alone sees only a 32-bit command carrier. This software-visible fact makes a
scoped diagnostic capture recipe derivable; it does not review the format or
qualify quantization on a model.

For a status-only check of configured tools and prebuilt simulator files, run
`python examples/gemmini/target/probe_oracles.py`. Set `MERLIN_CHIPYARD` to your
checkout; optional overrides are `MERLIN_GEMMINI_SPIKE`,
`MERLIN_GEMMINI_VERILATOR`, `MERLIN_RISCV_GCC` and `MERLIN_GEMMINI_HARNESS_DIR`.
Status mode runs nothing and is not a provenance or executable-validity check.
`--run spike` or `--run verilator` explicitly executes the selected prebuilt
test on provisioned resources; its exit status does not certify a generated compiler.
The generated harness ABI can separately be checked with
`MERLIN_RTL_FACTS=/generated/gemmini/source-1/facts.json python
build_tools/scripts/check_kernel_abi_arg_order.py --verbose`.

### Observe one selected-core DMA matmul numerically

[probe_native_dma.py](probe_native_dma.py) uses ModeLIR's existing Gemmini
DMA driver with an ARC library built from the selected core. It compares
every byte of one 16×16 INT8 matmul with an independent NumPy golden, then
freezes RTL, intermediate HW/Arc/LLVM files, the native library, tools,
driver source, inputs and outputs. Supply the exact artifact paths shown by
--help; the output directory is produced by the example, not hand-edited.

```sh
python -I /generated/gemmini/native-1/runner.py \
  --replay-bundle /generated/gemmini/native-1
```

That command verifies all frozen bytes and re-executes independently of the
original source checkout. The tested mvout_acc command saturates its INT32
accumulator to INT8 DRAM output; it does not qualify bit-preserving INT32
readout, the full Rocket SoC, or all shapes and values.

### Check generated contraction kernels on native simulators

[probe_native_kernel.py](probe_native_kernel.py) checks two generated Phase 0
contraction capsules against an independent scalar integer matmul, the capsules'
goldens, native Gemmini Spike, and native Gemmini Verilator. First produce a
corpus with `isa/SY_contraction_i8_aligned` and
`isa/SY_contraction_i8_partial`, and a selected source bundle whose hashes are
bound by that corpus's `_evidence/evidence-manifest.json`. Select the OOT
Gemmini support and Chipyard toolchain explicitly:

```sh
export MERLIN_TARGET_PATH=/selected/gemmini-mlir/merlin-support
export MERLIN_CHIPYARD=/selected/chipyard
python examples/gemmini/target/probe_native_kernel.py \
  --corpus /generated/gemmini/corpus-1 \
  --source-evidence /generated/gemmini/source-1 \
  --output-root out/artifacts/probes/gemmini-kernel-1
```

When refreshed, independently validated facts are stored outside the selected
source bundle, pass `--facts-evidence /generated/gemmini/facts-1`. The probe
checks that the facts validation binds the selected source hash and that the
Phase 0 manifest binds both facts and source/core HW bytes. Without the option,
the existing flat source bundle layout remains supported.

The output root must resolve beneath this checkout's ignored `out/` tree and
must not overlap either input bundle. `receipt.json` binds the selected source,
corpus manifest, probe, support files, tool binaries, and per-case program,
input and output hashes. Each case directory contains `command_buffer.json`,
`inputs.json`, `outputs.json`, `spike/main.c`,
`spike/merlin_gemmini_c0.elf`, `spike_console.log`,
`verilator_console.log`, and `spike_disassembly.txt`. For an existing output
root, add `--audit-existing` to re-execute the saved ELF on Spike; a saved
Verilator console is reused only when its hash and the ELF hash match the prior
receipt. Without that flag, both engines execute the newly compiled program.
For a frozen experiment run, point `--corpus` at its `phase0/capsules` directory;
the probe binds the adjacent `phase0/evidence-manifest.json`. Standalone corpora
with `capsules/_evidence/evidence-manifest.json` remain readable.

The current check covers exactly two fixed int8-input/int32-output shapes:
16×32 by 32×16 (256 outputs) and 16×31 by 31×15 (240 outputs). Their generated
stimuli use operand values 0–3, zero initial accumulator, and no epilogue. The
probe requires zero tile padding, compares all 496 outputs exactly, and checks
that the same ELF contains custom-3 RoCC instructions and runs on both engines.
Each simulator execution has a 180-second wall timeout. The selected Verilator
binary rejects the tested cycle-cap flags, so there is no independent simulated
cycle limit; `receipt.json` records those rejections. This is finite numerical
evidence, not a proof for every input, model, or operator. The generated
capsules' software admission remains `unknown`, and the receipt does not prove
that the selected Verilator binary was built from the exact selected RTL source.

To check that missing build link, run
[attest_native_simulator.py](attest_native_simulator.py) against a fresh ignored
output directory. It re-lowers the selected FIRRTL with the explicitly selected
Chipyard firtool, checks the Gemmini core RTL files byte-for-byte, regenerates
and compares the Verilator C++ model, then rebuilds and compares the entire
simulator executable to the one hashed in the kernel receipt:

```sh
python examples/gemmini/target/attest_native_simulator.py \
  --source-evidence /generated/gemmini/source-1 \
  --chipyard /selected/chipyard \
  --firtool /selected/chipyard/.conda-env/riscv-tools/bin/firtool \
  --verilator /selected/chipyard/.conda-env/bin/verilator \
  --kernel-receipt out/artifacts/probes/gemmini-kernel-1/receipt.json \
  --output-root out/artifacts/probes/gemmini-verilator-build-1
```

`receipt.json` in that output records input and tool hashes, the rebuild steps,
core RTL and C++ file counts, and the exact binary equality. The fresh RTL,
Verilated model, and rebuilt simulator remain alongside it. This is
reproducible build provenance for the selected core inside one full SoC binary,
not a proof of RTL behavior. Spike's `libgemmini.so` is a separately hashed
functional model: neither this build check nor agreement with its output makes
Spike RTL-derived. A different Chipyard build, toolchain, or kernel receipt
requires a new attestation.
On a read-only Chipyard tree, the attester copies the selected FIRRTL and projects
only the two hierarchy-output annotation filenames into its ignored artifact;
it records both annotation hashes and the exact path changes, then still requires
byte-for-byte equality of generated core RTL, Verilator C++, and the executable.

### Diagnose one headline-derived kernel window

`probe_headline_kernel.py` takes a captured `frontend-trace.json` and its exact
`model.mlir`, one prepared-graph source node ID, selected RTL facts, and a
capability contract. It refuses a stale trace or ambiguous operation. The
generated capsule records the original operation ordinal, operand types,
model/trace hashes and full matrix geometry, then derives a small integer
window from the RTL mesh edge. It accepts a traced `linalg.matmul`, or a
traced `linalg.generic` only when the exact captured MLIR has the signed
i8×i8→i32 matmul maps, iterators, provenance and reduction body. Other generic
loops are not treated as contractions. For example, TinyLlama's `k_proj` body
`g:prepared:root:n280` in the 8-token prefill capture has source geometry
8×2048×256; the selected 16-wide mesh gives an 8×32×16 diagnostic window.
No dimensions are copied into the generator.

```sh
python examples/gemmini/target/probe_headline_kernel.py \
  --capture "$HEADLINE_ROOT/tinyllama-prefill" \
  --source-node-id g:prepared:root:n280 \
  --rtl-facts "$RTL_ROOT/facts.json" \
  --support-contract "$MERLIN_TARGET_PATH/contracts/target_contract.yaml" \
  --output-root out/artifacts/probes/headline-tiny-kproj-1
```

The default generates `capsule.yaml`, `capsule.interface.mlir` and
`generation.json` without running a simulator. Add `--native` for independent
scalar-versus-Gemmini Spike output checks; add `--rtl` to execute the same ELF
on Verilator. Native mode requires the explicitly selected OOT support contract,
`MERLIN_TARGET_PATH` pointing to that support package, and `MERLIN_CHIPYARD`
pointing to the selected Chipyard toolchain/simulator build. The example's
Phase 0 contract supplies the authored corpus issue order; the probe checks
that its shared compute-unit and encoding declarations agree with the OOT
provider contract and records both contract hashes.
Before native execution, the probe checks that the source projection, capsule
and interface still match `generation.json`. The numerical receipt records the
SHA-256 of all three generated files, so a result cannot be silently reassigned
to a different generated diagnostic.
Each simulator has a 180-second wall timeout. The source capture supplies
geometry only: FP32/BF16 model captures do not establish an int8 model path.
An actual quantized capture and integerization evidence are required before
claiming the model invokes a corresponding int8 contraction, though they are
not required to check this synthetic Gemmini kernel's arithmetic.

For an integerized capture, inspect `frontend-trace.json`'s
`mlir.operations` for an `linalg.generic` with a prepared-graph source ID and
`tensor<...xi8>` operands. Pass that ID to `--source-node-id`; the probe checks
the parsed operation, not just those trace labels. For the selected ResNet50
W8A8 diagnostic capture, `g:prepared:root:n361` names a 12544×147×64
body; the 16-wide RTL mesh derives a 16×19×16 synthetic window. A native
Spike run of that window matched scalar arithmetic, but neither the model's
own values nor a whole-model accelerator route were executed. The selected
local Chipyard build did not expose Verilator or GSIM, so this is a functional
Spike check, not an RTL-simulator certificate.
