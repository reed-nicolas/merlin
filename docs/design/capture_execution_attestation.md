---
title: Capture execution attestation boundary
kind: design
status: current
owner: targetgen
last_verified: 2026-09-27
related: [phase0_specification, model2mlir, reproducibility]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/capture_execution_attestation.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/sealed_static.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/python_preflight.py
  - src/merlin/targetgen/application_inventory.py
---

# Capture execution attestation boundary

A model2MLIR `m2m.capture-receipt.v1` verifies the materialized bundle members against
recorded byte digests. Its `source_closure_verified: false` is a distinct result: the
receipt does not prove which loader, importer, framework, checkpoint, dependency or
ambient file the capture process read. A later digest of today's checkout cannot
prove what an earlier process executed. Existing captures must retain that status.

Merlin reserves `merlin.capture_execution_attestation.v1` for this separate claim.
The diagnostic implementation inventories explicitly selected source bytes and reports
the adjacent materialized receipt. It always writes
`status: diagnostic_only`, `fresh_execution: false`, and
`source_closure_verified: false` to a new evidence path outside the capture and
source trees. Its Phase 0 admission function accepts no verified issuer yet. Editing
these fields or copying an old receipt cannot make a capture admissible.

The separate `merlin.sealed-static-capture.v1` issuer exercises the isolation boundary
for a self-contained static ELF payload. It copies complete selected source and
runtime trees into a private run, rejects links and dynamic executables, and runs
with bubblewrap namespaces and a cleared environment. Only the snapshots are
visible read-only; only a new capture directory is writable. Its replay verifier
checks exact file and directory membership, issuer and bubblewrap bytes, reconstructs
the fixed sandbox policy, and reruns the payload to demand byte-identical output.
Its receipt says `local_sealed_static_execution`; replay returns
`replay_verified_static`. Both keep the generic `source_closure_verified: false`,
because unsigned JSON and replay cannot prove the historical issuing process.
The observed scope is `static_elf_process_only`. This is not a model2MLIR or
PyTorch capture attestation and is not wired into Phase 0 admission.

A future verified issuer must perform a *fresh* capture in a new output directory.
It must privately snapshot the complete loader/importer source, Python runtime and
packages, checkpoint and preprocessing inputs; bind their membership and bytes;
execute only those snapshots with the source and runtime read-only, no network, and
no ambient checkout, home or cache; then verify the source and output bytes again.
The issuer must bind the exact command, environment, isolation controls, fresh run
identity, capture artifact inventory, and materialized receipt into its result.
Only then may a reviewed Python/model issuer be added to the Phase 0 admission gate.

Bubblewrap being installed is insufficient: an ordinary Python virtual environment
may read dependencies and caches outside the declared source selection. Without
a sealed runtime/checkpoint root for the selected model capture, the diagnostic
path must not claim verified source closure. Phase 0's existing
coverage commitment remains blocked on `source_closure_verified: false` until a new
capture and independent verifier are ready.

For a proposed Python capture, run the separate preflight with explicit paths:

```sh
python -m merlin_experiments.capture_execution.python_preflight \
  --worker /selected/merlin/_m2m_capture_worker.py \
  --loader /selected/model2MLIR/workloads/model/loader.py \
  --m2m-root /selected/model2MLIR \
  --python /selected/model2MLIR/.venv/bin/python \
  --capture-receipt /selected/old-capture/capture_receipt.json \
  --output /generated/private-evidence/model-preflight.json
```

The write-once result inventories the selected interpreter and its symlink target,
venv startup hooks and editable imports, direct source files, loader environment
reads, and directly declared Python/Torch ELF dependencies. Supply exact data paths with
`--data-path` and declared loader settings with `--env NAME=VALUE`; the tool does
not inherit ambient environment variables. It always exits 2 with
`blocked_unsealed_python_capture`, `fresh_execution: false`, and
`source_closure_verified: false`. Missing paths and unselected loader inputs are
remediation data, not a closure proof. A feasible next build on a spacious
filesystem is a private, immutable snapshot of the selected venv, CPython base,
editable sources, OS/CUDA libraries, model inputs/checkpoints and capture worker,
then a fresh empty-root, network-disabled bubblewrap run. None of the current
materialized captures may be upgraded by this preflight.

The static loader scan sees literal environment reads in the loader file, not
reads inside imported helpers. For a known delegated requirement, add
`--require-env NAME` (for example, a token corpus path) so an unselected value
is reported. This is a caller declaration, clearly marked in the result; it
does not establish a complete dynamic environment or file-read inventory.

`--capture-receipt` is optional. It compares that older receipt's loader and named
M2M direct-owner digests against the **current** selected checkout, reporting exact
drift or missing files. A match only means those declared files match now: the
receipt does not enumerate transitive Python imports, runtime libraries, or data
reads, and the comparison does not authenticate the earlier process. Rerun a fresh
capture after any drift; never relabel the older one as source-closed.

The preflight also hashes every regular file and directory under the selected
`m2m` package and reports `.py` members absent from the older receipt's named
direct-owner list. It rejects links and special entries in that tree. This
current-tree inventory exposes a concrete gap such as a loader-imported helper
missing from the receipt, and supplies bytes for planning a private source
snapshot. It is still neither a historical source claim nor a complete Python
runtime/import closure.

When the receipt binds a sibling `meta.json`, the preflight verifies those
metadata bytes and cross-checks its observed M2M import-source hashes against
current files and the receipt's direct-owner list. This can expose an imported
helper omitted from that list. The worker records only modules newly imported
after its observation point, so an empty observed-M2M list does **not** prove
that no M2M modules were executed. Neither metadata nor the receipt authenticates
the historical process. Observed paths with symlinked ancestors or parent
traversal are rejected before their target bytes are read.

The same comparison also lists observed non-package modules under the selected
model2MLIR checkout, such as a workload loader imported by a thin Merlin example
adapter. Those entries appear as `selected_checkout_sources` with observed and
current hashes, separately from `selected_m2m_sources`. They are not silently
absorbed into the receipt's direct-owner list, and a matching hash still does not
prove a complete import or checkpoint-data closure.
