# TVM host setup

Build the [pinned independent TVM checkout](target.yaml) with host LLVM support, then select a workload recipe below. Host/frontend diagnostics, complete random integer ResNet functional checks and model-specific development results have separate gates; none establishes paper accuracy, deployment identity or timing.

The host cache enables the local `USE_HOST_ONLY_AUTO_COPY_GUARD` patch because this restricted checkout lacks `LowerAutoCopy`. The guard preserves unannotated IR and rejects automatic-copy annotations; it does not implement their optimization. This option defaults to OFF and must not coexist with the full implementation. Record the local patch and enabled mode with the base commit.

Run from the Merlin comparison checkout. Set absolute `TVM_ROOT` (independent source) and `TVM_BUILD` (host library) paths; TVM is not a Merlin submodule. Support outputs honor `MERLIN_OUT_ROOT` through `build_dir`. Tested host tools are Python 3.10.14, GCC 11.5, CMake 3.26.5 and Ninja 1.13.2. Override discovery with absolute `TVM_HOST_PYTHON`, `TVM_HOST_CMAKE`, `TVM_HOST_CC` or `TVM_HOST_CXX`; `TVM_BUILD_JOBS` defaults to 4. Do not source Chipyard's environment for this host build.

The newest source pin is a local development commit until the fork is published; use the existing checkout or a local clone containing that commit. A fresh remote clone can reproduce the new pin only after publication.

For a new TVM checkout, clone without checkout and apply the mandatory name exclusions before materializing files. These commands select the source commit recorded in `target.yaml`; they do not switch an existing checkout. If you already have that commit checked out, retain it and set the same two paths instead.

```bash
export TVM_ROOT=/absolute/path/to/tvm-gemmini
export TVM_BUILD=/absolute/path/to/tvm-host-build
tvm_commit=fc236e0b21e726fefd9411be9f18384425663537
git clone --filter=blob:none --no-checkout https://github.com/reed-nicolas/tvm.git "$TVM_ROOT"
git -C "$TVM_ROOT" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
git -C "$TVM_ROOT" checkout --detach "$tvm_commit"
```

In the same shell, verify the selected source pin and prepare the host environment:

```bash
: "${TVM_ROOT:?Set TVM_ROOT to the independent TVM checkout}"
: "${TVM_BUILD:?Set TVM_BUILD to the TVM host build directory}"
test "$(git -C "$TVM_ROOT" rev-parse HEAD)" = fc236e0b21e726fefd9411be9f18384425663537 || { echo 'TVM source pin mismatch' >&2; exit 1; }
merlin_root="$PWD"
setup_dir="$merlin_root/examples/gemmini/comparisons/tvm"
host_python="${TVM_HOST_PYTHON:-$(command -v python3.10)}"
host_cmake="${TVM_HOST_CMAKE:-$(command -v cmake)}"
host_cc="${TVM_HOST_CC:-$(command -v gcc)}"
host_cxx="${TVM_HOST_CXX:-$(command -v g++)}"
: "${host_python:?Set TVM_HOST_PYTHON to a Python 3.10 executable}"
: "${host_cmake:?Set TVM_HOST_CMAKE to a CMake executable}"
: "${host_cc:?Set TVM_HOST_CC to a GCC executable}"
: "${host_cxx:?Set TVM_HOST_CXX to a G++ executable}"
build_jobs="${TVM_BUILD_JOBS:-4}"
host_tools_path="$(dirname "$host_cmake"):$(dirname "$host_cc"):$(dirname "$host_cxx"):/usr/bin:/bin"
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
unset CMAKE_PREFIX_PATH PYTHONPATH PYTHONHOME CC CXX CFLAGS CXXFLAGS LDFLAGS
export PYTHONDONTWRITEBYTECODE=1
build_root="$(PYTHONPATH="$merlin_root/src" "$host_python" -c 'from merlin.common.paths import build_dir; print(build_dir() / "baselines" / "tvm-gemmini")')"
mkdir -p "$build_root"/{cache/pip,tmp,logs}
export PIP_CACHE_DIR="$build_root/cache/pip" TMPDIR="$build_root/tmp"
"$host_python" -m venv "$build_root/venv"
source "$build_root/venv/bin/activate"
python -m pip install -r "$setup_dir/host-requirements.txt"
host_ninja="$build_root/venv/bin/ninja"
export PATH="$build_root/venv/bin:$host_tools_path"
```

The three required TVM submodules are `dmlc-core`, `dlpack`, and `rang`. Initialize them at the recorded gitlinks with the workspace's required sparse exclusions, or preserve existing matching checkouts. Verify them before building:

```bash
git -C "$TVM_ROOT" submodule status -- 3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
```

A leading space in each status line means the pin matches; `-`, `+`, or `U` requires repair before building. For a fresh checkout, initialize only these three dependencies at the gitlinks recorded in TVM commit `fc236e0b21e726fefd9411be9f18384425663537`. Clone each dependency without checkout and apply the same exclusions before checkout:

```bash
git -C "$TVM_ROOT" submodule init -- 3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
while read -r dependency revision repository; do
  git clone --no-checkout "$repository" "$TVM_ROOT/3rdparty/$dependency"
  git -C "$TVM_ROOT/3rdparty/$dependency" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
  git -C "$TVM_ROOT/3rdparty/$dependency" checkout --detach "$revision"
done <<'DEPENDENCIES'
dmlc-core 3031e4a61a98f49f07a42cfdec6242340fb2fd8c https://github.com/dmlc/dmlc-core.git
dlpack e2bdd3bee8cb6501558042633fa59144cc8b7f5f https://github.com/dmlc/dlpack.git
rang cabe04d6d6b05356fa8f9741704924788f0dd762 https://github.com/agauniyal/rang.git
DEPENDENCIES
git -C "$TVM_ROOT" submodule status -- 3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
```

Preserve existing matching dependency checkouts. Do not initialize optional dependencies recursively.

A compatible complete LLVM 18 SDK can replace the source build below. Clone LLVM 18.1.8 (`3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff`) without checkout, then select only `llvm`, `cmake`, and `third-party`, excluding forbidden names before materializing files. Reuse the existing checkout when present rather than cloning over it.

```bash
llvm_source="$build_root/llvm-project"
llvm_build="$build_root/llvm-build"
git clone --filter=blob:none --no-checkout --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git "$llvm_source"
git -C "$llvm_source" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!/*/
/llvm/
/cmake/
/third-party/
!/llvm/test/
!/llvm/unittests/
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
git -C "$llvm_source" checkout --detach llvmorg-18.1.8
"$host_cmake" -S "$llvm_source/llvm" -B "$llvm_build" -G Ninja -DCMAKE_MAKE_PROGRAM="$host_ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$host_cc" -DCMAKE_CXX_COMPILER="$host_cxx" \
  '-DLLVM_TARGETS_TO_BUILD=X86;RISCV' -DLLVM_ENABLE_PROJECTS= -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_DYLIB_COMPONENTS=all -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_INCLUDE_TOOLS=ON -DLLVM_BUILD_TOOLS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF
"$host_cmake" --build "$llvm_build" --target LLVM llvm-config --parallel "$build_jobs"
# LLVM 18 creates this compatibility name during installation, but not in its build tree.
"$host_cmake" -E create_symlink libLLVM.so.18.1 "$llvm_build/lib/libLLVM-18.so"
"$llvm_build/bin/llvm-config" --version --targets-built --shared-mode --link-shared
```

Consume the LLVM build tree directly, retaining its source and generated include directories. `LLVM_INCLUDE_TOOLS=ON` is required for the shared-library target; requesting only `LLVM` and `llvm-config` avoids unrelated tools. Configure TVM with the checked-in host cache and an RPATH to this LLVM library directory.

```bash
"$host_cmake" -S "$TVM_ROOT" -B "$TVM_BUILD" -G Ninja -C "$setup_dir/host-config.cmake" -DCMAKE_MAKE_PROGRAM="$host_ninja" -DCMAKE_C_COMPILER="$host_cc" -DCMAKE_CXX_COMPILER="$host_cxx" -DUSE_LLVM="$llvm_build/bin/llvm-config --link-shared" -DCMAKE_BUILD_RPATH="$llvm_build/lib"
"$host_cmake" --build "$TVM_BUILD" --parallel "$build_jobs"
export PYTHONPATH="$TVM_ROOT/python"
export TVM_LIBRARY_PATH="$TVM_BUILD"
export TVM_FFI=ctypes
export PYTHONDONTWRITEBYTECODE=1
export TEST_DATA_ROOT_PATH="$build_root/cache/tvm-test-data"
python "$TVM_ROOT/apps/gemmini/verify_host.py" --output "$build_root/host-smoke.json"
```

To reuse a build in another shell, restore `merlin_root`, `setup_dir`, `build_root`, `TVM_ROOT` and `TVM_BUILD` for that checkout, activate `"$build_root/venv/bin/activate"`, and reapply the TVM environment exports above. A locally generated activation helper is optional and must match the current filesystem layout; this recipe does not require one.

Keep receipts, source patches, manifests and build logs under the build root. Each numerical run requires a fresh output directory and records the selected sources/artifacts, dependency locations and loaded library identity. Source commits do not relabel an older C++ build. Hashes are not complete runtime source closure; tensor budgets exclude executable/stack/platform storage unless explicitly accounted for. Host/export/functional results do not qualify deployment or timing.

## ONNX frontend verification

Install [frontend-requirements.txt](frontend-requirements.txt) into the host environment. The verified combination is Python 3.10.14, PyTorch 2.10.0+cu128 executing on CPU, ONNX 1.17.0, protobuf 5.29.5 and NumPy 1.26.4. ONNX 1.17 retains the `onnx.mapping` API used by this TVM revision. The verifier explicitly uses the legacy exporter (`dynamo=False`), matching the existing Merlin runner's primary path; it does not require ONNXScript, ONNX Runtime or torchvision.

```bash
python -m pip install -r "$setup_dir/frontend-requirements.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TVM_NUM_THREADS=1
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/current"
```

The pinned frontend packages execute on CPU even when the framework distribution includes CUDA support. Reports record actual package versions and locations; reusing an existing framework installation is optional.

The opset17/18 matmul, batched matmul, convolution, LayerNorm and RMSNorm fixtures compare Torch, ONNX ReferenceEvaluator and Relax LLVM with identical shapes/dtypes and `rtol=1e-4, atol=1e-5`. They are synthetic frontend checks, not model/device qualification. Receipts retain ONNX, imported IR, arrays and compiler/exporter/importer identities.

The loaded library retains its actual build revision even when later Python importer fixes are selected.

For a focused importer ablation, retrieve only the exact Python importer and load it through `--importer-source`; keep the current runtime and branch unchanged:

```bash
git -C "$TVM_ROOT" show c4dc0c29ff81ddae688da24625a603da3b4a2c0e:python/tvm/relax/frontend/onnx/onnx_frontend.py > "$build_root/frontend/apache-importer.py"
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/apache-importer" --importer-source "$build_root/frontend/apache-importer.py"
git -C "$TVM_ROOT" show d608061677e535f12b64c072d7a109a645bcca73:python/tvm/relax/frontend/onnx/onnx_frontend.py > "$build_root/frontend/reduction-fix-importer.py"
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/reduction-fix-importer" --importer-source "$build_root/frontend/reduction-fix-importer.py" --case rms_norm
```

The historical Apache importer ablation is expected to fail opset18 RMSNorm; the reduction-axis fix passes both RMSNorm opsets. These are Python-importer comparisons on the current runtime, not complete historical TVM builds.

## Focused importer regressions

```bash
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --suite importer --output-dir "$build_root/frontend/importer-dtype-fixed"
```

The independent schema-checked importer suite covers shape arithmetic, constant/runtime negative Gather, lower-rank Expand and scalar ConstantOfShape at both opsets. Integer dtype, shape and values must agree exactly with NumPy/ONNX reference results.

| Inherited change | Evidence and remaining limit |
| --- | --- |
| `537a5e0`: binary scalar/tensor folding | Apache's importer fails the valid Shape → Gather → Add graph. The inherited fix imports it but silently narrows int64 arrays to int32. The local follow-up passes `arr.dtype` explicitly to `relax.const`; the regression preserves `1099511627785`, previously truncated to `9`. |
| `75791d9`: negative Gather | Current constant/runtime integer-index cases pass. Historical importer overrides retain IR without executing unchecked negative indices. Mixed-dtype ScatterND was not tested because that case violates the ONNX schema. |
| `db359fe`: Gather casts/normalization | Valid runtime integer Gather passes, but its imported IR is identical with `75791d9` alone, so this case does not establish a need for the later normalization change. Float-index Gather is outside the valid ONNX gate. |
| `fb78e0e`: Expand/ConstantOfShape | Lower-rank Expand and scalar ConstantOfShape fail import immediately before this patch and pass with it. |

Historical importer evidence uses identical saved ONNX graphs. Current BF16 softmax/GELU/sigmoid probes and signed `MatMulInteger`/ties-to-even rounding support are separate focused checks; none establishes complete model compatibility.

## ResNet50 host structural diagnostic

Use `--graph-mode both` to compare ordinary and explicit graph-optimization paths on identical weights/images; `optimized` selects only TVM's `zero` plus recorded `default_build` pipeline. Baseline remains the default. Every mode saves actual IR/inventories and uses the same numerical gates.

[verify_resnet.py](verify_resnet.py) uses the complete torchvision ResNet50 v1.5 architecture (model2MLIR loader revision `7915e23475c6db446a3c404847b11e8bc72c8a27`) with seeded random weights and two synthetic images by default. Add torchvision 0.25.0+cu128 and Pillow 12.1.0 to the Python3.10 frontend pins; this does not invoke model2MLIR's Python3.12 capture environment.

```bash
model2mlir_source=/absolute/path/to/model2MLIR
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TVM_NUM_THREADS=4
python "$setup_dir/verify_resnet.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --output-dir "$build_root/resnet50-v1_5"
```

The diagnostic isolates inherited dataset/calibration settings, preserves zero-valued parameters and checks loaded state before/after. It validates the full [3,4,6,3] stage geometry and 25,557,032 parameters: FP32 NCHW `[1,3,224,224]` to `[1,1000]`, legacy ONNX opset17, 53 convolutions and one classifier. Both images/modes pass `rtol=atol=1e-4`; maximum recorded absolute error is `9.16e-5`. Saved graphs, IR, inventories and input/reference/output arrays bind loader/framework/model/input identities. This is a synthetic structural diagnostic.

Add `--batchnorm-folding --graph-mode both` to check a separate evaluation-only folded copy against the unchanged FP32 model before export. All 53 Conv/BatchNorm sites pass the random diagnostic at existing tolerances; receipts bind original/derived tensors and each epsilon. Training mode, missing statistics and incompatible/nonfinite values are rejected.

## Supplied ResNet50 artifacts

Supply all four options below together to compare a local checkpoint against every supplied image. Omitting them retains the two-image random diagnostic. This mode uses the same host dependencies and never downloads weights.

```bash
python "$setup_dir/verify_resnet.py" \
  --model2mlir-root /absolute/path/to/model2MLIR --tvm-source "$TVM_ROOT" \
  --checkpoint /absolute/path/to/resnet50-state-dict.pt \
  --inputs /absolute/path/to/images.npz \
  --input-source 'declared dataset/split/sample-list identifier' \
  --preprocessing 'declared preprocessing already applied' \
  --output-dir "$build_root/resnet50-supplied-new-run"
```

Use a new output directory for each invocation; an existing directory is rejected to preserve prior evidence. The checkpoint must be a plain torchvision ResNet50 state dict, loaded on CPU with `weights_only=True`; keys, tensor dtypes and shapes must match exactly, and all values must be finite. Nested training checkpoints and implicit conversion are rejected.

The NPZ must contain an `images` array of float32 values shaped `[N,3,224,224]` or `[N,1,3,224,224]`, with positive N and finite values. No dtype conversion or preprocessing is applied. One LLVM build checks every image against PyTorch at the same fixed tolerances as the diagnostic.

Optionally add `--labels /absolute/path/to/labels.json` in supplied-artifact mode for descriptive classification accuracy. The JSON must contain exactly `input_sha256`, `class_ids` and `samples`. `input_sha256` binds the complete supplied NPZ file. `class_ids` lists 1000 unique nonempty strings in classifier logit order. `samples` lists one object per image in NPZ order, each with exactly a nonempty string `id` and integer `label` from 0 to 999; Boolean labels are rejected. Duplicate sample IDs and repeated images are allowed because the image index identifies each occurrence. For example, a two-image stream uses this structure (replace the abbreviated class list with all 1000 identifiers):

```json
{"input_sha256": "<images.npz SHA-256>", "class_ids": ["class-0", "class-1", "...", "class-999"], "samples": [{"id": "sample-a", "label": 7}, {"id": "sample-b", "label": 2}]}
```

Reports bind file hashes, sample order and declared class mapping, with per-image top-five indices, top-one/top-five hits and full-stream `descriptive_accuracy`. Ties rank by class index; these descriptive metrics do not change FP32 numerical pass/fail.

Preprocessing, dataset/class/label truth and checkpoint training provenance remain declarations. FP32 mode performs no calibration and records `paper_validation: false`; omitting `--labels` keeps numerical-only verification.

Focused artifact tests cover complete-stream preservation, invalid checkpoint/image/label admission, environment isolation, ranking ties and aggregation:

```bash
PYTHONPATH="$merlin_root/src:$PYTHONPATH" python "$merlin_root/merlin/tests/gemmini/test_tvm_resnet_artifacts.py"
```

## Integer ResNet development recipe

[resnet_quantized.py](resnet_quantized.py) keeps the complete classifier and an independent integer reference. The current v6 development recipe uses signed symmetric INT8, power-of-two per-channel weight/per-tensor activation scales, checked INT32 contractions/bias, ties-away-from-zero requantization, the declared accumulator residual policy and integer pooling. Weight scales coarsen only when folded bias plus worst-case contraction would overflow; bias is never saturated. TRAIN-only activation calibration supports max-abs and bounded MSE selection, sole-consumer ReLU range propagation and pooled-output calibration with one rounding of the scaled spatial sum. Manifests record observed ranges, graph paths, chosen scales and integer bounds; evaluation tensors/labels never select coefficients. All 53 convolutions and the classifier use Gemmini; packing, requantization, residuals, pooling and FP32 logit dequantization are CPU work. Calibration differs from evaluated tensors by content, and the unchanged FP32 checkpoint is the quality reference.

Use the supplied-artifact command above with these additional options:

```bash
python "$setup_dir/verify_resnet.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" \
  --checkpoint /absolute/path/to/resnet50-state-dict.pt --inputs /absolute/path/to/evaluation.npz \
  --input-source 'declared evaluation split/sample list' --preprocessing 'declared resize/crop/normalization' \
  --labels /absolute/path/to/labels.json --quantized --calibration-inputs /absolute/path/to/calibration.npz \
  --calibration-source 'distinct calibration split/sample list' --activation-estimator mse --residual-policy accumulator \
  --graph-mode both --export-baremetal \
  --max-top1-drop 0.02 --memory-limit-bytes 268435456 --output-dir "$build_root/resnet50-integer-new-run"
```

Every supplied image is checked for exact raw int32 and dequantized FP32 agreement. Labels add a development top-one rate-loss gate versus the original FP32 model; the default 0.02 is a proposed development threshold, not approved paper accuracy. Dataset/class declarations and pretrained status remain unqualified by this generic verifier. Tensor-budget admission excludes executable/stack/platform reservations. The optional RV64 export does not execute or measure the model.

The full random 224×224 model passes both host modes/static exports and an optimized matched-Spike two-image repeat session. Recorded optimized tensor storage is 34,679,680 bytes; the linked diagnostic reserves 35,065,808 bytes in a requested 256-MiB map. These historical random-model receipts establish functional feasibility only. Trained v6 now passes both exact host modes and its development quality gate (16/20 top-one, matching FP32); frozen final100 also passes both modes, with top-one 87/100 versus FP32 88/100 at unchanged 0.02 loss gate. Selected development and final device images pass complete-model matched-Spike repeats/controls, with source bindings accepted. This is not device execution of the entire 100-image stream or canonical ImageNet paper accuracy. Numerical admission tests:

```bash
PYTHONPATH="$merlin_root/src:$PYTHONPATH" python "$merlin_root/merlin/tests/gemmini/test_tvm_resnet_quantized.py"
```

## Local Imagenette development inputs

[prepare_resnet_inputs.py](prepare_resnet_inputs.py) reads a local [Imagenette320 archive](https://github.com/fastai/imagenette) without extraction or downloads. It uses the installed [ResNet50 V2 transform](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), selects sorted train images for calibration and validation images for evaluation, and retains the full 1000-class head. The small diagnostic defaults to one calibration and two evaluation images per class; the TRAIN80/development20/final100 preparation below is the current frozen development recipe. Counts are explicit; no selected image is dropped during verification. Archive/member, RGB/tensor, category/transform and selected source hashes are recorded; duplicate/overlapping content and unsafe archive entries are refused. The ten-class subset is a development check, not canonical held-out ImageNet accuracy.

From the activated comparison checkout, the artifact cache resolves through the existing `artifacts_dir` helper:

```bash
if model_cache="$(PYTHONPATH="$merlin_root/src:$PYTHONPATH" python -c 'from merlin.common.paths import artifacts_dir; print(artifacts_dir() / "cache")')" && [ -n "$model_cache" ]; then
  mkdir -p "$model_cache/resnet50" &&
  curl --fail --location --retry 3 --continue-at - --output "$model_cache/resnet50/resnet50-11ad3fa6.pth" https://download.pytorch.org/models/resnet50-11ad3fa6.pth &&
  curl --fail --location --retry 3 --continue-at - --output "$model_cache/resnet50/imagenette2-320.tgz" https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
fi
input_run="$model_cache/resnet50/imagenette-v2-development"
python "$setup_dir/prepare_resnet_inputs.py" --archive "$model_cache/resnet50/imagenette2-320.tgz" --output-dir "$input_run"
python "$setup_dir/verify_resnet.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" \
  --checkpoint "$model_cache/resnet50/resnet50-11ad3fa6.pth" --inputs "$input_run/evaluation.npz" --labels "$input_run/labels.json" \
  --input-source 'Imagenette320 val; ordered samples in preparation.json' --preprocessing 'torchvision V2 RGB/resize232/crop224/normalization' \
  --quantized --calibration-inputs "$input_run/calibration.npz" --calibration-source 'Imagenette320 train; ordered samples in preparation.json' \
  --graph-mode both --export-baremetal --memory-limit-bytes 268435456 --output-dir "$build_root/resnet50-v2-integer-development"
```

Both commands require new output directories. The preparer binds supplied archive/member/transform bytes but does not authenticate training provenance or the official archive digest. Verification retains its development quality threshold and fails when the selected integer recipe exceeds it. Artifact tests cover V2 transforms, NPZ/labels, archive bounds, content separation and source mutation.

Integer exports emit `reference.npz` and `reference.json` from the independent integer oracle for every selected image. Follow the TVM [exported-graph simulator recipe](../../../../third_party/baselines/tvm-gemmini/apps/gemmini/README.md#complete-exported-graphs-in-gemmini-spike) with `baremetal_baseline` or `baremetal_optimized`. Detailed mode-specific export/memory artifacts are hashed; `results.json` summarizes storage. Export alone executes no device workload.

The optional `--activation-estimator mse` freezes power-of-two scales from bounded TRAIN reconstruction error; labels and evaluation tensors never choose scales. Prepare the larger disjoint development/final streams with:

```bash
python "$setup_dir/prepare_resnet_inputs.py" --archive "$model_cache/resnet50/imagenette2-320.tgz" \
  --calibration-per-class 8 --evaluation-per-class 2 --final-per-class 10 \
  --selection-seed resnet50-v2-train-final-v1 --output-dir "$model_cache/resnet50/train80-final100-new"
```

Set `RESNET_CHECKPOINT`, `RESNET_TRAIN_INPUTS` and a new `FROZEN_QUANTIZATION` JSON path. Create the v6 manifest from TRAIN images only, without evaluation or CLI-added source prose:

```bash
PYTHONPATH="$setup_dir:$merlin_root/src:$PYTHONPATH" python - <<'PYTHON'
import json, os
from pathlib import Path
import numpy as np
import torch, torchvision
import verify_resnet, resnet_quantized

torch.set_num_threads(2)
model = torchvision.models.resnet50(weights=None).cpu().eval()
verify_resnet.load_checkpoint(model, os.environ["RESNET_CHECKPOINT"], torch)
calibration, identity = verify_resnet.load_images(os.environ["RESNET_TRAIN_INPUTS"], np)
plan = resnet_quantized.calibrate_resnet50(
    model, calibration,
    calibration_ids=[f"{identity['sha256']}:{i}" for i in range(len(calibration))],
    checkpoint_declaration="local supplied state_dict",
    activation_estimator="mse", residual_policy="accumulator")
with Path(os.environ["FROZEN_QUANTIZATION"]).open("x") as stream:
    json.dump(plan.manifest(), stream, indent=2, sort_keys=True)
PYTHON
```

Add `--activation-estimator mse --residual-policy accumulator --frozen-quantization "$FROZEN_QUANTIZATION"` to the integer verification command. Recomputed TRAIN coefficients must match exactly before evaluation. Final evaluation additionally uses `--evaluation-role held-out`, `final.npz` and `final_labels.json`; keep that stream untouched during development. `--device-sample-indices 0,18` selects simulator fixtures only after every host image passes numerical/quality checks; device images remain a subset. Current v6 development and frozen final100 host gates pass. Final device fixtures use predeclared indices 0,90; selected complete-model simulation passes, with all numerical/quality gates unchanged.

## Stateful host sessions

[verify_session.py](verify_session.py) consumes a local factory's `get_model_and_inputs` and model2MLIR's `ExternalRuntimeSession` protocol. An optional `get_session_spec(model, inputs)` supplies single-program metadata; multi-program models expose their existing session capability. Stage order, streams, state routes and reset behavior come from that protocol. The verifier compiles separate static signatures when state shapes change and reuses them across repeats.

```bash
python "$setup_dir/verify_session.py" --model2mlir-root "$model2mlir_source" \
  --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
  --factory /absolute/path/to/loader.py --factory-arguments /absolute/path/to/arguments.json \
  --artifact /absolute/path/to/checkpoint --output-dir "$build_root/session-new-run"
```

Use a fresh output directory. Factory arguments are a JSON object of keyword arguments; omit the option if unnecessary. Each `--artifact` names a local file to hash and may be repeated. Supply the chosen factory's own dependencies and complete local checkpoints/inputs. Python network connections and HuggingFace downloads are disabled during factory loading and verification; this is not a process sandbox. No weights are downloaded, cast or quantized by the verifier.

PyTorch and TVM maintain independent routed state. Every stage output and state update must agree, and restoring initial inputs must reproduce the full session. Reports retain source/artifact hashes, loaded parameter digests, input identities, per-stage IR, compilation signatures and comparisons. Failed tensor comparisons record stage/index, dtype/shape, tolerances and error metrics without accepting the session. Validated failures also retain actual/reference snapshots in `failed_comparison.npz` when their combined payload fits 64 MiB; BF16 uses raw uint16 words with encoding metadata. Invalid tensors and oversized failures retain diagnostics without snapshots; write failures preserve the numerical failure and clean partial artifacts. The input-state byte count is diagnostic accounting, not deployed peak memory. Quality, paper, device and timing qualification remain false.

Original BF16 tensors use exact uint16 payload views through the optional `ml_dtypes` dependency; comparison casts do not alter their stored dtype or bits. Export runs on a deep copy of each source stage because modern tracing mutates ordinary runtime cache attributes into FakeTensor objects. The independent source stage and its reset behavior remain intact. These host export/VM copies consume additional memory; retained input-state bytes are not an allocation peak or target-fit claim.

`--onnx-exporter dynamo` explicitly selects the modern Torch exporter (default opset 18); the default legacy exporter remains opset 17. `--opset` overrides the schema when supported by the installed ONNX package. Native BF16 convolution requires opset 22. The tested optional host packages for the modern recurrence and TinyLlama path are:

```bash
python -m pip install 'ml_dtypes==0.5.4' 'onnxscript==0.5.6' 'onnx_ir==0.1.12'
```

Modern export also preserves original FP64 `scalar_tensor` values through a typed DOUBLE constant in [bf16_export.py](bf16_export.py), avoiding the built-in FLOAT-to-DOUBLE narrowing. Other supported scalar dtypes retain the built-in translation; implicit dtype follows Torch's current default. Non-CPU devices, nondefault layouts and pinned allocation are refused. This repair is independent of the optional BF16 math mode and is source-bound in stage receipts.

`--bf16-fused-export` is an optional modern-export candidate implemented in [bf16_export.py](bf16_export.py). It retains public ATen linear/attention/single-output LayerNorm/SiLU boundaries, original BF16 operands/results and FP32 internal math, and disables ONNX export optimization to preserve initializer dtypes. The receipt binds the translation source and retained ATen operation counts. Masks, fully masked rows, causal attention and explicit scale are checked; dropout, grouped-query expansion and incompatible dtypes are refused. LayerNorm validates static normalized/affine shapes, permits symbolic leading dimensions and preserves one BF16 terminal cast; native auxiliary statistics are outside the translation. SiLU avoids an added BF16 sigmoid store; directed tests pass both graph modes, including signed zero and unchanged default/FP32 exports. This mode does not reproduce every native CPU kernel's reduction/rounding and still must pass the unchanged source comparison. It selects no whole-model FP32 conversion or relaxed acceptance. Ordinary legacy and modern export remain the defaults.

Factories can expose `get_verification_artifacts(model, inputs)` to bind the checkpoint, tokenizer, input and supporting source files they actually select. Those files are hashed before session execution and checked again afterward. ONNX external tensor files are validated before loading, included in each stage's `onnx_artifacts` and rechecked after execution. The verifier uses the path-based ONNX checker to support external-weight models beyond the in-memory protobuf limit; the focused tests exercise smaller external-weight graphs, invalid locations/ranges/metadata and mutation rejection. These identities do not establish complete runtime source closure.

Focused session tests cover fused BF16 boundaries/masks, real LLVM recurrence/growing state, exact payload routing, external weights, cache/reset isolation and failed numerical receipts. Enable the modern export fixture with the optional dependencies above:

```bash
TVM_SESSION_MODERN_ONNX=1 TVM_SESSION_MODEL2MLIR="$model2mlir_source" PYTHONPATH="$merlin_root/src:$PYTHONPATH" \
  python "$merlin_root/merlin/tests/gemmini/test_tvm_session.py"
```

`get_onnx_export_options(torch)` may supply reviewed operator translations and an exporter optimization choice; it cannot override opset, exporter or verification controls. Integer factories use this hook for standard ONNX `MatMulInteger`, preserving signed INT8 operands and INT32 results. The importer implements ONNX ties-to-even rounding explicitly. Bind each supporting helper through `get_verification_artifacts` as well as the factory source.

After all host checks pass, `--export-baremetal --memory-limit-bytes BYTES` packages checked static signatures with shared constants and derives the declared semantic routes. It writes `baremetal/session/{route.json,fixture.npz,provenance.json}` using original Torch outputs; source reset must preserve exact bytes. The [device verifier](../../../../third_party/baselines/tvm-gemmini/apps/gemmini/README.md#complete-exported-graphs-in-gemmini-spike) consumes that package with `--session-route`, `--fixture` and `--fixture-provenance`. Host allclose does not imply its exact-byte device gate passes. BF16 fixtures retain raw uint16 payloads; implicit conversion, nonfinite tensors and hidden allocators are refused.

The adapter records aligned shared/packed constants, serial workspace and simultaneous prior state/current outputs from actual routes. The tensor budget includes that live plan; ELF, stack, runtime/platform storage and external scratch remain unknown, so production fit stays unqualified. The device verifier additionally retains every invocation output and guarded frozen streams. Its optional `--guest-memory-bytes` uses one requested budget for admission, linking, ELF checks and Spike; the default 256 MiB is a functional map, not deployed capacity. A bounded automatic INT8 session passes source-independent host/export and exact matched-Spike repeat/preservation/failure checks; complete models must pass their own gates before packaging.

For CPU math dependencies, add explicit repeatable `--cpu-runtime-library /absolute/path/to/library.a` arguments to TVM's exported-graph verifier, selecting the toolchain's libm/libc/libgcc closure. The verifier binds archive bytes/ABI and audits the final image; it selects no libraries by default. The bounded math identity smoke and BF16/FP64 device checks pass, while an independent nontrivial sinf fixture retains a one-ULP exact-byte failure. These establish bounded runtime support, not full-model numerical qualification.

## Complete TinyLlama host session

[tinyllama_session.py](tinyllama_session.py) requires the full [pinned TinyLlama checkpoint/tokenizer](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6) and a local UTF-8 corpus. It preserves all 22 layers, uses model2MLIR's functional prefill/decode cache stages, and checks those stages against original full-sequence Hugging Face logits before compilation. The development session uses eight prefill tokens plus three teacher-forced decode tokens; capacity/order/IDs and source hashes are recorded. FP32 explicitly converts the official BF16 weights and is a compiler diagnostic, not original-precision quality qualification.

The passing recipe uses Transformers 5.4.0's cache API and the modern exporter; 4.57.1 lacks the required cache state, and legacy tracing fails current attention-mask construction. Keep the additional dependencies isolated from other model environments:

```bash
python -m pip install --target "$build_root/tinyllama-python-v5" 'transformers==5.4.0' 'huggingface_hub==1.5.0' \
  'tokenizers==0.22.1' 'safetensors==0.6.2' 'numpy==1.26.4' 'onnx==1.17.0' \
  'ml_dtypes==0.5.4' 'onnxscript==0.5.6' 'onnx_ir==0.1.12'
export PYTHONPATH="$build_root/tinyllama-python-v5:$merlin_root/src:$PYTHONPATH"
```

Prepare a JSON arguments file with paths from your installation:

```json
{
  "checkpoint_dir": "/absolute/path/to/TinyLlama-checkpoint",
  "model2mlir_root": "/absolute/path/to/model2MLIR",
  "corpus_path": "/absolute/path/to/local-text.txt",
  "prefill_tokens": 8,
  "decode_tokens": 3,
  "precision": "float32",
  "cpu_threads": 2
}
```

```bash
python "$setup_dir/verify_session.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
  --factory "$setup_dir/tinyllama_session.py" --factory-arguments /absolute/path/to/tinyllama-arguments.json \
  --graph-mode baseline --onnx-exporter dynamo --repeats 2 --max-signatures 2 --output-dir "$build_root/tinyllama-new-run"
TVM_TINYLLAMA_INTEGER_ONNX=1 PYTHONPATH="$merlin_root/src:$PYTHONPATH" python "$merlin_root/merlin/tests/gemmini/test_tinyllama_session.py"
```

This distinct FP32-converted synthetic 8+3 recipe passes both compiled repeats/reset at `rtol=atol=1e-4`: maximum logit error `3.81e-5`, KV error `3.62e-5`, exact positions. The full checkpoint has 1,100,048,384 parameters and 4,400,193,536 converted FP32 parameter bytes. External ONNX weights and isolated source/cache state support export; this recipe supplies no integer-device or original-BF16 quality result.

Add `"quantized": true` to the factory arguments for the explicit W8A8 projection recipe. All 155 learned projections use per-output-channel INT8 weights, dynamic token INT8 activations, exact INT32 contractions and FP32 dequantization; attention, normalization, embedding, masks and cache operations remain explicit CPU FP32. `"activation_scale": "power_of_two"` is the default declared recipe; `"maxabs"` retains the earlier diagnostic. Source weights and sampled complete-K integer references are recorded.

The development language contract requires 128 prefill plus 32 teacher-forced decode tokens, two repeats/reset and at least eight distinct attributed 128-token articles. Its frozen checks are mean next-token NLL increase at most 0.20 nats and top-one agreement at least 0.80 versus the unchanged checkpoint after explicit FP32 conversion. These are development quantization checks, not a canonical paper language benchmark. Prepare the pinned [WikiText validation file](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/validation-00000-of-00001.parquet) locally; `pyarrow` is needed only for preparation:

```bash
PYTHONPATH="$setup_dir:$merlin_root/src:$PYTHONPATH" python - <<'PYTHON'
import json, os
from pathlib import Path
from tinyllama_integer import prepare_wikitext_corpus
arguments = prepare_wikitext_corpus(os.environ["WIKITEXT_PARQUET"], os.environ["TINYLLAMA_CHECKPOINT"], os.environ["CORPUS_OUTPUT"])
arguments.update(checkpoint_dir=os.environ["TINYLLAMA_CHECKPOINT"], model2mlir_root=os.environ["MODEL2MLIR_ROOT"], precision="float32", cpu_threads=2)
Path(os.environ["FACTORY_ARGUMENTS"]).write_text(json.dumps(arguments, indent=2) + "\n")
PYTHON
```

Set the five environment variables to selected local files/new output directories, then use the produced arguments JSON in the session command. Preparation checks the pinned parquet bytes and selects the first eight eligible articles without model feedback. Its fixed 1,016-prediction development quality check passes (NLL increase `0.02836` nats, top-one agreement `90.748%`), but real 128+32 cache/full-sequence comparison fails at length 150 (maximum logit error `0.8066`). Explicit `"workload": "synthetic_feasibility"` permits 8+3 or 128+32 with no quality corpus. Synthetic 8+3 passes the native source gate; its actual ONNX/TVM prefill fails unchanged `1e-4` tolerances (maximum error `1.1724`), before decode/reset/export. Independent scale/code and complete-K int64 checks pass on the actual inputs of 21 projections; a layer 1 MLP input crosses an RNE threshold. Separately, identical layer-6 Q/K/V expose native FP32 attention differences between full-query and cached execution that cross another threshold. No recipe or tolerance changes, and the quality proxy does not qualify the failed complete session.

The separately named `"arithmetic_policy": "fixed_tree_fp32_v1"` is an opt-in W8A8 candidate for QK and RMSNorm mean-square reductions. It uses separately rounded FP32 products followed by balanced adjacent-pair additions, without an initial zero addition, FMA, reassociation or hidden FP64. Other attention/norm operations, parameters, source quality reference and existing gates remain unchanged. Candidate-only wrappers preserve eager causal masks/cache/RoPE, refuse a missing causal mask before updating state, and register custom operators only on explicit use. The two compact `merlin.fixed_order` version-1 ONNX nodes lower to typed opaque CPU helpers with explicit private scratch counted in the static memory plan.

Native QK processes four query rows at a time; it replaces BLAS with elementwise products/tree additions and temporarily retains block outputs for concatenation. For the declared 128-query/160-key shape, product/tree/output storage is about 12.5 MiB excluding inputs, finite checks and allocator overhead; compiled scratch is only `4*K` bytes per helper plus declared outputs/reshape storage. CPU execution costs remain within the eventual timing boundary. Independent saved-operand/native/host and bounded target helper checks pass.

The full-checkpoint candidate passes native and compiled 8+3 sessions. Both compiled graph modes pass two repeats/reset with exact logits/positions and maximum KV error `1.91e-6` under the unchanged `1e-4` gate. The original language proxy also passes (NLL increase 0.02186, top-one agreement 91.339%). Native 128+32 recurrence still fails at length 139 (maximum logit error 0.8016): identical QK/probabilities expose query-shape-dependent rounding in unchanged FP32 probability/value matmul, with masks/RoPE/padding intact. This is not a demonstrated TVM defect. Representative 128+32 compiled/reset, static export and device gates remain. Keep `"arithmetic_policy": "native"` as the unchanged default and preserve its failed receipts; the candidate is not adopted or a measured performance winner.

The separate `"arithmetic_policy": "fixed_tree_fp32_v2_pv"` candidate adds ordered FP32 probability/value reduction. Every short, representative and quality call uses the same declared 160-token bound, padded with positive zero to a 256-leaf tree; padding can change signed zero. Oversized sessions and incompatible static cache/mask geometry are rejected before cache updates. Existing domain-1 helpers suffice. Native, v1, checkpoints, quantization and original quality gates remain unchanged. PV adds padding/tree work inside the eventual timing boundary, with 1 KiB helper scratch plus graph outputs and possible transpose workspace.

V2 passes all 33 native 128+32 recurrence checks with exact logits and the unchanged language proxy (NLL increase 0.01965, top-one agreement 89.862%). Its checkpoint-2 compiled baseline fails the first prefill at the unchanged `1e-4` gate (maximum logit error 1.03475). Compiled decode/repeats/reset and optimized mode were stopped. That historical receipt retains graphs, weights, IR and output hashes, without failed arrays/internal operands; the verifier now retains bounded failure snapshots for new runs. The candidate remains unadopted; static/device gates and deployed memory fit remain unqualified.

Bounded checks on the division-repaired compiler pass the layer-0 residual and identical-input ordered helpers in both modes. Actual saved RoPE initializer words explain layer-0 rotary bit drift before TVM compilation, while the residual stays exact. Separately, identical-native-input softmax/PV checks change one layer-5 INT8 code from 38 to 39; holding native probabilities removes the difference. Independent rounding, ordered-PV and integer-dot checks corroborate this local mechanism, but the projection output still passes `1e-4`. It does not establish the complete prefill failure's cause. These local findings motivated the isolated candidate below; they do not qualify the complete model.

Select `"arithmetic_policy": "fixed_tree_fp32_v3_softmax_reduce"` for the separate softmax-reduction candidate and `"rotary_policy": "source_table_v1"` to preserve native rotary words during export. Each can be selected independently. V3 uses an active-mask maximum, FP32 subtraction, backend exponential, positive-zero masked/tail leaves, the existing 256-leaf ordered sum and FP32 division. Native Torch and compiled TVM exponentials remain distinct; only the reduction order is fixed. Eager mask checks precede cache mutation; exported workloads require the admitted source-generated masks and positions. The rotary option records native configuration-derived tables for positions 0–159, verifies full/cached equality and exports indexed buffers, adding 80 KiB of constants. Softmax adds helper scratch and graph temporaries; CPU costs and final memory fit remain part of later qualification.

The combined candidate passes all 33 native 128+32 checks with exact logits and the unchanged language proxy (NLL increase `0.01698`, top-one agreement `90.059%`). Independent and combined local exports pass both graph modes: rotary Q/K and saved layer-5 INT8 codes/scales agree exactly, while probability/PV values retain small backend differences. Twenty selected exponential inputs agree with an independent high-precision oracle, including subnormal and underflow cases; this is no general correctly-rounded guarantee. Fresh complete baseline prefill still fails the unchanged `1e-4` gate (31,947/32,000 logits, maximum error `1.2591`). Failed actual/reference arrays are retained; compiled decode/repeats/reset and optimized mode were stopped. A propagated graph probe preserves the failed logits and localizes the first changed residual to layer 7 SiLU: one FP32 ULP changes one INT8 code. Replacing that cell with its native value restores the first residual exactly, without proving final-logit recovery. Independent staged arithmetic reproduces the compiled value; direct division gives the same result. A source-preserving nonlinear helper or separately reviewed shared policy remains to be specified; no generic compiler defect is established. The native default, existing v1/v2 recipes, original checkpoints, quality references and acceptance gates remain unchanged; v3 is not adopted.

## Original SmolVLA policy session

[smolvla_session.py](smolvla_session.py) stages the complete [pinned SmolVLA policy](https://huggingface.co/lerobot/smolvla_base/tree/c83c3163b8ca9b7e67c509fffd9121e66cb96205) with a supplied [SmolVLM backbone](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/dc831d5df58eb65ae2fc25b50b7f5a896edec069). Original LeRobot precision is preserved, including BF16 prefix/cache. The selected backbone files are hashed; its training-time revision is unverified. The complete neural session includes the declared cameras, prefix/cache, ten denoising steps and six-coordinate action slicing. Its boundary is preprocessed observations to policy action coordinates; robot execution and postprocessing remain outside the session. Original processor config/statistics/source identities and actual transforms are reported. The pinned checkpoint's namespaced statistics do not map to plain action/state keys, leaving the observed transforms as identity; no namespace is silently selected and normalization/physical units remain unqualified.

Keep this model in a separate Python 3.12 environment: LeRobot 0.5.1 requires its original Transformers 4.57.1 stack. The modern ONNX compatibility probes use Torch 2.10.0, ONNX 1.19.1, onnxscript 0.7.2, NumPy 1.26.4 and ml_dtypes 0.5.4. Do not combine its dependency path with TinyLlama's Transformers 5.4.0 target.

```bash
python3.12 -m venv "$build_root/smolvla-venv"
"$build_root/smolvla-venv/bin/python" -m pip install 'lerobot[smolvla]==0.5.1' 'torch==2.10.0' \
  'transformers==4.57.1' 'onnx==1.19.1' 'onnxscript==0.7.2' 'numpy==1.26.4' 'ml_dtypes==0.5.4' \
  'pyarrow==21.0.0' 'pandas==2.3.3' 'av==15.1.0'
```

Prepare a JSON arguments file with supplied local snapshots:

```json
{
  "checkpoint_dir": "/absolute/path/to/smolvla-policy",
  "backbone_dir": "/absolute/path/to/smolvlm-backbone",
  "model2mlir_root": "/absolute/path/to/model2MLIR",
  "fixture_kind": "synthetic",
  "seed": 0,
  "cpu_threads": 1,
  "precision": "original"
}
```

```bash
PYTHONPATH="$merlin_root/src:$TVM_ROOT/python" TVM_FFI=ctypes "$build_root/smolvla-venv/bin/python" \
  "$setup_dir/verify_session.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
  --factory "$setup_dir/smolvla_session.py" --factory-arguments /absolute/path/to/smolvla-arguments.json \
  --graph-mode baseline --onnx-exporter dynamo --opset 22 --repeats 2 --max-signatures 3 --output-dir "$build_root/smolvla-new-run"
MODEL2MLIR_ROOT="$model2mlir_source" PYTHONPATH="$merlin_root/src" "$build_root/smolvla-venv/bin/python" "$merlin_root/merlin/tests/gemmini/test_smolvla_session.py"
```

Focused staging/routing and native-backend diagnostic tests run through the existing SmolVLA test file. Complete eager staging agrees exactly with unchanged upstream `sample_actions`, including `[1,50,6]` actions and BF16 cache. Synthetic inputs establish reference staging only; attributed fixtures use `fixture_kind: dataset`, `input_npz` and `input_source`, with every selected tensor/camera validated.

The full compiled policy remains unqualified. Its saved integer prefix exports/imports/builds/runs, but fails the unchanged numerical gate before denoising/actions. Generic compiler fixes restore raw BF16 copies and explicit narrowing through fused internal buffers, with independent LLVM checks. Floating constant Pow now uses Relax lowering rather than NumPy pre-folding; a source-input local cache replay is raw-exact for both K/V outputs in both graph modes and repeats/reset. This local check does not qualify the complete prefix. Checkpoint precision and active tolerances remain unchanged.

[diagnose_smolvla_backends.py](diagnose_smolvla_backends.py) compares complete unchanged native CPU policies under default flash and forced MATH attention. It records prefix/cache, every denoising update and final actions, with identical observation/noise hashes, exact within-backend repeats, source/parameter/cache preservation and backend/RNG restoration. Profiling attributes the native attention calls to vision-prefix processing; denoising uses LeRobot's manual eager attention. Three synthetic development fixtures show differences propagating into final actions beyond the existing 1e-4 comparison. This informs a proposed BF16 acceptance contract; the active compiled-policy gate and original dtypes remain unchanged.

```bash
diagnostic_root="$(PYTHONPATH="$merlin_root/src" "$build_root/smolvla-venv/bin/python" -c 'from merlin.common.paths import artifacts_dir; print(artifacts_dir() / "verification/gemmini/tvm")')"
PYTHONPATH="$merlin_root/src" "$build_root/smolvla-venv/bin/python" "$setup_dir/diagnose_smolvla_backends.py" \
  --checkpoint-dir /absolute/path/to/smolvla-policy --backbone-dir /absolute/path/to/smolvlm-backbone \
  --seeds 0 1 17 --output-dir "$diagnostic_root/smolvla-native-backends-new-run"
```

The output directory must be fresh and beneath Merlin's configured artifacts root. For an attributed observation fixture, add `--fixture-kind dataset --input-npz /absolute/path/to/fixture.npz --input-source 'revision/episode/frame and preprocessing'` and select one seed; the supplied NPZ binds its own initial noise. Supplied synthetic fixtures also require an explicit input source. Saved BF16 tensors retain exact uint16 payloads; representable-step distances collapse signed zero and can grow large near zero/sign crossings. Raw encoding counts separately distinguish signed zero. No replacement tolerance, robot-quality criterion or compiled-policy acceptance is selected by this diagnostic.

For a recorded TRAIN/development observation, also supply `--input-manifest /absolute/path/to/observation.json`. The diagnostic includes a review-only numerical proposal and saved processor/config/statistics inventory. `normalization_metadata_inventory`, `original_processor_diagnostic` and `numerical_contract_proposal` can also generate that proposal from saved metadata without another policy run. The producer rejects held-out proposal inputs, reports `NO_ACCEPTANCE` and leaves operation-specific arithmetic proofs and action limits unresolved; its declared fixture identity is not independently verified by the producer. Exact integer checks use identical actual operands, while propagated floating-input/code drift and original-policy quality remain separate. Missing training camera/backbone identity limits application-reproduction claims; it does not prohibit functional compiler checks against the explicit pinned workload. The current `1e-4` gate remains unchanged.

For recorded observations, [prepare_smolvla_observation.py](prepare_smolvla_observation.py) uses the pinned [SO100 dataset](https://huggingface.co/datasets/lerobot/svla_so100_pickplace/tree/3d6d687a25cdf1565cdf24550814f72d999a861d), saved checkpoint processor and explicit top→camera1/wrist→camera2 mapping. It validates dataset schema/parquet identities, records video bytes/decoder and never invents a third camera or refits normalization. Camera training assignment and training-time backbone identity remain unverified. TRAIN/development/heldout use disjoint whole episodes; recorded replay does not establish closed-loop robot success.

```bash
export HF_DATASETS_CACHE="$model_cache/huggingface/datasets"
PYTHONPATH="$setup_dir:$merlin_root/src:$TVM_ROOT/python" "$build_root/smolvla-venv/bin/python" "$setup_dir/prepare_smolvla_observation.py" \
  --dataset-root "$SO100_DATASET" --checkpoint-dir "$SMOLVLA_CHECKPOINT" --backbone-dir "$SMOLVLM_BACKBONE" \
  --frame-index 0 --split train --noise-seed 0 --video-backend pyav --output-dir "$OBSERVATION_OUTPUT"
```

The recorded integer prefix remains under qualification. The FP64 scalar export repair clears all original-input suffix stages in both graph modes/two fresh resets, but complete velocity still fails the unchanged gate. A baseline expert-layer-0 diagnostic passes first attention; its sole RMSNorm-to-BF16 cast split is independently traced to FP32 reduction order, with activation codes/Q/K/V unchanged. That local split does not explain the complete velocity failure. These checks do not establish complete policy, quality or device acceptance; preserve failed receipts and the unchanged gates.

Actual expert-layer-0 isolation proves an extra BF16 sigmoid store in the original SiLU export and a separate TIR division-to-reciprocal rewrite changing four scale words and two INT8 codes. The opt-in fused translation matches all 102,400 SiLU source words; on the old compiler it alone leaves 582 residual failures. The rebuilt compiler now retains floating division: fresh actual scale/code/gate-up checks are raw-exact against independent arithmetic in both modes/two fresh VMs/reset. Applying both repairs makes all nine controlled MLP boundaries and all 36,000 residual values raw-exact. The legacy staged-SiLU control remains failed. This same-native-input cut does not qualify complete velocity/policy or change maxabs quantization, references or tolerances.

A fresh complete first-velocity check on identical original prefix/cache/noise/time still fails `rtol=atol=1e-4`: 1,552/1,600 outputs differ beyond the gate, maximum absolute error `0.0965571`. Two fresh VMs and repeated calls produce identical outputs while preserving inputs, source parameters and the native integer reference; this is not whole-policy reset qualification. In baseline mode, the corresponding source-input first-expert-layer terminal is raw-exact, including a check that instrumentation preserves it. One internal RMSNorm BF16 cast differs at a rounding midpoint, with both casts independently correct; it does not explain the full velocity failure. Full-graph instrumentation preserves the failed velocity. Native-suffix substitution clears its local residual error without changing final velocity. Native-layer-3-attention substitution restores that residual exactly and changes 895/1,600 final values, but the complete gate still fails (1,550 values versus 1,552 in the control). Layer-3 attention has seven differing BF16 values, two beyond the gate, despite exact Q/K/V inputs. These local differences do not establish a terminal repair or generic compiler defect; later causes remain unresolved. Optimized full velocity, fresh prefix and the dependent ten-step/action/state checks were stopped.

Use PyAV 15.1.0 for the declared decoder. Supply the resulting `observation.npz` and `observation.json` through factory `input_npz`, `input_manifest`, `fixture_kind: dataset`, and an attributed `input_source`; select `["observation.images.camera1", "observation.images.camera2"]`. For the separate W8A8 reference, add `quantized: true` and a fresh `integer_diagnostic_dir` beneath Merlin's artifacts root. The original policy stays intact while a target copy replaces learned linears with INT8/INT32 contractions, FP32 dequantization/bias and one original BF16/FP32 interface cast. Nonlinear/attention/vision-convolution/cache work keeps original CPU precision. Receipts separate executed/inactive projections, sampled complete-K integer checks, all-ten-step same-state velocity comparisons and independent final trajectories. Complete eager/reset checks pass on a recorded TRAIN observation; original-policy quality and compiled/device qualification remain pending.

`diagnose_smolvla_backends.py --velocity-replay` isolates denoiser arithmetic using identical states, timesteps, masks and prefix caches. A preserved-operand FP64 fused-linear diagnostic supplies a conditional dtype-aware arithmetic bound; it assumes FP32 operations/gradual underflow and one original BF16 output cast. That bound does not qualify native kernels, softmax/attention or full-trajectory amplification, and selects no replacement for the active 1e-4 gate.

## Provisional target checks

[target.yaml](target.yaml) declares the source/arithmetic/runtime profile and hypothetical memory; its checker does not establish hardware execution. From this checkout:

```bash
python "$setup_dir/check_target.py" --config "$setup_dir/target.yaml"
PYTHONPATH="$merlin_root/src:$PYTHONPATH" python "$merlin_root/merlin/tests/gemmini/test_tvm_target.py"
```

The checker rejects incompatible arithmetic, hardware-loop/WS allowances, invalid/overlapping ranges and requests beyond declared confirmed capacity. Source fields are declarations; adapter build verification separately checks exact header bytes and compiler includes. Successful checks leave paper/deployment qualification false.

Optionally add `--memory-plan /path/to/reservations.json`. Its `regions` list holds disjoint coarse reservations with `name`, integer `address` and integer `size_bytes`; account for code/static data, weights, arena/workspace, stack and runtime storage. For example, `{"regions":[{"name":"code_and_static","address":2147483648,"size_bytes":1048576}]}` reserves only the first MiB. Do not also list allocations within an arena as overlapping top-level regions. Unreserved gaps are not measured free memory, and omitted allocations remain unknown. The 8-GiB planning map is hypothetical; exported-graph functional checks use an explicit requested map, defaulting to 256 MiB.

## Pinned functional simulator

[build_simulator.py](build_simulator.py) stages exact integer libgemmini Git blobs, checks installed Spike headers and tests explicit plugin loading with a known CPU-only ELF. It binds sources/tools/dependencies without changing reference checkouts; loading alone executes no Gemmini arithmetic. Set these paths to existing selected installations and passing adapter/CPU builds:

```bash
python "$setup_dir/build_simulator.py" --libgemmini-repo "$LIBGEMMINI_REPO" --spike-prefix "$SPIKE_PREFIX" \
  --cxx "$HOST_CXX" --runtime-lib-dir "$SPIKE_SUPPORT/lib" --runtime-bin-dir "$SPIKE_SUPPORT/bin" \
  --smoke-elf "$CPU_PROOF/model.elf" --build-root "$build_root" --output-dir "$build_root/simulator-check"
python "$TVM_ROOT/apps/gemmini/verify_matmul_runtime.py" --adapter-receipt "$ADAPTER_BUILD/receipt.json" --simulator-receipt "$build_root/simulator-check/receipt.json" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$SPIKE_PREFIX/bin/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --plugin "$build_root/simulator-check/libgemmini.so" --output-dir "$build_root/matmul-runtime-check"
```

Use fresh output directories. Numerical fixtures check extremes, tails/strides, repeated-call recovery, buffer guards and intentional failures, recheck immutable identities and audit final ELFs for prohibited FSM instructions. This is the provisional functional model; deployed completion/coherence, complete-model capacity and timing require separate evidence.

## Scheduled graph and learned-search checks

The TVM fork owns graph optimization, primitive scheduling and bounded learned search. Use its `apps/gemmini/README.md` for the graph/simulator recipe and the contracts of `prepare_gemmini_graph` and `BoundedGemminiSearch`. Graph optimization folds mathematical constants before device substitution, fuses surrounding CPU operations and inlines eligible internal pointwise buffers while preserving Gemmini as an explicit boundary. Baremetal export rejects unresolved TVM workspace callbacks. This does not implement fusion through the device kernel.

Learned ranking uses TVM MetaSchedule's `PerStoreFeature` and `XGBModel`. In the isolated host environment, the additional tested dependencies are:

```bash
python -m pip install 'xgboost==1.7.6' 'pytest==8.3.5'
python -m pytest -c /dev/null -p no:cacheprovider "$TVM_ROOT/tests/python/relax/test_backend_contrib_gemmini_tuning.py" -q
```

`GemminiModelSearch` aggregates admitted shapes and host fallbacks, distributes a finite reproducible model-wide budget and emits candidate-bound requests with repeated-sample requirements. Measurements must match issued requests and one platform/compiler/adapter/protocol domain; invalid records leave search state unchanged. Per-shape choices feed `prepare_gemmini_graph` without changing defaults.

These packages are optional for ordinary host compilation and untrained candidate generation. Synthetic labels exercise the search machinery only. Actual timing records require consistent workload, compiler, adapter, platform and measurement-protocol bindings plus separate semantic/device correctness gates; these declarations do not authenticate the supplied evidence. Functional Spike execution supplies no timing labels. Record real target measurements separately before selecting a performance result.

The generated graph verifier passes both `--graph-mode baseline` and `--graph-mode optimized`, including final ELF no-FSM and intentional-failure checks. Add `--graph-mode optimized` to the TVM guide's existing command; replace its two tile options with `--search-seed 7` to verify an untrained proposal and record its candidate-specific simulator correctness. Use a fresh output directory for each mode. This is a bounded matmul/bias/ReLU fixture, not a full-model or performance qualification.

Use the same verifier with `--workload conv-residual --conv-kernel 3` for a bounded convolution/residual/requantization graph; kernels 1 and 7 are also checked. Baseline and optimized kernel-3 runs preserve exact raw convolution and final int8 outputs. A fresh `--search-seed 7` run also verifies an untrained convolution schedule proposal. Packing and restoration are TVM graph operations around the existing primitive matmul schedule. The tensor-memory preflight reports liveness, resident storage and caller outputs separately; final ELF/stack checks remain additional evidence. No selected quantized ResNet or deployment-memory claim follows.
