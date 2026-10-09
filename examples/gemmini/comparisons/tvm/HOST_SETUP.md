# TVM host setup

This recipe builds the [pinned independent TVM checkout](target.yaml) with host LLVM support on Linux. Checks recorded on 2026-10-07 passed CPU matmul, eight guard-rejection cases, twenty synthetic frontend cases and a full ResNet50 v1.5 random-weight host diagnostic. Pretrained/full-session qualification and Gemmini deployment remain pending.

The host cache enables the local `USE_HOST_ONLY_AUTO_COPY_GUARD` patch because this restricted checkout lacks `LowerAutoCopy`. The guard preserves unannotated IR and rejects automatic-copy annotations; it does not implement their optimization. This option defaults to OFF and must not coexist with the full implementation. Record the local patch and enabled mode with the base commit.

Run from the Merlin comparison checkout. Select absolute `TVM_ROOT` and `TVM_BUILD` paths for the independent source checkout and host library build. Merlin does not track or initialize this baseline as a submodule. Support artifacts use the configured Merlin build root, honoring `MERLIN_OUT_ROOT`; the TVM host build uses `TVM_BUILD`. The verified tool versions are Python 3.10.14, GCC 11.5, CMake 3.26.5 and Ninja 1.13.2. Select Python 3.10 and the host tools from your own installation; a Chipyard Conda environment is not required. The commands resolve tools from `PATH` before cleaning inherited build settings. Set `TVM_HOST_PYTHON`, `TVM_HOST_CMAKE`, `TVM_HOST_CC` or `TVM_HOST_CXX` to absolute executable paths to override discovery, and `TVM_BUILD_JOBS` to control parallelism (default 4; the recorded build used 16). Do not source Chipyard's environment for this host build.

The newest source pin is a local development commit until the fork is published; use the existing checkout or a local clone containing that commit. A fresh remote clone can reproduce the new pin only after publication.

For a new TVM checkout, clone without checkout and apply the mandatory name exclusions before materializing files. These commands select the published source commit recorded in `target.yaml`; they do not switch an existing checkout. If you already have that commit checked out, retain it and set the same two paths instead.

```bash
export TVM_ROOT=/absolute/path/to/tvm-gemmini
export TVM_BUILD=/absolute/path/to/tvm-host-build
tvm_commit=34821ba0d239f0b04da5ba511d3e61494e7f714c
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
test "$(git -C "$TVM_ROOT" rev-parse HEAD)" = 34821ba0d239f0b04da5ba511d3e61494e7f714c || { echo 'TVM source pin mismatch' >&2; exit 1; }
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

A leading space in each status line means the pin matches; `-`, `+`, or `U` requires repair before building. For a fresh checkout, initialize only these three dependencies at the gitlinks recorded in TVM commit `34821ba0d239f0b04da5ba511d3e61494e7f714c`. Clone each dependency without checkout and apply the same exclusions before checkout:

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

Keep `host-smoke.json`, `host-manifest.json`, `tvm-host.patch` and build logs under the build root. The receipt records the loaded library, LLVM support, guard mode and numerical/rejection checks; the manifest and patch identify the tested compiler beyond its base Git SHA. Local progress notes retain machine-specific setup history.

## ONNX frontend verification

Install [frontend-requirements.txt](frontend-requirements.txt) into the host environment. The verified combination is Python 3.10.14, PyTorch 2.10.0+cu128 executing on CPU, ONNX 1.17.0, protobuf 5.29.5 and NumPy 1.26.4. ONNX 1.17 retains the `onnx.mapping` API used by this TVM revision. The verifier explicitly uses the legacy exporter (`dynamo=False`), matching the existing Merlin runner's primary path; it does not require ONNXScript, ONNX Runtime or torchvision.

```bash
python -m pip install -r "$setup_dir/frontend-requirements.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TVM_NUM_THREADS=1
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/current"
```

A fresh setup installs the pinned packages into its own host environment. The recorded verification reused an existing PyTorch installation because of a local download restriction; that workaround is not a prerequisite. Reports record the actual dependency versions and resolved package locations. All checks execute on CPU, including when the installed framework distribution contains CUDA support.

All ten cases passed locally: matmul, batched matmul, convolution, LayerNorm and RMSNorm at opsets 17 and 18. Each compares PyTorch, ONNX ReferenceEvaluator and Relax LLVM CPU results, requiring equal output shapes/dtypes and `rtol=1e-4, atol=1e-5`. Maximum absolute Relax-versus-PyTorch error was `2.40e-5` (LayerNorm). These are small synthetic frontend checks, not full-model or Gemmini results.

The verifier writes `results.json` plus each case's ONNX graph, imported Relax IR and numerical arrays beneath its required output directory. It records exporter/importer identity, dependency locations, loaded TVM library and LLVM/guard metadata. The loaded library still reports its original build revision `fb78e0e...`; later source commits do not relabel that build.

For a focused importer ablation, retrieve only the exact Python importer and load it through `--importer-source`; keep the current runtime and branch unchanged:

```bash
git -C "$TVM_ROOT" show c4dc0c29ff81ddae688da24625a603da3b4a2c0e:python/tvm/relax/frontend/onnx/onnx_frontend.py > "$build_root/frontend/apache-importer.py"
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/apache-importer" --importer-source "$build_root/frontend/apache-importer.py"
git -C "$TVM_ROOT" show d608061677e535f12b64c072d7a109a645bcca73:python/tvm/relax/frontend/onnx/onnx_frontend.py > "$build_root/frontend/reduction-fix-importer.py"
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/reduction-fix-importer" --importer-source "$build_root/frontend/reduction-fix-importer.py" --case rms_norm
```

The Apache importer run is expected to exit nonzero: nine cases passed, but opset-18 RMSNorm produced incorrect values (maximum absolute error `1.19899`). Both RMSNorm opsets passed with just `d608061`, confirming the reduction-axis fix is required for this graph. This compares historical Python importers on the current runtime, not complete historical TVM builds. Preserve each variant's results in its own directory.

## Focused importer regressions

```bash
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --suite importer --output-dir "$build_root/frontend/importer-dtype-fixed"
```

This suite constructs schema-checked ONNX graphs independently of the exporter and compares NumPy expectations, ONNX ReferenceEvaluator and Relax execution. It covers shape-array arithmetic, negative Gather indices as runtime inputs and constants, lower-rank Expand, and scalar ConstantOfShape at both opsets. All ten importer cases and all ten exporter cases passed after the dtype fix below. Integer results require exact dtype, shape and value equality.

| Inherited change | Evidence and remaining limit |
| --- | --- |
| `537a5e0`: binary scalar/tensor folding | Apache's importer fails the valid Shape → Gather → Add graph. The inherited fix imports it but silently narrows int64 arrays to int32. The local follow-up passes `arr.dtype` explicitly to `relax.const`; the regression preserves `1099511627785`, previously truncated to `9`. |
| `75791d9`: negative Gather | Current constant/runtime integer-index cases pass. Historical importer overrides retain IR without executing unchecked negative indices. Mixed-dtype ScatterND was not tested because that case violates the ONNX schema. |
| `db359fe`: Gather casts/normalization | Valid runtime integer Gather passes, but its imported IR is identical with `75791d9` alone, so this case does not establish a need for the later normalization change. Float-index Gather is outside the valid ONNX gate. |
| `fb78e0e`: Expand/ConstantOfShape | Lower-rank Expand and scalar ConstantOfShape fail import immediately before this patch and pass with it. |

The dtype comparison uses identical saved ONNX graphs against the inherited and fixed importers. Reports remain under distinct `frontend/` subdirectories, including `inherited-large-shape`, `importer-dtype-fixed` and `torch-after-dtype-fix`. The BF16 sigmoid limitation and broader model coverage remain untested. The local Python importer fix changes source identity without rebuilding the C++ library.

## ResNet50 host structural diagnostic

Use `--graph-mode both` to compare the ordinary build with explicit graph optimization on identical weights and images. The optimized path runs TVM's `zero` pipeline (legalization, pattern annotation, constant folding and fusion), then records and compiles the actual `default_build` IR. `--graph-mode optimized` runs only that path; omitting the option preserves the baseline behavior. Each mode saves its IR, function/operator inventory and per-image comparisons. The full random-weight diagnostic passes both modes with reduced TIR function count after fusion. These are semantic checks, not performance measurements or pretrained-model qualification; Gemmini operator scheduling is separate.

`verify_resnet.py` exercises the full existing torchvision ResNet50 v1.5 architecture with seeded random weights and two distinct synthetic images. It makes no pretrained-accuracy, final paper-variant, quantization or accelerator claim. The loader comes from model2MLIR revision `7915e23475c6db446a3c404847b11e8bc72c8a27`; the verified environment adds torchvision 0.25.0+cu128 and Pillow 12.1.0 to the frontend pins above. Both CUDA-enabled framework distributions execute on CPU here. The standalone diagnostic uses the host Python 3.10 environment; it does not invoke model2MLIR's separate Python 3.12 capture pipeline.

```bash
model2mlir_source=/absolute/path/to/model2MLIR
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TVM_NUM_THREADS=4
python "$setup_dir/verify_resnet.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --output-dir "$build_root/resnet50-v1_5"
```

The verifier clears inherited ResNet dataset/calibration settings during loader construction and sets `RANDOM=1`, `PRETRAINED=0`, `PAPER_READY=0` and two session steps through the loader's named environment variables. Model initialization uses seed 194; images use the loader's independent seed 20260830. It preserves zero-valued parameters, checks the state hash before/after, and verifies the [3,4,6,3] stage depths and v1.5 downsampling strides.

The checked model has 25,557,032 parameters and 53 convolution modules, takes float32 NCHW `[1,3,224,224]` input and produces float32 `[1,1000]` logits. It exports legacy ONNX opset 17, runs strict ONNX checking and shape inference, imports frozen parameters into Relax, builds LLVM once, and checks both images against PyTorch. Both passed with preselected `rtol=1e-4, atol=1e-4`; maximum absolute errors were `8.39234e-5` and `9.15527e-5`.

| Exported ONNX operation | Count |
| --- | ---: |
| Conv | 53 (36 × 1×1, 16 × 3×3, 1 × 7×7) |
| Add | 16 |
| Relu | 49 |
| MaxPool / GlobalAveragePool / Flatten / Gemm | 1 each |
| Identity | 47 |

All inferred tensor shapes/dtypes are concrete. Convolutions use NCHW/OIHW and the classifier's stored weight uses OI with `transB=1`. Evaluation-mode BatchNorm is folded into convolution; this explains its absence from the exported graph. Relax contains 53 conv2d, one matmul, 70 adds (including biases), 49 ReLUs, one max pool, one mean, 54 reshapes and one permutation; ONNX Identity nodes disappear. This inventory identifies integration work without asserting Gemmini eligibility for any operation.

The output directory retains `results.json`, the exported and shape-inferred ONNX graphs, `onnx_inventory.json`, `relax_inventory.json`, imported Relax IR and two input/reference/output archives. Reports bind the verifier, loader, torchvision source, input/model hashes, frontend source and loaded TVM library. These are full-architecture compiler diagnostics; pretrained checkpoint fidelity, real-image quality, the intended multi-image paper session and Gemmini execution remain separate gates.

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
{"input_sha256": "<images.npz SHA-256>", "class_ids": ["class-0", "class-1", "...", "class-999"], "samples": [{"id": "sample-a", "label": 7}, {"id": "sample-a", "label": 2}]}
```

Reports record file hashes and the declared class mapping/sample order, plus per-image PyTorch and Relax top-five indices and top-one/top-five hits. Equal logits rank by ascending class index. Complete-stream counts and rates appear in `descriptive_accuracy`, including top-one disagreements. These metrics have no accuracy acceptance threshold; numerical comparison remains the pass/fail gate. The labels file is rehashed with the other supplied artifacts at the end.

Reports record SHA-256 digests of supplied artifact files and the loaded model state. Source, preprocessing, class order and labels are declarations, not independently verified facts; supplying a complete checkpoint does not establish pretrained status, training provenance or agreement between its classes and the manifest. The verifier performs no calibration, always records `paper_validation: false`, and does not qualify dataset truth or paper accuracy. Omitting `--labels` performs the existing numerical comparison without classification metrics. External qualification criteria remain to be selected and satisfied.

The supplied-file path passed a three-image CPU regression using a synthetic checkpoint distinct from the default initialization, including a repeated image to check stream preservation. Maximum absolute error was `9.16e-5` at `rtol=atol=1e-4`; this remains a synthetic regression. Focused tests cover artifact preservation, invalid checkpoints/images, environment isolation, label validation, ranking ties and count aggregation:

```bash
PYTHONPATH="$merlin_root/src:$PYTHONPATH" python "$merlin_root/merlin/tests/gemmini/test_tvm_resnet_artifacts.py"
```
