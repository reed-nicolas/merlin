# TVM host setup

This recipe builds the [pinned TVM checkout](../../../../third_party/baselines/README.md) with host LLVM support. Verified locally on 2026-10-07: a synthetic Relax CPU matmul matched NumPy within `7.5e-9` absolute error, and all eight guard-rejection checks passed. Model and Gemmini deployment validation remain pending.

The host cache enables the local `USE_HOST_ONLY_AUTO_COPY_GUARD` patch because this restricted checkout lacks `LowerAutoCopy`. The guard preserves unannotated IR and rejects automatic-copy annotations; it does not implement their optimization. This option defaults to OFF and must not coexist with the full implementation. Record the local patch and enabled mode with the base commit.

Run from the Merlin comparison checkout. All generated files use the configured build root. This setup uses Python 3.10.14, GCC 11.5, CMake 3.26.5, and Ninja 1.13.2; use their explicit paths without sourcing Chipyard's environment.

```bash
merlin_root="$PWD"
setup_dir="$merlin_root/examples/gemmini/comparisons/tvm"
tvm_source="$merlin_root/third_party/baselines/tvm-gemmini"
host_python=/bwrcq/C/reednicolas/ee194-sp26-chipyard/.conda-env/bin/python3.10
export PYTHONDONTWRITEBYTECODE=1
build_root="$(PYTHONPATH="$merlin_root/src" "$host_python" -c 'from merlin.common.paths import build_dir; print(build_dir() / "baselines" / "tvm-gemmini")')"
mkdir -p "$build_root"/{cache/pip,tmp,logs}
export PIP_CACHE_DIR="$build_root/cache/pip" TMPDIR="$build_root/tmp"
"$host_python" -m venv "$build_root/venv"
source "$build_root/venv/bin/activate"
python -m pip install -r "$setup_dir/host-requirements.txt"
host_ninja="$build_root/venv/bin/ninja"
export PATH="$build_root/venv/bin:/usr/bin:/bin"
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
unset CMAKE_PREFIX_PATH PYTHONPATH PYTHONHOME CC CXX CFLAGS CXXFLAGS LDFLAGS
```

The three required TVM submodules are `dmlc-core`, `dlpack`, and `rang`. They are initialized in this checkout at the recorded gitlinks with forbidden-name sparse exclusions. Preserve those checkouts and verify them before building:

```bash
git -C "$tvm_source" submodule status -- 3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
```

A leading space in each status line means the pin matches; `-`, `+`, or `U` requires repair before building. For a fresh checkout, initialize only these three dependencies using the [recorded pins](../../../../third_party/baselines/README.md), with `--no-checkout` and equivalent sparse exclusions established before checkout. Do not initialize optional dependencies recursively.

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
/usr/bin/cmake -S "$llvm_source/llvm" -B "$llvm_build" -G Ninja -DCMAKE_MAKE_PROGRAM="$host_ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  '-DLLVM_TARGETS_TO_BUILD=X86;RISCV' -DLLVM_ENABLE_PROJECTS= -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_DYLIB_COMPONENTS=all -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_INCLUDE_TOOLS=ON -DLLVM_BUILD_TOOLS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF
/usr/bin/cmake --build "$llvm_build" --target LLVM llvm-config --parallel 16
# LLVM 18 creates this compatibility name during installation, but not in its build tree.
/usr/bin/cmake -E create_symlink libLLVM.so.18.1 "$llvm_build/lib/libLLVM-18.so"
"$llvm_build/bin/llvm-config" --version --targets-built --shared-mode --link-shared
```

Consume the LLVM build tree directly, retaining its source and generated include directories. `LLVM_INCLUDE_TOOLS=ON` is required for the shared-library target; requesting only `LLVM` and `llvm-config` avoids unrelated tools. Configure TVM with the checked-in host cache and an RPATH to this LLVM library directory.

```bash
tvm_build="$build_root/host"
/usr/bin/cmake -S "$tvm_source" -B "$tvm_build" -G Ninja -C "$setup_dir/host-config.cmake" -DCMAKE_MAKE_PROGRAM="$host_ninja" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DUSE_LLVM="$llvm_build/bin/llvm-config --link-shared" -DCMAKE_BUILD_RPATH="$llvm_build/lib"
/usr/bin/cmake --build "$tvm_build" --parallel 16
export PYTHONPATH="$tvm_source/python"
export TVM_LIBRARY_PATH="$tvm_build"
export TVM_FFI=ctypes
export PYTHONDONTWRITEBYTECODE=1
export TEST_DATA_ROOT_PATH="$build_root/cache/tvm-test-data"
python "$tvm_source/apps/gemmini/verify_host.py" --output "$build_root/host-smoke.json"
```

The existing local environment can be restored with `source "$build_root/activate.sh"`. This generated helper contains machine-specific paths; use the commands above to reproduce the setup elsewhere.

Keep `host-smoke.json`, `host-manifest.json`, `tvm-host.patch` and build logs under the build root. The receipt records the loaded library, LLVM support, guard mode and numerical/rejection checks; the manifest and patch identify the tested compiler beyond its base Git SHA. Local progress notes retain machine-specific setup history.

## ONNX frontend verification

Install [frontend-requirements.txt](frontend-requirements.txt) into the host environment. The verified combination is Python 3.10.14, PyTorch 2.10.0+cu128 executing on CPU, ONNX 1.17.0, protobuf 5.29.5 and NumPy 1.26.4. ONNX 1.17 retains the `onnx.mapping` API used by this TVM revision. The verifier explicitly uses the legacy exporter (`dynamo=False`), matching the existing Merlin runner's primary path; it does not require ONNXScript, ONNX Runtime or torchvision.

```bash
python -m pip install -r "$setup_dir/frontend-requirements.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TVM_NUM_THREADS=1
python "$tvm_source/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/current"
```

The current machine reuses an existing PyTorch installation because the CPU wheel download domain is blocked. Its generated `$build_root/local-torch/` overlay contains symlinks to only `torch`, `torchgen`, `functorch`, `nvidia` and `torch-2.10.0.dist-info` from Chipyard's `.conda-env/lib/python3.10/site-packages/`. The remaining packages are pinned and installed in the isolated host environment. After sourcing `activate.sh`, add the existing overlay with `export PYTHONPATH="$PYTHONPATH:$build_root/local-torch"`. This overlay is machine-specific and is not part of a fresh installation recipe; the report records the resolved package locations. No GPU is used.

All ten cases passed locally: matmul, batched matmul, convolution, LayerNorm and RMSNorm at opsets 17 and 18. Each compares PyTorch, ONNX ReferenceEvaluator and Relax LLVM CPU results, requiring equal output shapes/dtypes and `rtol=1e-4, atol=1e-5`. Maximum absolute Relax-versus-PyTorch error was `2.40e-5` (LayerNorm). These are small synthetic frontend checks, not full-model or Gemmini results.

The verifier writes `results.json` plus each case's ONNX graph, imported Relax IR and numerical arrays beneath its required output directory. It records exporter/importer identity, dependency locations, loaded TVM library and LLVM/guard metadata. The loaded library still reports its original build revision `fb78e0e...`; later source commits do not relabel that build.

For a focused importer ablation, retrieve only the exact Python importer and load it through `--importer-source`; keep the current runtime and branch unchanged:

```bash
git -C "$tvm_source" show c4dc0c29ff81ddae688da24625a603da3b4a2c0e:python/tvm/relax/frontend/onnx/onnx_frontend.py > "$build_root/frontend/apache-importer.py"
python "$tvm_source/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/apache-importer" --importer-source "$build_root/frontend/apache-importer.py"
git -C "$tvm_source" show d608061677e535f12b64c072d7a109a645bcca73:python/tvm/relax/frontend/onnx/onnx_frontend.py > "$build_root/frontend/reduction-fix-importer.py"
python "$tvm_source/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/reduction-fix-importer" --importer-source "$build_root/frontend/reduction-fix-importer.py" --case rms_norm
```

The Apache importer run is expected to exit nonzero: nine cases passed, but opset-18 RMSNorm produced incorrect values (maximum absolute error `1.19899`). Both RMSNorm opsets passed with just `d608061`, confirming the reduction-axis fix is required for this graph. This compares historical Python importers on the current runtime, not complete historical TVM builds. The other four inherited patches and the reported BF16 sigmoid issue need separate probes; these cases do not establish their necessity or resolution. Preserve each variant's results in its own directory.
