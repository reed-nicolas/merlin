#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RISCV:?set RISCV to the pinned RV64GC/lp64d toolchain prefix}"
: "${MERLIN_CONSTANT_MATH_HELPER:?set the content-addressed Merlin constant math helper}"
: "${PYTHON:=python3}"
export HOST_MATH_OUT="${HOST_MATH_OUT:-$PWD/out}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$HOST_MATH_OUT"
# Each step fails closed; later steps re-check the hashes recorded by earlier ones.
for step in domains build_core_math run_rv64_probe build_execution_host_math check_execution_host_math check_mpfr_reference write_policy; do
  nice -n 10 "$PYTHON" "scripts/$step.py" > "$HOST_MATH_OUT/$step.log"
done
