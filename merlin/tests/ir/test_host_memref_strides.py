"""The host MLIR C interface can receive explicitly pitched memrefs safely."""

from __future__ import annotations

import ctypes

import pytest

from merlin.llvmlower.abi import HostModel, StridedMemRefArg
from merlin.llvmlower import toolchain


SOURCE = """
builtin.module {
  func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %e = tensor.empty() : tensor<2x3xf32>
    %y = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<2x3xf32>) outs(%e : tensor<2x3xf32>) {
    ^bb0(%a: f32, %unused: f32):
      %two = arith.constant 2.0 : f32
      %v = arith.mulf %a, %two : f32
      linalg.yield %v : f32
    } -> tensor<2x3xf32>
    func.return %y : tensor<2x3xf32>
  }
}
"""


def test_strided_memref_rejects_out_of_bounds_footprints():
    backing = (ctypes.c_float * 10)()
    ptr = ctypes.addressof(backing)
    with pytest.raises(ValueError, match="footprint"):
        StridedMemRefArg(ptr, (2, 3), (5, 1), storage_elements=7, dtype="f32", access="input")
    with pytest.raises(ValueError, match="same rank"):
        StridedMemRefArg(ptr, (2, 3), (5,), storage_elements=10, dtype="f32", access="input")
    with pytest.raises(ValueError, match="strides"):
        StridedMemRefArg(ptr, (2, 3), (5, -1), storage_elements=10, dtype="f32", access="input")
    with pytest.raises(ValueError, match="access"):
        StridedMemRefArg(ptr, (2, 3), (5, 1), storage_elements=10, dtype="f32", access="unspecified")
    for access in ("output", "inout"):
        with pytest.raises(ValueError, match="alias logical elements"):
            StridedMemRefArg(ptr, (2, 3), (1, 1), storage_elements=10, dtype="f32", access=access)
    # Broadcasting the same physical element to several logical *inputs* is safe.
    StridedMemRefArg(ptr, (2, 3), (1, 1), storage_elements=10, dtype="f32", access="input")


@pytest.mark.skipif(not toolchain.available(), reason="MLIR/LLVM host toolchain unavailable")
def test_native_lowered_model_reads_and_writes_pitched_buffers(tmp_path):
    """The actual lowered MLIR/LLVM entrypoint must honor both row pitches."""
    from merlin.llvmlower.lower import lower_model

    compiled = lower_model(SOURCE, tmp_path / "pitched", targets=("host",), textual=True)
    model = HostModel.load(str(compiled.host_so))
    x = (ctypes.c_float * 10)(1, 2, 3, -9, -9, 4, 5, 6, -9, -9)
    y = (ctypes.c_float * 14)(*([-7] * 14))
    model([
        StridedMemRefArg(ctypes.addressof(x), (2, 3), (5, 1), len(x), "f32", "input"),
        StridedMemRefArg(ctypes.addressof(y), (2, 3), (7, 1), len(y), "f32", "output"),
    ])
    assert list(y)[:3] == [2, 4, 6]
    assert list(y)[7:10] == [8, 10, 12]
    assert all(y[i] == -7 for i in (3, 4, 5, 6, 10, 11, 12, 13))
    dense_x = (ctypes.c_float * 6)(1, 2, 3, 4, 5, 6)
    dense_y = (ctypes.c_float * 6)()
    model([(ctypes.addressof(dense_x), (2, 3)), (ctypes.addressof(dense_y), (2, 3))])
    assert list(dense_y) == [2, 4, 6, 8, 10, 12]

    # Offset and non-unit inner stride take the generic element-copy path.
    separated_x = (ctypes.c_float * 20)(*([-9] * 20))
    separated_y = (ctypes.c_float * 20)(*([-7] * 20))
    for index, value in zip((1, 3, 5, 10, 12, 14), (1, 2, 3, 4, 5, 6)):
        separated_x[index] = value
    model([
        StridedMemRefArg(ctypes.addressof(separated_x), (2, 3), (9, 2), 20, "f32", "input", offset=1),
        StridedMemRefArg(ctypes.addressof(separated_y), (2, 3), (9, 2), 20, "f32", "output", offset=1),
    ])
    assert [separated_y[index] for index in (1, 3, 5, 10, 12, 14)] == [2, 4, 6, 8, 10, 12]
    assert all(separated_y[index] == -7 for index in set(range(20)) - {1, 3, 5, 10, 12, 14})

    # Staging would break alias semantics if an output copyback overwrote an
    # input or another output. Reject before invoking the native entrypoint.
    shared_input = StridedMemRefArg(ctypes.addressof(x), (2, 3), (5, 1), len(x), "f32", "input")
    shared_output = StridedMemRefArg(ctypes.addressof(x), (2, 3), (5, 1), len(x), "f32", "output")
    with pytest.raises(ValueError, match="overlapping physical storage"):
        model([shared_input, shared_output])
    with pytest.raises(ValueError, match="dense argument address"):
        model([(ctypes.addressof(x), (2, 3)), shared_output])
    assert list(x) == [1, 2, 3, -9, -9, 4, 5, 6, -9, -9]
