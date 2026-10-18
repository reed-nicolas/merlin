"""Compiled mask expansion preserves affine indexing and rejects invalid rows."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.compile.c_softmax_mask import render_masked_softmax_inputs
from merlin.semantic_compiler.model import IndexMap
from merlin.semantic_compiler.precision_values import FloatBoundarySpec


@pytest.mark.parametrize("reverse", [False, True])
def test_mask_packing_checks_all_values_before_writing(tmp_path, reverse):
    shape = (2, 3, 5)
    mapping = IndexMap(3, ((0, 1, 0), (0, 0, -1 if reverse else 1)), (0, 4 if reverse else 0))
    region = SimpleNamespace(shape=shape, mask_map=mapping)
    spec = FloatBoundarySpec("test", "bf16", "encode", admitted_code_ranges=((0, 0), (0x3E80, 0x4200)))
    helper = render_masked_softmax_inputs(region, (3, 5), spec, b"\x00\xc2", "mask")
    (tmp_path / "mask.c").write_text(
        helper
        + "\nint prepare(const float *a,const float *m,unsigned char *o,unsigned char *k) {\n"
        + "  return mask_prepare(a,m,o,k);\n}\n"
    )
    library_name = f"mask-{int(reverse)}.so"
    compiler = shutil.which("cc")
    assert compiler
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "mask.c", "-o", library_name],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(tmp_path / library_name))
    library.prepare.argtypes = [ctypes.c_void_p] * 4
    scores = np.arange(30, dtype="<f4").reshape(shape) / 4
    mask = np.array(
        [[0.0, -np.inf, 0.0, -np.inf, 0.0], [-np.inf, -np.inf, 0.0, 0.0, -np.inf], [0.0, 0.0, -np.inf, 0.0, -np.inf]],
        dtype="<f4",
    )
    encoded, keep = ctypes.create_string_buffer(60), ctypes.create_string_buffer(60)

    def call(a, m):
        return library.prepare(a.ctypes.data, m.ctypes.data, encoded, keep)

    assert call(scores, mask) == 0
    selected = mask[:, ::-1] if reverse else mask
    active = np.broadcast_to(selected == 0, shape)
    bits = scores.view("<u4")
    expected = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
    expected[~active] = 0xC200
    assert encoded.raw == expected.tobytes()
    assert keep.raw == np.where(active, 0x3F80, 0).astype("<u2").tobytes()
    for fault in ("last_mask", "all_masked", "nan_score", "domain_score"):
        a, m = scores.copy(), mask.copy()
        if fault == "last_mask":
            m[-1, -1] = -1.0
        elif fault == "all_masked":
            m[-1] = -np.inf
        elif fault == "nan_score":
            a.reshape(-1)[-1] = np.nan
        else:
            a.reshape(-1)[-1] = 64.0
        encoded.raw = b"\xa5" * len(encoded)
        keep.raw = b"\xcd" * len(keep)
        assert call(a, m) != 0
        assert encoded.raw == b"\xa5" * len(encoded)
        assert keep.raw == b"\xcd" * len(keep)
    with pytest.raises(ValueError, match="leaves its source"):
        render_masked_softmax_inputs(
            SimpleNamespace(shape=shape, mask_map=replace(mapping, offsets=(0, 5))), (3, 5), spec, b"\x00\xc2", "bad"
        )
