"""Pinned native CPU single-row BF16 QK halfway regression."""
import ctypes as ct
import _ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

LEFT = [15882, 16002, 16169, 16133, 48763, 49043, 48524, 48750, 49055, 48972, 16290, 48788, 16384, 15962, 48800, 49152, 48667, 48921, 48512, 16354, 16181, 48796, 47936, 16582, 48344, 48644, 16187, 16136, 15410, 49264, 16371, 16082, 16223, 16264, 15951, 49090, 48804, 48408, 16202, 49074, 16146, 48676, 16014, 16176, 48536, 15939, 49136, 16151, 48838, 48962, 16222, 16236, 16057, 16225, 48978, 16314, 16102, 48774, 48694, 48590, 15949, 48609, 48552, 48888]
RIGHT = [49317, 49157, 49048, 49334, 16480, 48928, 49282, 49254, 16358, 16240, 16461, 48876, 16398, 16137, 16286, 16280, 49143, 16433, 16436, 48856, 16243, 48918, 49283, 16231, 48998, 49034, 16272, 48360, 49030, 16576, 49283, 16449, 16580, 48980, 49136, 49113, 16474, 16062, 16339, 49195, 16181, 16191, 49265, 16340, 48850, 49195, 16382, 16435, 16274, 49281, 48688, 49211, 16044, 48984, 48928, 49461, 49062, 16116, 16275, 16144, 48977, 16389, 49082, 15940]

class DecodeQKTest(unittest.TestCase):
    def test_native_halfway_witness_and_zero_columns(self):
        source = Path(__file__).absolute().with_name("llama_host_bmm.c")
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "bmm.so"
            subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC", str(source),
                "-lm", "-o", str(library)], check=True, capture_output=True, timeout=120)
            lib = ct.CDLL(str(library))
            try:
                fn = lib.llama_host_bmm_bf16_pair
                fn.argtypes = [ct.c_void_p]*3 + [ct.c_size_t]*4 + [ct.c_int]
                a = np.array(LEFT, np.uint16).reshape(1, 1, 64)
                b = np.zeros((1, 129, 64), np.uint16)
                b[0, 121] = RIGHT
                output = np.full((1, 1, 129), 0x7fc0, np.uint16)
                self.assertEqual(fn(a.ctypes.data, b.ctypes.data, output.ctypes.data, 1, 1, 64, 129, 1), 0)
                # Original native c20a; old ascending pair order gives c20b.
                self.assertEqual(int(output[0, 0, 121]), 0xc20a)
                self.assertTrue(np.all(np.delete(output.reshape(-1), 121) == 0))
            finally:
                _ctypes.dlclose(lib._handle)

if __name__ == "__main__":
    unittest.main()
