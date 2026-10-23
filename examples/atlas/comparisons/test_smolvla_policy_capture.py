"""Focused native policy capture closure controls, without model loading."""
import unittest

import numpy as np

from capture_smolvla_policy import validate_trajectory


class CaptureClosureTest(unittest.TestCase):
    def test_complete_exact_trajectory(self):
        reference = {"final_actions": np.array([[0.0, 1.0]], np.float32)}
        validate_trajectory({key: value.copy() for key, value in reference.items()}, reference, list(range(10)))

    def test_missing_or_reordered_step_rejected(self):
        for steps in (list(range(9)), [1, 0, *range(2, 10)], list(range(11))):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                validate_trajectory({}, {}, steps)

    def test_missing_capture_rejected(self):
        with self.assertRaises(ValueError):
            validate_trajectory({}, {"final_actions": np.zeros(1, np.float32)}, list(range(10)))

    def test_raw_word_dtype_and_geometry_changes_rejected(self):
        expected = np.array([[0.0, 1.0]], np.float32)
        changed_bit = expected.copy()
        changed_bit.view(np.uint32)[0, 1] ^= 1
        for changed in (np.array([[-0.0, 1.0]], np.float32), changed_bit,
                        expected.astype(np.float64), expected.reshape(2)):
            with self.subTest(dtype=changed.dtype, shape=changed.shape), self.assertRaises(ValueError):
                validate_trajectory({"value": changed}, {"value": expected}, list(range(10)))


if __name__ == "__main__":
    unittest.main()
