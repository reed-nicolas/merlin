"""Directed policy-domain regressions without an instruction-model checkout."""
import sys
import unittest

import numpy as np

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "examples/atlas/rotary-host/reference"))
import rowspan_softmax as rs


def words(values):
    return (np.asarray([values], np.float32).view(np.uint32) >> 16).astype(np.uint16)


class RowspanPolicyTests(unittest.TestCase):
    def test_span96_requires_explicit_selection(self):
        scores = words([42.75, -48.75])
        keep = np.ones(scores.shape, np.bool_)
        with self.assertRaisesRegex(ValueError, "rowspan80"):
            rs.check_domain(scores, keep)
        rs.check_domain(scores, keep, policy=rs.POLICY96)

    def test_span96_inclusive_edges(self):
        for values in [[48, -48], [-32, -128], [0, -96]]:
            with self.subTest(values=values):
                scores = words(values)
                rs.check_domain(scores, np.ones(scores.shape, np.bool_), policy=rs.POLICY96)

    def test_span96_refuses_outside_domain(self):
        for values in [[48, -48.25], [-32.25, -40], [48.25, 0]]:
            with self.subTest(values=values):
                scores = words(values)
                with self.assertRaises(ValueError):
                    rs.check_domain(scores, np.ones(scores.shape, np.bool_), policy=rs.POLICY96)

    def test_masked_domain_checks_keep_and_nonempty_rows(self):
        scores = words([1, -float("inf")])
        rs.check_domain(scores, np.asarray([[True, False]]), policy=rs.POLICY96)
        with self.assertRaisesRegex(ValueError, "at least one"):
            rs.check_domain(scores, np.zeros(scores.shape, np.bool_), policy=rs.POLICY96)
        with self.assertRaisesRegex(ValueError, "normal"):
            rs.check_domain(scores, np.ones(scores.shape, np.bool_), policy=rs.POLICY96)

    def test_unrecognized_policy_refused(self):
        scores = words([0, -1])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            rs.check_domain(scores, np.ones(scores.shape, np.bool_), policy="span97")


if __name__ == "__main__":
    unittest.main()
