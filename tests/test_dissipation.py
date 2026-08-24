"""Tests for Lindblad dissipation channels."""

import unittest
import numpy as np
from bipartit_open_spin.config import ModelParams
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators
from bipartit_open_spin.core.operators import sigma_m1, sigma_m2


class TestDissipation(unittest.TestCase):
    def test_collapse_operators_construction(self):
        gamma = 0.25
        params = ModelParams(gamma=gamma)
        c_ops = build_collapse_operators(params)

        self.assertEqual(len(c_ops), 2)
        for op in c_ops:
            self.assertEqual(op.dims, [[2, 2], [2, 2]])
            self.assertEqual(op.shape, (4, 4))

        # Check exact numerical scaling: sqrt(0.25) * sigma_m
        expected_c1 = np.sqrt(gamma) * sigma_m1()
        expected_c2 = np.sqrt(gamma) * sigma_m2()

        self.assertTrue((c_ops[0] - expected_c1).norm("max") < 1e-12)
        self.assertTrue((c_ops[1] - expected_c2).norm("max") < 1e-12)

    def test_negative_gamma_raises(self):
        with self.assertRaises(ValueError):
            build_collapse_operators(ModelParams(gamma=-0.1))


if __name__ == "__main__":
    unittest.main()
