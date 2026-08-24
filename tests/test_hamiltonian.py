"""Tests for baseline Hamiltonian construction."""

import unittest
import numpy as np
from bipartit_open_spin.config import ModelParams
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian


class TestHamiltonian(unittest.TestCase):
    def test_hamiltonian_dimensions_and_hermiticity(self):
        params = ModelParams(omega=1.5, J=0.7, gamma=0.1)
        H = build_hamiltonian(params)

        self.assertEqual(H.dims, [[2, 2], [2, 2]])
        self.assertEqual(H.shape, (4, 4))
        self.assertTrue((H - H.dag()).norm("max") < 1e-12)

    def test_hamiltonian_matrix_elements(self):
        # Case 1: Pure single-spin detuning (J = 0)
        params_uncoupled = ModelParams(omega=2.0, J=0.0)
        H_uncoupled = build_hamiltonian(params_uncoupled)
        # H = 0.5 * 2 * (sz1 + sz2) = sz1 + sz2 = diag(1+1, 1-1, -1+1, -1-1) = diag(2, 0, 0, -2)
        expected_diag = np.diag([2.0, 0.0, 0.0, -2.0])
        np.testing.assert_allclose(H_uncoupled.full(), expected_diag, atol=1e-12)

        # Case 2: Pure coupling (omega = 0, J = 1.0)
        params_coupled = ModelParams(omega=0.0, J=1.0)
        H_coupled = build_hamiltonian(params_coupled)
        # sx1 * sx2 in computational basis flips both spins: |00><->|11|, |01><->|10|
        expected_coupled = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(H_coupled.full(), expected_coupled, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
