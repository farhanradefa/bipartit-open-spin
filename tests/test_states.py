"""Tests for quantum state generation and conversion."""

import unittest
import numpy as np
from qutip import Qobj, basis, sigmax, tensor
from bipartit_open_spin.core.states import (
    bell_phi_plus,
    computational_basis,
    to_density_matrix,
)


class TestStates(unittest.TestCase):
    def test_computational_basis_dimensions(self):
        for i in (0, 1):
            for j in (0, 1):
                state = computational_basis(i, j)
                self.assertTrue(state.isket)
                self.assertEqual(state.dims, [[2, 2], [1]])
                self.assertEqual(state.shape, (4, 1))
                self.assertAlmostEqual(state.norm(), 1.0)

    def test_computational_basis_invalid_indices(self):
        with self.assertRaises(ValueError):
            computational_basis(2, 0)
        with self.assertRaises(ValueError):
            computational_basis(0, -1)

    def test_bell_phi_plus(self):
        phi_plus = bell_phi_plus()
        self.assertTrue(phi_plus.isket)
        self.assertEqual(phi_plus.dims, [[2, 2], [1]])
        self.assertAlmostEqual(phi_plus.norm(), 1.0)

        # Expected vector: [1/sqrt(2), 0, 0, 1/sqrt(2)]^T
        expected = np.array([[1.0 / np.sqrt(2)], [0.0], [0.0], [1.0 / np.sqrt(2)]])
        np.testing.assert_allclose(phi_plus.full(), expected, atol=1e-12)

    def test_to_density_matrix_from_ket(self):
        phi_plus = bell_phi_plus()
        dm = to_density_matrix(phi_plus)
        self.assertTrue(dm.isoper)
        self.assertEqual(dm.dims, [[2, 2], [2, 2]])
        self.assertAlmostEqual(dm.tr(), 1.0)
        self.assertTrue((dm - dm.dag()).norm("max") < 1e-12)

    def test_to_density_matrix_from_oper(self):
        phi_plus = bell_phi_plus()
        dm_initial = phi_plus * phi_plus.dag()
        dm = to_density_matrix(dm_initial)
        self.assertEqual(dm.dims, [[2, 2], [2, 2]])
        self.assertAlmostEqual(dm.tr(), 1.0)

    def test_to_density_matrix_flat_dims_relabeling(self):
        # Shape (4, 4) with flat dims [[4], [4]]
        raw_4x4 = np.eye(4) / 4.0
        qobj_flat = Qobj(raw_4x4, dims=[[4], [4]])
        dm = to_density_matrix(qobj_flat)
        self.assertEqual(dm.shape, (4, 4))
        self.assertEqual(dm.dims, [[2, 2], [2, 2]])

    def test_to_density_matrix_incompatible_dimensions_raise(self):
        # 1-qubit ket
        with self.assertRaises(ValueError):
            to_density_matrix(basis(2, 0))

        # 1-qubit operator
        with self.assertRaises(ValueError):
            to_density_matrix(sigmax())

        # 3-qubit ket
        psi_3qubit = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
        with self.assertRaises(ValueError):
            to_density_matrix(psi_3qubit)

        # Bra vector (not ket, not oper)
        with self.assertRaises(ValueError):
            to_density_matrix(bell_phi_plus().dag())


if __name__ == "__main__":
    unittest.main()

