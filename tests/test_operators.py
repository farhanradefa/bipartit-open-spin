"""Tests for bipartite spin operators."""

import unittest
from bipartit_open_spin.core.operators import (
    identity_2qubit,
    sigma_m1,
    sigma_m2,
    sigma_x1,
    sigma_x2,
    sigma_z1,
    sigma_z2,
)


class TestOperators(unittest.TestCase):
    def test_operator_dimensions(self):
        operators = [
            identity_2qubit(),
            sigma_x1(),
            sigma_x2(),
            sigma_z1(),
            sigma_z2(),
            sigma_m1(),
            sigma_m2(),
        ]
        for op in operators:
            self.assertEqual(op.dims, [[2, 2], [2, 2]])
            self.assertEqual(op.shape, (4, 4))

    def test_hermiticity_and_involutions(self):
        I4 = identity_2qubit()
        sx1 = sigma_x1()
        sx2 = sigma_x2()
        sz1 = sigma_z1()
        sz2 = sigma_z2()

        # Check Hermiticity
        for op in [I4, sx1, sx2, sz1, sz2]:
            self.assertTrue((op - op.dag()).norm("max") < 1e-12)

        # Check Pauli squares: sigma_i^2 = I
        self.assertTrue(((sx1 * sx1) - I4).norm("max") < 1e-12)
        self.assertTrue(((sx2 * sx2) - I4).norm("max") < 1e-12)
        self.assertTrue(((sz1 * sz1) - I4).norm("max") < 1e-12)
        self.assertTrue(((sz2 * sz2) - I4).norm("max") < 1e-12)

    def test_subsystem_commutations(self):
        sx1 = sigma_x1()
        sx2 = sigma_x2()
        sz1 = sigma_z1()
        sz2 = sigma_z2()

        # Operators on different subsystems commute: [O_1, O_2] = 0
        comm_x1_x2 = sx1 * sx2 - sx2 * sx1
        comm_z1_z2 = sz1 * sz2 - sz2 * sz1
        comm_x1_z2 = sx1 * sz2 - sz2 * sx1

        self.assertTrue(comm_x1_x2.norm("max") < 1e-12)
        self.assertTrue(comm_z1_z2.norm("max") < 1e-12)
        self.assertTrue(comm_x1_z2.norm("max") < 1e-12)

    def test_lowering_operators(self):
        sm1 = sigma_m1()
        sm2 = sigma_m2()

        # sigma_- is not Hermitian
        self.assertFalse((sm1 - sm1.dag()).norm("max") < 1e-12)

        # Nilpotency: (sigma_-)^2 = 0
        self.assertTrue((sm1 * sm1).norm("max") < 1e-12)
        self.assertTrue((sm2 * sm2).norm("max") < 1e-12)


if __name__ == "__main__":
    unittest.main()
