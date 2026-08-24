"""Tests for entanglement measures."""

import unittest
import numpy as np
from bipartit_open_spin.core.states import bell_phi_plus, computational_basis, to_density_matrix
from bipartit_open_spin.analysis.entanglement import (
    concurrence,
    concurrence_trajectory,
    negativity,
    negativity_trajectory,
)


class TestEntanglement(unittest.TestCase):
    def test_bell_state_maximally_entangled(self):
        phi_plus = bell_phi_plus()
        rho_phi = to_density_matrix(phi_plus)

        # Test with ket input
        self.assertAlmostEqual(concurrence(phi_plus), 1.0, places=6)
        self.assertAlmostEqual(negativity(phi_plus), 0.5, places=6)

        # Test with density matrix input
        self.assertAlmostEqual(concurrence(rho_phi), 1.0, places=6)
        self.assertAlmostEqual(negativity(rho_phi), 0.5, places=6)

    def test_separable_product_states(self):
        for i in (0, 1):
            for j in (0, 1):
                prod_state = computational_basis(i, j)
                self.assertAlmostEqual(concurrence(prod_state), 0.0, places=6)
                self.assertAlmostEqual(negativity(prod_state), 0.0, places=6)

    def test_trajectory_evaluation(self):
        states = [bell_phi_plus(), computational_basis(0, 0)]
        c_traj = concurrence_trajectory(states)
        n_traj = negativity_trajectory(states)

        self.assertEqual(len(c_traj), 2)
        self.assertEqual(len(n_traj), 2)
        np.testing.assert_allclose(c_traj, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(n_traj, [0.5, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
