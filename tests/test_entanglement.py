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

    def test_negativity_tolerance_filtering(self):
        phi_plus = bell_phi_plus()
        # Normal Bell state has negativity 0.5 with default tol
        self.assertAlmostEqual(negativity(phi_plus, tol=1e-12), 0.5, places=6)

        # For Werner state rho = p |Phi+><Phi+| + (1-p)/4 I
        # partial transpose eigenvalues are (1-p)/4 (x3) and (1-3p)/4
        # For p = 0.4, (1-3*0.4)/4 = -0.05 (negative eigenvalue)
        p = 0.4
        rho_werner = p * to_density_matrix(phi_plus) + ((1.0 - p) / 4.0) * (
            to_density_matrix(computational_basis(0, 0))
            + to_density_matrix(computational_basis(0, 1))
            + to_density_matrix(computational_basis(1, 0))
            + to_density_matrix(computational_basis(1, 1))
        )
        # Negative eigenvalue is -0.05
        # If tol=0.01 (smaller than 0.05), it is counted (| -0.05 | = 0.05)
        self.assertAlmostEqual(negativity(rho_werner, tol=0.01), 0.05, places=6)
        # If tol=0.1 (larger than 0.05), eigenvalue -0.05 > -0.1, so it is filtered out
        self.assertEqual(negativity(rho_werner, tol=0.1), 0.0)

    def test_negativity_rejects_negative_tolerance(self):
        phi_plus = bell_phi_plus()
        with self.assertRaises(ValueError):
            negativity(phi_plus, tol=-1e-12)
        with self.assertRaises(ValueError):
            negativity_trajectory([phi_plus], tol=-1e-5)

    def test_entanglement_generation_from_separable_state(self):
        from bipartit_open_spin.config import ModelParams, SimulationConfig
        from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
        from bipartit_open_spin.dynamics.simulation import simulate_dynamics

        omega, J = 1.0, 0.5
        params = ModelParams(omega=omega, J=J, gamma=0.0)
        config = SimulationConfig(tlist=np.linspace(0.0, 5.0, 100))
        psi0 = computational_basis(0, 0)
        H = build_hamiltonian(params)

        states = simulate_dynamics(H, psi0, [], config)
        c_traj = concurrence_trajectory(states)

        # Initial separable state has zero concurrence
        self.assertAlmostEqual(c_traj[0], 0.0, places=6)

        # Entanglement is dynamically generated
        self.assertTrue(np.max(c_traj) > 0.0)

        # Analytical maximum concurrence for J <= omega is 2*omega*J / (omega^2 + J^2) = 0.8
        c_max_analytical = (2.0 * omega * J) / (omega**2 + J**2)
        self.assertAlmostEqual(np.max(c_traj), c_max_analytical, places=2)


if __name__ == "__main__":

    unittest.main()


