"""Unit tests for effective non-Hermitian Hamiltonian and quantum trajectory simulations."""

import unittest
import numpy as np
from qutip import Qobj

from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import computational_basis, bell_phi_plus
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators
from bipartit_open_spin.dynamics.hamiltonian import (
    build_hamiltonian,
    build_effective_hamiltonian,
)
from bipartit_open_spin.dynamics.simulation import (
    simulate_dynamics,
    simulate_no_jump_dynamics,
    simulate_quantum_trajectories,
)


class TestNonHermitian(unittest.TestCase):
    """Test suite for effective non-Hermitian dynamics and quantum trajectory unraveling."""

    def test_effective_hamiltonian_structure(self):
        """Test that H_eff = H - (i/2) sum_k L_k^dagger L_k and has correct non-Hermitian part."""
        params = ModelParams(omega=1.0, J=0.5, gamma=0.4)
        H = build_hamiltonian(params)
        c_ops = build_collapse_operators(params)
        H_eff = build_effective_hamiltonian(params, c_ops)

        # Check dimension and shape
        self.assertEqual(H_eff.dims, [[2, 2], [2, 2]])
        self.assertEqual(H_eff.shape, (4, 4))

        # Check that H_eff is not Hermitian
        diff_herm = np.linalg.norm((H_eff - H_eff.dag()).full())
        self.assertGreater(diff_herm, 1e-4)

        # Check the anti-Hermitian part: (H_eff - H_eff.dag()) / 2j == -0.5 * sum c_k^dagger c_k
        anti_herm = (H_eff - H_eff.dag()) / 2.0j
        expected_decay = Qobj(np.zeros((4, 4), dtype=complex), dims=[[2, 2], [2, 2]])
        for c in c_ops:
            expected_decay += -0.5 * c.dag() * c

        diff_decay = np.linalg.norm((anti_herm - expected_decay).full())
        self.assertAlmostEqual(diff_decay, 0.0, places=10)

    def test_basis_decay_rates(self):
        """Test that state-dependent decay rates match physical excitations in computational basis."""
        gamma = 0.6
        params = ModelParams(omega=1.0, J=0.0, gamma=gamma)
        c_ops = build_collapse_operators(params)
        H_eff = build_effective_hamiltonian(params, c_ops)

        mat_eff = H_eff.full()

        # In repository convention: |00>=[0], |01>=[1], |10>=[2], |11>=[3]
        # Imaginary parts correspond to -Gamma_ij / 2
        # |00>: 2 excitations -> Im = -gamma
        self.assertAlmostEqual(mat_eff[0, 0].imag, -gamma, places=10)
        # |01>: 1 excitation -> Im = -gamma / 2
        self.assertAlmostEqual(mat_eff[1, 1].imag, -gamma / 2.0, places=10)
        # |10>: 1 excitation -> Im = -gamma / 2
        self.assertAlmostEqual(mat_eff[2, 2].imag, -gamma / 2.0, places=10)
        # |11>: 0 excitation (ground state) -> Im = 0
        self.assertAlmostEqual(mat_eff[3, 3].imag, 0.0, places=10)

    def test_survival_probability_decay_and_derivative(self):
        """Test that P_no_jump(t) <= 1, is monotonically non-increasing, and matches theoretical loss rate."""
        params = ModelParams(omega=1.0, J=0.5, gamma=0.5)
        H_eff = build_effective_hamiltonian(params)
        c_ops = build_collapse_operators(params)

        tlist = np.linspace(0.0, 5.0, 500)
        config = SimulationConfig(tlist=tlist)
        psi0 = computational_basis(0, 0)

        res = simulate_no_jump_dynamics(H_eff, psi0, config, c_ops_for_loss=c_ops)
        p_surv = res["survival_probability"]
        loss_rate = res["theoretical_loss_rate"]

        # 1. P_no_jump(0) == 1.0 and P_no_jump(t) <= 1.0
        self.assertAlmostEqual(p_surv[0], 1.0, places=8)
        self.assertTrue(np.all(p_surv <= 1.0000001))
        self.assertTrue(np.all(p_surv >= 0.0))

        # 2. Monotonic non-increasing
        diff_p = np.diff(p_surv)
        self.assertTrue(np.all(diff_p <= 1e-7))

        # 3. Numerical derivative dP/dt matches -loss_rate
        dt = tlist[1] - tlist[0]
        dp_dt_num = np.gradient(p_surv, dt)
        err_deriv = np.max(np.abs(dp_dt_num[1:-1] - (-loss_rate[1:-1])))
        self.assertLess(err_deriv, 0.005)

    def test_no_jump_dark_ground_state(self):
        """Test that ground state |11> at J=0 does not decay under H_eff (P_no_jump == 1 for all t)."""
        params = ModelParams(omega=1.0, J=0.0, gamma=1.0)
        H_eff = build_effective_hamiltonian(params)

        tlist = np.linspace(0.0, 5.0, 50)
        config = SimulationConfig(tlist=tlist)
        psi11 = computational_basis(1, 1)

        res = simulate_no_jump_dynamics(H_eff, psi11, config)
        p_surv = res["survival_probability"]
        self.assertTrue(np.allclose(p_surv, 1.0, atol=1e-8))

    def test_quantum_trajectory_reconstruction(self):
        """Test that quantum trajectory ensemble average reconstructs Lindblad master equation dynamics."""
        params = ModelParams(omega=1.0, J=0.5, gamma=0.5)
        H = build_hamiltonian(params)
        c_ops = build_collapse_operators(params)

        tlist = np.linspace(0.0, 4.0, 50)
        config = SimulationConfig(tlist=tlist)
        psi0 = bell_phi_plus()

        # 1. Master equation
        states_lindblad = simulate_dynamics(H, psi0, c_ops, config)

        # 2. Quantum trajectories with ntraj=200
        mc_res = simulate_quantum_trajectories(H, psi0, c_ops, config, ntraj=200, seed=42)
        states_mc = mc_res.states

        # Check average distance ||rho_MC(t) - rho_Lindblad(t)||_F
        distances = [
            np.linalg.norm(rho_mc.full() - rho_lb.full())
            for rho_mc, rho_lb in zip(states_mc, states_lindblad)
        ]
        mean_dist = float(np.mean(distances))
        self.assertLess(mean_dist, 0.12)


if __name__ == "__main__":
    unittest.main()
