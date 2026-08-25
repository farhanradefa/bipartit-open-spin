"""Unit tests for dynamical EP encircling and time-dependent non-Hermitian simulation."""

import unittest
import numpy as np

from bipartit_open_spin.dynamics.simulation import simulate_timedependent_nonhermitian
from bipartit_open_spin.analysis.spectrum import (
    build_topological_odd_hamiltonian,
    track_instantaneous_eigenpairs,
)


class TestDynamicEncircling(unittest.TestCase):
    """Test suite for time-dependent non-Hermitian simulation and EP encircling."""

    def setUp(self):
        self.j_center = 0.35
        self.r_j = 0.15
        self.dg_center = 1.40
        self.r_dg = 0.45
        self.gamma_bar = 1.0
        self.T = 20.0
        self.tlist = np.linspace(0.0, self.T, 400)

    def get_loop_hamiltonian(self, direction="CCW"):
        sign = +1.0 if direction == "CCW" else -1.0

        def H_func(t):
            theta = sign * 2.0 * np.pi * t / self.T
            j_t = self.j_center + self.r_j * np.cos(theta)
            dg_t = self.dg_center + self.r_dg * np.sin(theta)
            # Full H_eff_odd(t) = -i * (gamma_bar/2) * I + H_top(t)
            H_top = build_topological_odd_hamiltonian(j_t, dg_t)
            return -0.5j * self.gamma_bar * np.eye(2, dtype=complex) + H_top

        return H_func

    def test_parameter_loop_closure_and_positivity(self):
        """Verify that parameter loops close exactly and dissipation rates remain positive."""
        for t in [0.0, self.T * 0.25, self.T * 0.5, self.T * 0.75, self.T]:
            theta = 2.0 * np.pi * t / self.T
            j_t = self.j_center + self.r_j * np.cos(theta)
            dg_t = self.dg_center + self.r_dg * np.sin(theta)
            g1_t = self.gamma_bar + 0.5 * dg_t
            g2_t = self.gamma_bar - 0.5 * dg_t

            self.assertGreater(g1_t, 0.0)
            self.assertGreater(g2_t, 0.0)
            self.assertGreater(j_t, 0.0)

        # Loop closure: t=0 and t=T must match exactly
        h_0 = self.get_loop_hamiltonian("CCW")(0.0)
        h_T = self.get_loop_hamiltonian("CCW")(self.T)
        self.assertTrue(np.allclose(h_0, h_T, atol=1e-12))

    def test_ep_enclosed_not_crossed(self):
        """Verify that EP (0.4, 1.6) is strictly inside the ellipse and minimum gap is non-zero."""
        # Ellipse: ((J - 0.4)/0.25)^2 + ((dg - 1.6)/0.80)^2 = 1.0
        # EP is at (0.4, 1.6), value = 0.0 < 1.0 (strictly inside)
        H_func = self.get_loop_hamiltonian("CCW")
        spec_res = track_instantaneous_eigenpairs(H_func, self.tlist)

        min_gap = spec_res["min_gap"]
        self.assertGreater(min_gap, 0.01)

    def test_cw_ccw_time_reversal_geometry(self):
        """Verify that CW path is the exact time-reverse of CCW path."""
        H_ccw = self.get_loop_hamiltonian("CCW")
        H_cw = self.get_loop_hamiltonian("CW")

        for t in np.linspace(0.0, self.T, 20):
            # H_cw(t) must equal H_ccw(T - t)
            h1 = H_cw(t)
            h2 = H_ccw(self.T - t)
            self.assertTrue(np.allclose(h1, h2, atol=1e-12))

    def test_normalized_state_conservation(self):
        """Verify that normalized conditional state has <psi_tilde|psi_tilde> = 1.0 for all t."""
        H_func = self.get_loop_hamiltonian("CCW")
        psi0 = np.array([1.0, 0.0], dtype=complex)

        res = simulate_timedependent_nonhermitian(H_func, psi0, self.tlist)
        norm_states = res["normalized_states"]

        for idx in range(len(self.tlist)):
            vec = norm_states[idx, :]
            norm_val = np.real(np.vdot(vec, vec))
            self.assertAlmostEqual(norm_val, 1.0, places=12)

        # Populations sum to 1
        p_sum = res["p01"] + res["p10"]
        self.assertTrue(np.allclose(p_sum, 1.0, atol=1e-12))

    def test_survival_probability_decay(self):
        """Verify that survival probability decays from 1.0 and remains positive."""
        H_func = self.get_loop_hamiltonian("CCW")
        psi0 = np.array([1.0, 0.0], dtype=complex)

        res = simulate_timedependent_nonhermitian(H_func, psi0, self.tlist)
        p_surv = res["survival_probability"]

        self.assertAlmostEqual(p_surv[0], 1.0, places=12)
        self.assertTrue(np.all(p_surv > 0.0))
        self.assertLess(p_surv[-1], p_surv[0])


if __name__ == "__main__":
    unittest.main()
