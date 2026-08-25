"""Unit tests for full open-system time-dependent Lindblad simulation and robustness checks."""

import unittest
import numpy as np
from bipartit_open_spin.core.operators import (
    sigma_z1,
    sigma_z2,
    sigma_x1,
    sigma_x2,
    sigma_m1,
    sigma_m2,
)
from bipartit_open_spin.dynamics.simulation import (
    simulate_timedependent_lindblad,
    simulate_timedependent_nonhermitian,
)
from bipartit_open_spin.validation.diagnostics import validate_density_matrix


class TestLindbladEncircling(unittest.TestCase):
    """Test suite for full open-system Lindblad dynamics under EP encircling."""

    def setUp(self):
        self.omega = 1.0
        self.j_center = 0.35
        self.r_j = 0.15
        self.dg_center = 1.40
        self.r_dg = 0.45
        self.gamma_bar = 1.0
        self.T = 15.0
        self.tlist = np.linspace(0.0, self.T, 300)

        # 4x4 Operators from validated core module
        self.sz1 = sigma_z1()
        self.sz2 = sigma_z2()
        self.sx1 = sigma_x1()
        self.sx2 = sigma_x2()
        self.sm1 = sigma_m1()
        self.sm2 = sigma_m2()

        self.H_diag = 0.5 * self.omega * (self.sz1 + self.sz2)
        self.H_int_base = self.sx1 * self.sx2

    def get_full_lindblad_model(self, direction="CCW", gamma_phi=0.0):
        sign = +1.0 if direction.upper() == "CCW" else -1.0

        def H_func(t):
            theta = sign * 2.0 * np.pi * t / self.T
            j_t = self.j_center + self.r_j * np.cos(theta)
            return self.H_diag + j_t * self.H_int_base

        def c_ops_func(t):
            theta = sign * 2.0 * np.pi * t / self.T
            dg_t = self.dg_center + self.r_dg * np.sin(theta)
            g1_t = self.gamma_bar + 0.5 * dg_t
            g2_t = self.gamma_bar - 0.5 * dg_t

            c_ops = [
                np.sqrt(max(0.0, g1_t)) * self.sm1,
                np.sqrt(max(0.0, g2_t)) * self.sm2,
            ]
            if gamma_phi > 0.0:
                c_ops.append(np.sqrt(0.5 * gamma_phi) * self.sz1)
                c_ops.append(np.sqrt(0.5 * gamma_phi) * self.sz2)
            return c_ops

        return H_func, c_ops_func

    def test_density_matrix_physical_validity(self):
        """Verify that density matrices along Lindblad trajectory satisfy Hermiticity, Trace=1, and Positivity."""
        H_func, c_ops_func = self.get_full_lindblad_model("CCW", gamma_phi=0.1)
        # Initial state |01> (basis index 1)
        psi0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)

        res = simulate_timedependent_lindblad(H_func, c_ops_func, psi0, self.tlist)

        for rho in res["states"]:
            val_res = validate_density_matrix(rho, tol=1e-6)
            self.assertTrue(val_res["valid"], f"Invalid density matrix: {val_res}")

    def test_parameter_rates_positivity(self):
        """Verify that gamma_1(t) and gamma_2(t) remain strictly positive for all t."""
        for t in np.linspace(0.0, self.T, 100):
            theta = 2.0 * np.pi * t / self.T
            dg_t = self.dg_center + self.r_dg * np.sin(theta)
            g1 = self.gamma_bar + 0.5 * dg_t
            g2 = self.gamma_bar - 0.5 * dg_t
            self.assertGreater(g1, 0.0)
            self.assertGreater(g2, 0.0)

    def test_lindblad_vs_nojump_short_time(self):
        """For short times t << 1/gamma_bar, Lindblad populations match conditional no-jump populations."""
        t_short = np.linspace(0.0, 0.1, 50)
        H_func, c_ops_func = self.get_full_lindblad_model("CCW", gamma_phi=0.0)
        psi0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)

        res_lindblad = simulate_timedependent_lindblad(H_func, c_ops_func, psi0, t_short)

        # 2x2 no-jump odd Hamiltonian
        def H_odd_func(t):
            theta = 2.0 * np.pi * t / self.T
            j_t = self.j_center + self.r_j * np.cos(theta)
            dg_t = self.dg_center + self.r_dg * np.sin(theta)
            return np.array([
                [-0.5j * (self.gamma_bar + 0.5 * dg_t), j_t],
                [j_t, -0.5j * (self.gamma_bar - 0.5 * dg_t)],
            ], dtype=complex)

        res_nojump = simulate_timedependent_nonhermitian(H_odd_func, np.array([1.0, 0.0]), t_short)
        raw_states = res_nojump["raw_states"]
        p01_nojump = np.abs(raw_states[:, 0]) ** 2

        diff = np.max(np.abs(res_lindblad["p01"] - p01_nojump))
        self.assertLess(diff, 0.01)

    def test_dephasing_odd_sector_damping(self):
        """Verify that pure dephasing accelerates decay of off-diagonal coherence rho_01,10."""
        psi_super = np.array([0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0], dtype=complex)

        H_func_0, c_ops_func_0 = self.get_full_lindblad_model("CCW", gamma_phi=0.0)
        H_func_phi, c_ops_func_phi = self.get_full_lindblad_model("CCW", gamma_phi=0.5)

        res_0 = simulate_timedependent_lindblad(H_func_0, c_ops_func_0, psi_super, self.tlist)
        res_phi = simulate_timedependent_lindblad(H_func_phi, c_ops_func_phi, psi_super, self.tlist)

        # Coherence with dephasing should be strictly smaller at t=T
        coh_0 = res_0["abs_coherence"][-1]
        coh_phi = res_phi["abs_coherence"][-1]
        self.assertLess(coh_phi, coh_0 + 1e-10)

    def test_control_loop_geometry(self):
        """Verify that the non-encircling control loop is shifted away from the EP line."""
        j_ctrl_center = 0.70
        dg_ctrl_center = 0.40
        r_j = 0.15
        r_dg = 0.45

        # Check distance to EP line: J = Delta_gamma / 4
        for theta in np.linspace(0, 2 * np.pi, 50):
            j_t = j_ctrl_center + r_j * np.cos(theta)
            dg_t = dg_ctrl_center + r_dg * np.sin(theta)
            j_ep_t = dg_t / 4.0
            # j_t is always well above j_ep_t
            self.assertGreater(j_t - j_ep_t, 0.20)


if __name__ == "__main__":
    unittest.main()
