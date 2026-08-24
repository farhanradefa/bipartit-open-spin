"""Unit tests for physical mechanism analysis, X-state decomposition, and Liouvillian superoperators."""

import unittest
import numpy as np
from qutip import Qobj, steadystate

from bipartit_open_spin.analysis.entanglement import (
    concurrence,
    x_state_concurrence_decomposition,
)
from bipartit_open_spin.config import ModelParams
from bipartit_open_spin.core.states import bell_phi_plus, computational_basis
from bipartit_open_spin.dynamics.dissipation import (
    build_collapse_operators,
    build_dephasing_collapse_operators,
    dissipative_jump_rates,
    liouvillian_action,
    liouvillian_superoperator_decomposition,
)
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian


class TestMechanism(unittest.TestCase):
    """Test suite for X-state concurrence decomposition and Liouvillian balance tools."""

    def test_x_state_decomposition_bell_state(self):
        """Test X-state decomposition on maximally entangled Bell state |Phi+>."""
        psi = bell_phi_plus()
        decomp = x_state_concurrence_decomposition(psi)

        self.assertAlmostEqual(decomp["p00"], 0.5, places=10)
        self.assertAlmostEqual(decomp["p11"], 0.5, places=10)
        self.assertAlmostEqual(decomp["p01"], 0.0, places=10)
        self.assertAlmostEqual(decomp["p10"], 0.0, places=10)
        self.assertAlmostEqual(decomp["abs_rho0011"], 0.5, places=10)
        self.assertAlmostEqual(decomp["abs_rho0110"], 0.0, places=10)
        self.assertAlmostEqual(decomp["threshold_even"], 0.0, places=10)
        self.assertAlmostEqual(decomp["e_even"], 0.5, places=10)
        self.assertAlmostEqual(decomp["c_x"], 1.0, places=10)
        self.assertAlmostEqual(decomp["concurrence"], 1.0, places=10)

    def test_x_state_decomposition_separable_state(self):
        """Test X-state decomposition on separable product state |00>."""
        psi = computational_basis(0, 0)
        decomp = x_state_concurrence_decomposition(psi)

        self.assertAlmostEqual(decomp["p00"], 1.0, places=10)
        self.assertAlmostEqual(decomp["p11"], 0.0, places=10)
        self.assertAlmostEqual(decomp["abs_rho0011"], 0.0, places=10)
        self.assertAlmostEqual(decomp["c_x"], 0.0, places=10)
        self.assertAlmostEqual(decomp["concurrence"], 0.0, places=10)

    def test_x_state_decomposition_werner_state(self):
        """Test X-state decomposition on Werner-like state with known analytical concurrence."""
        # rho = p |Phi+><Phi+| + (1-p)/4 I
        p = 0.8
        bell_dm = bell_phi_plus().proj()
        ident = Qobj(np.eye(4) / 4.0, dims=[[2, 2], [2, 2]])
        rho_w = p * bell_dm + (1 - p) * ident

        decomp = x_state_concurrence_decomposition(rho_w)
        c_expected = max(0.0, 1.5 * p - 0.5)

        self.assertAlmostEqual(decomp["c_x"], c_expected, places=7)
        self.assertAlmostEqual(decomp["concurrence"], c_expected, places=7)

    def test_liouvillian_decomposition_linearity_and_steady_state(self):
        """Test that L_tot = L_H + L_amp + L_phi and L_tot(rho_ss) == 0 at steady state."""
        params = ModelParams(omega=1.0, J=0.85, gamma=1.80)
        H = build_hamiltonian(params)
        c_ops_amp = build_collapse_operators(params)
        c_ops_phi = build_dephasing_collapse_operators(0.50)

        c_ops_all = c_ops_amp + c_ops_phi
        rho_ss = steadystate(H, c_ops_all)

        decomp = liouvillian_superoperator_decomposition(H, c_ops_amp, c_ops_phi, rho_ss)
        L_H = decomp["L_H"]
        L_amp = decomp["L_amp"]
        L_phi = decomp["L_phi"]
        L_tot = decomp["L_tot"]

        # 1. Linearity: L_tot == L_H + L_amp + L_phi
        sum_components = L_H + L_amp + L_phi
        diff_linearity = np.linalg.norm(L_tot.full() - sum_components.full())
        self.assertAlmostEqual(diff_linearity, 0.0, places=10)

        # 2. Direct liouvillian_action matches L_tot
        L_direct = liouvillian_action(H, c_ops_all, rho_ss)
        diff_direct = np.linalg.norm(L_tot.full() - L_direct.full())
        self.assertAlmostEqual(diff_direct, 0.0, places=10)

        # 3. Steady state condition: ||L_tot(rho_ss)|| ~ 0
        norm_ss = np.linalg.norm(L_tot.full())
        self.assertLess(norm_ss, 1e-6)

    def test_dissipative_jump_rates(self):
        """Test calculation of dissipative jump rates."""
        params = ModelParams(omega=1.0, J=0.0, gamma=2.0)
        c_ops_amp = build_collapse_operators(params)

        # In QuTiP convention, sigma_- |0> = |1>, so excited state is |0>
        psi_00 = computational_basis(0, 0)
        rates_00 = dissipative_jump_rates(c_ops_amp, psi_00)
        self.assertAlmostEqual(rates_00[0], 2.0, places=10)
        self.assertAlmostEqual(rates_00[1], 2.0, places=10)

        # Ground state |11> has no emission
        psi_11 = computational_basis(1, 1)
        rates_11 = dissipative_jump_rates(c_ops_amp, psi_11)
        self.assertAlmostEqual(rates_11[0], 0.0, places=10)
        self.assertAlmostEqual(rates_11[1], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
