"""Unit tests for spectral non-Hermitian physics and exceptional point diagnostics."""

import unittest
import numpy as np

from bipartit_open_spin.config import ModelParams
from bipartit_open_spin.core.operators import sigma_m1, sigma_m2
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators
from bipartit_open_spin.dynamics.hamiltonian import (
    build_hamiltonian,
    build_effective_hamiltonian,
)
from bipartit_open_spin.analysis.spectrum import (
    parity_block_decomposition,
    analytical_eigenvalues,
    compute_complex_spectrum,
    detect_exceptional_point,
)


class TestSpectrum(unittest.TestCase):
    """Test suite for non-Hermitian spectrum, parity decomposition, and EP diagnostics."""

    def test_hermitian_limit_spectrum(self):
        """Test that at gamma=0, eigenvalues are strictly real and match Hermitian spectrum."""
        params = ModelParams(omega=1.0, J=0.5, gamma=0.0)
        H_eff = build_effective_hamiltonian(params)
        spec = compute_complex_spectrum(H_eff)
        evals = spec["eigenvalues"]

        # All imaginary parts must vanish
        self.assertTrue(np.allclose(evals.imag, 0.0, atol=1e-12))

        # Expected Hermitian eigenvalues:
        # Odd sector: pm J -> pm 0.5
        # Even sector: pm sqrt(omega^2 + J^2) -> pm sqrt(1.25) approx pm 1.1180
        expected = np.sort([-np.sqrt(1.25), -0.5, 0.5, np.sqrt(1.25)])
        self.assertTrue(np.allclose(np.sort(evals.real), expected, atol=1e-12))

    def test_uncoupled_j0_spectrum(self):
        """Test that at J=0, eigenvalues match uncoupled basis energies and decay rates."""
        omega = 1.0
        gamma = 0.8
        params = ModelParams(omega=omega, J=0.0, gamma=gamma)
        H_eff = build_effective_hamiltonian(params)
        spec = compute_complex_spectrum(H_eff)
        evals = spec["eigenvalues"]

        # Expected uncoupled eigenvalues:
        # |00>: omega - i*gamma = 1.0 - 0.8j
        # |01>: 0 - i*gamma/2 = -0.4j
        # |10>: 0 - i*gamma/2 = -0.4j
        # |11>: -omega - 0j = -1.0
        expected = [-1.0 + 0j, -0.4j, -0.4j, 1.0 - 0.8j]
        # Compare sorted by real then imag
        evals_sorted = np.sort(evals)
        expected_sorted = np.sort(expected)
        self.assertTrue(np.allclose(evals_sorted, expected_sorted, atol=1e-12))

    def test_parity_block_decomposition(self):
        """Test that H_eff exactly decomposes into independent 2x2 even and odd blocks."""
        params = ModelParams(omega=1.2, J=0.7, gamma=0.4)
        H_eff = build_effective_hamiltonian(params)
        H_even, H_odd = parity_block_decomposition(H_eff)

        self.assertEqual(H_even.shape, (2, 2))
        self.assertEqual(H_odd.shape, (2, 2))

        # Reconstruct and compare with full matrix
        mat = H_eff.full()
        # Non-block elements must be zero
        self.assertAlmostEqual(mat[0, 1], 0.0, places=12)
        self.assertAlmostEqual(mat[0, 2], 0.0, places=12)
        self.assertAlmostEqual(mat[3, 1], 0.0, places=12)
        self.assertAlmostEqual(mat[3, 2], 0.0, places=12)

        # Block diagonal matches
        self.assertTrue(np.allclose(mat[np.ix_([0, 3], [0, 3])], H_even, atol=1e-12))
        self.assertTrue(np.allclose(mat[np.ix_([1, 2], [1, 2])], H_odd, atol=1e-12))

    def test_analytical_vs_numerical_eigenvalues(self):
        """Test analytical eigenvalue formulas against numerical diagonalization."""
        omega = 1.0
        for J in [0.2, 0.5, 1.5]:
            for g1, g2 in [(0.3, 0.3), (0.1, 0.9), (1.2, 0.4)]:
                c_ops = [np.sqrt(g1) * sigma_m1(), np.sqrt(g2) * sigma_m2()]
                params = ModelParams(omega=omega, J=J, gamma=g1)
                H_eff = build_effective_hamiltonian(params, c_ops)

                spec = compute_complex_spectrum(H_eff)
                num_evals = spec["eigenvalues"]

                ana_even, ana_odd = analytical_eigenvalues(omega, J, g1, g2)
                ana_all = np.concatenate([ana_even, ana_odd])

                num_sorted = np.sort(num_evals)
                ana_sorted = np.sort(ana_all)

                self.assertTrue(np.allclose(num_sorted, ana_sorted, atol=1e-10))

    def test_symmetric_model_no_ep(self):
        """Verify that the symmetric model (gamma1=gamma2) contains NO exceptional points for J > 0."""
        omega = 1.0
        for J in np.linspace(0.1, 2.0, 10):
            for gamma in np.linspace(0.1, 3.0, 10):
                params = ModelParams(omega=omega, J=J, gamma=gamma)
                H_eff = build_effective_hamiltonian(params)
                ep_diag = detect_exceptional_point(H_eff)
                self.assertFalse(ep_diag["is_exceptional_point"])
                self.assertGreater(ep_diag["min_gap"], 1e-4)

    def test_asymmetric_ep_detection(self):
        """Verify that the asymmetric-loss model possesses an exact EP2 at J = |gamma1 - gamma2| / 4."""
        omega = 1.0
        gamma1 = 2.5
        gamma2 = 0.5
        delta_gamma = gamma1 - gamma2  # = 2.0
        j_ep = delta_gamma / 4.0  # = 0.5

        c_ops_ep = [np.sqrt(gamma1) * sigma_m1(), np.sqrt(gamma2) * sigma_m2()]
        params = ModelParams(omega=omega, J=j_ep, gamma=gamma1)
        H_eff = build_effective_hamiltonian(params, c_ops_ep)

        ep_diag = detect_exceptional_point(H_eff, tol_gap=1e-4, tol_cond=50.0)

        # Must detect an exceptional point
        self.assertTrue(ep_diag["is_exceptional_point"])
        self.assertTrue(ep_diag["rank_defect"])
        self.assertGreater(ep_diag["eigenvector_cond"], 50.0)
        self.assertLess(ep_diag["min_gap"], 1e-4)


if __name__ == "__main__":
    unittest.main()
