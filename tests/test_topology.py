"""Unit tests for static non-Hermitian spectral topology and branch tracking around EP2."""

import unittest
import numpy as np

from bipartit_open_spin.analysis.spectrum import (
    build_topological_odd_hamiltonian,
    compute_biorthogonal_eigenpairs,
    track_eigenpairs_along_loop,
)


class TestTopology(unittest.TestCase):
    """Test suite for topological Hamiltonian, biorthogonal vectors, and EP branch tracking."""

    def test_topological_hamiltonian_structure(self):
        """Test that H_top is traceless, non-Hermitian, and has correct matrix elements."""
        J = 0.4
        delta_gamma = 1.6
        H_top = build_topological_odd_hamiltonian(J, delta_gamma)

        self.assertEqual(H_top.shape, (2, 2))
        # Traceless
        self.assertAlmostEqual(np.trace(H_top), 0.0, places=14)
        # Elements
        self.assertAlmostEqual(H_top[0, 0], -0.4j, places=14)
        self.assertAlmostEqual(H_top[1, 1], +0.4j, places=14)
        self.assertAlmostEqual(H_top[0, 1], 0.4, places=14)
        self.assertAlmostEqual(H_top[1, 0], 0.4, places=14)

    def test_biorthogonal_normalization(self):
        """Test that left and right eigenvectors satisfy biorthogonality away from EP."""
        J = 0.6
        delta_gamma = 1.6  # J_EP = 0.4, so J=0.6 is well separated
        H_top = build_topological_odd_hamiltonian(J, delta_gamma)
        res = compute_biorthogonal_eigenpairs(H_top)

        R = res["right_eigenvectors"]
        L = res["left_eigenvectors"]

        # Check <L_m | R_n> = delta_mn
        overlap_matrix = L.conj().T @ R
        self.assertTrue(np.allclose(overlap_matrix, np.eye(2), atol=1e-10))

        # Check Petermann factors are finite and >= 1.0
        K = res["petermann_factors"]
        self.assertTrue(np.all(K >= 1.0 - 1e-10))

    def test_enclosing_loop_eigenvalue_swap(self):
        """Verify that an EP-enclosing loop swaps eigenvalues after 2*pi and returns after 4*pi."""
        # EP is at (J_EP, delta_gamma_EP) = (0.4, 1.6), delta_omega = 0
        j_ep = 0.4
        dg_ep = 1.6
        r_j = 0.15
        r_omega = 0.15

        theta_grid = np.linspace(0, 4.0 * np.pi, 801)

        def H_loop(th):
            j_th = j_ep + r_j * np.cos(th)
            domega_th = r_omega * np.sin(th)
            return np.array([
                [domega_th - 0.25j * dg_ep, j_th],
                [j_th, -domega_th + 0.25j * dg_ep],
            ], dtype=complex)

        res = track_eigenpairs_along_loop(H_loop, theta_grid)

        # 1-loop swap check: lambda_1(2pi) approx lambda_2(0) and lambda_2(2pi) approx lambda_1(0)
        idx_2pi = np.argmin(np.abs(theta_grid - 2.0 * np.pi))
        evals_0 = res["eigenvalues"][0, :]
        evals_2pi = res["eigenvalues"][idx_2pi, :]

        self.assertTrue(res["permutation_1_loop"])
        self.assertLess(np.abs(evals_2pi[0] - evals_0[1]), 1e-3)
        self.assertLess(np.abs(evals_2pi[1] - evals_0[0]), 1e-3)

        # 2-loop return check: lambda_1(4pi) approx lambda_1(0)
        idx_4pi = np.argmin(np.abs(theta_grid - 4.0 * np.pi))
        evals_4pi = res["eigenvalues"][idx_4pi, :]
        self.assertLess(np.abs(evals_4pi[0] - evals_0[0]), 1e-3)
        self.assertLess(np.abs(evals_4pi[1] - evals_0[1]), 1e-3)

    def test_avoiding_loop_no_swap(self):
        """Verify that a loop avoiding the EP does NOT swap eigenvalues after 2*pi."""
        # Loop centered at (0.7, 1.0) with radii (0.1, 0.3) -> stays away from EP line J = dg / 4 = 0.25
        j_center = 0.7
        dg_center = 1.0
        r_j = 0.1
        r_dg = 0.3

        theta_grid = np.linspace(0, 2.0 * np.pi, 400)

        def H_loop(th):
            j_th = j_center + r_j * np.cos(th)
            dg_th = dg_center + r_dg * np.sin(th)
            return build_topological_odd_hamiltonian(j_th, dg_th)

        res = track_eigenpairs_along_loop(H_loop, theta_grid)

        self.assertFalse(res["permutation_1_loop"])
        evals_0 = res["eigenvalues"][0, :]
        evals_2pi = res["eigenvalues"][-1, :]
        self.assertLess(np.abs(evals_2pi[0] - evals_0[0]), 1e-3)
        self.assertLess(np.abs(evals_2pi[1] - evals_0[1]), 1e-3)

    def test_branch_tracking_continuity(self):
        """Verify continuous evolution without step-to-step discontinuities."""
        j_ep = 0.4
        dg_ep = 1.6
        r_j = 0.1
        r_dg = 0.4
        theta_grid = np.linspace(0, 2.0 * np.pi, 500)

        def H_loop(th):
            j_th = j_ep + r_j * np.cos(th)
            dg_th = dg_ep + r_dg * np.sin(th)
            return build_topological_odd_hamiltonian(j_th, dg_th)

        res = track_eigenpairs_along_loop(H_loop, theta_grid)
        evals = res["eigenvalues"]

        # Step-to-step diffs
        diffs_0 = np.abs(np.diff(evals[:, 0]))
        diffs_1 = np.abs(np.diff(evals[:, 1]))

        # Max step jump must be smooth (< 0.05)
        self.assertLess(np.max(diffs_0), 0.05)
        self.assertLess(np.max(diffs_1), 0.05)


if __name__ == "__main__":
    unittest.main()
