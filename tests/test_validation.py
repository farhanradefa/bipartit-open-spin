"""Tests for numerical validation, state diagnostics, and physical benchmarks."""

import unittest
import numpy as np
from qutip import Qobj
from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import bell_phi_plus, computational_basis, to_density_matrix
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators
from bipartit_open_spin.dynamics.simulation import simulate_dynamics
from bipartit_open_spin.validation.diagnostics import (
    check_hermiticity,
    check_positivity,
    check_trace_preservation,
    validate_density_matrix,
    validate_state_trajectory,
    verify_excited_population_decay,
)


class TestValidation(unittest.TestCase):
    def test_valid_physical_states(self):
        rho = to_density_matrix(bell_phi_plus())
        diag = validate_density_matrix(rho)

        self.assertTrue(diag["trace_preservation"])
        self.assertTrue(diag["hermiticity"])
        self.assertTrue(diag["positivity"])
        self.assertTrue(diag["valid"])

    def test_invalid_trace_detected(self):
        rho_bad_trace = 1.5 * to_density_matrix(bell_phi_plus())
        self.assertFalse(check_trace_preservation(rho_bad_trace))
        self.assertFalse(validate_density_matrix(rho_bad_trace)["valid"])

    def test_non_hermitian_detected(self):
        raw = to_density_matrix(bell_phi_plus()).full()
        raw[0, 1] += 0.5  # Break Hermiticity
        rho_non_hermitian = Qobj(raw, dims=[[2, 2], [2, 2]])
        self.assertFalse(check_hermiticity(rho_non_hermitian))
        self.assertFalse(check_positivity(rho_non_hermitian))
        self.assertFalse(validate_density_matrix(rho_non_hermitian)["valid"])

    def test_negative_eigenvalues_detected(self):
        # Diagonal matrix with negative eigenvalue
        raw = np.diag([1.2, 0.0, 0.0, -0.2])
        rho_indefinite = Qobj(raw, dims=[[2, 2], [2, 2]])
        self.assertFalse(check_positivity(rho_indefinite))
        self.assertFalse(validate_density_matrix(rho_indefinite)["valid"])

    def test_positivity_numerical_tolerance_boundary(self):
        # Matrix with small negative eigenvalue within tolerance
        raw = np.diag([1.0 + 1e-8, 0.0, 0.0, -1e-8])
        rho_near_positive = Qobj(raw, dims=[[2, 2], [2, 2]])
        self.assertTrue(check_positivity(rho_near_positive, tol=1e-6))
        self.assertFalse(check_positivity(rho_near_positive, tol=1e-10))

    def test_diagnostics_reject_negative_tolerance(self):
        rho = to_density_matrix(bell_phi_plus())
        with self.assertRaises(ValueError):
            check_trace_preservation(rho, tol=-1e-6)
        with self.assertRaises(ValueError):
            check_hermiticity(rho, tol=-1e-6)
        with self.assertRaises(ValueError):
            check_positivity(rho, tol=-1e-6)
        with self.assertRaises(ValueError):
            validate_density_matrix(rho, tol=-1e-6)
        with self.assertRaises(ValueError):
            validate_state_trajectory([rho], tol=-1e-6)

    def test_empty_trajectory_validation(self):
        traj_val = validate_state_trajectory([])
        self.assertFalse(traj_val["valid"])
        self.assertFalse(traj_val["trace_preservation"])
        self.assertFalse(traj_val["hermiticity"])
        self.assertFalse(traj_val["positivity"])

    def test_verify_excited_population_decay_validation(self):
        tlist = np.linspace(0, 10, 50)
        p_excited = np.exp(-0.2 * tlist)

        # Valid input passes
        self.assertTrue(verify_excited_population_decay(gamma=0.2, tlist=tlist, p_excited=p_excited, tol=1e-3))

        # Negative gamma raises ValueError
        with self.assertRaises(ValueError):
            verify_excited_population_decay(gamma=-0.1, tlist=tlist, p_excited=p_excited)

        # Negative tol raises ValueError
        with self.assertRaises(ValueError):
            verify_excited_population_decay(gamma=0.2, tlist=tlist, p_excited=p_excited, tol=-1e-3)

        # Length mismatch raises ValueError
        with self.assertRaises(ValueError):
            verify_excited_population_decay(gamma=0.2, tlist=tlist, p_excited=p_excited[:-1])

        # Empty arrays return False
        self.assertFalse(verify_excited_population_decay(gamma=0.2, tlist=np.array([]), p_excited=np.array([])))

    def test_open_system_simulation_and_trajectory_validation(self):

        params = ModelParams(omega=1.0, J=0.5, gamma=0.1)
        config = SimulationConfig(tlist=np.linspace(0, 5, 50))

        H = build_hamiltonian(params)
        c_ops = build_collapse_operators(params)
        psi0 = bell_phi_plus()

        states = simulate_dynamics(H, psi0, c_ops, config)

        self.assertEqual(len(states), 50)
        for s in states:
            self.assertEqual(s.dims, [[2, 2], [2, 2]])

        traj_val = validate_state_trajectory(states)
        self.assertTrue(traj_val["valid"])
        self.assertTrue(traj_val["trace_preservation"])
        self.assertTrue(traj_val["hermiticity"])
        self.assertTrue(traj_val["positivity"])

    def test_uncoupled_excited_population_decay_benchmark(self):
        # Uncoupled system: J = 0, gamma = 0.2
        gamma = 0.2
        params = ModelParams(omega=1.0, J=0.0, gamma=gamma)
        tlist = np.linspace(0, 10, 100)
        config = SimulationConfig(tlist=tlist)

        H = build_hamiltonian(params)
        c_ops = build_collapse_operators(params)

        # Initial state: |01> (spin 1 in upper/excited state |0>, spin 2 in lower/ground state |1>)
        psi0 = computational_basis(0, 1)
        states = simulate_dynamics(H, psi0, c_ops, config)

        # In standard computational basis (00, 01, 10, 11), index 0 is |00> and index 1 is |01|
        # P_excited for spin 1 is the sum of populations in |00> and |01|:
        p_excited_spin1 = np.array([abs(s.full()[0, 0] + s.full()[1, 1]) for s in states])

        is_valid_decay = verify_excited_population_decay(
            gamma=gamma,
            tlist=tlist,
            p_excited=p_excited_spin1,
            tol=1e-3,
        )
        self.assertTrue(is_valid_decay)


if __name__ == "__main__":
    unittest.main()
