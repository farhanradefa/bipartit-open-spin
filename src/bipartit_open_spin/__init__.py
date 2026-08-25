"""Bipartite Open Spin System: Dissipative Entanglement Dynamics."""

from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import (
    bell_phi_plus,
    computational_basis,
    to_density_matrix,
)
from bipartit_open_spin.core.operators import (
    identity_2qubit,
    sigma_m1,
    sigma_m2,
    sigma_x1,
    sigma_x2,
    sigma_z1,
    sigma_z2,
)
from bipartit_open_spin.dynamics.hamiltonian import (
    build_effective_hamiltonian,
    build_hamiltonian,
)
from bipartit_open_spin.dynamics.dissipation import (
    build_collapse_operators,
    build_dephasing_collapse_operators,
    dissipative_jump_rates,
    liouvillian_action,
    liouvillian_superoperator_decomposition,
)
from bipartit_open_spin.dynamics.simulation import (
    simulate_dynamics,
    simulate_no_jump_dynamics,
    simulate_quantum_trajectories,
)
from bipartit_open_spin.analysis.entanglement import (
    concurrence,
    concurrence_trajectory,
    negativity,
    negativity_trajectory,
    x_state_concurrence_decomposition,
)
from bipartit_open_spin.analysis.spectrum import (
    analytical_eigenvalues,
    build_topological_odd_hamiltonian,
    compute_biorthogonal_eigenpairs,
    compute_complex_spectrum,
    detect_exceptional_point,
    parity_block_decomposition,
    track_eigenpairs_along_loop,
)
from bipartit_open_spin.validation.diagnostics import (
    check_hermiticity,
    check_positivity,
    check_trace_preservation,
    validate_density_matrix,
    validate_state_trajectory,
    verify_excited_population_decay,
)


def main() -> None:
    print("Bipartite Open Spin System: Baseline Module Initialized.")


__all__ = [
    "ModelParams",
    "SimulationConfig",
    "computational_basis",
    "bell_phi_plus",
    "to_density_matrix",
    "identity_2qubit",
    "sigma_x1",
    "sigma_x2",
    "sigma_z1",
    "sigma_z2",
    "sigma_m1",
    "sigma_m2",
    "build_hamiltonian",
    "build_effective_hamiltonian",
    "build_collapse_operators",
    "build_dephasing_collapse_operators",
    "liouvillian_action",
    "liouvillian_superoperator_decomposition",
    "dissipative_jump_rates",
    "simulate_dynamics",
    "simulate_no_jump_dynamics",
    "simulate_quantum_trajectories",
    "concurrence",
    "negativity",
    "concurrence_trajectory",
    "negativity_trajectory",
    "x_state_concurrence_decomposition",
    "parity_block_decomposition",
    "analytical_eigenvalues",
    "compute_complex_spectrum",
    "detect_exceptional_point",
    "build_topological_odd_hamiltonian",
    "compute_biorthogonal_eigenpairs",
    "track_eigenpairs_along_loop",
    "check_trace_preservation",
    "check_hermiticity",
    "check_positivity",
    "validate_density_matrix",
    "validate_state_trajectory",
    "verify_excited_population_decay",
    "main",
]
