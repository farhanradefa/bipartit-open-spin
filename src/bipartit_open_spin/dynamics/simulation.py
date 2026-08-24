"""Simulation engine wrapping QuTiP master equation solver."""

from qutip import Qobj, mesolve
from bipartit_open_spin.config import SimulationConfig
from bipartit_open_spin.core.states import to_density_matrix


def simulate_dynamics(
    H: Qobj,
    psi0: Qobj,
    c_ops: list[Qobj],
    config: SimulationConfig,
) -> list[Qobj]:
    """Simulate open-system time evolution via the Lindblad master equation.

    All states along the trajectory are guaranteed to be density matrices with
    subsystem dimensions [[2, 2], [2, 2]].

    Args:
        H: System Hamiltonian (Qobj).
        psi0: Initial state (ket or density matrix).
        c_ops: List of Lindblad collapse operators.
        config: SimulationConfig containing time grid and solver options.

    Returns:
        List of density matrix Qobj instances for each time point in config.tlist.
    """
    kwargs = {}
    if config.options is not None:
        kwargs["options"] = config.options

    result = mesolve(
        H,
        psi0,
        config.tlist,
        c_ops,
        **kwargs,
    )

    # Ensure every state in trajectory is a density matrix with correct dims
    density_matrices = [to_density_matrix(s) for s in result.states]
    return density_matrices
