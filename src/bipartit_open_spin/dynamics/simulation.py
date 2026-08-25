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


def simulate_no_jump_dynamics(
    H_eff: Qobj,
    psi0: Qobj,
    config: SimulationConfig,
    c_ops_for_loss: list[Qobj] = None,
) -> dict:
    """Simulate conditional no-jump time evolution governed by the non-Hermitian Hamiltonian H_eff.

    Evolves: i d|psi>/dt = H_eff |psi>.
    Tracks unnormalized states, survival probability P_no_jump(t) = <psi(t)|psi(t)>,
    and conditional normalized states |psi_c(t)> = |psi(t)> / sqrt(P_no_jump(t)).

    Args:
        H_eff: Effective non-Hermitian Hamiltonian (Qobj).
        psi0: Initial pure state ket (Qobj).
        config: SimulationConfig containing time grid.
        c_ops_for_loss: Optional list of jump operators L_k to compute theoretical loss rate.

    Returns:
        dict containing:
            'unnormalized_states': list of unnormalized Qobj kets
            'survival_probability': 1D np.ndarray P_no_jump(t)
            'conditional_states': list of normalized Qobj kets
            'theoretical_loss_rate': 1D np.ndarray sum_k <psi(t)|L_k^dagger L_k|psi(t)>
    """
    import numpy as np

    options = {"normalize_output": False}
    if config.options is not None:
        if isinstance(config.options, dict):
            options.update(config.options)
        else:
            options = config.options

    # Solve non-Hermitian Schrödinger equation with c_ops=[]
    result = mesolve(
        H_eff,
        psi0,
        config.tlist,
        [],
        options=options,
    )

    unnormalized_states = result.states
    survival_prob = np.zeros(len(config.tlist), dtype=float)
    conditional_states = []
    loss_rate = np.zeros(len(config.tlist), dtype=float)

    for idx, psi in enumerate(unnormalized_states):
        # norm squared = <psi|psi>
        p_surv = float(np.real(psi.norm() ** 2))
        survival_prob[idx] = p_surv

        if p_surv > 1e-15:
            psi_cond = (psi / np.sqrt(p_surv)).unit()
        else:
            psi_cond = psi

        conditional_states.append(psi_cond)

        if c_ops_for_loss is not None:
            r_loss = 0.0
            for c in c_ops_for_loss:
                cdc = c.dag() * c
                r_loss += float(np.real(cdc.matrix_element(psi, psi)))
            loss_rate[idx] = r_loss

    return {
        "unnormalized_states": unnormalized_states,
        "survival_probability": survival_prob,
        "conditional_states": conditional_states,
        "theoretical_loss_rate": loss_rate,
    }


def simulate_quantum_trajectories(
    H: Qobj,
    psi0: Qobj,
    c_ops: list[Qobj],
    config: SimulationConfig,
    ntraj: int = 500,
    seed: int = None,
):
    """Simulate stochastic quantum trajectories via the Monte Carlo wavefunction solver (mcsolve).

    Args:
        H: System Hamiltonian (Qobj).
        psi0: Initial pure state ket (Qobj).
        c_ops: List of Lindblad collapse operators.
        config: SimulationConfig containing time grid.
        ntraj: Number of Monte Carlo trajectories (default 500).
        seed: Optional random seed for reproducibility.

    Returns:
        QuTiP McResult object containing individual trajectories and ensemble averages.
    """
    from qutip import mcsolve

    options = {"progress_bar": None}
    if config.options is not None:
        if isinstance(config.options, dict):
            options.update(config.options)
        else:
            options = config.options

    kwargs = {}
    if seed is not None:
        kwargs["seeds"] = seed

    result = mcsolve(
        H,
        psi0,
        config.tlist,
        c_ops,
        ntraj=ntraj,
        options=options,
        **kwargs,
    )
    return result
