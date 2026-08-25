"""Simulation engine wrapping QuTiP master equation solver."""

import numpy as np
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


def simulate_timedependent_nonhermitian(
    H_func,
    psi0: np.ndarray,
    tlist: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict:
    """Solve the time-dependent non-Hermitian Schrödinger equation:

    d psi / dt = -i H(t) psi(t)

    Computes both unnormalized conditional states |psi(t)> and normalized
    states |psi_tilde(t)>, as well as survival probabilities and subspace populations.

    Args:
        H_func: Callable t -> 2x2 complex ndarray.
        psi0: 1D complex ndarray of shape (2,) representing initial state.
        tlist: 1D ndarray of time points.
        rtol: Relative error tolerance for ODE solver.
        atol: Absolute error tolerance for ODE solver.

    Returns:
        dict containing:
            'tlist': 1D float ndarray
            'raw_states': (len(tlist), 2) complex ndarray
            'normalized_states': (len(tlist), 2) complex ndarray
            'survival_probability': 1D float ndarray of <psi(t)|psi(t)>
            'p01': 1D float ndarray of |<01|psi_tilde(t)>|^2
            'p10': 1D float ndarray of |<10|psi_tilde(t)>|^2
    """
    from scipy.integrate import solve_ivp

    psi0_vec = np.asarray(psi0, dtype=complex).flatten()
    if len(psi0_vec) != 2:
        raise ValueError(f"psi0 must be a 2-element state vector, got length {len(psi0_vec)}")

    # Ensure initial vector is normalized
    norm_0 = np.linalg.norm(psi0_vec)
    if norm_0 > 1e-15:
        psi0_vec = psi0_vec / norm_0

    def rhs(t, y):
        H_t = H_func(t)
        return -1j * (H_t @ y)

    sol = solve_ivp(
        rhs,
        (tlist[0], tlist[-1]),
        psi0_vec,
        t_eval=tlist,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # sol.y has shape (2, len(tlist))
    raw_states = sol.y.T  # (len(tlist), 2)
    n_t = len(tlist)

    survival_prob = np.zeros(n_t, dtype=float)
    normalized_states = np.zeros((n_t, 2), dtype=complex)
    p01 = np.zeros(n_t, dtype=float)
    p10 = np.zeros(n_t, dtype=float)

    for idx in range(n_t):
        vec = raw_states[idx, :]
        norm_sq = float(np.real(np.vdot(vec, vec)))
        survival_prob[idx] = norm_sq

        if norm_sq > 1e-30:
            vec_norm = vec / np.sqrt(norm_sq)
        else:
            vec_norm = vec

        normalized_states[idx, :] = vec_norm
        p01[idx] = float(np.abs(vec_norm[0]) ** 2)
        p10[idx] = float(np.abs(vec_norm[1]) ** 2)

    return {
        "tlist": tlist,
        "raw_states": raw_states,
        "normalized_states": normalized_states,
        "survival_probability": survival_prob,
        "p01": p01,
        "p10": p10,
    }
