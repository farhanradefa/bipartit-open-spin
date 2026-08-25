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


def simulate_timedependent_lindblad(
    H_func,
    c_ops_func,
    rho0,
    tlist: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> dict:
    """Solve the full time-dependent Lindblad master equation for a 2-qubit system (4x4 density matrix):

    d rho / dt = -i [H(t), rho] + sum_k ( L_k(t) rho L_k(t)^dagger - 0.5 * {L_k(t)^dagger L_k(t), rho} )

    Args:
        H_func: Callable t -> 4x4 complex ndarray or Qobj.
        c_ops_func: Callable t -> list of 4x4 complex ndarrays or Qobjs.
        rho0: Initial density matrix or ket (Qobj or 4x4 / (4,) ndarray).
        tlist: 1D float ndarray of time points.
        rtol: Relative error tolerance for ODE solver.
        atol: Absolute error tolerance for ODE solver.

    Returns:
        dict containing:
            'tlist': 1D float ndarray
            'states': list of Qobj density matrices of dimensions [[2, 2], [2, 2]]
            'p00': 1D float ndarray of <00|rho(t)|00>
            'p01': 1D float ndarray of <01|rho(t)|01>
            'p10': 1D float ndarray of <10|rho(t)|10>
            'p11': 1D float ndarray of <11|rho(t)|11>
            's_odd': 1D float ndarray of P_01(t) + P_10(t)
            'coherence_01_10': 1D complex ndarray of rho_{01,10}(t)
            'abs_coherence': 1D float ndarray of |rho_{01,10}(t)|
            'imbalance_z': 1D float ndarray of (P01 - P10) / (P01 + P10)
    """
    from scipy.integrate import solve_ivp
    from bipartit_open_spin.core.states import to_density_matrix

    # Convert initial state to 4x4 density matrix ndarray
    if isinstance(rho0, Qobj):
        rho0_mat = to_density_matrix(rho0).full()
    else:
        rho0_arr = np.asarray(rho0, dtype=complex)
        if rho0_arr.ndim == 1:
            rho0_arr = rho0_arr / np.linalg.norm(rho0_arr)
            rho0_mat = np.outer(rho0_arr, np.conj(rho0_arr))
        else:
            rho0_mat = rho0_arr

    y0 = rho0_mat.flatten()

    def rhs(t, y):
        rho_t = y.reshape((4, 4))
        H_raw = H_func(t)
        H_t = H_raw.full() if isinstance(H_raw, Qobj) else np.asarray(H_raw, dtype=complex)

        c_ops_raw = c_ops_func(t)
        c_ops_t = [
            op.full() if isinstance(op, Qobj) else np.asarray(op, dtype=complex)
            for op in c_ops_raw
        ]

        # Commutator -i [H, rho]
        drho = -1j * (H_t @ rho_t - rho_t @ H_t)

        # Dissipators sum_k ( L rho L^dagger - 0.5 {L^dagger L, rho} )
        for L in c_ops_t:
            L_dag = L.conj().T
            L_dag_L = L_dag @ L
            drho += L @ rho_t @ L_dag - 0.5 * (L_dag_L @ rho_t + rho_t @ L_dag_L)

        return drho.flatten()

    sol = solve_ivp(
        rhs,
        (tlist[0], tlist[-1]),
        y0,
        t_eval=tlist,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Lindblad ODE integration failed: {sol.message}")

    n_t = len(tlist)
    states = []
    p00 = np.zeros(n_t, dtype=float)
    p01 = np.zeros(n_t, dtype=float)
    p10 = np.zeros(n_t, dtype=float)
    p11 = np.zeros(n_t, dtype=float)
    coherence_01_10 = np.zeros(n_t, dtype=complex)
    abs_coherence = np.zeros(n_t, dtype=float)
    imbalance_z = np.zeros(n_t, dtype=float)
    s_odd = np.zeros(n_t, dtype=float)

    for idx in range(n_t):
        rho_mat = sol.y[:, idx].reshape((4, 4))
        # Enforce exact Hermiticity
        rho_mat = 0.5 * (rho_mat + rho_mat.conj().T)
        q_rho = Qobj(rho_mat, dims=[[2, 2], [2, 2]])
        states.append(q_rho)

        p00[idx] = float(np.real(rho_mat[0, 0]))
        p01[idx] = float(np.real(rho_mat[1, 1]))
        p10[idx] = float(np.real(rho_mat[2, 2]))
        p11[idx] = float(np.real(rho_mat[3, 3]))

        odd_sum = p01[idx] + p10[idx]
        s_odd[idx] = odd_sum

        coh = rho_mat[1, 2]
        coherence_01_10[idx] = coh
        abs_coherence[idx] = float(np.abs(coh))

        if odd_sum > 1e-15:
            imbalance_z[idx] = (p01[idx] - p10[idx]) / odd_sum
        else:
            imbalance_z[idx] = 0.0

    return {
        "tlist": tlist,
        "states": states,
        "p00": p00,
        "p01": p01,
        "p10": p10,
        "p11": p11,
        "s_odd": s_odd,
        "coherence_01_10": coherence_01_10,
        "abs_coherence": abs_coherence,
        "imbalance_z": imbalance_z,
    }
