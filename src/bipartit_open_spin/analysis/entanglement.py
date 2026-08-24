"""Entanglement and quantum correlation measures for bipartite spin density matrices."""

import numpy as np
from qutip import Qobj, concurrence as qutip_concurrence, partial_transpose
from bipartit_open_spin.core.states import to_density_matrix


def concurrence(rho: Qobj) -> float:
    """Compute Wootters concurrence C(rho) for a bipartite two-qubit density matrix.

    Args:
        rho: Qobj density matrix with dims [[2, 2], [2, 2]] (or ket convertible to dm).

    Returns:
        float concurrence in [0, 1].
    """
    dm = to_density_matrix(rho)
    val = float(qutip_concurrence(dm))
    return max(0.0, min(1.0, val))


def negativity(rho: Qobj, tol: float = 1e-12) -> float:
    """Compute Peres-Horodecki negativity N(rho) for a bipartite two-qubit density matrix.

    N(rho) = (||rho^{T_A}||_1 - 1) / 2 = sum_{lambda_i < -tol} |lambda_i|

    Args:
        rho: Qobj density matrix with dims [[2, 2], [2, 2]] (or ket convertible to dm).
        tol: Numerical tolerance threshold for negative eigenvalues (default 1e-12).

    Returns:
        float negativity in [0, 0.5].

    Raises:
        ValueError: If tol is negative.
    """
    if tol < 0:
        raise ValueError(f"Tolerance must be non-negative, got tol={tol}")
    dm = to_density_matrix(rho)
    # Partial transpose with respect to subsystem A
    rho_pt = partial_transpose(dm, [1, 0])
    eigenvalues = rho_pt.eigenenergies()
    negative_evals = eigenvalues[eigenvalues < -tol]
    neg = float(np.sum(np.abs(negative_evals)))
    return max(0.0, neg)


def concurrence_trajectory(states: list[Qobj]) -> np.ndarray:
    """Compute concurrence along a trajectory of quantum states.

    Args:
        states: Sequence of Qobj density matrices or kets.

    Returns:
        1D numpy array of concurrence values.
    """
    return np.array([concurrence(state) for state in states], dtype=float)


def negativity_trajectory(states: list[Qobj], tol: float = 1e-12) -> np.ndarray:
    """Compute negativity along a trajectory of quantum states.

    Args:
        states: Sequence of Qobj density matrices or kets.
        tol: Numerical tolerance threshold for negative eigenvalues (default 1e-12).

    Returns:
        1D numpy array of negativity values.

    Raises:
        ValueError: If tol is negative.
    """
    if tol < 0:
        raise ValueError(f"Tolerance must be non-negative, got tol={tol}")
    return np.array([negativity(state, tol=tol) for state in states], dtype=float)

