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


def x_state_concurrence_decomposition(rho: Qobj) -> dict[str, float]:
    """Decompose Wootters concurrence for a bipartite two-qubit density matrix assuming X-state form.

    For an X-state in computational basis {|00>, |01>, |10>, |11>}:
        E_even = |rho_00,11| - sqrt(P_01 * P_10)
        E_odd = |rho_01,10| - sqrt(P_00 * P_11)
        C_X = 2 * max(0.0, E_even, E_odd)

    Args:
        rho: Qobj density matrix with dims [[2, 2], [2, 2]] (or ket convertible to dm).

    Returns:
        Dictionary containing populations, coherences, threshold terms, and concurrence.
    """
    dm = to_density_matrix(rho)
    mat = dm.full()

    p00 = float(np.real(mat[0, 0]))
    p01 = float(np.real(mat[1, 1]))
    p10 = float(np.real(mat[2, 2]))
    p11 = float(np.real(mat[3, 3]))

    rho0011 = complex(mat[0, 3])
    rho0110 = complex(mat[1, 2])

    abs_rho0011 = float(np.abs(rho0011))
    abs_rho0110 = float(np.abs(rho0110))

    threshold_even = float(np.sqrt(max(0.0, p01 * p10)))
    threshold_odd = float(np.sqrt(max(0.0, p00 * p11)))

    e_even = abs_rho0011 - threshold_even
    e_odd = abs_rho0110 - threshold_odd

    c_x = 2.0 * max(0.0, e_even, e_odd)
    c_standard = concurrence(dm)

    return {
        "p00": p00,
        "p01": p01,
        "p10": p10,
        "p11": p11,
        "rho0011": rho0011,
        "rho0110": rho0110,
        "abs_rho0011": abs_rho0011,
        "abs_rho0110": abs_rho0110,
        "re_rho0011": float(np.real(rho0011)),
        "im_rho0011": float(np.imag(rho0011)),
        "re_rho0110": float(np.real(rho0110)),
        "im_rho0110": float(np.imag(rho0110)),
        "threshold_even": threshold_even,
        "threshold_odd": threshold_odd,
        "e_even": e_even,
        "e_odd": e_odd,
        "c_x": c_x,
        "concurrence": c_standard,
    }

