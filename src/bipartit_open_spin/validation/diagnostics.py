"""Validation and diagnostic tools for bipartite open quantum systems."""

import numpy as np
from qutip import Qobj
from bipartit_open_spin.core.states import to_density_matrix


def check_trace_preservation(rho: Qobj, tol: float = 1e-6) -> bool:
    """Check if the density matrix satisfies Tr(rho) == 1 within numerical tolerance.

    Args:
        rho: Qobj density matrix (or ket).
        tol: Absolute tolerance for trace deviation (must be >= 0).

    Returns:
        True if |Tr(rho) - 1.0| < tol, else False.

    Raises:
        ValueError: If tol is negative.
    """
    if tol < 0:
        raise ValueError(f"Tolerance must be non-negative, got tol={tol}")
    dm = to_density_matrix(rho)
    tr_val = dm.tr()
    # Handle real or complex scalar
    tr_real = float(np.real(tr_val))
    tr_imag = float(np.imag(tr_val))
    return bool(abs(tr_real - 1.0) < tol and abs(tr_imag) < tol)


def check_hermiticity(rho: Qobj, tol: float = 1e-6) -> bool:
    """Check if the density matrix satisfies rho == rho^dagger within numerical tolerance.

    Args:
        rho: Qobj density matrix (or ket).
        tol: Maximum element-wise norm difference tolerance (must be >= 0).

    Returns:
        True if ||rho - rho^dagger||_inf < tol, else False.

    Raises:
        ValueError: If tol is negative.
    """
    if tol < 0:
        raise ValueError(f"Tolerance must be non-negative, got tol={tol}")
    dm = to_density_matrix(rho)
    diff = dm - dm.dag()
    max_diff = float(np.max(np.abs(diff.full())))
    return bool(max_diff < tol)


def check_positivity(rho: Qobj, tol: float = 1e-6) -> bool:
    """Check if the density matrix is positive semi-definite (all eigenvalues >= -tol).

    First verifies Hermiticity within tolerance; returns False if non-Hermitian.
    For Hermitian matrices, uses np.linalg.eigvalsh to compute real eigenvalues.

    Args:
        rho: Qobj density matrix (or ket).
        tol: Tolerance threshold for negative eigenvalue drift and Hermiticity (must be >= 0).

    Returns:
        True if matrix is Hermitian and positive semi-definite within tolerance, else False.

    Raises:
        ValueError: If tol is negative.
    """
    if tol < 0:
        raise ValueError(f"Tolerance must be non-negative, got tol={tol}")
    if not check_hermiticity(rho, tol=tol):
        return False
    dm = to_density_matrix(rho)
    eigenvalues = np.linalg.eigvalsh(dm.full())
    min_eval = float(np.min(eigenvalues))
    return bool(min_eval >= -tol)


def validate_density_matrix(rho: Qobj, tol: float = 1e-6) -> dict[str, bool]:
    """Perform comprehensive physical validation on a bipartite density matrix.

    Checks trace preservation, Hermiticity, and positive semi-definiteness.

    Args:
        rho: Qobj density matrix (or ket).
        tol: Numerical tolerance for diagnostic checks (must be >= 0).

    Returns:
        Dictionary mapping validation criteria names to boolean results,
        including a composite 'valid' key.

    Raises:
        ValueError: If tol is negative.
    """
    if tol < 0:
        raise ValueError(f"Tolerance must be non-negative, got tol={tol}")
    trace_ok = check_trace_preservation(rho, tol=tol)
    hermitian_ok = check_hermiticity(rho, tol=tol)
    positive_ok = check_positivity(rho, tol=tol)

    return {
        "trace_preservation": trace_ok,
        "hermiticity": hermitian_ok,
        "positivity": positive_ok,
        "valid": bool(trace_ok and hermitian_ok and positive_ok),
    }


def validate_state_trajectory(states: list[Qobj], tol: float = 1e-6) -> dict[str, bool]:
    """Validate all states along an open-system time evolution trajectory.

    Args:
        states: Sequence of density matrices along the time trajectory.
        tol: Numerical tolerance for diagnostic checks (must be >= 0).

    Returns:
        Dictionary mapping validation criteria names to boolean results.
        Empty trajectories return valid=False.

    Raises:
        ValueError: If tol is negative.
    """
    if tol < 0:
        raise ValueError(f"Tolerance must be non-negative, got tol={tol}")

    if len(states) == 0:
        return {
            "trace_preservation": False,
            "hermiticity": False,
            "positivity": False,
            "valid": False,
        }

    all_trace = True
    all_hermitian = True
    all_positive = True

    for s in states:
        res = validate_density_matrix(s, tol=tol)
        if not res["trace_preservation"]:
            all_trace = False
        if not res["hermiticity"]:
            all_hermitian = False
        if not res["positivity"]:
            all_positive = False

    return {
        "trace_preservation": all_trace,
        "hermiticity": all_hermitian,
        "positivity": all_positive,
        "valid": bool(all_trace and all_hermitian and all_positive),
    }


def verify_excited_population_decay(
    gamma: float,
    tlist: np.ndarray,
    p_excited: np.ndarray,
    tol: float = 1e-3,
) -> bool:
    """Verify that uncoupled single-spin excited population matches analytical decay e^{-gamma * t}.

    For an uncoupled single-spin dissipation channel with initial excited state population P_e(0),
    the analytical decay is P_e(t) = P_e(0) * exp(-gamma * t).

    Args:
        gamma: Spontaneous emission rate (must be >= 0).
        tlist: Array of time points.
        p_excited: Array of numerically calculated excited state populations P_e(t).
        tol: Maximum absolute deviation allowed between numerical and analytical values (must be >= 0).

    Returns:
        True if max|P_numerical(t) - P_analytical(t)| < tol, else False.

    Raises:
        ValueError: If gamma < 0, tol < 0, or if tlist and p_excited have different lengths.
    """
    if gamma < 0:
        raise ValueError(f"Decay rate gamma must be non-negative, got gamma={gamma}")
    if tol < 0:
        raise ValueError(f"Tolerance must be non-negative, got tol={tol}")

    tlist_arr = np.asarray(tlist)
    p_excited_arr = np.asarray(p_excited)

    if len(tlist_arr) != len(p_excited_arr):
        raise ValueError(
            f"Length mismatch: tlist has length {len(tlist_arr)}, "
            f"but p_excited has length {len(p_excited_arr)}"
        )

    if len(p_excited_arr) == 0 or len(tlist_arr) == 0:
        return False

    p0 = float(p_excited_arr[0])
    analytical = p0 * np.exp(-gamma * tlist_arr)
    max_err = float(np.max(np.abs(p_excited_arr - analytical)))
    return bool(max_err < tol)

