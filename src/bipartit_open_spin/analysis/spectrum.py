"""Spectral analysis tools for effective non-Hermitian Hamiltonians and exceptional points."""

import numpy as np
from qutip import Qobj


def parity_block_decomposition(H_eff: Qobj) -> tuple[np.ndarray, np.ndarray]:
    """Decompose H_eff into even and odd parity blocks in computational basis.

    Basis ordering:
        |00> -> index 0 (even)
        |01> -> index 1 (odd)
        |10> -> index 2 (odd)
        |11> -> index 3 (even)

    Returns:
        H_even: 2x2 complex ndarray on span{|00>, |11>}
        H_odd: 2x2 complex ndarray on span{|01>, |10>}
    """
    mat = H_eff.full() if isinstance(H_eff, Qobj) else np.asarray(H_eff)
    even_idx = [0, 3]
    odd_idx = [1, 2]

    H_even = mat[np.ix_(even_idx, even_idx)]
    H_odd = mat[np.ix_(odd_idx, odd_idx)]
    return H_even, H_odd


def analytical_eigenvalues(
    omega: float,
    J: float,
    gamma1: float,
    gamma2: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute analytical eigenvalues for even and odd parity sectors of H_eff.

    Args:
        omega: Qubit transition frequency.
        J: Spin-spin coupling strength.
        gamma1: Dissipation rate for qubit 1.
        gamma2: Dissipation rate for qubit 2 (defaults to gamma1 if None).

    Returns:
        evals_even: 1D array of 2 complex eigenvalues for the even sector.
        evals_odd: 1D array of 2 complex eigenvalues for the odd sector.
    """
    if gamma2 is None:
        gamma2 = gamma1

    gamma_bar = (gamma1 + gamma2) / 2.0
    delta_gamma = gamma1 - gamma2

    # Even sector eigenvalues:
    # lambda_even,pm = -i*gamma_bar/2 pm sqrt((omega - i*gamma_bar/2)^2 + J^2)
    disc_even = (omega - 0.5j * gamma_bar) ** 2 + J ** 2
    sqrt_even = np.sqrt(disc_even)
    evals_even = np.array([
        -0.5j * gamma_bar + sqrt_even,
        -0.5j * gamma_bar - sqrt_even,
    ], dtype=complex)

    # Odd sector eigenvalues:
    # lambda_odd,pm = -i*gamma_bar/2 pm sqrt(J^2 - (delta_gamma / 4)^2)
    disc_odd = J ** 2 - (delta_gamma / 4.0) ** 2 + 0j
    sqrt_odd = np.sqrt(disc_odd)
    evals_odd = np.array([
        -0.5j * gamma_bar + sqrt_odd,
        -0.5j * gamma_bar - sqrt_odd,
    ], dtype=complex)

    return evals_even, evals_odd


def compute_complex_spectrum(H_eff: Qobj) -> dict:
    """Compute sorted complex eigenvalues, eigenvectors, gaps, and condition numbers.

    Args:
        H_eff: Effective non-Hermitian Hamiltonian (Qobj or 4x4 array).

    Returns:
        dict containing:
            'eigenvalues': 1D np.ndarray of 4 complex eigenvalues (sorted by real part then imag)
            'eigenvectors': 4x4 np.ndarray where column j is eigenvector j
            'min_gap': float, minimum pairwise Euclidean distance between eigenvalues in complex plane
            'pairwise_gaps': 4x4 np.ndarray of distances |lambda_i - lambda_j|
            'eigenvector_overlaps': 4x4 np.ndarray of inner products |<v_i|v_j>|
            'eigenvector_cond': float, condition number of eigenvector matrix V
            'matrix_cond': float, condition number of H_eff
    """
    mat = H_eff.full() if isinstance(H_eff, Qobj) else np.asarray(H_eff)

    evals, evecs = np.linalg.eig(mat)

    # Sort eigenvalues systematically (by real part, then imaginary part)
    sort_idx = np.lexsort((evals.imag, evals.real))
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    # Normalize eigenvectors
    for i in range(evecs.shape[1]):
        norm_v = np.linalg.norm(evecs[:, i])
        if norm_v > 1e-15:
            evecs[:, i] = evecs[:, i] / norm_v

    # Pairwise gaps and overlaps
    n = len(evals)
    pairwise_gaps = np.zeros((n, n), dtype=float)
    eigenvector_overlaps = np.zeros((n, n), dtype=float)

    min_gap = float("inf")
    for i in range(n):
        for j in range(n):
            gap_ij = float(np.abs(evals[i] - evals[j]))
            pairwise_gaps[i, j] = gap_ij
            if i != j and gap_ij < min_gap:
                min_gap = gap_ij

            overlap_ij = float(np.abs(np.vdot(evecs[:, i], evecs[:, j])))
            eigenvector_overlaps[i, j] = overlap_ij

    # Eigenvector matrix condition number (diverges near exceptional points)
    try:
        evec_cond = float(np.linalg.cond(evecs))
    except Exception:
        evec_cond = float("inf")

    # Matrix condition number
    try:
        mat_cond = float(np.linalg.cond(mat))
    except Exception:
        mat_cond = float("inf")

    return {
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "min_gap": min_gap,
        "pairwise_gaps": pairwise_gaps,
        "eigenvector_overlaps": eigenvector_overlaps,
        "eigenvector_cond": evec_cond,
        "matrix_cond": mat_cond,
    }


def detect_exceptional_point(
    H_eff: Qobj,
    tol_gap: float = 1e-4,
    tol_cond: float = 100.0,
) -> dict:
    """Rigorous diagnostic test for non-Hermitian Exceptional Points (EPs).

    Tests:
    1. Eigenvalue coalescence: min |lambda_i - lambda_j| < tol_gap.
    2. Eigenvector coalescence: max_{i != j} |<v_i|v_j>| > 0.99.
    3. Eigenvector condition number: cond(V) > tol_cond.
    4. Jordan defect: rank(H_eff - lambda_EP I) < 4 - 1.

    Returns:
        dict with boolean 'is_exceptional_point', candidate pairs, and diagnostic metrics.
    """
    spec = compute_complex_spectrum(H_eff)
    mat = H_eff.full() if isinstance(H_eff, Qobj) else np.asarray(H_eff)
    evals = spec["eigenvalues"]
    overlaps = spec["eigenvector_overlaps"]
    evec_cond = spec["eigenvector_cond"]
    min_gap = spec["min_gap"]

    is_ep = False
    coalesced_pair = None
    rank_defect = False

    if min_gap < tol_gap:
        # Check which pair coalesced
        for i in range(len(evals)):
            for j in range(i + 1, len(evals)):
                if np.abs(evals[i] - evals[j]) < tol_gap:
                    # Check eigenvector overlap and conditioning
                    if overlaps[i, j] > 0.95 or evec_cond > tol_cond:
                        # Rank test: check geometric multiplicity of candidate eigenvalue
                        lambda_cand = 0.5 * (evals[i] + evals[j])
                        shifted_mat = mat - lambda_cand * np.eye(mat.shape[0], dtype=complex)
                        # SVD singular values to evaluate numerical rank
                        s = np.linalg.svd(shifted_mat, compute_uv=False)
                        nullity = int(np.sum(s < tol_gap * 10))
                        # For an EP2 (algebraic mult 2), geometric mult (nullity) must be 1 (< algebraic mult 2)
                        if nullity < 2:
                            is_ep = True
                            rank_defect = True
                            coalesced_pair = (i, j)

    return {
        "is_exceptional_point": is_ep,
        "coalesced_pair": coalesced_pair,
        "min_gap": min_gap,
        "max_overlap": float(np.max(overlaps - np.eye(len(evals)))),
        "eigenvector_cond": evec_cond,
        "rank_defect": rank_defect,
    }
