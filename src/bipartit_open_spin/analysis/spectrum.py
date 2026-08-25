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


def build_topological_odd_hamiltonian(J: float, delta_gamma: float) -> np.ndarray:
    """Construct the centered, traceless 2x2 topological Hamiltonian H_top.

    H_top = J * sigma_x - i * (delta_gamma / 4) * sigma_z
          = [[-i * delta_gamma / 4, J],
             [J,                   +i * delta_gamma / 4]]

    Args:
        J: Coherent coupling strength.
        delta_gamma: Dissipation asymmetry (gamma1 - gamma2).

    Returns:
        2x2 complex ndarray.
    """
    return np.array([
        [-0.25j * delta_gamma, J],
        [J, 0.25j * delta_gamma],
    ], dtype=complex)


def compute_biorthogonal_eigenpairs(H: np.ndarray) -> dict:
    """Compute right and left eigenvectors with biorthogonal normalization.

    H |R_n> = lambda_n |R_n>
    <L_n| H = lambda_n <L_n|  <=>  H^dagger |L_n> = lambda_n^* |L_n>

    Biorthogonal normalization:
        <L_m | R_n> = delta_mn

    Args:
        H: 2x2 or NxN complex ndarray.

    Returns:
        dict containing:
            'eigenvalues': 1D array of eigenvalues
            'right_eigenvectors': NxN array of right eigenvectors (columns)
            'left_eigenvectors': NxN array of left eigenvectors (columns)
            'petermann_factors': 1D array of Petermann factors K_n = <L_n|L_n> <R_n|R_n>
    """
    H = np.asarray(H, dtype=complex)
    n = H.shape[0]

    # Right eigenvectors
    evals_R, R = np.linalg.eig(H)
    # Left eigenvectors via H^dagger
    evals_L, L = np.linalg.eig(H.conj().T)

    # Sort and match left eigenvectors to right eigenvalues
    matched_L = np.zeros_like(L)
    for i, lam_R in enumerate(evals_R):
        match_idx = np.argmin(np.abs(evals_L.conj() - lam_R))
        matched_L[:, i] = L[:, match_idx]

    # Normalize right eigenvectors to unit 2-norm
    for i in range(n):
        norm_r = np.linalg.norm(R[:, i])
        if norm_r > 1e-15:
            R[:, i] = R[:, i] / norm_r

    # Biorthogonal scaling for left eigenvectors: <L_i|R_i> = 1
    petermann_factors = np.zeros(n, dtype=float)
    for i in range(n):
        overlap = np.vdot(matched_L[:, i], R[:, i])
        if np.abs(overlap) > 1e-12:
            matched_L[:, i] = matched_L[:, i] / overlap.conj()
            # Petermann factor K = ||L_i||^2 ||R_i||^2 / |<L_i|R_i>|^2
            petermann_factors[i] = float(
                np.linalg.norm(matched_L[:, i]) ** 2 * np.linalg.norm(R[:, i]) ** 2
            )
        else:
            petermann_factors[i] = float("inf")

    return {
        "eigenvalues": evals_R,
        "right_eigenvectors": R,
        "left_eigenvectors": matched_L,
        "petermann_factors": petermann_factors,
    }


def track_eigenpairs_along_loop(
    hamiltonian_func,
    theta_grid: np.ndarray,
) -> dict:
    """Continuously track complex eigenvalues and eigenvectors along a closed parameter loop.

    Implements continuous analytic branch continuation by minimizing step-to-step
    Euclidean distance between eigenvalues, and aligns eigenvector global phases.

    Args:
        hamiltonian_func: Callable theta -> 2x2 complex ndarray or list/array of 2x2 matrices.
        theta_grid: 1D array of parameter angles (e.g. from 0 to 2*pi or 4*pi).

    Returns:
        dict containing:
            'theta_grid': 1D ndarray of theta values
            'eigenvalues': (N_theta, 2) complex ndarray of tracked eigenvalue branches
            'eigenvectors': (N_theta, 2, 2) complex ndarray of tracked right eigenvectors
            'overlaps': dict with O11, O12, O21, O22 overlaps vs initial state
            'permutation_1_loop': bool (True if branch 1 and branch 2 swapped after theta=2*pi)
            'permutation_error_1_loop': float, |lambda_1(2pi) - lambda_2(0)|
            'return_error_2_loop': float, |lambda_1(4pi) - lambda_1(0)| (if theta_max >= 4*pi)
    """
    n_theta = len(theta_grid)
    tracked_evals = np.zeros((n_theta, 2), dtype=complex)
    tracked_evecs = np.zeros((n_theta, 2, 2), dtype=complex)

    for k, theta in enumerate(theta_grid):
        if callable(hamiltonian_func):
            H_k = hamiltonian_func(theta)
        else:
            H_k = hamiltonian_func[k]

        evals_k, evecs_k = np.linalg.eig(H_k)

        # Normalize eigenvectors
        for i in range(2):
            norm_v = np.linalg.norm(evecs_k[:, i])
            if norm_v > 1e-15:
                evecs_k[:, i] = evecs_k[:, i] / norm_v

        if k == 0:
            # Initial sort by real part
            sort_idx = np.argsort(evals_k.real)
            tracked_evals[0, :] = evals_k[sort_idx]
            tracked_evecs[0, :, 0] = evecs_k[:, sort_idx[0]]
            tracked_evecs[0, :, 1] = evecs_k[:, sort_idx[1]]
        else:
            prev_evals = tracked_evals[k - 1, :]
            prev_evecs = tracked_evecs[k - 1, :, :]

            # Permutation matching: compare direct assignment (0->0, 1->1) vs swap (0->1, 1->0)
            d_direct = np.abs(evals_k[0] - prev_evals[0]) ** 2 + np.abs(evals_k[1] - prev_evals[1]) ** 2
            d_swap = np.abs(evals_k[1] - prev_evals[0]) ** 2 + np.abs(evals_k[0] - prev_evals[1]) ** 2

            if d_swap < d_direct:
                current_evals = np.array([evals_k[1], evals_k[0]])
                current_evecs = np.column_stack((evecs_k[:, 1], evecs_k[:, 0]))
            else:
                current_evals = np.array([evals_k[0], evals_k[1]])
                current_evecs = np.column_stack((evecs_k[:, 0], evecs_k[:, 1]))

            # Phase alignment with previous step: v(k) -> exp(-i*arg(<v(k-1)|v(k)>)) * v(k)
            for i in range(2):
                inner_prod = np.vdot(prev_evecs[:, i], current_evecs[:, i])
                if np.abs(inner_prod) > 1e-15:
                    phase = np.angle(inner_prod)
                    current_evecs[:, i] = current_evecs[:, i] * np.exp(-1j * phase)

            tracked_evals[k, :] = current_evals
            tracked_evecs[k, :, :] = current_evecs

    # Overlaps with initial state |v_1(0)> and |v_2(0)>
    v1_0 = tracked_evecs[0, :, 0]
    v2_0 = tracked_evecs[0, :, 1]

    o11 = np.array([np.abs(np.vdot(v1_0, tracked_evecs[k, :, 0])) for k in range(n_theta)])
    o12 = np.array([np.abs(np.vdot(v1_0, tracked_evecs[k, :, 1])) for k in range(n_theta)])
    o21 = np.array([np.abs(np.vdot(v2_0, tracked_evecs[k, :, 0])) for k in range(n_theta)])
    o22 = np.array([np.abs(np.vdot(v2_0, tracked_evecs[k, :, 1])) for k in range(n_theta)])

    # Check 1-loop index (closest to 2*pi)
    idx_2pi = np.argmin(np.abs(theta_grid - 2.0 * np.pi))
    d_no_swap = np.abs(tracked_evals[idx_2pi, 0] - tracked_evals[0, 0])
    d_swap = np.abs(tracked_evals[idx_2pi, 0] - tracked_evals[0, 1])
    perm_1_loop = bool(d_swap < d_no_swap)

    # Check 2-loop index (closest to 4*pi if present)
    idx_4pi = np.argmin(np.abs(theta_grid - 4.0 * np.pi))
    return_err_2_loop = float(np.abs(tracked_evals[idx_4pi, 0] - tracked_evals[0, 0]))

    return {
        "theta_grid": theta_grid,
        "eigenvalues": tracked_evals,
        "eigenvectors": tracked_evecs,
        "overlaps": {
            "O11": o11,
            "O12": o12,
            "O21": o21,
            "O22": o22,
        },
        "permutation_1_loop": perm_1_loop,
        "permutation_error_1_loop": float(d_swap),
        "return_error_2_loop": return_err_2_loop,
    }


def track_instantaneous_eigenpairs(
    H_func,
    tlist: np.ndarray,
) -> dict:
    """Track instantaneous eigenvalues and normalized eigenvectors along a time-dependent trajectory.

    Args:
        H_func: Callable t -> 2x2 complex ndarray.
        tlist: 1D ndarray of time points.

    Returns:
        dict containing:
            'tlist': 1D ndarray of times
            'eigenvalues': (len(tlist), 2) complex ndarray of tracked instantaneous eigenvalues
            'eigenvectors': (len(tlist), 2, 2) complex ndarray of tracked normalized eigenvectors
            'gaps': 1D float array of |lambda_+(t) - lambda_-(t)|
            'min_gap': float
    """
    res = track_eigenpairs_along_loop(H_func, tlist)
    evals = res["eigenvalues"]
    evecs = res["eigenvectors"]

    gaps = np.array([float(np.abs(evals[k, 0] - evals[k, 1])) for k in range(len(tlist))])
    min_gap = float(np.min(gaps))

    return {
        "tlist": tlist,
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "gaps": gaps,
        "min_gap": min_gap,
    }
