"""Lindblad dissipation channels and collapse operators."""

import numpy as np
from qutip import Qobj
from bipartit_open_spin.config import ModelParams
from bipartit_open_spin.core.operators import sigma_m1, sigma_m2, sigma_z1, sigma_z2


def build_collapse_operators(params: ModelParams) -> list[Qobj]:
    """Construct baseline Lindblad collapse operators for spontaneous emission.

    C_1 = sqrt(gamma) * sigma_m1
    C_2 = sqrt(gamma) * sigma_m2

    Args:
        params: ModelParams instance containing gamma.

    Returns:
        List of Qobj collapse operators with dims [[2, 2], [2, 2]].
    """
    if params.gamma < 0:
        raise ValueError(f"Dissipation rate gamma must be non-negative, got {params.gamma}")

    rate_factor = np.sqrt(params.gamma)
    c_ops = [
        rate_factor * sigma_m1(),
        rate_factor * sigma_m2(),
    ]
    return c_ops


def build_dephasing_collapse_operators(gamma_phi: float) -> list[Qobj]:
    """Construct local pure-dephasing Lindblad collapse operators.

    L_phi^(1) = sqrt(gamma_phi / 2) * sigma_z1
    L_phi^(2) = sqrt(gamma_phi / 2) * sigma_z2

    Convention:
        With rate factor sqrt(gamma_phi / 2), the single-qubit off-diagonal
        coherence decays as rho_01(t) = rho_01(0) * exp(-gamma_phi * t).

    Args:
        gamma_phi: Pure dephasing rate (must be non-negative).

    Returns:
        List of Qobj collapse operators with dims [[2, 2], [2, 2]].
    """
    if gamma_phi < 0:
        raise ValueError(f"Dephasing rate gamma_phi must be non-negative, got {gamma_phi}")

    rate_factor = np.sqrt(gamma_phi / 2.0)
    c_ops = [
        rate_factor * sigma_z1(),
        rate_factor * sigma_z2(),
    ]
    return c_ops


def liouvillian_action(H: Qobj, c_ops: list[Qobj], rho: Qobj) -> Qobj:
    """Evaluate the action of the Lindblad generator on a density matrix rho: L(rho).

    L(rho) = -i [H, rho] + sum_k (C_k rho C_k^dagger - 1/2 {C_k^dagger C_k, rho})

    Args:
        H: Hamiltonian Qobj (or None/zero operator).
        c_ops: List of collapse operators Qobj.
        rho: Density matrix Qobj.

    Returns:
        Qobj representing d rho / dt with dims [[2, 2], [2, 2]].
    """
    from bipartit_open_spin.core.states import to_density_matrix

    dm = to_density_matrix(rho)

    if H is not None and H.norm() > 0:
        comm = -1j * (H * dm - dm * H)
    else:
        comm = Qobj(np.zeros((4, 4), dtype=complex), dims=[[2, 2], [2, 2]])

    diss = Qobj(np.zeros((4, 4), dtype=complex), dims=[[2, 2], [2, 2]])
    for c in c_ops:
        cdc = c.dag() * c
        diss += c * dm * c.dag() - 0.5 * (cdc * dm + dm * cdc)

    return comm + diss


def liouvillian_superoperator_decomposition(
    H: Qobj,
    c_ops_amp: list[Qobj],
    c_ops_phi: list[Qobj],
    rho: Qobj,
) -> dict[str, Qobj]:
    """Decompose d rho / dt into Hamiltonian, amplitude damping, and pure dephasing contributions.

    d rho / dt = L_H(rho) + L_amp(rho) + L_phi(rho)

    Args:
        H: Hamiltonian Qobj.
        c_ops_amp: List of amplitude damping collapse operators.
        c_ops_phi: List of pure dephasing collapse operators.
        rho: Density matrix Qobj.

    Returns:
        dict with keys 'L_H', 'L_amp', 'L_phi', 'L_tot'.
    """
    from bipartit_open_spin.core.states import to_density_matrix

    dm = to_density_matrix(rho)

    L_H = -1j * (H * dm - dm * H) if H is not None else Qobj(np.zeros((4, 4), dtype=complex), dims=[[2, 2], [2, 2]])

    L_amp = Qobj(np.zeros((4, 4), dtype=complex), dims=[[2, 2], [2, 2]])
    for c in c_ops_amp:
        cdc = c.dag() * c
        L_amp += c * dm * c.dag() - 0.5 * (cdc * dm + dm * cdc)

    L_phi = Qobj(np.zeros((4, 4), dtype=complex), dims=[[2, 2], [2, 2]])
    for c in c_ops_phi:
        cdc = c.dag() * c
        L_phi += c * dm * c.dag() - 0.5 * (cdc * dm + dm * cdc)

    L_tot = L_H + L_amp + L_phi

    return {
        "L_H": L_H,
        "L_amp": L_amp,
        "L_phi": L_phi,
        "L_tot": L_tot,
    }


def dissipative_jump_rates(c_ops: list[Qobj], rho: Qobj) -> list[float]:
    """Compute the expectation values of dissipative jump rates R_k = Tr[C_k^dagger C_k rho].

    Args:
        c_ops: List of collapse operators Qobj.
        rho: Density matrix Qobj.

    Returns:
        List of float jump rates R_k.
    """
    from bipartit_open_spin.core.states import to_density_matrix

    dm = to_density_matrix(rho)
    rates = []
    for c in c_ops:
        cdc = c.dag() * c
        r_val = float(np.real((cdc * dm).tr()))
        rates.append(r_val)
    return rates

