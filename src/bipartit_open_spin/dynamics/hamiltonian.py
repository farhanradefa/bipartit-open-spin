"""Hamiltonian construction for the bipartite spin system."""

from qutip import Qobj
from bipartit_open_spin.config import ModelParams
from bipartit_open_spin.core.operators import sigma_x1, sigma_x2, sigma_z1, sigma_z2


def build_hamiltonian(params: ModelParams) -> Qobj:
    """Construct the baseline bipartite Hamiltonian.

    H = (omega / 2) * (sigma_z1 + sigma_z2) + J * (sigma_x1 * sigma_x2)

    Args:
        params: ModelParams instance containing omega and J.

    Returns:
        Qobj representing the Hamiltonian with dims [[2, 2], [2, 2]].
    """
    sz1 = sigma_z1()
    sz2 = sigma_z2()
    sx1 = sigma_x1()
    sx2 = sigma_x2()

    H = 0.5 * params.omega * (sz1 + sz2) + params.J * (sx1 * sx2)
    return H


def build_effective_hamiltonian(params: ModelParams, c_ops: list[Qobj] = None) -> Qobj:
    """Construct the effective non-Hermitian Hamiltonian H_eff = H - (i/2) sum_k L_k^dagger L_k.

    Args:
        params: ModelParams instance containing omega, J, and gamma.
        c_ops: Optional list of collapse operators. If None, builds default amplitude damping operators.

    Returns:
        Qobj representing the non-Hermitian effective Hamiltonian with dims [[2, 2], [2, 2]].
    """
    import numpy as np
    from bipartit_open_spin.dynamics.dissipation import build_collapse_operators

    H = build_hamiltonian(params)
    if c_ops is None:
        c_ops = build_collapse_operators(params)

    decay_term = Qobj(np.zeros((4, 4), dtype=complex), dims=[[2, 2], [2, 2]])
    for c in c_ops:
        decay_term += c.dag() * c

    H_eff = H - 0.5j * decay_term
    return H_eff
