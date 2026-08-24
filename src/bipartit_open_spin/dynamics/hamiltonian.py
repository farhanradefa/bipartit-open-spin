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
