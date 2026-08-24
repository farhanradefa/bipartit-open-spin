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

