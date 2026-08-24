"""Lindblad dissipation channels and collapse operators."""

import numpy as np
from qutip import Qobj
from bipartit_open_spin.config import ModelParams
from bipartit_open_spin.core.operators import sigma_m1, sigma_m2


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
