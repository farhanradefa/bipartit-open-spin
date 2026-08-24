"""Configuration dataclasses for model parameters and simulation settings."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class ModelParams:
    """Physical parameters for the bipartite open spin system.

    Attributes:
        omega: Transition frequency / single-qubit energy splitting (hbar = 1).
        J: Spin-spin coupling strength along the x-direction.
        gamma: Dissipation / spontaneous emission rate.
    """

    omega: float = 1.0
    J: float = 0.5
    gamma: float = 0.1


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical configuration for time evolution.

    Attributes:
        tlist: Array of time points for the simulation.
        options: Optional dictionary of solver options passed to QuTiP mesolve.
    """

    tlist: np.ndarray = field(default_factory=lambda: np.linspace(0, 20, 400))
    options: dict | None = None
