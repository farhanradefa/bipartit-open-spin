# Open Bipartite Spin Dynamics: Entanglement, Dissipation, Non-Hermitian Dynamics, and Exceptional Points

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![QuTiP](https://img.shields.io/badge/QuTiP-5.3+-green.svg)](https://qutip.org/)
[![Tests](https://img.shields.io/badge/unit%20tests-64%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

A comprehensive computational and analytical research framework investigating the nonequilibrium physics of two interacting spin - $1/2$ qubits across closed unitary, open dissipative, and effective non-Hermitian dynamical regimes. The repository implements exact Lindblad master-equation solvers, stochastic Monte-Carlo quantum trajectories, non-Hermitian spectral topology mapping, and dynamic parameter-space encircling of second-order exceptional points ($\mathrm{EP2}$).

---

## Table of Contents

1. [Scientific Motivation](#1-scientific-motivation)
2. [Physical Model](#2-physical-model)
3. [Open Quantum System Framework](#3-open-quantum-system-framework)
4. [Effective Non-Hermitian Framework](#4-effective-non-hermitian-framework)
5. [Experimental Roadmap](#5-experimental-roadmap)
6. [Experiment Summaries (01 to 06c)](#6-experiment-summaries-01-to-06c)
7. [Key Physical Findings](#7-key-physical-findings)
8. [Key Results (Figures)](#8-key-results-figures)
9. [Repository Structure](#9-repository-structure)
10. [Installation & Setup](#10-installation--setup)
11. [Running Experiments](#11-running-experiments)
12. [Reproducibility & Validation](#12-reproducibility--validation)
13. [Research Architecture](#13-research-architecture)
14. [Future Directions](#14-future-directions)

---

## 1. Scientific Motivation

Quantum information processing and open quantum systems research rely on understanding how coherent unitary interactions compete with environmental decoherence. In closed systems, bipartite exchange interactions generate maximal quantum entanglement. In realistic physical implementations—such as superconducting circuits, trapped ions, and nitrogen-vacancy centers—spontaneous emission (amplitude damping) and phase noise (pure dephasing) degrade quantum coherence.

However, open systems are not merely noisy versions of closed systems. When dissipation is structured, the interplay between coherent coupling and local relaxation can:
1. **Stabilize Entangled Non-Equilibrium Steady States (NESS):** Counter-intuitively generating persistent entanglement without continuous external control.
2. **Induce Effective Non-Hermitian Dynamics:** Governing conditional no-jump quantum trajectories.
3. **Generate Exceptional Points (EPs):** Spectral singularities where eigenvalues and their corresponding eigenvectors simultaneously coalesce.
4. **Enable Direction-Dependent (Chiral) State Transfer:** Breaking adiabatic symmetry when parameters dynamically encircle an exceptional point.

This repository systematically charts this physical hierarchy from foundational unitary dynamics to full master-equation robustness of topological EP encircling.

---


## 2. Experimental Roadmap

| Experiment | Scientific Objective | Physical Regime | Key Physical Result | Script |
| :--- | :--- | :--- | :--- | :--- |
| **01** | Unitary Bell state dynamics | Closed system, $\gamma=0$ | Coherent oscillations of Concurrence and Negativity; invariant trace and purity. | [`01_unitary_entanglement.py`](experiments/01_unitary_entanglement.py) |
| **01b** | Entanglement generation | Closed system, initial $\|00\rangle$ | $J$-driven generation of maximal entanglement with period $\pi/J$. | [`01b_unitary_entanglement_generation.py`](experiments/01b_unitary_entanglement_generation.py) |
| **02a** | Pure dissipative decay | Open system, $J=0, \gamma>0$ | Monotonic exponential entanglement decay with rate $2\gamma$; asymptotic dark state $\|11\rangle$. | [`02a_pure_dissipative_entanglement.py`](experiments/02a_pure_dissipative_entanglement.py) |
| **02b** | Coherent-dissipative competition | Open system, $J>0, \gamma>0$ | Crossover from underdamped oscillatory decay to overdamped asymptotic state. | [`02b_coherent_dissipative_competition.py`](experiments/02b_coherent_dissipative_competition.py) |
| **02c** | Steady-state phase map | Asymptotic NESS, $(J, \gamma)$ | Discovery of an entangled NESS island with peak Concurrence $C_{\mathrm{ss}} \approx 0.309$. | [`02c_steady_state_entanglement_phase_map.py`](experiments/02c_steady_state_entanglement_phase_map.py) |
| **03a** | Pure dephasing baseline | Open system, $\gamma_1=0, \gamma_\phi>0$ | Population-preserving phase coherence damping; asymptotic entanglement is zero. | [`03a_pure_dephasing_baseline.py`](experiments/03a_pure_dephasing_baseline.py) |
| **03b** | Mixed damping and dephasing | Open system, $\gamma_1>0, \gamma_\phi>0$ | Dephasing suppresses NESS entanglement; smooth crossover boundary mapped. | [`03b_combined_damping_dephasing.py`](experiments/03b_combined_damping_dephasing.py) |
| **03c** | NESS mechanism and scaling | Liouvillian spectral analysis | Analytical proof of NESS entanglement via X-state coherence; universal $(J/\gamma_1)$ scaling. | [`03c_physical_mechanism_scaling.py`](experiments/03c_physical_mechanism_scaling.py) |
| **04** | Quantum trajectories & $H_{\mathrm{eff}}$ | Stochastic unraveling | $N_{\mathrm{traj}} \ge 500$ trajectory average reconstructs master equation; no-jump norm decay. | [`04_effective_nonhermitian_trajectories.py`](experiments/04_effective_nonhermitian_trajectories.py) |
| **05** | Spectral non-Hermitian physics | Odd/even $H_{\mathrm{eff}}$ blocks | Symmetric model has no EPs; asymmetric loss ($\gamma_1 \neq \gamma_2$) produces an $\mathrm{EP2}$ at $J = \|\Delta\gamma\|/4$. | [`05_spectral_nonhermitian_physics.py`](experiments/05_spectral_nonhermitian_physics.py) |
| **06a** | Static EP2 topology | Parameter loops in $(J, \Delta\gamma)$ | $2\pi$ loop swaps eigenpairs ($\lambda_1 \leftrightarrow \lambda_2$); $4\pi$ loop restores identity on Riemann surface. | [`06a_static_ep_topology.py`](experiments/06a_static_ep_topology.py) |
| **06b** | Dynamic EP encircling | Time-dependent no-jump drift | Direction-dependent chiral state transfer; funnels all initial states to single mode with $\chi_{\max} = 0.96$. | [`06b_dynamic_ep_encircling.py`](experiments/06b_dynamic_ep_encircling.py) |
| **06c** | Full open-system robustness | Time-dependent Lindblad | Chirality survives quantum jumps; survival-weighted chirality peaks at $T \approx 19.4\,\omega^{-1}$. | [`06c_ep_encircling_open_system_robustness.py`](experiments/06c_ep_encircling_open_system_robustness.py) |

---


## 3. Key Physical Findings

1. **Dissipative Entanglement Stabilization:** Amplitude damping does not merely destroy entanglement; when balanced against coherent exchange $J$, it continuously purges the separable double-excited component $\rho_{00,00}$, sustaining an entangled NESS ($C_{\mathrm{ss}} \approx 0.309$).
2. **Dephasing-Induced Coherence Destruction:** Pure dephasing specifically attacks off-diagonal density matrix coherences without affecting state populations, extinguishing NESS entanglement via a smooth crossover.
3. **Symmetry Requirement for Exceptional Points:** Symmetric dissipation ($\gamma_1 = \gamma_2$) cannot induce exceptional points in the transverse Ising model. Introducing asymmetric dissipation ($\Delta\gamma = \gamma_1 - \gamma_2 \neq 0$) breaks chiral symmetry and generates an $\mathrm{EP2}$ at $J = |\Delta\gamma|/4$.
4. **Static Topology vs Dynamic Funneling:** While static parameter continuation produces a cyclic state swap ($\lambda_1 \leftrightarrow \lambda_2$) after $2\pi$, real-time dynamical encircling exhibits non-adiabatic chiral mode selection due to differential exponential amplification $\sim \exp\left(\int \mathrm{Im}(\lambda(t)) dt\right)$.
5. **Open-System Survival Trade-Off:** In full Lindblad open systems, the observability of EP chirality is governed by the survival-weighted metric $\chi_{\mathrm{eff}}(T) = \chi(T) \cdot S_{\mathrm{odd}}(T)$, balancing non-adiabatic fidelity at short $T$ against dissipative ground-state depletion at long $T$.

---

## 4. Key Results (Figures)

### Closed Dynamics & Entangled Steady States
| Unitary Entanglement Generation (Exp 01b) | Steady-State Concurrence Phase Map (Exp 02c) |
| :---: | :---: |
| ![Exp 01b Entanglement](results/figures/experiment_01b_entanglement_generation.png) | ![Exp 02c Phase Map](results/figures/experiment_02c_steady_state_concurrence_phase_map.png) |

### NESS Scaling & Quantum Trajectories
| Dimensionless Scaling & Robustness (Exp 03c) | Trajectory Ensemble Convergence (Exp 04) |
| :---: | :---: |
| ![Exp 03c Scaling](results/figures/experiment_03c_dimensionless_scaling.png) | ![Exp 04 Convergence](results/figures/experiment_04_density_matrix_convergence.png) |

### Exceptional Point Physics & Chiral Encircling
| Asymmetric Dissipation EP2 Map (Exp 05) | Real Eigenvalue Riemann Surface (Exp 06a) |
| :---: | :---: |
| ![Exp 05 EP Map](results/figures/experiment_05_asymmetric_loss_ep_map.png) | ![Exp 06a Riemann Surface](results/figures/experiment_06a_riemann_surface_real.png) |

| Conditional Chiral State Transfer (Exp 06b) | Full Open-System Mechanism Flowchart (Exp 06c) |
| :---: | :---: |
| ![Exp 06b Chirality](results/figures/experiment_06b_chirality_vs_period.png) | ![Exp 06c Mechanism](results/figures/experiment_06c_physical_mechanism_diagram.png) |

---

## 5. Repository Structure

```
bipartit-open-spin/
├── pyproject.toml              # Build configuration and dependency specifications
├── README.md                   # Project documentation and research report
├── src/
│   └── bipartit_open_spin/     # Modular quantum dynamics core package
│       ├── __init__.py         # Public API exports
│       ├── config.py           # SimulationConfig and ModelParams dataclasses
│       ├── analysis/           # Entanglement, concurrence, negativity, and spectra
│       │   ├── __init__.py
│       │   ├── entanglement.py # Concurrence, Negativity, X-state decomposition
│       │   └── spectrum.py     # Complex eigenvalues, biorthogonal vectors, EP detection
│       ├── core/               # Hilbert space basis, states, and Pauli operators
│       │   ├── __init__.py
│       │   ├── operators.py    # Pauli matrices, lowering operators, identity
│       │   └── states.py       # Density matrices, Bell states, computational basis
│       ├── dynamics/           # Master equation, non-Hermitian drift, trajectories
│       │   ├── __init__.py
│       │   ├── dissipation.py  # Lindblad collapse operators, jump rates, Liouvillians
│       │   ├── hamiltonian.py  # Bipartite and effective non-Hermitian Hamiltonians
│       │   └── simulation.py   # mesolve, mcsolve, DOP853 ODE integration
│       └── validation/         # Physical state verification and sanity checks
│           ├── __init__.py
│           └── diagnostics.py  # Trace, Hermiticity, and positive semi-definiteness
├── experiments/                # Executable experiment scripts (01 to 06c)
│   ├── 01_unitary_entanglement.py
│   ├── 01b_unitary_entanglement_generation.py
│   ├── 02a_pure_dissipative_entanglement.py
│   ├── 02b_coherent_dissipative_competition.py
│   ├── 02c_steady_state_entanglement_phase_map.py
│   ├── 03a_pure_dephasing_baseline.py
│   ├── 03b_combined_damping_dephasing.py
│   ├── 03c_physical_mechanism_scaling.py
│   ├── 04_effective_nonhermitian_trajectories.py
│   ├── 05_spectral_nonhermitian_physics.py
│   ├── 06a_static_ep_topology.py
│   ├── 06b_dynamic_ep_encircling.py
│   └── 06c_ep_encircling_open_system_robustness.py
├── tests/                      # Automated unit test suite (64 tests)
│   ├── test_core.py
│   ├── test_dynamics.py
│   ├── test_entanglement.py
│   ├── test_hamiltonian.py
│   ├── test_lindblad_encircling.py
│   ├── test_mechanism.py
│   ├── test_nonhermitian.py
│   ├── test_operators.py
│   ├── test_spectrum.py
│   ├── test_states.py
│   ├── test_topology.py
│   └── test_validation.py
└── results/
    ├── data/                   # Compressed NumPy (.npz) numerical archives
    └── figures/                # High-resolution (300 DPI) publication figures
```

---

## 6. Installation & Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic Python environment management.

### Prerequisites
- Python $\ge 3.13$
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/farhanradefa/bipartit-open-spin.git
   cd bipartit-open-spin
   ```

2. **Set up the virtual environment and install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using standard pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

### Core Dependencies
- `qutip >= 5.3.1` (Quantum Toolbox in Python)
- `numpy >= 2.5.2` (Numerical array computing)
- `scipy >= 1.18.1` (High-order ODE solvers and matrix algebra)
- `matplotlib >= 3.11.1` (Publication figure rendering)
- `pandas >= 3.0.5` (Structured parameter scan management)

---

## 7. Running Experiments

All experiment scripts can be executed directly from the project root.

```bash
# Run closed-system unitary dynamics
uv run python experiments/01_unitary_entanglement.py
uv run python experiments/01b_unitary_entanglement_generation.py

# Run open-system dissipative competition & steady-state maps
uv run python experiments/02a_pure_dissipative_entanglement.py
uv run python experiments/02b_coherent_dissipative_competition.py
uv run python experiments/02c_steady_state_entanglement_phase_map.py

# Run dephasing noise & analytical mechanism analyses
uv run python experiments/03a_pure_dephasing_baseline.py
uv run python experiments/03b_combined_damping_dephasing.py
uv run python experiments/03c_physical_mechanism_scaling.py

# Run quantum trajectories & non-Hermitian physics
uv run python experiments/04_effective_nonhermitian_trajectories.py
uv run python experiments/05_spectral_nonhermitian_physics.py

# Run EP topological mapping & dynamic encircling
uv run python experiments/06a_static_ep_topology.py
uv run python experiments/06b_dynamic_ep_encircling.py
uv run python experiments/06c_ep_encircling_open_system_robustness.py
```

Generated datasets are saved to `results/data/*.npz` and high-resolution publication figures are output to `results/figures/*.png`.

---

## 12. Reproducibility & Validation

The codebase enforces strict physical sanity checks and numerical validation across all modules:

### Diagnostic Checks
- **Trace Preservation:** Checks that $|\mathrm{Tr}(\rho) - 1.0| < 10^{-6}$ for all density matrices.
- **Hermiticity:** Verifies $\|\rho - \rho^\dagger\|_\infty < 10^{-6}$.
- **Positive Semi-Definiteness:** Computes eigenvalues via `np.linalg.eigvalsh` to verify $\lambda_{\min}(\rho) \ge -10^{-6}$.
- **Positivity of Decay Rates:** Ensures $\gamma_1(t) \ge 0$ and $\gamma_2(t) \ge 0$ throughout parameter loops.

### Running the Test Suite
The repository includes 64 automated unit tests covering operator algebra, state validation, Liouvillian decomposition, biorthogonal eigenvectors, and EP branch tracking:

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 8. Research Architecture

The logical progression of the research framework is organized as follows:

```
Closed Quantum Dynamics
        ↓
Dissipative Open-System Dynamics
        ↓
Decoherence and Noise Competition
        ↓
Quantum Trajectories
        ↓
Effective Non-Hermitian Hamiltonian
        ↓
Exceptional Points
        ↓
Dynamical EP Encircling and Chiral Transfer
```

---

## 9. Future Directions

The following topics represent prospective avenues for extending this research:

1. **Continuous Repumping & Chiral NESS Engines:** Investigating active incoherent or coherent repumping ($|11\rangle \to |00\rangle$) to continuously replenish the odd sector and stabilize a stationary chiral quantum engine.
2. **Topological Berry & Geometric Phases:** Computing non-Abelian holonomies and geometric phases accumulated during multi-mode non-Hermitian cyclic transport.
3. **Non-Markovian Structured Reservoirs:** Extending the bath coupling beyond Lindblad master equations to examine memory kernels and colored noise effects on EP stability.
4. **Higher-Order Spin Networks ($N \ge 3$):** Scaling to multi-qubit topologies exhibiting third-order ($\mathrm{EP3}$) and higher-order exceptional surfaces.
5. **Experimental Implementation Mapping:** Formulating exact pulse sequences and parameter schedules tailored for circuit QED and trapped-ion quantum simulators.

---

## License

This project is licensed under the MIT License — see the repository files for details.
