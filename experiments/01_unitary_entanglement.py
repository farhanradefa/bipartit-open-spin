"""Experiment 01: Unitary Entanglement Dynamics.

This experiment investigates the coherent time evolution of two-qubit entanglement
under the baseline Hamiltonian in the absence of dissipation (unitary limit, gamma = 0):

    H = (omega / 2) * (sigma_z1 + sigma_z2) + J * (sigma_x1 * sigma_x2)

Initial State:
    |Phi+> = (|00> + |11>) / sqrt(2)

Entanglement Metrics:
    1. Wootters Concurrence C(t) in [0, 1]
    2. Peres-Horodecki Negativity N(t) in [0, 0.5]
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless reproducibility
import matplotlib.pyplot as plt
import numpy as np

from bipartit_open_spin.analysis.entanglement import (
    concurrence_trajectory,
    negativity_trajectory,
)
from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import bell_phi_plus
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.dynamics.simulation import simulate_dynamics
from bipartit_open_spin.validation.diagnostics import validate_state_trajectory


def run_experiment() -> dict:
    """Execute Experiment 01 and generate output figures."""
    print("=" * 60)
    print("EXPERIMENT 01: UNITARY ENTANGLEMENT DYNAMICS")
    print("=" * 60)

    # 1. Define physical model parameters (Unitary limit: gamma = 0.0)
    params = ModelParams(omega=1.0, J=0.5, gamma=0.0)
    config = SimulationConfig(tlist=np.linspace(0, 20, 400))
    print(f"Model Parameters: omega={params.omega}, J={params.J}, gamma={params.gamma}")
    print(f"Time Grid: t in [0.0, {config.tlist[-1]:.1f}], {len(config.tlist)} points")

    # 2. Initial state: Bell state |Phi+>
    psi0 = bell_phi_plus()
    print(f"Initial State: |Phi+> (dims: {psi0.dims})")

    # 3. Construct Hamiltonian
    H = build_hamiltonian(params)
    print(f"Hamiltonian constructed (shape: {H.shape}, dims: {H.dims})")

    # 4. Collapse operators (empty list in unitary limit)
    c_ops = []

    # 5. Simulate dynamics
    print("\nSimulating unitary time evolution...")
    states = simulate_dynamics(H, psi0, c_ops, config)
    print(f"Simulation completed. Generated {len(states)} state density matrices.")

    # 6. Physical validation of state trajectory
    print("\nRunning diagnostic validation on state trajectory...")
    diag = validate_state_trajectory(states, tol=1e-6)
    print(f"  - Trace preservation : {'PASSED' if diag['trace_preservation'] else 'FAILED'}")
    print(f"  - Hermiticity        : {'PASSED' if diag['hermiticity'] else 'FAILED'}")
    print(f"  - Positivity         : {'PASSED' if diag['positivity'] else 'FAILED'}")
    print(f"  - Composite validity : {'PASSED' if diag['valid'] else 'FAILED'}")

    if not diag["valid"]:
        raise RuntimeError(
            f"Physical state validation failed for trajectory: {diag}"
        )

    # 7. Calculate entanglement trajectories
    print("\nComputing entanglement metrics...")
    c_traj = concurrence_trajectory(states)
    n_traj = negativity_trajectory(states)

    # 8. Physical sanity checks
    c0, c_final = float(c_traj[0]), float(c_traj[-1])
    n0, n_final = float(n_traj[0]), float(n_traj[-1])
    c_min, c_max = float(np.min(c_traj)), float(np.max(c_traj))
    n_min, n_max = float(np.min(n_traj)), float(np.max(n_traj))

    print(f"\nEntanglement Metrics Summary:")
    print(f"  Initial Concurrence C(0) : {c0:.6f} (Expected: ~1.000000)")
    print(f"  Initial Negativity N(0)  : {n0:.6f} (Expected: ~0.500000)")
    print(f"  Final Concurrence C(t_f) : {c_final:.6f}")
    print(f"  Final Negativity N(t_f)  : {n_final:.6f}")
    print(f"  Concurrence Range [min, max] : [{c_min:.6f}, {c_max:.6f}]")
    print(f"  Negativity Range [min, max]  : [{n_min:.6f}, {n_max:.6f}]")

    # Strict physical boundary assertions
    if not np.isclose(c0, 1.0, atol=1e-5):
        raise AssertionError(f"Initial concurrence C(0) = {c0} deviates from 1.0")
    if not np.isclose(n0, 0.5, atol=1e-5):
        raise AssertionError(f"Initial negativity N(0) = {n0} deviates from 0.5")

    if np.any(c_traj < -1e-12) or np.any(c_traj > 1.0 + 1e-12):
        raise AssertionError(f"Concurrence out of physical bounds [0, 1]: min={c_min}, max={c_max}")
    if np.any(n_traj < -1e-12) or np.any(n_traj > 0.5 + 1e-12):
        raise AssertionError(f"Negativity out of physical bounds [0, 0.5]: min={n_min}, max={n_max}")

    # 9. Plot publication-quality figure
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "experiment_01_entanglement.png"

    # Analytical benchmark curve
    omega, J = params.omega, params.J
    omega_eff = np.sqrt(omega**2 + J**2)
    sin_sq = np.sin(omega_eff * config.tlist) ** 2
    c_analytical = np.sqrt(np.maximum(0.0, 1.0 - ((2.0 * omega * J) / (omega**2 + J**2))**2 * (sin_sq**2)))
    n_analytical = 0.5 * c_analytical

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Plot numerical curves
    ax.plot(config.tlist, c_traj, label=r"Concurrence $C(t)$ (Numerical)", color="#1f77b4", linewidth=2.2)
    ax.plot(config.tlist, n_traj, label=r"Negativity $N(t)$ (Numerical)", color="#ff7f0e", linewidth=2.2)

    # Overlay analytical solution (dashed) to prove exact dynamical agreement
    ax.plot(config.tlist, c_analytical, "--", label=r"Analytical $C(t)$", color="#0b3c5d", linewidth=1.2, alpha=0.8)
    ax.plot(config.tlist, n_analytical, "--", label=r"Analytical $N(t)$", color="#b35400", linewidth=1.2, alpha=0.8)

    ax.set_title(
        r"Experiment 01: Unitary Entanglement Dynamics" "\n"
        rf"$\omega = {params.omega:.1f},\ J = {params.J:.1f},\ \gamma = {params.gamma:.1f}\quad|\psi(0)\rangle = |\Phi^+\rangle$",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Time $t$", fontsize=11, fontweight="medium")
    ax.set_ylabel("Entanglement Measure", fontsize=11, fontweight="medium")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(config.tlist[0], config.tlist[-1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    # Add annotation box with numerical extrema
    textstr = "\n".join((
        r"$\mathbf{Extrema:}$",
        rf"$C_{{\max}} = {c_max:.4f},\ C_{{\min}} = {c_min:.4f}$",
        rf"$N_{{\max}} = {n_max:.4f},\ N_{{\min}} = {n_min:.4f}$",
        rf"$\Omega = \sqrt{{\omega^2 + J^2}} \approx {omega_eff:.4f}$",
    ))
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
    ax.text(0.03, 0.08, textstr, transform=ax.transAxes, fontsize=9.5, verticalalignment="bottom", bbox=props)

    plt.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"\nFigure saved successfully to: {figure_path}")

    return {
        "params": params,
        "c0": c0,
        "n0": n0,
        "c_final": c_final,
        "n_final": n_final,
        "c_min": c_min,
        "c_max": c_max,
        "n_min": n_min,
        "n_max": n_max,
        "validation": diag,
        "figure_path": str(figure_path),
    }


if __name__ == "__main__":
    run_experiment()
