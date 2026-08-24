"""Experiment 01b: Unitary Entanglement Generation from a Separable State.

This experiment investigates whether and how the baseline bipartite Hamiltonian:

    H = (omega / 2) * (sigma_z1 + sigma_z2) + J * (sigma_x1 * sigma_x2)

dynamically generates quantum entanglement from an initially separable state:

    |psi(0)> = |00>

in the unitary limit (gamma = 0).

Key Analytical Features:
    The evolution is strictly confined to the even-parity subspace H_even = span{|00>, |11>}:
        |psi(t)> = alpha(t)|00> + beta(t)|11>
    where:
        Omega = sqrt(omega^2 + J^2)
        alpha(t) = cos(Omega * t) - i * (omega / Omega) * sin(Omega * t)
        beta(t)  = -i * (J / Omega) * sin(Omega * t)

    Concurrence:
        C(t) = 2 * |alpha(t) * beta(t)| = 2 * (J / Omega) * |sin(Omega * t)| * sqrt(1 - (J^2 / Omega^2) * sin^2(Omega * t))
    Negativity:
        N(t) = C(t) / 2
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np

from bipartit_open_spin.analysis.entanglement import (
    concurrence_trajectory,
    negativity_trajectory,
)
from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import computational_basis
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.dynamics.simulation import simulate_dynamics
from bipartit_open_spin.validation.diagnostics import validate_state_trajectory


def run_experiment() -> dict:
    """Execute Experiment 01b and generate numerical data and figures."""
    print("=" * 65)
    print("EXPERIMENT 01B: UNITARY ENTANGLEMENT GENERATION FROM |00>")
    print("=" * 65)

    # 1. Define physical model parameters (Unitary limit: gamma = 0.0)
    params = ModelParams(omega=1.0, J=0.5, gamma=0.0)
    config = SimulationConfig(tlist=np.linspace(0.0, 20.0, 400))
    print(f"Model Parameters : omega={params.omega}, J={params.J}, gamma={params.gamma}")
    print(f"Time Grid        : t in [0.0, {config.tlist[-1]:.1f}], {len(config.tlist)} points")

    # 2. Initial state: Separable state |00>
    psi0 = computational_basis(0, 0)
    print(f"Initial State    : |00> (dims: {psi0.dims}, shape: {psi0.shape})")

    # 3. Construct Hamiltonian
    H = build_hamiltonian(params)
    print(f"Hamiltonian      : shape={H.shape}, dims={H.dims}")

    # 4. Collapse operators (empty list in unitary limit)
    c_ops = []

    # 5. Simulate dynamics
    print("\nSimulating unitary time evolution from |00>...")
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
        raise RuntimeError(f"Physical state validation failed on trajectory: {diag}")

    # 7. Calculate entanglement trajectories
    print("\nComputing numerical entanglement metrics...")
    c_traj = concurrence_trajectory(states)
    n_traj = negativity_trajectory(states)

    # 8. Analytical state amplitudes and entanglement curves
    omega, J = params.omega, params.J
    omega_eff = np.sqrt(omega**2 + J**2)
    tlist = config.tlist

    sin_wt = np.sin(omega_eff * tlist)
    cos_wt = np.cos(omega_eff * tlist)

    alpha_t = cos_wt - 1j * (omega / omega_eff) * sin_wt
    beta_t = -1j * (J / omega_eff) * sin_wt

    c_analytical = 2.0 * np.abs(alpha_t * beta_t)
    n_analytical = 0.5 * c_analytical

    # Maximum errors between numerical and analytical
    max_err_c = float(np.max(np.abs(c_traj - c_analytical)))
    max_err_n = float(np.max(np.abs(n_traj - n_analytical)))

    # Metrics summary
    c0, c_final = float(c_traj[0]), float(c_traj[-1])
    n0, n_final = float(n_traj[0]), float(n_traj[-1])
    c_min, c_max = float(np.min(c_traj)), float(np.max(c_traj))
    n_min, n_max = float(np.min(n_traj)), float(np.max(n_traj))

    idx_c_max = int(np.argmax(c_traj))
    t_c_max = float(tlist[idx_c_max])
    idx_n_max = int(np.argmax(n_traj))
    t_n_max = float(tlist[idx_n_max])

    # Theoretical maximum concurrence: C_max = 2*omega*J / (omega^2 + J^2) for J <= omega
    c_max_theory = (2.0 * omega * J) / (omega**2 + J**2)
    n_max_theory = 0.5 * c_max_theory
    t_first_max_theory = np.pi / (2.0 * omega_eff)

    print("\nEntanglement Dynamics Summary:")
    print(f"  Initial Concurrence C(0)        : {c0:.6f} (Expected: 0.000000)")
    print(f"  Initial Negativity N(0)         : {n0:.6f} (Expected: 0.000000)")
    print(f"  Max Numerical Concurrence C_max : {c_max:.6f} at t = {t_c_max:.4f} s")
    print(f"  Theoretical Max Concurrence     : {c_max_theory:.6f} (first peak at t = {t_first_max_theory:.4f} s)")
    print(f"  Max Numerical Negativity N_max  : {n_max:.6f} at t = {t_n_max:.4f} s")
    print(f"  Theoretical Max Negativity      : {n_max_theory:.6f}")
    print(f"  Final Concurrence C(t_f)        : {c_final:.6f}")
    print(f"  Final Negativity N(t_f)         : {n_final:.6f}")
    print(f"  Max Error |C_num - C_ana|       : {max_err_c:.2e}")
    print(f"  Max Error |N_num - N_ana|       : {max_err_n:.2e}")

    # Strict physical assertions
    if not np.isclose(c0, 0.0, atol=1e-10):
        raise AssertionError(f"Initial state |00> must have C(0) = 0, got {c0}")
    if not np.isclose(n0, 0.0, atol=1e-10):
        raise AssertionError(f"Initial state |00> must have N(0) = 0, got {n0}")
    if not np.isclose(c_max, c_max_theory, atol=1e-3):
        raise AssertionError(f"Observed C_max = {c_max:.6f} deviates from theoretical {c_max_theory:.6f}")
    if not np.isclose(n_max, n_max_theory, atol=1e-3):
        raise AssertionError(f"Observed N_max = {n_max:.6f} deviates from theoretical {n_max_theory:.6f}")

    if np.any(c_traj < -1e-12) or np.any(c_traj > 1.0 + 1e-12):
        raise AssertionError(f"Concurrence out of physical bounds [0, 1]: [{c_min}, {c_max}]")
    if np.any(n_traj < -1e-12) or np.any(n_traj > 0.5 + 1e-12):
        raise AssertionError(f"Negativity out of physical bounds [0, 0.5]: [{n_min}, {n_max}]")

    # 9. Plot publication-quality figure
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "experiment_01b_entanglement_generation.png"

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)

    # Numerical curves
    ax.plot(tlist, c_traj, label=r"Concurrence $C(t)$ (Numerical)", color="#1f77b4", linewidth=2.2)
    ax.plot(tlist, n_traj, label=r"Negativity $N(t)$ (Numerical)", color="#ff7f0e", linewidth=2.2)

    # Analytical overlays
    ax.plot(tlist, c_analytical, "--", label=r"Analytical $C(t)$", color="#0b3c5d", linewidth=1.2, alpha=0.85)
    ax.plot(tlist, n_analytical, "--", label=r"Analytical $N(t)$", color="#b35400", linewidth=1.2, alpha=0.85)

    # Theoretical upper ceiling line
    ax.axhline(c_max_theory, color="#1f77b4", linestyle=":", alpha=0.6, label=rf"$C_{{\max}} = \frac{{2\omega J}}{{\omega^2 + J^2}} = {c_max_theory:.2f}$")
    ax.axhline(n_max_theory, color="#ff7f0e", linestyle=":", alpha=0.6, label=rf"$N_{{\max}} = \frac{{\omega J}}{{\omega^2 + J^2}} = {n_max_theory:.2f}$")

    ax.set_title(
        r"Experiment 01b: Unitary Entanglement Generation from Separable State $|\psi(0)\rangle = |00\rangle$" "\n"
        rf"$\omega = {params.omega:.1f},\ J = {params.J:.1f},\ \gamma = {params.gamma:.1f}\quad (\Omega = \sqrt{{\omega^2 + J^2}} \approx {omega_eff:.4f})$",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Time $t$", fontsize=11, fontweight="medium")
    ax.set_ylabel("Entanglement Measure", fontsize=11, fontweight="medium")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(tlist[0], tlist[-1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    # Add text annotation box
    textstr = "\n".join((
        r"$\mathbf{Entanglement\ Generation\ Metrics:}$",
        rf"$C(0) = {c0:.4f},\ N(0) = {n0:.4f}$ (Separable)",
        rf"$C_{{\max}} = {c_max:.4f}\quad (\text{{Theory: }} {c_max_theory:.4f})$",
        rf"$N_{{\max}} = {n_max:.4f}\quad (\text{{Theory: }} {n_max_theory:.4f})$",
        rf"$t(C_{{\max}}) \approx {t_c_max:.4f}\text{{ s}}\quad (\pi / 2\Omega \approx {t_first_max_theory:.4f}\text{{ s}})$",
        rf"$\text{{Max Error}} < {max(max_err_c, max_err_n):.1e}$",
        r"$\mathbf{Note:}\ C_{{\max}} < 1\text{ due to detuning } \omega \neq J$",
    ))
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
    ax.text(0.03, 0.42, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="bottom", bbox=props)

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
        "t_c_max": t_c_max,
        "n_min": n_min,
        "n_max": n_max,
        "t_n_max": t_n_max,
        "c_max_theory": c_max_theory,
        "n_max_theory": n_max_theory,
        "max_err_c": max_err_c,
        "max_err_n": max_err_n,
        "validation": diag,
        "figure_path": str(figure_path),
    }


if __name__ == "__main__":
    run_experiment()
