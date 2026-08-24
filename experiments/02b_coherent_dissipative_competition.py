"""Experiment 02b: Coherent-Dissipative Competition in Bipartite Spin Entanglement.

This experiment investigates the competition between coherent entanglement generation
driven by the non-local exchange Hamiltonian:

    H = (omega / 2) * (sigma_z1 + sigma_z2) + J * (sigma_x1 * sigma_x2)

and local spontaneous-emission dissipation:

    C_1 = sqrt(gamma) * (sigma_- tensor I)
    C_2 = sqrt(gamma) * (I tensor sigma_-)

from an initially separable state:

    |psi(0)> = |00>   (C(0) = 0, N(0) = 0)

Physical Regimes Characterized via Dimensionless Ratio r = gamma / J:
    1. Coherence-dominated regime (r << 1, e.g. gamma = 0.05, r = 0.1)
    2. Intermediate competition regime (r ~ 1, e.g. gamma = 0.5, r = 1.0)
    3. Dissipation-dominated / overdamped regime (r >> 1, e.g. gamma = 2.0, r = 4.0)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend for reproducibility
import matplotlib.pyplot as plt
import numpy as np

from bipartit_open_spin.analysis.entanglement import (
    concurrence_trajectory,
    negativity_trajectory,
)
from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import computational_basis
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.dynamics.simulation import simulate_dynamics
from bipartit_open_spin.validation.diagnostics import validate_state_trajectory


def run_experiment() -> dict:
    """Execute Experiment 02b sweep, validate trajectories, and generate all figures."""
    print("=" * 75)
    print("EXPERIMENT 02B: COHERENT-DISSIPATIVE ENTANGLEMENT COMPETITION")
    print("=" * 75)

    # 1. Model & Simulation Parameters
    omega = 1.0
    J = 0.5
    gamma_list = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]
    tlist = np.linspace(0.0, 20.0, 800)
    config = SimulationConfig(tlist=tlist)
    psi0 = computational_basis(0, 0)

    print(f"\nBaseline Model Parameters: omega = {omega:.1f}, J = {J:.1f}")
    print(f"Initial State            : |00> (Separable, C(0) = 0, N(0) = 0)")
    print(f"Time Grid                : t in [0.0, {tlist[-1]:.1f}], {len(tlist)} points")
    print(f"Dissipation Sweep (gamma): {gamma_list}")

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 2. Execute Simulation Sweep
    print("\n--- Running Parameter Sweep & State Validation ---")
    for gamma in gamma_list:
        params = ModelParams(omega=omega, J=J, gamma=gamma)
        H = build_hamiltonian(params)
        c_ops = build_collapse_operators(params) if gamma > 0 else []

        states = simulate_dynamics(H, psi0, c_ops, config)

        # State trajectory validation
        diag = validate_state_trajectory(states, tol=1e-6)
        if not diag["valid"]:
            raise RuntimeError(f"Physical state validation failed for gamma = {gamma}: {diag}")

        # Entanglement measures
        c_traj = concurrence_trajectory(states)
        n_traj = negativity_trajectory(states)

        # Populations
        p00 = np.array([abs(s.full()[0, 0]) for s in states])
        p01 = np.array([abs(s.full()[1, 1]) for s in states])
        p10 = np.array([abs(s.full()[2, 2]) for s in states])
        p11 = np.array([abs(s.full()[3, 3]) for s in states])

        idx_peak = int(np.argmax(c_traj))
        c_max = float(c_traj[idx_peak])
        t_peak = float(tlist[idx_peak])
        n_max = float(n_traj[int(np.argmax(n_traj))])
        c_final = float(c_traj[-1])
        n_final = float(n_traj[-1])
        r = gamma / J

        results[gamma] = {
            "params": params,
            "r": r,
            "states": states,
            "c_traj": c_traj,
            "n_traj": n_traj,
            "p00": p00,
            "p01": p01,
            "p10": p10,
            "p11": p11,
            "c_max": c_max,
            "t_peak": t_peak,
            "n_max": n_max,
            "c_final": c_final,
            "n_final": n_final,
            "validation": diag,
        }

        print(f"  gamma = {gamma:4.2f} (r = {r:4.2f}) | C_max = {c_max:6.4f} at t = {t_peak:5.2f} s | "
              f"C(t_f) = {c_final:8.2e}, N(t_f) = {n_final:8.2e} | Validation: PASSED")

    # 3. Benchmark gamma = 0 against Analytical Solution (Experiment 01b Reference)
    omega_eff = np.sqrt(omega**2 + J**2)
    sin_wt = np.sin(omega_eff * tlist)
    cos_wt = np.cos(omega_eff * tlist)
    alpha_t = cos_wt - 1j * (omega / omega_eff) * sin_wt
    beta_t = -1j * (J / omega_eff) * sin_wt
    c_ana_0 = 2.0 * np.abs(alpha_t * beta_t)
    n_ana_0 = 0.5 * c_ana_0

    max_err_c0 = float(np.max(np.abs(results[0.0]["c_traj"] - c_ana_0)))
    max_err_n0 = float(np.max(np.abs(results[0.0]["n_traj"] - n_ana_0)))
    print(f"\nUnitary Benchmark (gamma = 0 vs Analytical Experiment 01b):")
    print(f"  Max Error |C_num - C_ana| = {max_err_c0:.2e}")
    print(f"  Max Error |N_num - N_ana| = {max_err_n0:.2e}")

    # 4. Color palette for figures
    gamma_colors = {
        0.0: "#1f77b4",
        0.05: "#2ca02c",
        0.2: "#ff7f0e",
        0.5: "#d62728",
        1.0: "#9467bd",
        2.0: "#8c564b",
    }

    # -------------------------------------------------------------
    # Figure A: Concurrence vs Time for all gamma
    # -------------------------------------------------------------
    fig_a, ax_a = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    for g in gamma_list:
        res = results[g]
        ax_a.plot(tlist, res["c_traj"], label=rf"$\gamma = {g:.2f}\ (r = {res['r']:.2f})$",
                  color=gamma_colors[g], linewidth=2.0 if g > 0 else 1.8,
                  linestyle="-" if g > 0 else "--")

    ax_a.set_title(
        r"Experiment 02b: Concurrence Dynamics $C(t)$ under Coherent–Dissipative Competition" "\n"
        rf"$\omega = {omega:.1f},\ J = {J:.1f},\ |\psi(0)\rangle = |00\rangle$",
        fontsize=12, fontweight="bold", pad=12
    )
    ax_a.set_xlabel("Time $t$", fontsize=11)
    ax_a.set_ylabel("Wootters Concurrence $C(t)$", fontsize=11)
    ax_a.set_xlim(0, 20)
    ax_a.set_ylim(-0.02, 0.85)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    plt.tight_layout()
    fig_a_path = output_dir / "experiment_02b_concurrence_vs_gamma.png"
    fig_a.savefig(fig_a_path, dpi=300)
    plt.close(fig_a)
    print(f"\nFigure A saved to: {fig_a_path}")

    # -------------------------------------------------------------
    # Figure B: Negativity vs Time for all gamma
    # -------------------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    for g in gamma_list:
        res = results[g]
        ax_b.plot(tlist, res["n_traj"], label=rf"$\gamma = {g:.2f}\ (r = {res['r']:.2f})$",
                  color=gamma_colors[g], linewidth=2.0 if g > 0 else 1.8,
                  linestyle="-" if g > 0 else "--")

    ax_b.set_title(
        r"Experiment 02b: Negativity Dynamics $N(t)$ under Coherent–Dissipative Competition" "\n"
        rf"$\omega = {omega:.1f},\ J = {J:.1f},\ |\psi(0)\rangle = |00\rangle$",
        fontsize=12, fontweight="bold", pad=12
    )
    ax_b.set_xlabel("Time $t$", fontsize=11)
    ax_b.set_ylabel("Peres-Horodecki Negativity $N(t)$", fontsize=11)
    ax_b.set_xlim(0, 20)
    ax_b.set_ylim(-0.01, 0.43)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    plt.tight_layout()
    fig_b_path = output_dir / "experiment_02b_negativity_vs_gamma.png"
    fig_b.savefig(fig_b_path, dpi=300)
    plt.close(fig_b)
    print(f"Figure B saved to: {fig_b_path}")

    # -------------------------------------------------------------
    # Figure C: Maximum Entanglement vs Dissipation Rate gamma
    # -------------------------------------------------------------
    fig_c, ax_c = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    g_arr = np.array(gamma_list)
    c_max_arr = np.array([results[g]["c_max"] for g in gamma_list])
    n_max_arr = np.array([results[g]["n_max"] for g in gamma_list])

    # Plot lines with markers
    ax_c.plot(g_arr, c_max_arr, "o-", label=r"Max Concurrence $C_{\max}(\gamma)$", color="#1f77b4", linewidth=2.2, markersize=7)
    ax_c.plot(g_arr, n_max_arr, "s-", label=r"Max Negativity $N_{\max}(\gamma)$", color="#ff7f0e", linewidth=2.2, markersize=7)

    # Operational suppression threshold line (C_max < 0.05)
    ax_c.axhline(0.05, color="#d62728", linestyle=":", linewidth=1.5,
                 label=r"Operational Suppression Threshold ($C_{\max} = 0.05$)")

    # Shaded regime backgrounds
    ax_c.axvspan(-0.05, 0.1, alpha=0.12, color="#2ca02c", label=r"Coherence-dominated ($r \ll 1$)")
    ax_c.axvspan(0.1, 0.75, alpha=0.12, color="#ff7f0e", label=r"Intermediate Competition ($r \sim 1$)")
    ax_c.axvspan(0.75, 2.1, alpha=0.12, color="#d62728", label=r"Dissipation-dominated ($r \gg 1$)")

    ax_c.set_title(
        r"Experiment 02b: Peak Generated Entanglement vs Dissipation Rate $\gamma$" "\n"
        rf"$\omega = {omega:.1f},\ J = {J:.1f}\quad (r = \gamma / J)$",
        fontsize=12, fontweight="bold", pad=12
    )
    ax_c.set_xlabel(r"Dissipation Rate $\gamma$ (units of $\omega$)", fontsize=11)
    ax_c.set_ylabel("Maximum Entanglement", fontsize=11)
    ax_c.set_xlim(-0.05, 2.1)
    ax_c.set_ylim(-0.01, 0.85)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper right", framealpha=0.95, fontsize=9.2)

    # Annotation box
    textstr_c = "\n".join((
        r"$\mathbf{Entanglement\ Summary:}$",
        rf"$\gamma = 0.00 \ (r = 0.0) \to C_{{\max}} = {results[0.0]['c_max']:.4f}$",
        rf"$\gamma = 0.05 \ (r = 0.1) \to C_{{\max}} = {results[0.05]['c_max']:.4f}$",
        rf"$\gamma = 0.50 \ (r = 1.0) \to C_{{\max}} = {results[0.5]['c_max']:.4f}$",
        rf"$\gamma = 2.00 \ (r = 4.0) \to C_{{\max}} = {results[2.0]['c_max']:.4f}$",
    ))
    props_c = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
    ax_c.text(0.48, 0.42, textstr_c, transform=ax_c.transAxes, fontsize=9.2, verticalalignment="bottom", bbox=props_c)


    plt.tight_layout()
    fig_c_path = output_dir / "experiment_02b_max_entanglement_vs_gamma.png"
    fig_c.savefig(fig_c_path, dpi=300)
    plt.close(fig_c)
    print(f"Figure C saved to: {fig_c_path}")

    # -------------------------------------------------------------
    # Figure D: Population Dynamics across 3 Representative Regimes
    # -------------------------------------------------------------
    rep_gammas = [0.05, 0.5, 2.0]
    fig_d, axes_d = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300, sharey=True)

    titles_d = [
        r"(a) $\gamma = 0.05\ (r = 0.10)$: Underdamped Coherent Regime",
        r"(b) $\gamma = 0.50\ (r = 1.00)$: Intermediate Competition",
        r"(c) $\gamma = 2.00\ (r = 4.00)$: Overdamped Dissipative Regime",
    ]

    for ax, g, title in zip(axes_d, rep_gammas, titles_d):
        res = results[g]
        ax.plot(tlist, res["p00"], label=r"$P_{00}(t)$", color="#1f77b4", linewidth=2.0)
        ax.plot(tlist, res["p01"], label=r"$P_{01}(t)$", color="#ff7f0e", linewidth=1.8)
        ax.plot(tlist, res["p10"], "--", label=r"$P_{10}(t)$", color="#2ca02c", linewidth=1.8)
        ax.plot(tlist, res["p11"], label=r"$P_{11}(t)$", color="#d62728", linewidth=2.0)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time $t$", fontsize=10.5)
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        if g == 0.05:
            ax.set_ylabel("Subspace Population", fontsize=11)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    fig_d.suptitle(
        r"Experiment 02b: Population Dynamics $P_{ij}(t)$ across Regimes ($\omega = 1.0, J = 0.5$)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig_d_path = output_dir / "experiment_02b_population_dynamics.png"
    fig_d.savefig(fig_d_path, dpi=300)
    plt.close(fig_d)
    print(f"Figure D saved to: {fig_d_path}")

    # 5. Summary Table Printout
    print("\n" + "=" * 80)
    print("EXPERIMENT 02B NUMERICAL SUMMARY TABLE")
    print("=" * 80)
    print(f"{'gamma':>8} | {'gamma/J (r)':>12} | {'C_max':>8} | {'t_peak (s)':>11} | {'C(t_final)':>12} | {'N(t_final)':>12} | {'State Valid':>11}")
    print("-" * 80)
    for g in gamma_list:
        res = results[g]
        print(f"{g:8.2f} | {res['r']:12.2f} | {res['c_max']:8.4f} | {res['t_peak']:11.2f} | {res['c_final']:12.2e} | {res['n_final']:12.2e} | {'PASSED':>11}")
    print("=" * 80)

    return {
        "results": results,
        "max_err_c0": max_err_c0,
        "max_err_n0": max_err_n0,
        "fig_a_path": str(fig_a_path),
        "fig_b_path": str(fig_b_path),
        "fig_c_path": str(fig_c_path),
        "fig_d_path": str(fig_d_path),
    }


if __name__ == "__main__":
    run_experiment()
