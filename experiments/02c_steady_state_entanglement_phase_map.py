"""Experiment 02c: Steady-State Entanglement Phase Map.

This experiment constructs a comprehensive 2D parameter-space phase map of the
non-equilibrium steady-state (NESS) bipartite entanglement as a function of:
    1. Coherent exchange coupling strength J in [0.0, 2.0]
    2. Local dissipation rate gamma in [0.0, 2.0]

Model:
    H = (omega / 2) * (sigma_z1 + sigma_z2) + J * (sigma_x1 * sigma_x2)  [omega = 1.0]
    C_1 = sqrt(gamma) * (sigma_- tensor I)
    C_2 = sqrt(gamma) * (I tensor sigma_-)

QuTiP Operator Convention:
    sigma_- |0> = |1>  (decay from upper level |0> to ground level |1>)
    sigma_- |1> = 0    (ground level / dark state)

Steady-State Determination:
    For gamma > 0, the NESS density matrix rho_ss satisfies L(rho_ss) = 0.
    For gamma = 0 (unitary limit), no dissipative steady state exists (perpetual oscillations).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend for publication reproducibility
import matplotlib.pyplot as plt
import numpy as np
from qutip import steadystate

from bipartit_open_spin.analysis.entanglement import concurrence, negativity
from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import computational_basis
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.dynamics.simulation import simulate_dynamics
from bipartit_open_spin.validation.diagnostics import validate_density_matrix


def run_experiment() -> dict:
    """Execute 2D phase map scan, cross-validate with dynamics, and save data and figures."""
    print("=" * 75)
    print("EXPERIMENT 02C: STEADY-STATE ENTANGLEMENT PHASE MAP")
    print("=" * 75)

    # 1. Parameter Grid
    omega = 1.0
    N_J = 41
    N_gamma = 41
    J_values = np.linspace(0.0, 2.0, N_J)
    gamma_values = np.linspace(0.0, 2.0, N_gamma)

    print(f"\n--- [1] Scan Configuration ---")
    print(f"  Single-Spin Frequency : omega = {omega:.1f}")
    print(f"  Coupling Range (J)    : [{J_values[0]:.2f}, {J_values[-1]:.2f}], {N_J} points")
    print(f"  Dissipation (gamma)   : [{gamma_values[0]:.2f}, {gamma_values[-1]:.2f}], {N_gamma} points")
    print(f"  Total Phase Map Grid  : {N_J} x {N_gamma} = {N_J * N_gamma} points")

    # Output directories
    fig_dir = Path("results/figures")
    data_dir = Path("results/data")
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Storage arrays (shape: [N_gamma, N_J] for row=gamma, col=J)
    concurrence_ss = np.full((N_gamma, N_J), np.nan)
    negativity_ss = np.full((N_gamma, N_J), np.nan)
    convergence_mask = np.zeros((N_gamma, N_J), dtype=bool)

    # 2. Compute 2D Phase Map
    print("\n--- [2] Computing Steady-State Entanglement Grid ---")
    validation_failures = 0

    for i, gamma in enumerate(gamma_values):
        for j, J in enumerate(J_values):
            if np.isclose(gamma, 0.0):
                # Unitary limit: No dissipative steady state exists
                concurrence_ss[i, j] = 0.0
                negativity_ss[i, j] = 0.0
                convergence_mask[i, j] = False
                continue

            params = ModelParams(omega=omega, J=J, gamma=gamma)
            H = build_hamiltonian(params)
            c_ops = build_collapse_operators(params)

            # Direct Liouvillian steady state
            rho_ss = steadystate(H, c_ops)

            # Physical validation of steady state density matrix
            v_res = validate_density_matrix(rho_ss, tol=1e-6)
            if not v_res["valid"]:
                validation_failures += 1

            c_val = concurrence(rho_ss)
            n_val = negativity(rho_ss)

            concurrence_ss[i, j] = c_val
            negativity_ss[i, j] = n_val
            convergence_mask[i, j] = True

    print(f"  Grid scan complete. Validation failures: {validation_failures}")
    print(f"  Max Steady-State Concurrence : {np.nanmax(concurrence_ss):.4f}")
    print(f"  Max Steady-State Negativity  : {np.nanmax(negativity_ss):.4f}")

    # 3. Cross-Validation against Time-Dependent Dynamics (mesolve)
    print("\n--- [3] Cross-Validating Representative Points with mesolve Dynamics ---")
    rep_points = [
        (0.0, 0.50),  # J=0, gamma=0.5 (Uncoupled dissipative)
        (0.2, 0.20),  # J=0.2, gamma=0.2 (Weak coupling)
        (0.5, 0.05),  # J=0.5, gamma=0.05 (Coherence-dominated)
        (0.5, 0.50),  # J=0.5, gamma=0.5 (Intermediate)
        (0.5, 2.00),  # J=0.5, gamma=2.0 (Overdamped)
        (1.0, 0.50),  # J=1.0, gamma=0.5 (Strong coupling)
        (1.0, 2.00),  # J=1.0, gamma=2.0 (Strong coupling + strong dissipation)
        (2.0, 0.05),  # J=2.0, gamma=0.05 (Large J, weak dissipation)
        (2.0, 2.00),  # J=2.0, gamma=2.0 (Large J, strong dissipation)
    ]

    psi0 = computational_basis(0, 0)
    rep_table_data = []

    print(f"{'J':>6} | {'gamma':>6} | {'C_ss (Direct)':>14} | {'C_ss (Dynamics)':>16} | {'Diff ||rho_dyn - rho_ss||':>26} | {'Status':>8}")
    print("-" * 85)

    for J_rep, gamma_rep in rep_points:
        params_rep = ModelParams(omega=omega, J=J_rep, gamma=gamma_rep)
        H_rep = build_hamiltonian(params_rep)
        c_ops_rep = build_collapse_operators(params_rep)
        rho_ss_direct = steadystate(H_rep, c_ops_rep)
        c_direct = concurrence(rho_ss_direct)
        n_direct = negativity(rho_ss_direct)

        # Long-time dynamics to ensure relaxation
        t_sim_max = max(50.0, 15.0 / gamma_rep)
        config_rep = SimulationConfig(tlist=np.linspace(0.0, t_sim_max, 400))
        states_rep = simulate_dynamics(H_rep, psi0, c_ops_rep, config_rep)

        rho_final_dyn = states_rep[-1]
        c_dyn = concurrence(rho_final_dyn)
        diff_norm = np.linalg.norm(rho_final_dyn.full() - rho_ss_direct.full())
        v_dyn = validate_density_matrix(rho_final_dyn, tol=1e-5)

        is_converged = bool(diff_norm < 1e-3 and v_dyn["valid"])
        status = "PASSED" if is_converged else "WARN"

        print(f"{J_rep:6.2f} | {gamma_rep:6.2f} | {c_direct:14.4f} | {c_dyn:16.4f} | {diff_norm:26.2e} | {status:>8}")

        rep_table_data.append({
            "J": J_rep,
            "gamma": gamma_rep,
            "c_ss": c_direct,
            "n_ss": n_direct,
            "c_dyn": c_dyn,
            "diff_norm": diff_norm,
            "converged": is_converged,
            "valid": v_dyn["valid"],
        })

    # 4. Save Numerical Phase Map Data
    data_file = data_dir / "experiment_02c_steady_state_phase_map.npz"
    np.savez_compressed(
        data_file,
        J_values=J_values,
        gamma_values=gamma_values,
        concurrence_ss=concurrence_ss,
        negativity_ss=negativity_ss,
        convergence_mask=convergence_mask,
        omega=omega,
    )
    print(f"\nPhase map numerical data saved to: {data_file}")

    # -------------------------------------------------------------
    # 5. FIGURE 1: Concurrence Phase Map
    # -------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.2), dpi=300)
    J_grid, G_grid = np.meshgrid(J_values, gamma_values)

    # Mask gamma = 0 for accurate dissipative heatmap
    c_plot = np.where(G_grid == 0.0, np.nan, concurrence_ss)
    cmap_c = plt.cm.viridis.copy()
    cmap_c.set_bad(color="#d3d3d3")  # Light gray for unitary boundary gamma=0

    mesh1 = ax1.pcolormesh(J_grid, G_grid, c_plot, cmap=cmap_c, shading="auto", vmin=0.0, vmax=0.35)
    cbar1 = fig1.colorbar(mesh1, ax=ax1, pad=0.03)
    cbar1.set_label("Steady-State Concurrence $C_{\\mathrm{ss}}$", fontsize=11, fontweight="medium")

    # Overlay contour lines
    contours1 = ax1.contour(J_grid, G_grid, c_plot, levels=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                            colors="white", linewidths=1.0, alpha=0.75)
    ax1.clabel(contours1, inline=True, fontsize=8.5, fmt="%.2f")

    # Mark unitary line
    ax1.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.5, label=r"$\gamma = 0$ (Unitary Limit, No NESS)")

    # Mark maximum entanglement ridge
    max_idx = np.unravel_index(np.nanargmax(c_plot), c_plot.shape)
    opt_gamma = gamma_values[max_idx[0]]
    opt_J = J_values[max_idx[1]]
    ax1.plot(opt_J, opt_gamma, "r*", markersize=12, label=rf"Global Peak: $C_{{\mathrm{{ss}}}}={c_plot[max_idx]:.4f}\ (J={opt_J:.2f}, \gamma={opt_gamma:.2f})$")

    ax1.set_title(
        r"Experiment 02c: Steady-State Concurrence Phase Map $C_{\mathrm{ss}}(J, \gamma)$" "\n"
        rf"$\omega = {omega:.1f},\ H = \frac{{\omega}}{{2}}(\sigma_z^1 + \sigma_z^2) + J\sigma_x^1\sigma_x^2,\ C_k = \sqrt{{\gamma}}\sigma_-^k$",
        fontsize=11.5, fontweight="bold", pad=12
    )
    ax1.set_xlabel(r"Exchange Coupling Strength $J$ (units of $\omega$)", fontsize=11)
    ax1.set_ylabel(r"Dissipation Rate $\gamma$ (units of $\omega$)", fontsize=11)
    ax1.set_xlim(0.0, 2.0)
    ax1.set_ylim(0.0, 2.0)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.2)

    plt.tight_layout()
    fig1_path = fig_dir / "experiment_02c_steady_state_concurrence_phase_map.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"Figure 1 saved to: {fig1_path}")

    # -------------------------------------------------------------
    # 6. FIGURE 2: Negativity Phase Map
    # -------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.5, 6.2), dpi=300)
    n_plot = np.where(G_grid == 0.0, np.nan, negativity_ss)
    cmap_n = plt.cm.plasma.copy()
    cmap_n.set_bad(color="#d3d3d3")

    mesh2 = ax2.pcolormesh(J_grid, G_grid, n_plot, cmap=cmap_n, shading="auto", vmin=0.0, vmax=0.18)
    cbar2 = fig2.colorbar(mesh2, ax=ax2, pad=0.03)
    cbar2.set_label("Steady-State Negativity $N_{\\mathrm{ss}}$", fontsize=11, fontweight="medium")

    contours2 = ax2.contour(J_grid, G_grid, n_plot, levels=[0.025, 0.05, 0.075, 0.10, 0.125, 0.15],
                            colors="white", linewidths=1.0, alpha=0.75)
    ax2.clabel(contours2, inline=True, fontsize=8.5, fmt="%.3f")

    ax2.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.5, label=r"$\gamma = 0$ (Unitary Limit, No NESS)")
    ax2.plot(opt_J, opt_gamma, "w*", markersize=12, label=rf"Global Peak: $N_{{\mathrm{{ss}}}}={n_plot[max_idx]:.4f}\ (J={opt_J:.2f}, \gamma={opt_gamma:.2f})$")

    ax2.set_title(
        r"Experiment 02c: Steady-State Negativity Phase Map $N_{\mathrm{ss}}(J, \gamma)$" "\n"
        rf"$\omega = {omega:.1f},\ H = \frac{{\omega}}{{2}}(\sigma_z^1 + \sigma_z^2) + J\sigma_x^1\sigma_x^2,\ C_k = \sqrt{{\gamma}}\sigma_-^k$",
        fontsize=11.5, fontweight="bold", pad=12
    )
    ax2.set_xlabel(r"Exchange Coupling Strength $J$ (units of $\omega$)", fontsize=11)

    ax2.set_ylabel(r"Dissipation Rate $\gamma$ (units of $\omega$)", fontsize=11)
    ax2.set_xlim(0.0, 2.0)
    ax2.set_ylim(0.0, 2.0)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.2)

    plt.tight_layout()
    fig2_path = fig_dir / "experiment_02c_steady_state_negativity_phase_map.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"Figure 2 saved to: {fig2_path}")

    # -------------------------------------------------------------
    # 7. FIGURE 3: Parameter Cuts
    # -------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    selected_gammas = [0.05, 0.20, 0.50, 1.00, 2.00]
    cut_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    # Panel (a): C_ss vs J for selected gamma
    for g_cut, col in zip(selected_gammas, cut_colors):
        # find nearest gamma index
        idx_g = int(np.argmin(np.abs(gamma_values - g_cut)))
        actual_g = gamma_values[idx_g]
        c_cut = concurrence_ss[idx_g, :]
        ax3a.plot(J_values, c_cut, label=rf"$\gamma = {actual_g:.2f}$", color=col, linewidth=2.0)

    ax3a.set_title(r"(a) Steady-State Concurrence $C_{\mathrm{ss}}$ vs $J$", fontsize=12, fontweight="bold")
    ax3a.set_xlabel("Exchange Coupling $J$", fontsize=11)
    ax3a.set_ylabel("Concurrence $C_{\\mathrm{ss}}$", fontsize=11)
    ax3a.set_xlim(0.0, 2.0)
    ax3a.set_ylim(-0.01, 0.35)
    ax3a.grid(True, linestyle="--", alpha=0.5)
    ax3a.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    # Panel (b): N_ss vs J for selected gamma
    for g_cut, col in zip(selected_gammas, cut_colors):
        idx_g = int(np.argmin(np.abs(gamma_values - g_cut)))
        actual_g = gamma_values[idx_g]
        n_cut = negativity_ss[idx_g, :]
        ax3b.plot(J_values, n_cut, label=rf"$\gamma = {actual_g:.2f}$", color=col, linewidth=2.0)

    ax3b.set_title(r"(b) Steady-State Negativity $N_{\mathrm{ss}}$ vs $J$", fontsize=12, fontweight="bold")
    ax3b.set_xlabel("Exchange Coupling $J$", fontsize=11)
    ax3b.set_ylabel("Negativity $N_{\\mathrm{ss}}$", fontsize=11)
    ax3b.set_xlim(0.0, 2.0)
    ax3b.set_ylim(-0.005, 0.18)
    ax3b.grid(True, linestyle="--", alpha=0.5)
    ax3b.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    fig3.suptitle(
        r"Experiment 02c: 1D Parameter Cuts of Steady-State Entanglement ($\omega = 1.0$)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig3_path = fig_dir / "experiment_02c_parameter_cuts.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"Figure 3 saved to: {fig3_path}")

    return {
        "J_values": J_values,
        "gamma_values": gamma_values,
        "concurrence_ss": concurrence_ss,
        "negativity_ss": negativity_ss,
        "opt_J": opt_J,
        "opt_gamma": opt_gamma,
        "opt_C": c_plot[max_idx],
        "rep_table_data": rep_table_data,
        "fig1_path": str(fig1_path),
        "fig2_path": str(fig2_path),
        "fig3_path": str(fig3_path),
        "data_file": str(data_file),
    }


if __name__ == "__main__":
    run_experiment()
