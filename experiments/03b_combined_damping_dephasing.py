"""Experiment 03b: Combined Amplitude Damping and Pure Dephasing.

Investigates the competition between coherent interaction, amplitude damping,
and pure dephasing in generating non-equilibrium steady-state (NESS) entanglement.

Part A: Robustness of the optimal NESS (J=0.85, gamma_1=1.80) against pure dephasing gamma_phi.
Part B: 2D parameter map of NESS concurrence over (gamma_1, gamma_phi) at fixed J=0.85.
Part C: Dynamical trajectories of entanglement for specific noise regimes.

Model:
    H = (omega / 2) * (sigma_z1 + sigma_z2) + J * (sigma_x1 * sigma_x2)  [omega = 1.0]
    C_1 = sqrt(gamma_1) * (sigma_- tensor I)
    C_2 = sqrt(gamma_1) * (I tensor sigma_-)
    L_phi1 = sqrt(gamma_phi / 2) * (sigma_z tensor I)
    L_phi2 = sqrt(gamma_phi / 2) * (I tensor sigma_z)
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
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators, build_dephasing_collapse_operators
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.dynamics.simulation import simulate_dynamics
from bipartit_open_spin.validation.diagnostics import validate_density_matrix


def run_experiment() -> dict:
    print("=" * 75)
    print("EXPERIMENT 03b: COMBINED AMPLITUDE DAMPING AND PURE DEPHASING")
    print("=" * 75)

    # Base parameters
    omega = 1.0
    J_opt = 0.85
    gamma_1_opt = 1.80

    fig_dir = Path("results/figures")
    data_dir = Path("results/data")
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Part A: Robustness of 02c optimal NESS vs. gamma_phi
    # =========================================================================
    print("\n--- [Part A] 1D Robustness Scan ---")
    N_phi = 41
    gamma_phi_1d = np.linspace(0.0, 2.0, N_phi)
    c_ss_1d = np.zeros(N_phi)
    n_ss_1d = np.zeros(N_phi)

    for i, g_phi in enumerate(gamma_phi_1d):
        params = ModelParams(omega=omega, J=J_opt, gamma=gamma_1_opt)
        H = build_hamiltonian(params)
        # Combine both amplitude damping and pure dephasing collapse operators
        c_ops = build_collapse_operators(params) + build_dephasing_collapse_operators(g_phi)
        
        rho_ss = steadystate(H, c_ops)
        c_ss_1d[i] = concurrence(rho_ss)
        n_ss_1d[i] = negativity(rho_ss)

    figA, axA = plt.subplots(figsize=(8.5, 6.0), dpi=300)
    axA.plot(gamma_phi_1d, c_ss_1d, 'b-', linewidth=2.5, label='Concurrence $C_{\\mathrm{ss}}$')
    axA.plot(gamma_phi_1d, n_ss_1d, 'r--', linewidth=2.5, label='Negativity $N_{\\mathrm{ss}}$')
    axA.set_title(
        f"Part A: NESS Entanglement Robustness to Pure Dephasing\n"
        f"($J={J_opt}$, $\\gamma_1={gamma_1_opt}$, $\\omega={omega}$)",
        fontsize=12, fontweight='bold'
    )
    axA.set_xlabel(r"Pure Dephasing Rate $\gamma_\phi$ (units of $\omega$)", fontsize=11)
    axA.set_ylabel("Steady-State Entanglement", fontsize=11)
    axA.grid(True, linestyle="--", alpha=0.6)
    axA.legend(loc="upper right", fontsize=10)
    axA.set_xlim(0, 2.0)
    axA.set_ylim(0, np.max(c_ss_1d)*1.1)

    figA_path = fig_dir / "experiment_03b_partA_robustness.png"
    figA.tight_layout()
    figA.savefig(figA_path)
    plt.close(figA)
    print(f"  Part A figure saved to: {figA_path}")

    # =========================================================================
    # Part B: 2D Parameter Map (gamma_1, gamma_phi) at fixed J = 0.85
    # =========================================================================
    print("\n--- [Part B] 2D Phase Map ---")
    N_g1 = 41
    N_gphi = 41
    gamma_1_vals = np.linspace(0.0, 2.5, N_g1)
    gamma_phi_vals = np.linspace(0.0, 2.5, N_gphi)

    c_ss_2d = np.zeros((N_gphi, N_g1))

    for i, g_phi in enumerate(gamma_phi_vals):
        for j, g_1 in enumerate(gamma_1_vals):
            if np.isclose(g_1, 0.0):
                # When gamma_1 = 0, we only have pure dephasing. 
                # As shown in 03a, pure dephasing drives the system to an unentangled mixture (C_ss = 0).
                # steadystate() might fail due to singular Liouvillian if H doesn't lift degeneracies fully.
                c_ss_2d[i, j] = 0.0
                continue
            
            params = ModelParams(omega=omega, J=J_opt, gamma=g_1)
            H = build_hamiltonian(params)
            c_ops = build_collapse_operators(params) + build_dephasing_collapse_operators(g_phi)
            rho_ss = steadystate(H, c_ops)
            c_ss_2d[i, j] = concurrence(rho_ss)

    figB, axB = plt.subplots(figsize=(8.5, 6.2), dpi=300)
    G1_grid, Gphi_grid = np.meshgrid(gamma_1_vals, gamma_phi_vals)

    c_plot_2d = np.where(G1_grid == 0.0, np.nan, c_ss_2d)
    cmap_b = plt.cm.viridis.copy()
    cmap_b.set_bad(color="#d3d3d3")

    meshB = axB.pcolormesh(G1_grid, Gphi_grid, c_plot_2d, cmap=cmap_b, shading="auto")
    cbarB = figB.colorbar(meshB, ax=axB, pad=0.03)
    cbarB.set_label("Steady-State Concurrence $C_{\\mathrm{ss}}$", fontsize=11)

    contoursB = axB.contour(G1_grid, Gphi_grid, c_plot_2d, levels=[0.05, 0.1, 0.15, 0.2, 0.25],
                            colors="white", linewidths=1.0, alpha=0.8)
    axB.clabel(contoursB, inline=True, fontsize=8.5, fmt="%.2f")

    axB.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5, label=r"$\gamma_1 = 0$ (No NESS)")
    axB.plot(gamma_1_opt, 0.0, "r*", markersize=12, label="Optimal NESS without dephasing")

    axB.set_title(
        f"Part B: Steady-State Concurrence Phase Map\n($J={J_opt}$, $\\omega={omega}$)",
        fontsize=12, fontweight="bold"
    )
    axB.set_xlabel(r"Amplitude Damping Rate $\gamma_1$", fontsize=11)
    axB.set_ylabel(r"Pure Dephasing Rate $\gamma_\phi$", fontsize=11)
    axB.set_xlim(0, 2.5)
    axB.set_ylim(0, 2.5)
    axB.legend(loc="upper right")

    figB_path = fig_dir / "experiment_03b_partB_2d_map.png"
    figB.tight_layout()
    figB.savefig(figB_path)
    plt.close(figB)
    print(f"  Part B figure saved to: {figB_path}")

    # =========================================================================
    # Part C: Dynamical Trajectories
    # =========================================================================
    print("\n--- [Part C] Dynamical Trajectories ---")
    regimes = [
        {"name": "No Dephasing (Opt NESS)", "gamma_1": 1.80, "gamma_phi": 0.0},
        {"name": "Equal Noise", "gamma_1": 1.80, "gamma_phi": 1.80},
        {"name": "Dephasing Dominated", "gamma_1": 0.50, "gamma_phi": 2.50},
    ]

    tlist = np.linspace(0.0, 30.0, 400)
    config = SimulationConfig(tlist=tlist)
    psi0 = computational_basis(0, 0)

    figC, axC = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for r, col in zip(regimes, colors):
        g_1 = r["gamma_1"]
        g_phi = r["gamma_phi"]
        
        params = ModelParams(omega=omega, J=J_opt, gamma=g_1)
        H = build_hamiltonian(params)
        c_ops = build_collapse_operators(params) + build_dephasing_collapse_operators(g_phi)
        
        states = simulate_dynamics(H, psi0, c_ops, config)
        c_dyn = [concurrence(rho) for rho in states]
        
        axC.plot(tlist, c_dyn, label=rf"{r['name']} ($\gamma_1={g_1}, \gamma_\phi={g_phi}$)", color=col, linewidth=2)

    axC.set_title(
        f"Part C: Dynamical Trajectories from |00>\n($J={J_opt}$, $\\omega={omega}$)",
        fontsize=12, fontweight="bold"
    )
    axC.set_xlabel(r"Time $t$", fontsize=11)
    axC.set_ylabel(r"Concurrence $C(t)$", fontsize=11)
    axC.set_xlim(0, tlist[-1])
    axC.set_ylim(0, 0.45)
    axC.grid(True, linestyle="--", alpha=0.6)
    axC.legend(loc="upper right")

    figC_path = fig_dir / "experiment_03b_partC_dynamics.png"
    figC.tight_layout()
    figC.savefig(figC_path)
    plt.close(figC)
    print(f"  Part C figure saved to: {figC_path}")

    # Save all data
    data_file = data_dir / "experiment_03b_data.npz"
    np.savez_compressed(
        data_file,
        gamma_phi_1d=gamma_phi_1d,
        c_ss_1d=c_ss_1d,
        n_ss_1d=n_ss_1d,
        gamma_1_vals=gamma_1_vals,
        gamma_phi_vals=gamma_phi_vals,
        c_ss_2d=c_ss_2d
    )
    print(f"  Data saved to: {data_file}")

    return {
        "figA_path": str(figA_path),
        "figB_path": str(figB_path),
        "figC_path": str(figC_path),
        "data_file": str(data_file),
    }

if __name__ == "__main__":
    run_experiment()
