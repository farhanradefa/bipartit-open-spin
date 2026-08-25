"""Experiment 06b: Dynamical Encircling of an Exceptional Point and Direction-Dependent State Transfer.

This experiment investigates real-time non-Hermitian dynamics when system parameters
(J(t), Delta_gamma(t)) dynamically encircle the EP2:
1. Time-dependent Schrödinger integration: d psi / dt = -i H_eff_odd(t) psi(t).
2. Comparison of Counter-Clockwise (CCW) vs Clockwise (CW) encircling.
3. Analysis of both initial instantaneous eigenstates |phi_+(0)> and |phi_-(0)>.
4. Fast (T=5), Intermediate (T=25), and Slow (T=100) encircling periods.
5. Continuous tracking of raw norm (P_survival), normalized state (|psi_tilde>), and instantaneous overlaps.
6. Quantitative chirality metric chi(T) vs encircling period T in [2, 120].
7. Generation of 8 publication-quality figures and numerical .npz dataset.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bipartit_open_spin.analysis.spectrum import (
    build_topological_odd_hamiltonian,
    track_instantaneous_eigenpairs,
)
from bipartit_open_spin.dynamics.simulation import simulate_timedependent_nonhermitian

# Output directories
FIGURES_DIR = Path("results/figures")
DATA_DIR = Path("results/data")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "lines.linewidth": 1.6,
})


# ==============================================================================
# PARAMETER LOOP AND TIME-DEPENDENT HAMILTONIAN CONSTRUCTOR
# ==============================================================================

def make_loop_hamiltonian(
    T: float,
    direction: str = "CCW",
    j_center: float = 0.35,
    r_j: float = 0.15,
    dg_center: float = 1.40,
    r_dg: float = 0.45,
    gamma_bar: float = 1.0,
):
    """Construct callable H(t) for the closed elliptical parameter loop around EP."""
    sign = +1.0 if direction.upper() == "CCW" else -1.0

    def H_func(t):
        theta = sign * 2.0 * np.pi * (t / T)
        j_t = j_center + r_j * np.cos(theta)
        dg_t = dg_center + r_dg * np.sin(theta)
        H_top = build_topological_odd_hamiltonian(j_t, dg_t)
        return -0.5j * gamma_bar * np.eye(2, dtype=complex) + H_top

    return H_func


# ==============================================================================
# PART 1: DYNAMICAL SIMULATION FOR REGIMES (FAST, MID, SLOW)
# ==============================================================================

def run_regime_simulations(
    t_periods: dict[str, float] = None,
    n_steps: int = 1500,
    j_center: float = 0.35,
    r_j: float = 0.15,
    dg_center: float = 1.40,
    r_dg: float = 0.45,
    gamma_bar: float = 1.0,
) -> dict:
    """Run dynamical simulations across Fast, Intermediate, and Slow encircling periods."""
    if t_periods is None:
        t_periods = {"fast": 5.0, "mid": 25.0, "slow": 100.0}

    print("=" * 80)
    print("PART 1: TIME-DEPENDENT NON-HERMITIAN INTEGRATION (CW vs CCW)")
    print("=" * 80)

    regime_results = {}

    for reg_name, T in t_periods.items():
        print(f"\n--- Simulating {reg_name.upper()} regime (T = {T:.1f}) ---")
        tlist = np.linspace(0.0, T, n_steps)

        H_ccw = make_loop_hamiltonian(T, "CCW", j_center, r_j, dg_center, r_dg, gamma_bar)
        H_cw = make_loop_hamiltonian(T, "CW", j_center, r_j, dg_center, r_dg, gamma_bar)

        # Track instantaneous eigenmodes along CCW and CW
        spec_ccw = track_instantaneous_eigenpairs(H_ccw, tlist)
        spec_cw = track_instantaneous_eigenpairs(H_cw, tlist)

        # Initial eigenmodes at t=0
        # phi_plus(0) is eigenmode 1 (higher real energy), phi_minus(0) is eigenmode 0
        phi_minus_0 = spec_ccw["eigenvectors"][0, :, 0]
        phi_plus_0 = spec_ccw["eigenvectors"][0, :, 1]

        cases = {
            ("CCW", "init_plus"): (H_ccw, phi_plus_0, spec_ccw),
            ("CCW", "init_minus"): (H_ccw, phi_minus_0, spec_ccw),
            ("CW", "init_plus"): (H_cw, phi_plus_0, spec_cw),
            ("CW", "init_minus"): (H_cw, phi_minus_0, spec_cw),
        }

        sim_data = {}
        for (direction, init_mode), (H_fn, psi0, spec_fn) in cases.items():
            res = simulate_timedependent_nonhermitian(H_fn, psi0, tlist)

            # Compute instantaneous eigenstate projections: |<phi_pm(t)|psi_tilde(t)>|^2
            norm_states = res["normalized_states"]
            evecs_t = spec_fn["eigenvectors"]
            p_plus_t = np.zeros(len(tlist), dtype=float)
            p_minus_t = np.zeros(len(tlist), dtype=float)

            for idx in range(len(tlist)):
                psi_t = norm_states[idx, :]
                evec_minus = evecs_t[idx, :, 0]
                evec_plus = evecs_t[idx, :, 1]
                p_minus_t[idx] = float(np.abs(np.vdot(evec_minus, psi_t)) ** 2)
                p_plus_t[idx] = float(np.abs(np.vdot(evec_plus, psi_t)) ** 2)

            sim_data[(direction, init_mode)] = {
                "tlist": tlist,
                "tau": tlist / T,
                "raw_states": res["raw_states"],
                "normalized_states": norm_states,
                "survival_probability": res["survival_probability"],
                "p01": res["p01"],
                "p10": res["p10"],
                "p_plus": p_plus_t,
                "p_minus": p_minus_t,
                "final_p01": res["p01"][-1],
                "final_p10": res["p10"][-1],
                "final_p_plus": p_plus_t[-1],
                "final_p_minus": p_minus_t[-1],
            }

        # Calculate final state fidelities
        # Compare final state with instantaneous eigenstates at t=T
        psi_final_ccw_plus = sim_data[("CCW", "init_plus")]["normalized_states"][-1]
        psi_final_ccw_minus = sim_data[("CCW", "init_minus")]["normalized_states"][-1]
        psi_final_cw_plus = sim_data[("CW", "init_plus")]["normalized_states"][-1]
        psi_final_cw_minus = sim_data[("CW", "init_minus")]["normalized_states"][-1]

        phi_plus_T_ccw = spec_ccw["eigenvectors"][-1, :, 1]
        phi_minus_T_ccw = spec_ccw["eigenvectors"][-1, :, 0]
        phi_plus_T_cw = spec_cw["eigenvectors"][-1, :, 1]
        phi_minus_T_cw = spec_cw["eigenvectors"][-1, :, 0]

        fidelity_matrix_ccw = np.array([
            [float(np.abs(np.vdot(phi_plus_T_ccw, psi_final_ccw_plus)) ** 2),
             float(np.abs(np.vdot(phi_minus_T_ccw, psi_final_ccw_plus)) ** 2)],
            [float(np.abs(np.vdot(phi_plus_T_ccw, psi_final_ccw_minus)) ** 2),
             float(np.abs(np.vdot(phi_minus_T_ccw, psi_final_ccw_minus)) ** 2)],
        ])

        fidelity_matrix_cw = np.array([
            [float(np.abs(np.vdot(phi_plus_T_cw, psi_final_cw_plus)) ** 2),
             float(np.abs(np.vdot(phi_minus_T_cw, psi_final_cw_plus)) ** 2)],
            [float(np.abs(np.vdot(phi_plus_T_cw, psi_final_cw_minus)) ** 2),
             float(np.abs(np.vdot(phi_minus_T_cw, psi_final_cw_minus)) ** 2)],
        ])

        # Chirality metric for this T
        # chi = |P01_CW(T) - P01_CCW(T)| for initial state |phi_+(0)>
        p01_cw = sim_data[("CW", "init_plus")]["final_p01"]
        p01_ccw = sim_data[("CCW", "init_plus")]["final_p01"]
        chirality_val = float(np.abs(p01_cw - p01_ccw))

        print(f"  CCW Final P01 (init +): {p01_ccw:.4f} | CW Final P01 (init +): {p01_cw:.4f}")
        print(f"  Chirality Metric chi(T={T:.1f}) = {chirality_val:.4f}")

        regime_results[reg_name] = {
            "T": T,
            "tlist": tlist,
            "sim_data": sim_data,
            "spec_ccw": spec_ccw,
            "spec_cw": spec_cw,
            "fidelity_matrix_ccw": fidelity_matrix_ccw,
            "fidelity_matrix_cw": fidelity_matrix_cw,
            "chirality": chirality_val,
        }

    return regime_results


# ==============================================================================
# PART 2: CONTINUOUS PERIOD SCAN CHIRALITY CURVE chi(T)
# ==============================================================================

def run_period_scan(
    t_min: float = 2.0,
    t_max: float = 120.0,
    n_t: int = 40,
    j_center: float = 0.35,
    r_j: float = 0.15,
    dg_center: float = 1.40,
    r_dg: float = 0.45,
    gamma_bar: float = 1.0,
) -> dict:
    """Scan encircling period T in [t_min, t_max] to map the chirality curve chi(T)."""
    print("\n" + "=" * 80)
    print("PART 2: PERIOD SCAN & CHIRAL STATE TRANSFER MAPPING chi(T)")
    print("=" * 80)

    t_grid = np.linspace(t_min, t_max, n_t)
    chirality_p01_plus = np.zeros(n_t, dtype=float)
    chirality_p01_minus = np.zeros(n_t, dtype=float)
    final_p01_cw = np.zeros(n_t, dtype=float)
    final_p01_ccw = np.zeros(n_t, dtype=float)

    # Initial eigenmodes at t=0
    H_0 = make_loop_hamiltonian(10.0, "CCW", j_center, r_j, dg_center, r_dg, gamma_bar)(0.0)
    _, evecs_0 = np.linalg.eig(H_0)
    sort_idx = np.argsort(np.linalg.eigvals(H_0).real)
    phi_minus_0 = evecs_0[:, sort_idx[0]] / np.linalg.norm(evecs_0[:, sort_idx[0]])
    phi_plus_0 = evecs_0[:, sort_idx[1]] / np.linalg.norm(evecs_0[:, sort_idx[1]])

    for idx, T in enumerate(t_grid):
        tlist = np.linspace(0.0, T, 800)
        H_ccw = make_loop_hamiltonian(T, "CCW", j_center, r_j, dg_center, r_dg, gamma_bar)
        H_cw = make_loop_hamiltonian(T, "CW", j_center, r_j, dg_center, r_dg, gamma_bar)

        res_ccw_plus = simulate_timedependent_nonhermitian(H_ccw, phi_plus_0, tlist)
        res_cw_plus = simulate_timedependent_nonhermitian(H_cw, phi_plus_0, tlist)
        res_ccw_minus = simulate_timedependent_nonhermitian(H_ccw, phi_minus_0, tlist)
        res_cw_minus = simulate_timedependent_nonhermitian(H_cw, phi_minus_0, tlist)

        p01_ccw_p = res_ccw_plus["p01"][-1]
        p01_cw_p = res_cw_plus["p01"][-1]
        p01_ccw_m = res_ccw_minus["p01"][-1]
        p01_cw_m = res_cw_minus["p01"][-1]

        final_p01_ccw[idx] = p01_ccw_p
        final_p01_cw[idx] = p01_cw_p
        chirality_p01_plus[idx] = np.abs(p01_cw_p - p01_ccw_p)
        chirality_p01_minus[idx] = np.abs(p01_cw_m - p01_ccw_m)

    print(f"  Period scan completed: {n_t} points from T={t_min:.1f} to T={t_max:.1f}")
    print(f"  Max population chirality: {np.max(chirality_p01_plus):.4f} (at T={t_grid[np.argmax(chirality_p01_plus)]:.1f})")

    return {
        "t_grid": t_grid,
        "chirality_p01_plus": chirality_p01_plus,
        "chirality_p01_minus": chirality_p01_minus,
        "final_p01_cw": final_p01_cw,
        "final_p01_ccw": final_p01_ccw,
    }


# ==============================================================================
# FIGURE GENERATION (8 PUBLICATION FIGURES)
# ==============================================================================

def plot_figure_1_parameter_loop(j_center=0.35, r_j=0.15, dg_center=1.40, r_dg=0.45):
    """Figure 1: Parameter-Space Elliptical Loop in (J, Delta_gamma) with EP."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    theta = np.linspace(0, 2 * np.pi, 300)
    J_loop = j_center + r_j * np.cos(theta)
    DG_loop = dg_center + r_dg * np.sin(theta)

    # EP boundary line J = Delta_gamma / 4
    dg_line = np.linspace(0.5, 2.5, 100)
    j_line = dg_line / 4.0
    ax.plot(j_line, dg_line, "k--", linewidth=1.8, label=r"EP Line $J = \Delta\gamma / 4$")

    # Plot loop
    ax.plot(J_loop, DG_loop, "b-", linewidth=2.5, label=r"Parameter Loop $\mathcal{C}$")

    # Mark CCW and CW arrows
    ax.annotate("", xy=(j_center - 0.05, dg_center + r_dg), xytext=(j_center + 0.05, dg_center + r_dg),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2.2))
    ax.text(j_center, dg_center + r_dg + 0.06, "CCW", color="blue", fontweight="bold", ha="center", fontsize=9.5)

    ax.annotate("", xy=(j_center + 0.05, dg_center - r_dg), xytext=(j_center - 0.05, dg_center - r_dg),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.2))
    ax.text(j_center, dg_center - r_dg - 0.10, "CW", color="red", fontweight="bold", ha="center", fontsize=9.5)

    # Mark the enclosed EP (0.35, 1.40)
    ax.plot(j_center, dg_center, "r*", markersize=14, label=r"Enclosed EP2 $(0.35, 1.40)$")
    ax.plot(j_center + r_j, dg_center, "go", markersize=8, label=r"Start/Finish $(t=0, t=T)$")

    ax.set_title(r"Dynamical Parameter Loop Encircling the EP2 in $(J, \Delta\gamma)$ Space")
    ax.set_xlabel(r"Coupling Strength $J(t)$ [$\omega$]")
    ax.set_ylabel(r"Dissipation Asymmetry $\Delta\gamma(t)$ [$\omega$]")
    ax.set_xlim(0.10, 0.60)
    ax.set_ylim(0.70, 2.10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06b_parameter_loop.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Figure 1 saved to: {fig_path}")


def plot_figure_2_eigenvalue_trajectories(regime_data):
    """Figure 2: Instantaneous Complex Eigenvalue Trajectories in Complex Plane."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    spec_ccw = regime_data["mid"]["spec_ccw"]
    evals = spec_ccw["eigenvalues"]

    # Trajectories of both branches
    ax.plot(evals[:, 0].real, evals[:, 0].imag, "b-", linewidth=2.2, label=r"Instantaneous Mode $\lambda_-(t)$")
    ax.plot(evals[:, 1].real, evals[:, 1].imag, "r-", linewidth=2.2, label=r"Instantaneous Mode $\lambda_+(t)$")

    # Start points (t=0)
    ax.scatter([evals[0, 0].real], [evals[0, 0].imag], color="blue", marker="o", s=80, zorder=5, label=r"$\lambda_-(0)$")
    ax.scatter([evals[0, 1].real], [evals[0, 1].imag], color="red", marker="s", s=80, zorder=5, label=r"$\lambda_+(0)$")

    # End points (t=T) showing branch swap
    ax.scatter([evals[-1, 0].real], [evals[-1, 0].imag], color="red", marker="x", s=100, linewidth=2.5, zorder=6, label=r"$\lambda_-(T) \to \lambda_+(0)$")
    ax.scatter([evals[-1, 1].real], [evals[-1, 1].imag], color="blue", marker="+", s=100, linewidth=2.5, zorder=6, label=r"$\lambda_+(T) \to \lambda_-(0)$")

    ax.axhline(-0.5, color="gray", linestyle=":", alpha=0.6, label=r"$\mathrm{Im}(\lambda) = -\bar{\gamma}/2 = -0.5$")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6)

    ax.set_title(r"Instantaneous Eigenvalue Flow in the Complex Plane ($0 \leq t \leq T$)")
    ax.set_xlabel(r"Real Energy $\mathrm{Re}(\lambda_\pm(t))$ [$\omega$]")
    ax.set_ylabel(r"Decay Rate $\mathrm{Im}(\lambda_\pm(t))$ [$\omega$]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06b_eigenvalue_trajectories.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Figure 2 saved to: {fig_path}")


def plot_figure_3_eigenvalue_flow_time(regime_data):
    """Figure 3: Real and Imaginary Eigenvalue Components vs Normalized Time t/T."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    spec_ccw = regime_data["mid"]["spec_ccw"]
    tlist = regime_data["mid"]["tlist"]
    T = regime_data["mid"]["T"]
    tau = tlist / T
    evals = spec_ccw["eigenvalues"]

    # Panel A: Real Parts
    ax0 = axes[0]
    ax0.plot(tau, evals[:, 1].real, "r-", linewidth=2.2, label=r"$\mathrm{Re}(\lambda_+)$")
    ax0.plot(tau, evals[:, 0].real, "b--", linewidth=2.2, label=r"$\mathrm{Re}(\lambda_-)$")
    ax0.set_title(r"(a) Real Eigenvalue Dispersion $\mathrm{Re}(\lambda_\pm(t))$")
    ax0.set_xlabel(r"Normalized Time $t/T$")
    ax0.set_ylabel(r"$\mathrm{Re}(\lambda)$ [$\omega$]")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.5)

    # Panel B: Imaginary Parts
    ax1 = axes[1]
    ax1.plot(tau, evals[:, 1].imag, "r-", linewidth=2.2, label=r"$\mathrm{Im}(\lambda_+)$ (Amplified / Low-Loss)")
    ax1.plot(tau, evals[:, 0].imag, "b--", linewidth=2.2, label=r"$\mathrm{Im}(\lambda_-)$ (Damped / High-Loss)")
    ax1.axhline(-0.5, color="gray", linestyle=":", label=r"Uniform Loss $-\bar{\gamma}/2$")
    ax1.set_title(r"(b) Imaginary Decay Rates $\mathrm{Im}(\lambda_\pm(t))$")
    ax1.set_xlabel(r"Normalized Time $t/T$")
    ax1.set_ylabel(r"$\mathrm{Im}(\lambda)$ [$\omega$]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06b_eigenvalue_flow_time.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Figure 3 saved to: {fig_path}")


def plot_figure_4_normalized_populations(regime_data):
    """Figure 4: Subspace Populations P_01(t) and P_10(t) for CW vs CCW Encircling."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.0), sharex=True)

    reg_name = "slow"  # T=100 (Clean adiabatic mode selection)
    sim = regime_data[reg_name]["sim_data"]
    tau = sim[("CCW", "init_plus")]["tau"]

    # (0, 0): CCW, Initial |phi_+(0)>
    ax00 = axes[0, 0]
    ax00.plot(tau, sim[("CCW", "init_plus")]["p01"], "b-", label=r"$P_{01}(t)$")
    ax00.plot(tau, sim[("CCW", "init_plus")]["p10"], "r--", label=r"$P_{10}(t)$")
    ax00.set_title(r"(a) CCW [Init $|\phi_+(0)\rangle$] $\to$ Final State Selection")
    ax00.set_ylabel("Normalized Population")
    ax00.set_ylim(-0.02, 1.05)
    ax00.grid(True, alpha=0.3)
    ax00.legend(loc="center right", fontsize=8.5)

    # (0, 1): CW, Initial |phi_+(0)>
    ax01 = axes[0, 1]
    ax01.plot(tau, sim[("CW", "init_plus")]["p01"], "b-", label=r"$P_{01}(t)$")
    ax01.plot(tau, sim[("CW", "init_plus")]["p10"], "r--", label=r"$P_{10}(t)$")
    ax01.set_title(r"(b) CW [Init $|\phi_+(0)\rangle$] $\to$ Opposite Mode Selection")
    ax01.set_ylim(-0.02, 1.05)
    ax01.grid(True, alpha=0.3)
    ax01.legend(loc="center right", fontsize=8.5)

    # (1, 0): CCW, Initial |phi_-(0)>
    ax10 = axes[1, 0]
    ax10.plot(tau, sim[("CCW", "init_minus")]["p01"], "b-", label=r"$P_{01}(t)$")
    ax10.plot(tau, sim[("CCW", "init_minus")]["p10"], "r--", label=r"$P_{10}(t)$")
    ax10.set_title(r"(c) CCW [Init $|\phi_-(0)\rangle$] $\to$ Funnels to SAME Mode as (a)")
    ax10.set_xlabel(r"Normalized Time $t/T$")
    ax10.set_ylabel("Normalized Population")
    ax10.set_ylim(-0.02, 1.05)
    ax10.grid(True, alpha=0.3)
    ax10.legend(loc="center right", fontsize=8.5)

    # (1, 1): CW, Initial |phi_-(0)>
    ax11 = axes[1, 1]
    ax11.plot(tau, sim[("CW", "init_minus")]["p01"], "b-", label=r"$P_{01}(t)$")
    ax11.plot(tau, sim[("CW", "init_minus")]["p10"], "r--", label=r"$P_{10}(t)$")
    ax11.set_title(r"(d) CW [Init $|\phi_-(0)\rangle$] $\to$ Funnels to SAME Mode as (b)")
    ax11.set_xlabel(r"Normalized Time $t/T$")
    ax11.set_ylim(-0.02, 1.05)
    ax11.grid(True, alpha=0.3)
    ax11.legend(loc="center right", fontsize=8.5)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06b_normalized_populations.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Figure 4 saved to: {fig_path}")


def plot_figure_5_eigenstate_overlaps(regime_data):
    """Figure 5: Instantaneous Eigenstate Overlaps P_+(t) and P_-(t)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    reg_name = "slow"
    sim = regime_data[reg_name]["sim_data"]
    tau = sim[("CCW", "init_plus")]["tau"]

    # Panel A: CCW Encircling
    ax0 = axes[0]
    ax0.plot(tau, sim[("CCW", "init_plus")]["p_plus"], "r-", label=r"$P_+(t)$ (Init $+$)")
    ax0.plot(tau, sim[("CCW", "init_plus")]["p_minus"], "b--", label=r"$P_-(t)$ (Init $+$)")
    ax0.plot(tau, sim[("CCW", "init_minus")]["p_plus"], "m:", linewidth=2.0, label=r"$P_+(t)$ (Init $-$)")
    ax0.plot(tau, sim[("CCW", "init_minus")]["p_minus"], "c-.", linewidth=2.0, label=r"$P_-(t)$ (Init $-$)")
    ax0.set_title(r"(a) CCW Encircling: Convergence to Mode $+$")
    ax0.set_xlabel(r"Normalized Time $t/T$")
    ax0.set_ylabel(r"Eigenstate Overlap $|\langle\phi_\pm(t)|\tilde{\psi}(t)\rangle|^2$")
    ax0.set_ylim(-0.02, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="center right", fontsize=8.2, framealpha=0.9)

    # Panel B: CW Encircling
    ax1 = axes[1]
    ax1.plot(tau, sim[("CW", "init_plus")]["p_plus"], "r-", label=r"$P_+(t)$ (Init $+$)")
    ax1.plot(tau, sim[("CW", "init_plus")]["p_minus"], "b--", label=r"$P_-(t)$ (Init $+$)")
    ax1.plot(tau, sim[("CW", "init_minus")]["p_plus"], "m:", linewidth=2.0, label=r"$P_+(t)$ (Init $-$)")
    ax1.plot(tau, sim[("CW", "init_minus")]["p_minus"], "c-.", linewidth=2.0, label=r"$P_-(t)$ (Init $-$)")
    ax1.set_title(r"(b) CW Encircling: Convergence to Mode $-$")
    ax1.set_xlabel(r"Normalized Time $t/T$")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right", fontsize=8.2, framealpha=0.9)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06b_eigenstate_overlaps.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Figure 5 saved to: {fig_path}")


def plot_figure_6_survival_probability(regime_data):
    """Figure 6: Raw No-Jump Survival Probability P_survival(t) (Semilog)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)

    regimes = ["fast", "mid", "slow"]
    titles = [r"(a) Fast ($T=5$)", r"(b) Intermediate ($T=25$)", r"(c) Slow ($T=100$)"]

    for idx, reg in enumerate(regimes):
        ax = axes[idx]
        sim = regime_data[reg]["sim_data"]
        tau = sim[("CCW", "init_plus")]["tau"]

        ax.semilogy(tau, sim[("CCW", "init_plus")]["survival_probability"], "b-", label=r"CCW (Init $+$)")
        ax.semilogy(tau, sim[("CCW", "init_minus")]["survival_probability"], "b--", label=r"CCW (Init $-$)")
        ax.semilogy(tau, sim[("CW", "init_plus")]["survival_probability"], "r-", label=r"CW (Init $+$)")
        ax.semilogy(tau, sim[("CW", "init_minus")]["survival_probability"], "r--", label=r"CW (Init $-$)")

        ax.set_title(titles[idx])
        ax.set_xlabel(r"Normalized Time $t/T$")
        if idx == 0:
            ax.set_ylabel(r"Survival Probability $P_{\mathrm{surv}}(t) = \langle\psi(t)|\psi(t)\rangle$")
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=7.8, framealpha=0.9)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06b_survival_probability.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Figure 6 saved to: {fig_path}")


def plot_figure_7_fidelity_matrix(regime_data):
    """Figure 7: Final-State Fidelity Matrices for CW and CCW Traversal."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    f_ccw = regime_data["slow"]["fidelity_matrix_ccw"]
    f_cw = regime_data["slow"]["fidelity_matrix_cw"]

    labels_in = [r"Init $|\phi_+(0)\rangle$", r"Init $|\phi_-(0)\rangle$"]
    labels_out = [r"Final $|\phi_+(T)\rangle$", r"Final $|\phi_-(T)\rangle$"]

    # Panel A: CCW
    ax0 = axes[0]
    im0 = ax0.imshow(f_ccw, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im0, ax=ax0, label="Fidelity")
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xticklabels(labels_out)
    ax0.set_yticklabels(labels_in)
    ax0.set_title("(a) CCW Encircling ($T=100$)\nFunnels All Inputs to Mode $+$")
    for i in range(2):
        for j in range(2):
            ax0.text(j, i, f"{f_ccw[i, j]:.3f}", ha="center", va="center", color="white" if f_ccw[i, j] > 0.5 else "black", fontweight="bold")

    # Panel B: CW
    ax1 = axes[1]
    im1 = ax1.imshow(f_cw, cmap="Reds", vmin=0.0, vmax=1.0)
    fig.colorbar(im1, ax=ax1, label="Fidelity")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels_out)
    ax1.set_yticklabels(labels_in)
    ax1.set_title("(b) CW Encircling ($T=100$)\nFunnels All Inputs to Mode $-$")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{f_cw[i, j]:.3f}", ha="center", va="center", color="white" if f_cw[i, j] > 0.5 else "black", fontweight="bold")

    fig.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06b_fidelity_matrix.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Figure 7 saved to: {fig_path}")


def plot_figure_8_chirality_vs_period(period_data):
    """Figure 8: Chirality Metric vs Encircling Period T."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    t_grid = period_data["t_grid"]
    chi_p = period_data["chirality_p01_plus"]
    chi_m = period_data["chirality_p01_minus"]
    p01_cw = period_data["final_p01_cw"]
    p01_ccw = period_data["final_p01_ccw"]

    # Panel A: Direction-Dependent Final Population P01(T)
    ax0 = axes[0]
    ax0.plot(t_grid, p01_ccw, "b-o", markersize=4.5, label=r"CCW Traversal: $P_{01}^{\mathrm{CCW}}(T)$")
    ax0.plot(t_grid, p01_cw, "r-s", markersize=4.5, label=r"CW Traversal: $P_{01}^{\mathrm{CW}}(T)$")
    ax0.set_title(r"(a) Final Population $P_{01}(T)$ vs Encircling Period $T$")
    ax0.set_xlabel(r"Encircling Period $T$ [$\omega^{-1}$]")
    ax0.set_ylabel(r"Final State Population $P_{01}(T)$")
    ax0.set_ylim(-0.02, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="center right", fontsize=8.5, framealpha=0.9)

    # Panel B: Quantitative Chirality Metric chi(T)
    ax1 = axes[1]
    ax1.plot(t_grid, chi_p, "k-", linewidth=2.4, label=r"Chirality $\chi(T) = |P_{01}^{\mathrm{CW}} - P_{01}^{\mathrm{CCW}}|$ (Init $+$)")
    ax1.plot(t_grid, chi_m, "g--", linewidth=2.0, label=r"Chirality $\chi(T)$ (Init $-$)")

    # Mark Dynamical Regimes
    ax1.axvspan(0, 10, color="gray", alpha=0.15, label="Fast / Diabatic Regime")
    ax1.axvspan(10, 45, color="orange", alpha=0.15, label="Intermediate / Crossover")
    ax1.axvspan(45, 120, color="green", alpha=0.15, label="Slow / Chiral State Transfer")

    ax1.set_title(r"(b) Chirality Metric $\chi(T)$ vs Encircling Period $T$")
    ax1.set_xlabel(r"Encircling Period $T$ [$\omega^{-1}$]")
    ax1.set_ylabel(r"Chirality Metric $\chi(T)$")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06b_chirality_vs_period.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Figure 8 saved to: {fig_path}")


# ==============================================================================
# MAIN EXECUTION WORKFLOW
# ==============================================================================

def main(force_recompute: bool = False):
    print("=" * 80)
    print("EXPERIMENT 06b: DYNAMICAL EP ENCYCLING & CHIRAL STATE TRANSFER")
    print("=" * 80)

    # 1. Parameter loop geometry
    j_center = 0.35
    r_j = 0.15
    dg_center = 1.40
    r_dg = 0.45
    gamma_bar = 1.0

    data_path = DATA_DIR / "experiment_06b_dynamic_ep_encircling.npz"

    # Always compute regime results (fast: ~1-2 seconds) for high-res trajectory plotting
    regime_results = run_regime_simulations(
        t_periods={"fast": 5.0, "mid": 25.0, "slow": 100.0},
        n_steps=1500,
        j_center=j_center,
        r_j=r_j,
        dg_center=dg_center,
        r_dg=r_dg,
        gamma_bar=gamma_bar,
    )

    # Load period scan from existing dataset if available, else run scan
    if data_path.exists() and not force_recompute:
        print(f"\nLoading existing period scan dataset from: {data_path}")
        loaded = np.load(data_path)
        period_results = {
            "t_grid": loaded["period_grid"],
            "chirality_p01_plus": loaded["chirality_p01_plus"],
            "chirality_p01_minus": loaded["chirality_p01_minus"],
            "final_p01_cw": loaded["final_p01_cw"],
            "final_p01_ccw": loaded["final_p01_ccw"],
        }
    else:
        period_results = run_period_scan(
            t_min=2.0,
            t_max=120.0,
            n_t=40,
            j_center=j_center,
            r_j=r_j,
            dg_center=dg_center,
            r_dg=r_dg,
            gamma_bar=gamma_bar,
        )

        npz_data = {
            "j_center": j_center,
            "r_j": r_j,
            "dg_center": dg_center,
            "r_dg": r_dg,
            "gamma_bar": gamma_bar,
            "t_fast_tau": regime_results["fast"]["sim_data"][("CCW", "init_plus")]["tau"],
            "t_fast_p01_ccw": regime_results["fast"]["sim_data"][("CCW", "init_plus")]["p01"],
            "t_fast_p01_cw": regime_results["fast"]["sim_data"][("CW", "init_plus")]["p01"],
            "t_mid_tau": regime_results["mid"]["sim_data"][("CCW", "init_plus")]["tau"],
            "t_mid_p01_ccw": regime_results["mid"]["sim_data"][("CCW", "init_plus")]["p01"],
            "t_mid_p01_cw": regime_results["mid"]["sim_data"][("CW", "init_plus")]["p01"],
            "t_slow_tau": regime_results["slow"]["sim_data"][("CCW", "init_plus")]["tau"],
            "t_slow_p01_ccw_plus": regime_results["slow"]["sim_data"][("CCW", "init_plus")]["p01"],
            "t_slow_p01_cw_plus": regime_results["slow"]["sim_data"][("CW", "init_plus")]["p01"],
            "t_slow_p01_ccw_minus": regime_results["slow"]["sim_data"][("CCW", "init_minus")]["p01"],
            "t_slow_p01_cw_minus": regime_results["slow"]["sim_data"][("CW", "init_minus")]["p01"],
            "fidelity_matrix_ccw_slow": regime_results["slow"]["fidelity_matrix_ccw"],
            "fidelity_matrix_cw_slow": regime_results["slow"]["fidelity_matrix_cw"],
            "period_grid": period_results["t_grid"],
            "chirality_p01_plus": period_results["chirality_p01_plus"],
            "chirality_p01_minus": period_results["chirality_p01_minus"],
            "final_p01_cw": period_results["final_p01_cw"],
            "final_p01_ccw": period_results["final_p01_ccw"],
        }
        np.savez_compressed(data_path, **npz_data)
        print(f"\nSaved numerical dataset to: {data_path}")

    # Generate Figures (8 Publication Figures)
    print("\n" + "=" * 80)
    print("PART 3: GENERATING PUBLICATION FIGURES (8 FIGURES)")
    print("=" * 80)
    plot_figure_1_parameter_loop(j_center, r_j, dg_center, r_dg)
    plot_figure_2_eigenvalue_trajectories(regime_results)
    plot_figure_3_eigenvalue_flow_time(regime_results)
    plot_figure_4_normalized_populations(regime_results)
    plot_figure_5_eigenstate_overlaps(regime_results)
    plot_figure_6_survival_probability(regime_results)
    plot_figure_7_fidelity_matrix(regime_results)
    plot_figure_8_chirality_vs_period(period_results)

    # Print summary metrics
    print("\n" + "=" * 80)
    print("NUMERICAL RESULTS SUMMARY:")
    print("=" * 80)
    print(f"  FAST Chirality (T = 5.0):   {regime_results['fast']['chirality']:.4f}")
    print(f"  MID Chirality  (T = 25.0):  {regime_results['mid']['chirality']:.4f}")
    print(f"  SLOW Chirality (T = 100.0): {regime_results['slow']['chirality']:.4f}")
    max_idx = np.argmax(period_results["chirality_p01_plus"])
    print(f"  Maximum Chirality:          {period_results['chirality_p01_plus'][max_idx]:.4f}")
    print(f"  Period at Max Chirality:    T = {period_results['t_grid'][max_idx]:.1f}")
    print("=" * 80)
    print("EXPERIMENT 06b COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
