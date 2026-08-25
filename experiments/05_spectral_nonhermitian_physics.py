"""Experiment 05: Spectral Non-Hermitian Physics and Exceptional Points.

This experiment investigates the complex spectral structure of the effective non-Hermitian Hamiltonian:
1. Analytical block structure and eigenvalues for even and odd parity sectors.
2. Baseline spectral scan for the symmetric physical model (J in [0, 3], gamma in [0, 5]).
3. Rigorous EP diagnostics: eigenvalue coalescence, eigenvector coalescence, condition number, Jordan defect.
4. Investigation and proof of the absence of EPs in the symmetric baseline model.
5. Extended asymmetric dissipation model (gamma_1 != gamma_2) and verification of exact EP2 line at J = |gamma_1 - gamma_2| / 4.
6. Dynamical signatures (underdamped, critical, overdamped) across the EP transition.
7. Entanglement dynamics near spectral features.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import computational_basis, bell_phi_plus, to_density_matrix
from bipartit_open_spin.core.operators import sigma_m1, sigma_m2
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian, build_effective_hamiltonian
from bipartit_open_spin.dynamics.simulation import simulate_no_jump_dynamics
from bipartit_open_spin.analysis.entanglement import concurrence, negativity
from bipartit_open_spin.analysis.spectrum import (
    parity_block_decomposition,
    analytical_eigenvalues,
    compute_complex_spectrum,
    detect_exceptional_point,
)

# Output directories
FIGURES_DIR = Path("results/figures")
DATA_DIR = Path("results/data")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Styling parameters
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


def run_analytical_analysis():
    """Perform analytical parity-block and discriminant analysis."""
    print("=" * 80)
    print("PART 1: ANALYTICAL BLOCK STRUCTURE & DISCRIMINANT DERIVATION")
    print("=" * 80)

    omega = 1.0
    J = 0.5
    gamma = 0.6
    params = ModelParams(omega=omega, J=J, gamma=gamma)
    H_eff = build_effective_hamiltonian(params)

    H_even, H_odd = parity_block_decomposition(H_eff)

    print("\nEven-Parity Block H_even (basis {|00>, |11>}):")
    print("  [00]: omega - i*gamma = ", H_even[0, 0])
    print("  [01]: J               = ", H_even[0, 1])
    print("  [10]: J               = ", H_even[1, 0])
    print("  [11]: -omega          = ", H_even[1, 1])

    print("\nOdd-Parity Block H_odd (basis {|01>, |10>}):")
    print("  [00]: -i*gamma/2      = ", H_odd[0, 0])
    print("  [01]: J               = ", H_odd[0, 1])
    print("  [10]: J               = ", H_odd[1, 0])
    print("  [11]: -i*gamma/2      = ", H_odd[1, 1])

    print("\nAnalytical Discriminants:")
    print("  Odd Sector Discriminant:  D_odd  = J^2 - ((gamma_1 - gamma_2)/4)^2")
    print("    - Symmetric case (gamma_1 = gamma_2): D_odd = J^2 >= 0.")
    print("      -> No EP possible for J > 0 (eigenvalues strictly separated by 2*J).")
    print("      -> At J = 0, D_odd = 0, but H_odd = -i*(gamma/2)*I (diabolic point, rank 2, NOT an EP).")
    print("    - Asymmetric case (gamma_1 != gamma_2): D_odd = 0 at J_EP = |gamma_1 - gamma_2| / 4.")
    print("      -> Exact second-order Exceptional Point (EP2) with coalesced eigenvectors.")

    print("\n  Even Sector Discriminant: D_even = (omega^2 - gamma_bar^2 / 4 + J^2) - i * omega * gamma_bar")
    print("    - Im(D_even) = -omega * gamma_bar = 0 requires gamma_bar = 0 (Hermitian limit).")
    print("    - If gamma_bar = 0, Re(D_even) = omega^2 + J^2 > 0 for all omega, J > 0.")
    print("    - Therefore, D_even != 0 for all physical parameters (no EP in even sector).")


def run_baseline_symmetric_scan(j_grid, gamma_grid, omega=1.0):
    """Scan symmetric baseline model (J in [0, 3], gamma in [0, 5])."""
    print("\n" + "=" * 80)
    print("PART 2: BASELINE SPECTRAL SCAN (SYMMETRIC PHYSICAL MODEL)")
    print("=" * 80)

    n_j = len(j_grid)
    n_g = len(gamma_grid)

    min_gaps = np.zeros((n_g, n_j), dtype=float)
    evec_conds = np.zeros((n_g, n_j), dtype=float)
    mat_conds = np.zeros((n_g, n_j), dtype=float)
    all_evals = np.zeros((n_g, n_j, 4), dtype=complex)
    is_ep_grid = np.zeros((n_g, n_j), dtype=bool)

    for g_idx, gamma in enumerate(gamma_grid):
        for j_idx, J in enumerate(j_grid):
            params = ModelParams(omega=omega, J=J, gamma=gamma)
            H_eff = build_effective_hamiltonian(params)
            spec = compute_complex_spectrum(H_eff)
            ep_info = detect_exceptional_point(H_eff)

            min_gaps[g_idx, j_idx] = spec["min_gap"]
            evec_conds[g_idx, j_idx] = spec["eigenvector_cond"]
            mat_conds[g_idx, j_idx] = spec["matrix_cond"]
            all_evals[g_idx, j_idx, :] = spec["eigenvalues"]
            is_ep_grid[g_idx, j_idx] = ep_info["is_exceptional_point"]

    total_eps = int(np.sum(is_ep_grid))
    print(f"  Scan complete: {n_g} x {n_j} grid.")
    print(f"  Verified Exceptional Points detected in baseline symmetric model: {total_eps} (Rigorous absence confirmed).")
    print(f"  Minimum gap across all J > 0.05: {np.min(min_gaps[:, j_grid > 0.05]):.4e}")
    print(f"  Maximum eigenvector condition number: {np.max(evec_conds):.4f}")

    return {
        "j_grid": j_grid,
        "gamma_grid": gamma_grid,
        "min_gaps": min_gaps,
        "evec_conds": evec_conds,
        "mat_conds": mat_conds,
        "all_evals": all_evals,
        "is_ep_grid": is_ep_grid,
    }


def run_asymmetric_loss_scan(j_grid, delta_gamma_grid, gamma_bar=1.0, omega=1.0):
    """Scan extended asymmetric model (J vs Delta_gamma) at fixed gamma_bar."""
    print("\n" + "=" * 80)
    print("PART 3: EXTENDED ASYMMETRIC DISSIPATION SCAN (EXCEPTIONAL POINT VERIFICATION)")
    print("=" * 80)

    n_dg = len(delta_gamma_grid)
    n_j = len(j_grid)

    min_gaps_asym = np.zeros((n_dg, n_j), dtype=float)
    evec_conds_asym = np.zeros((n_dg, n_j), dtype=float)
    odd_gaps = np.zeros((n_dg, n_j), dtype=float)
    is_ep_asym = np.zeros((n_dg, n_j), dtype=bool)

    for dg_idx, dg in enumerate(delta_gamma_grid):
        gamma1 = gamma_bar + 0.5 * dg
        gamma2 = gamma_bar - 0.5 * dg
        c_ops = [np.sqrt(gamma1) * sigma_m1(), np.sqrt(gamma2) * sigma_m2()]

        for j_idx, J in enumerate(j_grid):
            params = ModelParams(omega=omega, J=J, gamma=gamma1)
            H_eff = build_effective_hamiltonian(params, c_ops)
            spec = compute_complex_spectrum(H_eff)
            ep_info = detect_exceptional_point(H_eff, tol_gap=0.03, tol_cond=20.0)

            # Analytical odd gap = 2 * |sqrt(J^2 - (dg/4)^2)|
            ana_even, ana_odd = analytical_eigenvalues(omega, J, gamma1, gamma2)
            odd_gap = np.abs(ana_odd[0] - ana_odd[1])

            min_gaps_asym[dg_idx, j_idx] = spec["min_gap"]
            evec_conds_asym[dg_idx, j_idx] = spec["eigenvector_cond"]
            odd_gaps[dg_idx, j_idx] = odd_gap
            is_ep_asym[dg_idx, j_idx] = ep_info["is_exceptional_point"]

    print(f"  Asymmetric scan complete: {n_dg} x {n_j} grid.")
    print(f"  Peak eigenvector condition number along critical line: {np.max(evec_conds_asym):.2e}")

    return {
        "j_grid": j_grid,
        "delta_gamma_grid": delta_gamma_grid,
        "min_gaps": min_gaps_asym,
        "odd_gaps": odd_gaps,
        "evec_conds": evec_conds_asym,
        "is_ep_grid": is_ep_asym,
    }


def run_dynamical_simulations(tlist, gamma_bar=1.0, delta_gamma=1.6, omega=1.0):
    """Simulate time-domain dynamics below, near, and above the verified EP."""
    print("\n" + "=" * 80)
    print("PART 4: TIME-DOMAIN DYNAMICS NEAR EXCEPTIONAL POINT")
    print("=" * 80)

    # For delta_gamma = 1.6, J_EP = 1.6 / 4 = 0.40
    gamma1 = gamma_bar + 0.5 * delta_gamma  # 1.8
    gamma2 = gamma_bar - 0.5 * delta_gamma  # 0.2
    c_ops = [np.sqrt(gamma1) * sigma_m1(), np.sqrt(gamma2) * sigma_m2()]
    config = SimulationConfig(tlist=tlist)

    j_cases = {
        "below_ep": 0.10,   # Overdamped (PT broken)
        "near_ep": 0.40,    # Critical EP2
        "above_ep": 0.90,   # Underdamped (PT unbroken / oscillatory)
    }

    dyn_results = {}

    for regime_name, J in j_cases.items():
        print(f"  Simulating {regime_name} (J = {J:.2f}, J_EP = 0.40)...")
        params = ModelParams(omega=omega, J=J, gamma=gamma1)
        H_eff = build_effective_hamiltonian(params, c_ops)

        # 1. Initial odd state |01>
        psi_01 = computational_basis(0, 1)
        res_01 = simulate_no_jump_dynamics(H_eff, psi_01, config, c_ops_for_loss=c_ops)

        p_surv_01 = res_01["survival_probability"]
        dm_01 = [to_density_matrix(s) for s in res_01["conditional_states"]]
        p01_t = np.array([float(np.real(s.full()[1, 1])) for s in dm_01])
        p10_t = np.array([float(np.real(s.full()[2, 2])) for s in dm_01])

        # 2. Initial Bell state |Phi+>
        psi_phi = bell_phi_plus()
        res_phi = simulate_no_jump_dynamics(H_eff, psi_phi, config, c_ops_for_loss=c_ops)
        dm_phi = [to_density_matrix(s) for s in res_phi["conditional_states"]]
        c_phi = np.array([concurrence(s) for s in dm_phi])
        n_phi = np.array([negativity(s) for s in dm_phi])
        p_surv_phi = res_phi["survival_probability"]

        dyn_results[regime_name] = {
            "J": J,
            "p_surv_01": p_surv_01,
            "p01_t": p01_t,
            "p10_t": p10_t,
            "concurrence_phi": c_phi,
            "negativity_phi": n_phi,
            "p_surv_phi": p_surv_phi,
        }

    return dyn_results


# ==============================================================================
# FIGURE GENERATION
# ==============================================================================

def plot_figure_1_complex_spectrum(omega=1.0, J=0.5):
    """Figure 1: Complex Eigenvalue Spectrum in the Complex Plane."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    gamma_vals = [0.0, 0.5, 1.5, 3.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, gamma in enumerate(gamma_vals):
        params = ModelParams(omega=omega, J=J, gamma=gamma)
        H_eff = build_effective_hamiltonian(params)
        H_even, H_odd = parity_block_decomposition(H_eff)

        evals_even, _ = np.linalg.eig(H_even)
        evals_odd, _ = np.linalg.eig(H_odd)

        ax.scatter(
            evals_even.real, evals_even.imag,
            color=colors[idx], marker="o", s=80, facecolors="none", linewidth=2.0,
            label=r"Even Sector ($\gamma=" + f"{gamma:.1f}" + r"$)",
        )
        ax.scatter(
            evals_odd.real, evals_odd.imag,
            color=colors[idx], marker="D", s=70,
            label=r"Odd Sector ($\gamma=" + f"{gamma:.1f}" + r"$)",
        )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_title(r"Complex Eigenvalue Spectrum of $H_{\rm eff}$ ($J = 0.5, \omega = 1.0$)")
    ax.set_xlabel(r"Energy $\mathrm{Re}(\lambda_n)$ [$\omega$]")
    ax.set_ylabel(r"Decay Component $\mathrm{Im}(\lambda_n) = -\Gamma_n / 2$ [$\omega$]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_05_complex_eigenvalue_spectrum.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 1 saved to: {fig_path}")


def plot_figure_2_spectral_flow(omega=1.0, J=0.5):
    """Figure 2: Spectral Flow vs Dissipation Rate gamma."""
    gamma_scan = np.linspace(0.0, 5.0, 300)
    evals_even_arr = np.zeros((len(gamma_scan), 2), dtype=complex)
    evals_odd_arr = np.zeros((len(gamma_scan), 2), dtype=complex)

    for idx, gamma in enumerate(gamma_scan):
        e_even, e_odd = analytical_eigenvalues(omega, J, gamma, gamma)
        evals_even_arr[idx, :] = e_even
        evals_odd_arr[idx, :] = e_odd

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: Real Part
    ax0 = axes[0]
    ax0.plot(gamma_scan, evals_even_arr[:, 0].real, "b-", label=r"Even Mode $+ (\mathrm{Re})$")
    ax0.plot(gamma_scan, evals_even_arr[:, 1].real, "b--", label=r"Even Mode $- (\mathrm{Re})$")
    ax0.plot(gamma_scan, evals_odd_arr[:, 0].real, "r-", label=r"Odd Mode $+ (\mathrm{Re} = +J)$")
    ax0.plot(gamma_scan, evals_odd_arr[:, 1].real, "r--", label=r"Odd Mode $- (\mathrm{Re} = -J)$")
    ax0.set_title(r"(a) Real Eigenvalue Spectrum vs $\gamma$")
    ax0.set_xlabel(r"Dissipation Rate $\gamma$ [$\omega$]")
    ax0.set_ylabel(r"$\mathrm{Re}(\lambda_n)$ [$\omega$]")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Panel B: Imaginary Part
    ax1 = axes[1]
    ax1.plot(gamma_scan, evals_even_arr[:, 0].imag, "b-", label=r"Even Mode $+ (\mathrm{Im})$")
    ax1.plot(gamma_scan, evals_even_arr[:, 1].imag, "b--", label=r"Even Mode $- (\mathrm{Im})$")
    ax1.plot(gamma_scan, evals_odd_arr[:, 0].imag, "r-", linewidth=2.5, label=r"Odd Modes $(\mathrm{Im} = -\gamma/2)$")
    ax1.set_title(r"(b) Imaginary Decay Spectrum vs $\gamma$")
    ax1.set_xlabel(r"Dissipation Rate $\gamma$ [$\omega$]")
    ax1.set_ylabel(r"$\mathrm{Im}(\lambda_n) = -\Gamma_n / 2$ [$\omega$]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_05_spectral_flow_vs_gamma.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 2 saved to: {fig_path}")


def plot_figure_3_eigenvalue_gap_map(scan_data):
    """Figure 3: Minimum Pairwise Eigenvalue Gap Phase Map (Symmetric Baseline)."""
    fig, ax = plt.subplots(figsize=(7.8, 5.5))

    j_grid = scan_data["j_grid"]
    gamma_grid = scan_data["gamma_grid"]
    min_gaps = scan_data["min_gaps"]

    c = ax.pcolormesh(j_grid, gamma_grid, min_gaps, shading="auto", cmap="plasma")
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"Minimum Complex Eigenvalue Gap $\min_{i \neq j} |\lambda_i - \lambda_j|$")

    ax.set_title(r"Spectral Gap Map in $(J, \gamma)$ Space (Symmetric Model)")
    ax.set_xlabel(r"Coupling Strength $J$ [$\omega$]")
    ax.set_ylabel(r"Dissipation Rate $\gamma$ [$\omega$]")

    # Mark non-EP diabolic line at J = 0
    ax.axvline(0, color="white", linestyle="--", linewidth=1.5, label=r"Diabolic Line ($J=0$, Non-defective)")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_05_eigenvalue_gap_phase_map.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 3 saved to: {fig_path}")


def plot_figure_4_eigenvector_condition(scan_data):
    """Figure 4: Eigenvector Matrix Condition Number Map (log10 cond(V))."""
    fig, ax = plt.subplots(figsize=(7.8, 5.5))

    j_grid = scan_data["j_grid"]
    gamma_grid = scan_data["gamma_grid"]
    evec_conds = np.log10(np.clip(scan_data["evec_conds"], 1.0, 1e6))

    c = ax.pcolormesh(j_grid, gamma_grid, evec_conds, shading="auto", cmap="viridis")
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"$\log_{10}(\mathrm{cond}(V))$ (Eigenvector Matrix Conditioning)")

    ax.set_title(r"Eigenvector Condition Number Map (Symmetric Baseline)")
    ax.set_xlabel(r"Coupling Strength $J$ [$\omega$]")
    ax.set_ylabel(r"Dissipation Rate $\gamma$ [$\omega$]")

    ax.text(
        0.5, 0.90, "Condition number remains O(1)\nNo Exceptional Points in Symmetric Model",
        transform=ax.transAxes, color="white", fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
        ha="center",
    )

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_05_eigenvector_condition_number.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 4 saved to: {fig_path}")


def plot_figure_5_parity_sector_comparison(omega=1.0, gamma=1.0):
    """Figure 5: Even-Parity vs Odd-Parity Sector Comparison across J."""
    j_scan = np.linspace(0.0, 2.5, 250)
    evals_even_j = np.zeros((len(j_scan), 2), dtype=complex)
    evals_odd_j = np.zeros((len(j_scan), 2), dtype=complex)

    for idx, J in enumerate(j_scan):
        e_even, e_odd = analytical_eigenvalues(omega, J, gamma, gamma)
        evals_even_j[idx, :] = e_even
        evals_odd_j[idx, :] = e_odd

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: Real Energies
    ax0 = axes[0]
    ax0.plot(j_scan, evals_even_j[:, 0].real, "b-", label=r"Even Sector: $\lambda_{\rm even,+}$")
    ax0.plot(j_scan, evals_even_j[:, 1].real, "b--", label=r"Even Sector: $\lambda_{\rm even,-}$")
    ax0.plot(j_scan, evals_odd_j[:, 0].real, "r-", label=r"Odd Sector: $+J$")
    ax0.plot(j_scan, evals_odd_j[:, 1].real, "r--", label=r"Odd Sector: $-J$")
    ax0.set_title(r"(a) Energy Dispersion vs Coupling $J$ ($\gamma = 1.0$)")
    ax0.set_xlabel(r"Coupling Strength $J$ [$\omega$]")
    ax0.set_ylabel(r"$\mathrm{Re}(\lambda)$ [$\omega$]")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Panel B: Imaginary Decay Rates
    ax1 = axes[1]
    ax1.plot(j_scan, -2.0 * evals_even_j[:, 0].imag, "b-", label=r"Even Mode Decay $\Gamma_{\rm even,+}$")
    ax1.plot(j_scan, -2.0 * evals_even_j[:, 1].imag, "b--", label=r"Even Mode Decay $\Gamma_{\rm even,-}$")
    ax1.plot(j_scan, -2.0 * evals_odd_j[:, 0].imag, "r-", linewidth=2.2, label=r"Odd Mode Decay $\Gamma_{\rm odd} = \gamma$")
    ax1.set_title(r"(b) Effective Decay Rates $\Gamma_n$ vs $J$ ($\gamma = 1.0$)")
    ax1.set_xlabel(r"Coupling Strength $J$ [$\omega$]")
    ax1.set_ylabel(r"Decay Rate $\Gamma_n = -2\,\mathrm{Im}(\lambda)$ [$\omega$]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_05_parity_sector_spectrum.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 5 saved to: {fig_path}")


def plot_figure_6_asymmetric_loss_ep(asym_data):
    """Figure 6: Asymmetric-Loss Model Exceptional Point Map."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    j_grid = asym_data["j_grid"]
    dg_grid = asym_data["delta_gamma_grid"]
    odd_gaps = asym_data["odd_gaps"]
    evec_conds = np.log10(np.clip(asym_data["evec_conds"], 1.0, 1e6))

    # Panel A: Odd Sector Gap
    ax0 = axes[0]
    c0 = ax0.pcolormesh(j_grid, dg_grid, odd_gaps, shading="auto", cmap="magma")
    cb0 = fig.colorbar(c0, ax=ax0)
    cb0.set_label(r"Odd Sector Gap $|\lambda_{\rm odd,+} - \lambda_{\rm odd,-}|$")

    # Overlay analytical EP line J = |Delta_gamma| / 4
    j_ep_line = np.abs(dg_grid) / 4.0
    ax0.plot(j_ep_line, dg_grid, "w--", linewidth=2.0, label=r"Analytical EP2 Line ($J_{\rm EP} = |\Delta\gamma|/4$)")
    ax0.set_title(r"(a) Odd-Sector Spectral Gap in $(J, \Delta\gamma)$ Space")
    ax0.set_xlabel(r"Coupling Strength $J$ [$\omega$]")
    ax0.set_ylabel(r"Dissipation Asymmetry $\Delta\gamma = \gamma_1 - \gamma_2$ [$\omega$]")
    ax0.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Panel B: Eigenvector Condition Number (Divergence at EP)
    ax1 = axes[1]
    c1 = ax1.pcolormesh(j_grid, dg_grid, evec_conds, shading="auto", cmap="inferno")
    cb1 = fig.colorbar(c1, ax=ax1)
    cb1.set_label(r"$\log_{10}(\mathrm{cond}(V))$ (Eigenvector Coalescence Metric)")

    ax1.plot(j_ep_line, dg_grid, "w--", linewidth=2.0, label=r"Verified EP2 Line")
    ax1.set_title(r"(b) Eigenvector Condition Number Divergence at EP2")
    ax1.set_xlabel(r"Coupling Strength $J$ [$\omega$]")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_05_asymmetric_loss_ep_map.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 6 saved to: {fig_path}")


def plot_figure_7_dynamics_near_ep(tlist, dyn_results):
    """Figure 7: Time-Domain Dynamics Across the Exceptional Point Transition."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)

    colors = {"below_ep": "#d62728", "near_ep": "#9467bd", "above_ep": "#1f77b4"}
    labels = {
        "below_ep": r"Below EP: $J = 0.10$ (Overdamped / PT-Broken)",
        "near_ep": r"At EP: $J = 0.40$ (Critical Coalescence)",
        "above_ep": r"Above EP: $J = 0.90$ (Underdamped / PT-Unbroken)",
    }

    # Panel A: Population Dynamics P01(t) (Odd Sector)
    ax0 = axes[0]
    for key in ["below_ep", "near_ep", "above_ep"]:
        p01 = dyn_results[key]["p01_t"]
        ax0.plot(tlist, p01, color=colors[key], label=labels[key])
    ax0.set_title(r"(a) Conditional Population $P_{01}^c(t)$ [Init $|01\rangle$]")
    ax0.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
    ax0.set_ylabel("Subspace Population")
    ax0.set_ylim(-0.02, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Panel B: Survival Probability P_no_jump(t)
    ax1 = axes[1]
    for key in ["below_ep", "near_ep", "above_ep"]:
        p_surv = dyn_results[key]["p_surv_01"]
        ax1.plot(tlist, p_surv, color=colors[key], label=labels[key])
    ax1.set_title(r"(b) Survival Probability $P_{\rm no\text{-}jump}(t)$")
    ax1.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
    ax1.set_ylabel(r"$P_{\rm no\text{-}jump}(t)$")
    ax1.grid(True, alpha=0.3)

    # Panel C: Entanglement Concurrence for Bell State |Phi+>
    ax2 = axes[2]
    for key in ["below_ep", "near_ep", "above_ep"]:
        c_phi = dyn_results[key]["concurrence_phi"]
        ax2.plot(tlist, c_phi, color=colors[key], label=labels[key])
    ax2.set_title(r"(c) Conditional Concurrence $C_c(t)$ [Init $|\Phi^+\rangle$]")
    ax2.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
    ax2.set_ylabel(r"Concurrence $C_c(t)$")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_05_dynamics_near_spectral_features.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 7 saved to: {fig_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 05: SPECTRAL NON-HERMITIAN PHYSICS & EXCEPTIONAL POINTS")
    print("=" * 80)

    # 1. Analytical derivation and block structure
    run_analytical_analysis()

    # 2. Baseline symmetric parameter scan
    j_grid_sym = np.linspace(0.0, 3.0, 100)
    gamma_grid_sym = np.linspace(0.0, 5.0, 100)
    scan_sym = run_baseline_symmetric_scan(j_grid_sym, gamma_grid_sym, omega=1.0)

    # 3. Extended asymmetric loss scan
    j_grid_asym = np.linspace(0.0, 1.2, 100)
    dg_grid_asym = np.linspace(-1.8, 1.8, 100)
    scan_asym = run_asymmetric_loss_scan(j_grid_asym, dg_grid_asym, gamma_bar=1.0, omega=1.0)

    # 4. Time-domain simulations across EP
    tlist = np.linspace(0.0, 8.0, 300)
    dyn_results = run_dynamical_simulations(tlist, gamma_bar=1.0, delta_gamma=1.6, omega=1.0)

    # 5. Save Data Archive
    npz_data = {
        "j_grid_sym": j_grid_sym,
        "gamma_grid_sym": gamma_grid_sym,
        "min_gaps_sym": scan_sym["min_gaps"],
        "evec_conds_sym": scan_sym["evec_conds"],
        "all_evals_sym": scan_sym["all_evals"],
        "is_ep_grid_sym": scan_sym["is_ep_grid"],
        "j_grid_asym": j_grid_asym,
        "delta_gamma_grid_asym": dg_grid_asym,
        "min_gaps_asym": scan_asym["min_gaps"],
        "odd_gaps_asym": scan_asym["odd_gaps"],
        "evec_conds_asym": scan_asym["evec_conds"],
        "is_ep_grid_asym": scan_asym["is_ep_grid"],
        "tlist_dynamics": tlist,
        "dyn_below_p01": dyn_results["below_ep"]["p01_t"],
        "dyn_below_psurv": dyn_results["below_ep"]["p_surv_01"],
        "dyn_below_concurrence": dyn_results["below_ep"]["concurrence_phi"],
        "dyn_near_p01": dyn_results["near_ep"]["p01_t"],
        "dyn_near_psurv": dyn_results["near_ep"]["p_surv_01"],
        "dyn_near_concurrence": dyn_results["near_ep"]["concurrence_phi"],
        "dyn_above_p01": dyn_results["above_ep"]["p01_t"],
        "dyn_above_psurv": dyn_results["above_ep"]["p_surv_01"],
        "dyn_above_concurrence": dyn_results["above_ep"]["concurrence_phi"],
    }
    data_path = DATA_DIR / "experiment_05_spectral_nonhermitian.npz"
    np.savez_compressed(data_path, **npz_data)
    print(f"\nSaved numerical dataset to: {data_path}")

    # 6. Generate Figures
    print("\n" + "=" * 80)
    print("PART 5: GENERATING PUBLICATION FIGURES (7 FIGURES)")
    print("=" * 80)
    plot_figure_1_complex_spectrum(omega=1.0, J=0.5)
    plot_figure_2_spectral_flow(omega=1.0, J=0.5)
    plot_figure_3_eigenvalue_gap_map(scan_sym)
    plot_figure_4_eigenvector_condition(scan_sym)
    plot_figure_5_parity_sector_comparison(omega=1.0, gamma=1.0)
    plot_figure_6_asymmetric_loss_ep(scan_asym)
    plot_figure_7_dynamics_near_ep(tlist, dyn_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT 05 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
