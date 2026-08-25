"""Experiment 06a: Static Non-Hermitian Spectral Topology and Riemann-Surface Mapping Around EP2.

This experiment maps the static complex eigenvalue topology of the second-order
exceptional point (EP2) in the asymmetric-loss odd-parity sector:
1. Dense 2D parameter scan over (J, Delta_gamma) computing eigenvalues of H_top and H_eff.
2. 3D Riemann surface generation for Re(lambda_pm) and Im(lambda_pm).
3. Complex eigenvalue gap map and logarithmic gap map sharply revealing the EP boundary J = Delta_gamma / 4.
4. Continuous branch tracking along EP-enclosing and EP-avoiding closed parameter loops.
5. Demonstration of square-root branch point topology: one-loop eigenvalue permutation (2pi) and two-loop return (4pi).
6. Continuous eigenvector tracking and overlap metrics.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bipartit_open_spin.analysis.spectrum import (
    build_topological_odd_hamiltonian,
    compute_biorthogonal_eigenpairs,
    track_eigenpairs_along_loop,
)

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
# PART A: 2D PARAMETER SCAN & SPECTRAL SURFACE MAPPING
# ==============================================================================

def run_2d_spectral_grid_scan(
    j_min: float = 0.0,
    j_max: float = 1.0,
    dg_min: float = 0.0,
    dg_max: float = 3.0,
    n_pts: int = 201,
    gamma_bar: float = 1.0,
) -> dict:
    """Compute eigenvalues, gaps, and condition numbers on a dense 2D (J, Delta_gamma) grid."""
    print("=" * 80)
    print("PART A: 2D SPECTRAL GRID SCAN & RIEMANN SHEET COMPUTATION")
    print("=" * 80)

    j_grid = np.linspace(j_min, j_max, n_pts)
    dg_grid = np.linspace(dg_min, dg_max, n_pts)
    J_mesh, DG_mesh = np.meshgrid(j_grid, dg_grid)

    evals_top_plus = np.zeros_like(J_mesh, dtype=complex)
    evals_top_minus = np.zeros_like(J_mesh, dtype=complex)
    gap_mesh = np.zeros_like(J_mesh, dtype=float)

    for i in range(n_pts):
        for j in range(n_pts):
            J_val = J_mesh[i, j]
            dg_val = DG_mesh[i, j]

            # Analytical / numerical values for H_top
            H_top = build_topological_odd_hamiltonian(J_val, dg_val)
            evals, _ = np.linalg.eig(H_top)

            # Sort by real part, then imaginary part
            sort_idx = np.lexsort((evals.imag, evals.real))
            evals = evals[sort_idx]

            evals_top_minus[i, j] = evals[0]
            evals_top_plus[i, j] = evals[1]
            gap_mesh[i, j] = np.abs(evals[1] - evals[0])

    print(f"  2D Grid scan completed: {n_pts} x {n_pts} points.")
    print(f"  Minimum spectral gap on grid: {np.min(gap_mesh):.4e}")
    print(f"  Maximum spectral gap on grid: {np.max(gap_mesh):.4e}")

    return {
        "j_grid": j_grid,
        "dg_grid": dg_grid,
        "J_mesh": J_mesh,
        "DG_mesh": DG_mesh,
        "evals_top_plus": evals_top_plus,
        "evals_top_minus": evals_top_minus,
        "gap_mesh": gap_mesh,
        "gamma_bar": gamma_bar,
    }


# ==============================================================================
# PART B, C, D, E: PARAMETER LOOPS & CONTINUOUS BRANCH TRACKING
# ==============================================================================

def run_loop_analysis(
    j_ep: float = 0.4,
    dg_ep: float = 1.6,
    gamma_bar: float = 1.0,
    n_theta: int = 801,
) -> dict:
    """Track eigenvalues and eigenvectors continuously along EP-enclosing and EP-avoiding loops."""
    print("\n" + "=" * 80)
    print("PART B-E: CONTINUOUS BRANCH TRACKING ALONG CLOSED PARAMETER LOOPS")
    print("=" * 80)

    # 4*pi grid for two full cycles
    theta_grid_4pi = np.linspace(0.0, 4.0 * np.pi, n_theta)
    # 2*pi grid for one full cycle
    theta_grid_2pi = np.linspace(0.0, 2.0 * np.pi, (n_theta // 2) + 1)

    # Loop 1: EP-Enclosing Loop around (J_EP, Delta_gamma_EP, delta_omega=0)
    r_j1 = 0.15
    r_omega1 = 0.15

    def H_loop_enclosing(th):
        j_th = j_ep + r_j1 * np.cos(th)
        domega_th = r_omega1 * np.sin(th)
        return np.array([
            [domega_th - 0.25j * dg_ep, j_th],
            [j_th, -domega_th + 0.25j * dg_ep],
        ], dtype=complex)

    res_enclosing = track_eigenpairs_along_loop(H_loop_enclosing, theta_grid_4pi)

    # Loop 2: EP-Avoiding Loop centered away from EP
    j_center2 = 0.70
    r_j2 = 0.15
    r_omega2 = 0.15

    def H_loop_avoiding(th):
        j_th = j_center2 + r_j2 * np.cos(th)
        domega_th = r_omega2 * np.sin(th)
        return np.array([
            [domega_th - 0.25j * dg_ep, j_th],
            [j_th, -domega_th + 0.25j * dg_ep],
        ], dtype=complex)

    res_avoiding = track_eigenpairs_along_loop(H_loop_avoiding, theta_grid_4pi)

    # Loop 3: Small EP-Approaching Loop
    r_j3 = 0.04
    r_omega3 = 0.04

    def H_loop_small(th):
        j_th = j_ep + r_j3 * np.cos(th)
        domega_th = r_omega3 * np.sin(th)
        return np.array([
            [domega_th - 0.25j * dg_ep, j_th],
            [j_th, -domega_th + 0.25j * dg_ep],
        ], dtype=complex)

    res_small = track_eigenpairs_along_loop(H_loop_small, theta_grid_4pi)

    # Print summary
    print(f"  Loop 1 (EP-Enclosing):")
    print(f"    1-Loop Permutation: {res_enclosing['permutation_1_loop']} (Eigenvalues SWAP after 2*pi)")
    print(f"    1-Loop Swap Error |lambda_1(2pi) - lambda_2(0)|: {res_enclosing['permutation_error_1_loop']:.4e}")
    print(f"    2-Loop Return Error |lambda_1(4pi) - lambda_1(0)|: {res_enclosing['return_error_2_loop']:.4e}")

    half_idx = len(theta_grid_4pi) // 2
    err_return_av = float(np.abs(res_avoiding['eigenvalues'][half_idx, 0] - res_avoiding['eigenvalues'][0, 0]))
    print(f"\n  Loop 2 (EP-Avoiding):")
    print(f"    1-Loop Permutation: {res_avoiding['permutation_1_loop']} (NO swap, remains on original branch)")
    print(f"    1-Loop Return Error |lambda_1(2pi) - lambda_1(0)|: {err_return_av:.4e}")

    return {
        "theta_grid": theta_grid_4pi,
        "res_enclosing": res_enclosing,
        "res_avoiding": res_avoiding,
        "res_small": res_small,
        "loop_enclosing_params": {"j_ep": j_ep, "dg_ep": dg_ep, "r_j": r_j1, "r_omega": r_omega1},
        "loop_avoiding_params": {"j_center": j_center2, "dg_ep": dg_ep, "r_j": r_j2, "r_omega": r_omega2},
    }


# ==============================================================================
# FIGURE GENERATION
# ==============================================================================

def plot_figure_1_real_surfaces(grid_data):
    """Figure 1: 3D Surface of Re(lambda_pm) over (J, Delta_gamma)."""
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    J_mesh = grid_data["J_mesh"]
    DG_mesh = grid_data["DG_mesh"]
    re_plus = grid_data["evals_top_plus"].real
    re_minus = grid_data["evals_top_minus"].real

    surf1 = ax.plot_surface(J_mesh, DG_mesh, re_plus, cmap="viridis", alpha=0.85, edgecolor="none")
    surf2 = ax.plot_surface(J_mesh, DG_mesh, re_minus, cmap="plasma", alpha=0.85, edgecolor="none")

    # Plot EP line J = Delta_gamma / 4
    dg_line = np.linspace(0, 3.0, 100)
    j_line = dg_line / 4.0
    valid = j_line <= 1.0
    ax.plot(j_line[valid], dg_line[valid], np.zeros_like(j_line[valid]), "r--", linewidth=2.5, label=r"EP Boundary $J = \Delta\gamma / 4$")

    ax.set_title(r"Real Eigenvalue Surfaces $\mathrm{Re}(\lambda_\pm)$ of $H_{\rm top}$", pad=15)
    ax.set_xlabel(r"Coupling Strength $J$ [$\omega$]", labelpad=8)
    ax.set_ylabel(r"Dissipation Asymmetry $\Delta\gamma$ [$\omega$]", labelpad=8)
    ax.set_zlabel(r"$\mathrm{Re}(\lambda)$ [$\omega$]", labelpad=8)
    ax.view_init(elev=28, azim=-125)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06a_real_eigenvalue_surfaces.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 1 saved to: {fig_path}")


def plot_figure_2_imag_surfaces(grid_data):
    """Figure 2: 3D Surface of Im(lambda_pm) over (J, Delta_gamma)."""
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    J_mesh = grid_data["J_mesh"]
    DG_mesh = grid_data["DG_mesh"]
    im_plus = grid_data["evals_top_plus"].imag
    im_minus = grid_data["evals_top_minus"].imag

    surf1 = ax.plot_surface(J_mesh, DG_mesh, im_plus, cmap="coolwarm", alpha=0.85, edgecolor="none")
    surf2 = ax.plot_surface(J_mesh, DG_mesh, im_minus, cmap="cividis", alpha=0.85, edgecolor="none")

    dg_line = np.linspace(0, 3.0, 100)
    j_line = dg_line / 4.0
    valid = j_line <= 1.0
    ax.plot(j_line[valid], dg_line[valid], np.zeros_like(j_line[valid]), "r--", linewidth=2.5, label=r"EP Boundary $J = \Delta\gamma / 4$")

    ax.set_title(r"Imaginary Eigenvalue Surfaces $\mathrm{Im}(\lambda_\pm)$ of $H_{\rm top}$", pad=15)
    ax.set_xlabel(r"Coupling Strength $J$ [$\omega$]", labelpad=8)
    ax.set_ylabel(r"Dissipation Asymmetry $\Delta\gamma$ [$\omega$]", labelpad=8)
    ax.set_zlabel(r"$\mathrm{Im}(\lambda)$ [$\omega$]", labelpad=8)
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06a_imag_eigenvalue_surfaces.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 2 saved to: {fig_path}")


def plot_figure_3_gap_map(grid_data):
    """Figure 3: Complex Eigenvalue Gap Map and Logarithmic Gap Map."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    J_mesh = grid_data["J_mesh"]
    DG_mesh = grid_data["DG_mesh"]
    gap_mesh = grid_data["gap_mesh"]

    # Panel A: Linear Gap Map
    ax0 = axes[0]
    c0 = ax0.pcolormesh(J_mesh, DG_mesh, gap_mesh, shading="auto", cmap="viridis")
    cb0 = fig.colorbar(c0, ax=ax0)
    cb0.set_label(r"Complex Spectral Gap $\Delta\lambda = |\lambda_+ - \lambda_-|$")

    dg_line = np.linspace(0, 3.0, 100)
    j_line = dg_line / 4.0
    valid = j_line <= 1.0
    ax0.plot(j_line[valid], dg_line[valid], "r--", linewidth=2.0, label=r"EP Boundary $J = \Delta\gamma / 4$")
    ax0.plot(0.4, 1.6, "w*", markersize=12, label=r"Target EP $(0.4, 1.6)$")
    ax0.set_title(r"(a) Spectral Gap $\Delta\lambda$ in $(J, \Delta\gamma)$ Space")
    ax0.set_xlabel(r"Coupling Strength $J$ [$\omega$]")
    ax0.set_ylabel(r"Dissipation Asymmetry $\Delta\gamma$ [$\omega$]")
    ax0.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Panel B: Logarithmic Gap Map (Sharp EP Valley)
    ax1 = axes[1]
    log_gap = np.log10(np.clip(gap_mesh, 1e-4, 2.0))
    c1 = ax1.pcolormesh(J_mesh, DG_mesh, log_gap, shading="auto", cmap="magma")
    cb1 = fig.colorbar(c1, ax=ax1)
    cb1.set_label(r"$\log_{10}(\Delta\lambda)$")

    ax1.plot(j_line[valid], dg_line[valid], "w--", linewidth=2.0, label=r"Zero-Gap Valley (EP2 Line)")
    ax1.plot(0.4, 1.6, "c*", markersize=12, label=r"Target EP $(0.4, 1.6)$")
    ax1.set_title(r"(b) Log-Scaled Gap Map (Singularity Valley)")
    ax1.set_xlabel(r"Coupling Strength $J$ [$\omega$]")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06a_eigenvalue_gap_map.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 3 saved to: {fig_path}")


def plot_figure_4_local_ep_zoom(grid_data):
    """Figure 4: High-Resolution Local Zoom around the Exceptional Point."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Zoom window: J in [0.2, 0.6], Delta_gamma in [1.0, 2.2]
    j_mask = (grid_data["j_grid"] >= 0.2) & (grid_data["j_grid"] <= 0.6)
    dg_mask = (grid_data["dg_grid"] >= 1.0) & (grid_data["dg_grid"] <= 2.2)

    J_sub = grid_data["J_mesh"][np.ix_(dg_mask, j_mask)]
    DG_sub = grid_data["DG_mesh"][np.ix_(dg_mask, j_mask)]
    re_diff = (grid_data["evals_top_plus"].real - grid_data["evals_top_minus"].real)[np.ix_(dg_mask, j_mask)]
    im_diff = (grid_data["evals_top_plus"].imag - grid_data["evals_top_minus"].imag)[np.ix_(dg_mask, j_mask)]

    # Panel A: Real Splitting
    ax0 = axes[0]
    c0 = ax0.pcolormesh(J_sub, DG_sub, re_diff, shading="auto", cmap="Blues")
    fig.colorbar(c0, ax=ax0, label=r"$\mathrm{Re}(\lambda_+) - \mathrm{Re}(\lambda_-)$")
    ax0.plot(grid_data["dg_grid"][dg_mask] / 4.0, grid_data["dg_grid"][dg_mask], "r--", linewidth=2.0, label="EP Boundary")
    ax0.plot(0.4, 1.6, "k*", markersize=14, label="EP2 (0.4, 1.6)")
    ax0.set_title(r"(a) Real Splitting: $\mathrm{Re}(\Delta\lambda)$ (Oscillatory Sector)")
    ax0.set_xlabel(r"$J$ [$\omega$]")
    ax0.set_ylabel(r"$\Delta\gamma$ [$\omega$]")
    ax0.legend(loc="upper left", fontsize=8.5)

    # Panel B: Imaginary Splitting
    ax1 = axes[1]
    c1 = ax1.pcolormesh(J_sub, DG_sub, im_diff, shading="auto", cmap="Reds")
    fig.colorbar(c1, ax=ax1, label=r"$\mathrm{Im}(\lambda_+) - \mathrm{Im}(\lambda_-)$")
    ax1.plot(grid_data["dg_grid"][dg_mask] / 4.0, grid_data["dg_grid"][dg_mask], "r--", linewidth=2.0, label="EP Boundary")
    ax1.plot(0.4, 1.6, "k*", markersize=14, label="EP2 (0.4, 1.6)")
    ax1.set_title(r"(b) Imaginary Splitting: $\mathrm{Im}(\Delta\lambda)$ (Decay Sector)")
    ax1.set_xlabel(r"$J$ [$\omega$]")
    ax1.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06a_local_ep_zoom.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 4 saved to: {fig_path}")


def plot_figure_5_riemann_surface_3d(grid_data, loop_data):
    """Figure 5: 3D Real Riemann Surface with Enclosing Loop Trajectory."""
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    J_mesh = grid_data["J_mesh"]
    DG_mesh = grid_data["DG_mesh"]
    re_plus = grid_data["evals_top_plus"].real
    re_minus = grid_data["evals_top_minus"].real

    # Subsample for smooth 3D rendering
    step = 3
    ax.plot_surface(
        J_mesh[::step, ::step], DG_mesh[::step, ::step], re_plus[::step, ::step],
        cmap="viridis", alpha=0.55, edgecolor="none"
    )
    ax.plot_surface(
        J_mesh[::step, ::step], DG_mesh[::step, ::step], re_minus[::step, ::step],
        cmap="plasma", alpha=0.55, edgecolor="none"
    )

    # Plot EP line
    dg_line = np.linspace(0, 3.0, 100)
    j_line = dg_line / 4.0
    valid = j_line <= 1.0
    ax.plot(j_line[valid], dg_line[valid], np.zeros_like(j_line[valid]), "r--", linewidth=2.5, label=r"Branch Cut $J = \Delta\gamma/4$")

    # Mark the reference EP
    ax.scatter([0.4], [1.6], [0.0], color="red", s=100, marker="o", label=r"EP2 $(0.4, 1.6)$")

    ax.set_title(r"Two-Sheeted Riemann Surface $\mathrm{Re}(\lambda_\pm(J, \Delta\gamma))$", pad=15)
    ax.set_xlabel(r"Coupling Strength $J$ [$\omega$]", labelpad=8)
    ax.set_ylabel(r"Dissipation Asymmetry $\Delta\gamma$ [$\omega$]", labelpad=8)
    ax.set_zlabel(r"$\mathrm{Re}(\lambda)$ [$\omega$]", labelpad=8)
    ax.view_init(elev=26, azim=-130)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06a_riemann_surface_real.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 5 saved to: {fig_path}")


def plot_figure_6_complex_eigenvalue_encircling(loop_data):
    """Figure 6: Complex Eigenvalue Trajectory in the Complex Plane during Encircling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    theta_grid = loop_data["theta_grid"]
    res_enc = loop_data["res_enclosing"]
    res_av = loop_data["res_avoiding"]

    # Panel A: EP-Enclosing Loop Trajectory in Complex Plane
    ax0 = axes[0]
    evals_enc = res_enc["eigenvalues"]
    half_idx = len(theta_grid) // 2

    # First loop (0 -> 2pi)
    ax0.plot(evals_enc[:half_idx, 0].real, evals_enc[:half_idx, 0].imag, "b-", linewidth=2.2, label=r"Branch 1 ($0 \to 2\pi$)")
    ax0.plot(evals_enc[:half_idx, 1].real, evals_enc[:half_idx, 1].imag, "r-", linewidth=2.2, label=r"Branch 2 ($0 \to 2\pi$)")
    # Second loop (2pi -> 4pi)
    ax0.plot(evals_enc[half_idx:, 0].real, evals_enc[half_idx:, 0].imag, "b--", linewidth=2.0, label=r"Branch 1 ($2\pi \to 4\pi$)")
    ax0.plot(evals_enc[half_idx:, 1].real, evals_enc[half_idx:, 1].imag, "r--", linewidth=2.0, label=r"Branch 2 ($2\pi \to 4\pi$)")

    # Start points and endpoints
    ax0.scatter([evals_enc[0, 0].real], [evals_enc[0, 0].imag], color="blue", marker="o", s=80, zorder=5, label=r"$\lambda_1(0)$")
    ax0.scatter([evals_enc[0, 1].real], [evals_enc[0, 1].imag], color="red", marker="s", s=80, zorder=5, label=r"$\lambda_2(0)$")
    ax0.scatter([evals_enc[half_idx, 0].real], [evals_enc[half_idx, 0].imag], color="red", marker="x", s=100, linewidth=2.5, zorder=6, label=r"$\lambda_1(2\pi) \to \lambda_2(0)$")
    ax0.scatter([evals_enc[half_idx, 1].real], [evals_enc[half_idx, 1].imag], color="blue", marker="+", s=100, linewidth=2.5, zorder=6, label=r"$\lambda_2(2\pi) \to \lambda_1(0)$")

    ax0.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax0.axvline(0, color="gray", linestyle=":", alpha=0.6)
    ax0.set_title(r"(a) EP-Enclosing Loop (Branch Permutation: $\lambda_1 \leftrightarrow \lambda_2$)")
    ax0.set_xlabel(r"$\mathrm{Re}(\lambda)$ [$\omega$]")
    ax0.set_ylabel(r"$\mathrm{Im}(\lambda)$ [$\omega$]")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=7.8, framealpha=0.9)

    # Panel B: EP-Avoiding Loop Trajectory
    ax1 = axes[1]
    evals_av = res_av["eigenvalues"]
    ax1.plot(evals_av[:half_idx, 0].real, evals_av[:half_idx, 0].imag, "b-", linewidth=2.2, label=r"Branch 1 ($0 \to 2\pi$)")
    ax1.plot(evals_av[:half_idx, 1].real, evals_av[:half_idx, 1].imag, "r-", linewidth=2.2, label=r"Branch 2 ($0 \to 2\pi$)")
    ax1.scatter([evals_av[0, 0].real], [evals_av[0, 0].imag], color="blue", marker="o", s=80, zorder=5, label=r"$\lambda_1(0) = \lambda_1(2\pi)$")
    ax1.scatter([evals_av[0, 1].real], [evals_av[0, 1].imag], color="red", marker="s", s=80, zorder=5, label=r"$\lambda_2(0) = \lambda_2(2\pi)$")

    ax1.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.6)
    ax1.set_title(r"(b) EP-Avoiding Loop (No Permutation: Closed Orbits)")
    ax1.set_xlabel(r"$\mathrm{Re}(\lambda)$ [$\omega$]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06a_complex_eigenvalue_encircling.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 6 saved to: {fig_path}")


def plot_figure_7_eigenvalue_permutation(loop_data):
    """Figure 7: Real and Imaginary Eigenvalue Flow vs Parameter Angle theta."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)

    theta_grid = loop_data["theta_grid"]
    res_enc = loop_data["res_enclosing"]
    res_av = loop_data["res_avoiding"]

    # Panel A: Real Parts (Enclosing)
    ax00 = axes[0, 0]
    ax00.plot(theta_grid / np.pi, res_enc["eigenvalues"][:, 0].real, "b-", label=r"Tracked Branch 1")
    ax00.plot(theta_grid / np.pi, res_enc["eigenvalues"][:, 1].real, "r--", label=r"Tracked Branch 2")
    ax00.axvline(2.0, color="gray", linestyle=":", label=r"$\theta = 2\pi$ (1 Loop)")
    ax00.axvline(4.0, color="k", linestyle="--", label=r"$\theta = 4\pi$ (2 Loops)")
    ax00.set_title(r"(a) $\mathrm{Re}(\lambda)$ vs $\theta$ [EP-Enclosing Loop]")
    ax00.set_ylabel(r"$\mathrm{Re}(\lambda)$ [$\omega$]")
    ax00.grid(True, alpha=0.3)
    ax00.legend(loc="lower right", fontsize=8)

    # Panel B: Imaginary Parts (Enclosing)
    ax10 = axes[1, 0]
    ax10.plot(theta_grid / np.pi, res_enc["eigenvalues"][:, 0].imag, "b-", label=r"Tracked Branch 1")
    ax10.plot(theta_grid / np.pi, res_enc["eigenvalues"][:, 1].imag, "r--", label=r"Tracked Branch 2")
    ax10.axvline(2.0, color="gray", linestyle=":")
    ax10.axvline(4.0, color="k", linestyle="--")
    ax10.set_title(r"(b) $\mathrm{Im}(\lambda)$ vs $\theta$ [EP-Enclosing Loop]")
    ax10.set_xlabel(r"Parameter Angle $\theta$ [$\pi$ rad]")
    ax10.set_ylabel(r"$\mathrm{Im}(\lambda)$ [$\omega$]")
    ax10.grid(True, alpha=0.3)

    # Panel C: Real Parts (Avoiding)
    ax01 = axes[0, 1]
    ax01.plot(theta_grid / np.pi, res_av["eigenvalues"][:, 0].real, "b-", label=r"Branch 1")
    ax01.plot(theta_grid / np.pi, res_av["eigenvalues"][:, 1].real, "r--", label=r"Branch 2")
    ax01.axvline(2.0, color="gray", linestyle=":")
    ax01.axvline(4.0, color="k", linestyle="--")
    ax01.set_title(r"(c) $\mathrm{Re}(\lambda)$ vs $\theta$ [EP-Avoiding Loop]")
    ax01.grid(True, alpha=0.3)
    ax01.legend(loc="upper right", fontsize=8)

    # Panel D: Imaginary Parts (Avoiding)
    ax11 = axes[1, 1]
    ax11.plot(theta_grid / np.pi, res_av["eigenvalues"][:, 0].imag, "b-", label=r"Branch 1")
    ax11.plot(theta_grid / np.pi, res_av["eigenvalues"][:, 1].imag, "r--", label=r"Branch 2")
    ax11.axvline(2.0, color="gray", linestyle=":")
    ax11.axvline(4.0, color="k", linestyle="--")
    ax11.set_title(r"(d) $\mathrm{Im}(\lambda)$ vs $\theta$ [EP-Avoiding Loop]")
    ax11.set_xlabel(r"Parameter Angle $\theta$ [$\pi$ rad]")
    ax11.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06a_eigenvalue_permutation.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 7 saved to: {fig_path}")


def plot_figure_8_eigenvector_overlap(loop_data):
    """Figure 8: Eigenvector Overlaps during Closed Loop Encircling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    theta_grid = loop_data["theta_grid"]
    res_enc = loop_data["res_enclosing"]
    res_av = loop_data["res_avoiding"]

    # Panel A: Overlaps for EP-Enclosing Loop
    ax0 = axes[0]
    o11 = res_enc["overlaps"]["O11"]
    o12 = res_enc["overlaps"]["O12"]

    ax0.plot(theta_grid / np.pi, o11, "b-", linewidth=2.0, label=r"Same-Branch: $|\langle v_1(0)|v_1(\theta)\rangle|$")
    ax0.plot(theta_grid / np.pi, o12, "r--", linewidth=2.0, label=r"Cross-Branch: $|\langle v_1(0)|v_2(\theta)\rangle|$")
    ax0.axvline(2.0, color="gray", linestyle=":", label=r"$\theta = 2\pi$ (1-Loop Swap)")
    ax0.axvline(4.0, color="k", linestyle="--", label=r"$\theta = 4\pi$ (2-Loop Return)")

    ax0.set_title(r"(a) Eigenvector Overlaps [EP-Enclosing Loop]")
    ax0.set_xlabel(r"Parameter Angle $\theta$ [$\pi$ rad]")
    ax0.set_ylabel(r"Overlap Metric")
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="center right", fontsize=8.5, framealpha=0.9)

    # Panel B: Overlaps for EP-Avoiding Loop
    ax1 = axes[1]
    o11_av = res_av["overlaps"]["O11"]
    o12_av = res_av["overlaps"]["O12"]

    ax1.plot(theta_grid / np.pi, o11_av, "b-", linewidth=2.0, label=r"Same-Branch: $|\langle v_1(0)|v_1(\theta)\rangle|$")
    ax1.plot(theta_grid / np.pi, o12_av, "r--", linewidth=2.0, label=r"Cross-Branch: $|\langle v_1(0)|v_2(\theta)\rangle|$")
    ax1.axvline(2.0, color="gray", linestyle=":")
    ax1.axvline(4.0, color="k", linestyle="--")

    ax1.set_title(r"(b) Eigenvector Overlaps [EP-Avoiding Loop]")
    ax1.set_xlabel(r"Parameter Angle $\theta$ [$\pi$ rad]")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_06a_eigenvector_overlap.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 8 saved to: {fig_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 06a: STATIC NON-HERMITIAN SPECTRAL TOPOLOGY & RIEMANN SURFACES")
    print("=" * 80)

    # 1. 2D Parameter Grid Scan (201x201)
    grid_data = run_2d_spectral_grid_scan(
        j_min=0.0, j_max=1.0, dg_min=0.0, dg_max=3.0, n_pts=201, gamma_bar=1.0
    )

    # 2. Continuous Loop Analysis (Enclosing vs Avoiding vs Small)
    loop_data = run_loop_analysis(j_ep=0.4, dg_ep=1.6, gamma_bar=1.0, n_theta=801)

    # 3. Save Data Archive
    npz_data = {
        "j_grid": grid_data["j_grid"],
        "delta_gamma_grid": grid_data["dg_grid"],
        "evals_top_plus": grid_data["evals_top_plus"],
        "evals_top_minus": grid_data["evals_top_minus"],
        "gap_mesh": grid_data["gap_mesh"],
        "loop_theta_grid": loop_data["theta_grid"],
        "evals_enclosing": loop_data["res_enclosing"]["eigenvalues"],
        "evecs_enclosing": loop_data["res_enclosing"]["eigenvectors"],
        "o11_enclosing": loop_data["res_enclosing"]["overlaps"]["O11"],
        "o12_enclosing": loop_data["res_enclosing"]["overlaps"]["O12"],
        "evals_avoiding": loop_data["res_avoiding"]["eigenvalues"],
        "evecs_avoiding": loop_data["res_avoiding"]["eigenvectors"],
        "o11_avoiding": loop_data["res_avoiding"]["overlaps"]["O11"],
        "o12_avoiding": loop_data["res_avoiding"]["overlaps"]["O12"],
        "evals_small": loop_data["res_small"]["eigenvalues"],
        "gamma_bar": grid_data["gamma_bar"],
        "j_ep": 0.4,
        "delta_gamma_ep": 1.6,
    }
    data_path = DATA_DIR / "experiment_06a_ep_topology.npz"
    np.savez_compressed(data_path, **npz_data)
    print(f"\nSaved numerical dataset to: {data_path}")

    # 4. Generate Figures (8 Publication Figures)
    print("\n" + "=" * 80)
    print("PART F & L: GENERATING PUBLICATION FIGURES (8 FIGURES)")
    print("=" * 80)
    plot_figure_1_real_surfaces(grid_data)
    plot_figure_2_imag_surfaces(grid_data)
    plot_figure_3_gap_map(grid_data)
    plot_figure_4_local_ep_zoom(grid_data)
    plot_figure_5_riemann_surface_3d(grid_data, loop_data)
    plot_figure_6_complex_eigenvalue_encircling(loop_data)
    plot_figure_7_eigenvalue_permutation(loop_data)
    plot_figure_8_eigenvector_overlap(loop_data)

    print("\n" + "=" * 80)
    print("EXPERIMENT 06a COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
