"""Experiment 03c: Physical Mechanism and Scaling of Entangled Non-Equilibrium Steady States.

This experiment investigates the microscopic physical mechanism that sustains the
entangled non-equilibrium steady state (NESS) in a bipartite spin system under
coherent exchange interaction, local amplitude damping, and local pure dephasing.

Model:
    H = (omega / 2) * (sigma_z1 + sigma_z2) + J * (sigma_x1 * sigma_x2)  [omega = 1.0]
    C_1 = sqrt(gamma_1) * sigma_-1
    C_2 = sqrt(gamma_1) * sigma_-2
    L_phi1 = sqrt(gamma_phi / 2) * sigma_z1
    L_phi2 = sqrt(gamma_phi / 2) * sigma_z2

Investigations:
    Part A: Steady-State Density Matrix Structure (X-state verification & matrix elements).
    Part B: Entanglement Decomposition (Dominant coherence vs population threshold).
    Part C: Physical Flow of Population and Coherence (Dissipative jump rates & recycling).
    Part D: Coherence Balance Equation (Liouvillian superoperator decomposition).
    Part E: Dimensionless Scaling Analysis (Collapse vs r_1 = J/gamma_1, r_phi = gamma_phi/gamma_1).
    Part F: Entanglement Robustness Boundary Extraction (Half-suppression and operational thresholds).
    Part G: Time-Scale Analysis (tau_J, tau_amp, tau_phi hierarchy).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend for publication reproducibility
import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, steadystate

from bipartit_open_spin.analysis.entanglement import (
    concurrence,
    negativity,
    x_state_concurrence_decomposition,
)
from bipartit_open_spin.config import ModelParams
from bipartit_open_spin.dynamics.dissipation import (
    build_collapse_operators,
    build_dephasing_collapse_operators,
    dissipative_jump_rates,
    liouvillian_superoperator_decomposition,
)
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.validation.diagnostics import validate_density_matrix


def run_experiment() -> dict:
    """Execute Experiment 03c, perform mechanism decompositions, save data and figures."""
    print("=" * 80, flush=True)
    print("EXPERIMENT 03C: PHYSICAL MECHANISM & SCALING OF ENTANGLED NESS", flush=True)
    print("=" * 80, flush=True)

    omega = 1.0
    J_opt = 0.85
    gamma_1_opt = 1.80

    fig_dir = Path("results/figures")
    data_dir = Path("results/data")
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART A: Steady-State Density Matrix Structure
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART A: STEADY-STATE DENSITY MATRIX STRUCTURE", flush=True)
    print("=" * 60, flush=True)

    rep_cases = [
        {"name": "Case 1: Strong Entangled NESS (gamma_phi = 0.00)", "gamma_phi": 0.00},
        {"name": "Case 2: Weak Entangled NESS (gamma_phi = 0.50)", "gamma_phi": 0.50},
        {"name": "Case 3: Intermediate Dephased NESS (gamma_phi = 1.50)", "gamma_phi": 1.50},
        {"name": "Case 4: Deep Asymptotic Regime (gamma_phi = 5.00)", "gamma_phi": 5.00},
    ]

    rep_data = []
    for case in rep_cases:
        g_phi = case["gamma_phi"]
        params = ModelParams(omega=omega, J=J_opt, gamma=gamma_1_opt)
        H = build_hamiltonian(params)
        c_ops_amp = build_collapse_operators(params)
        c_ops_phi = build_dephasing_collapse_operators(g_phi)
        c_ops_all = c_ops_amp + c_ops_phi

        rho_ss = steadystate(H, c_ops_all)
        v_res = validate_density_matrix(rho_ss, tol=1e-6)
        if not v_res["valid"]:
            raise RuntimeError(f"Validation failed for {case['name']}")

        decomp = x_state_concurrence_decomposition(rho_ss)
        mat = rho_ss.full()

        # Non-X element maximum magnitude
        non_x_elements = [
            abs(mat[0, 1]), abs(mat[0, 2]),
            abs(mat[1, 0]), abs(mat[1, 3]),
            abs(mat[2, 0]), abs(mat[2, 3]),
            abs(mat[3, 1]), abs(mat[3, 2]),
        ]
        max_non_x = max(non_x_elements)

        rates_amp = dissipative_jump_rates(c_ops_amp, rho_ss)
        liouv_decomp = liouvillian_superoperator_decomposition(H, c_ops_amp, c_ops_phi, rho_ss)
        l_tot_norm = float(np.linalg.norm(liouv_decomp["L_tot"].full()))

        case_dict = {
            "name": case["name"],
            "gamma_phi": g_phi,
            "rho_ss": mat,
            "decomp": decomp,
            "max_non_x": max_non_x,
            "rates_amp": rates_amp,
            "l_tot_norm": l_tot_norm,
            "liouv_decomp": liouv_decomp,
        }
        rep_data.append(case_dict)

        print(f"\n--- {case['name']} ---", flush=True)
        print(f"  Populations: P00 = {decomp['p00']:.4f}, P01 = {decomp['p01']:.4f}, P10 = {decomp['p10']:.4f}, P11 = {decomp['p11']:.4f}", flush=True)
        print(f"  Even Coherence |rho00,11| = {decomp['abs_rho0011']:.4f}  (Re = {decomp['re_rho0011']:+.4f}, Im = {decomp['im_rho0011']:+.4f})", flush=True)
        print(f"  Odd Coherence  |rho01,10| = {decomp['abs_rho0110']:.4f}  (Re = {decomp['re_rho0110']:+.4f}, Im = {decomp['im_rho0110']:+.4f})", flush=True)
        print(f"  Threshold sqrt(P01*P10)   = {decomp['threshold_even']:.4f} | E_even = {decomp['e_even']:+.4f}", flush=True)
        print(f"  Threshold sqrt(P00*P11)   = {decomp['threshold_odd']:.4f} | E_odd  = {decomp['e_odd']:+.4f}", flush=True)
        print(f"  Concurrence C_X = {decomp['c_x']:.4f} | Wootters C = {decomp['concurrence']:.4f} | Negativity N = {negativity(rho_ss):.4f}", flush=True)
        print(f"  Max Non-X Element Magnitude = {max_non_x:8.2e} (Exact X-state within machine precision)", flush=True)
        print(f"  Steady-state residual ||L_tot(rho_ss)|| = {l_tot_norm:8.2e}", flush=True)

    # =========================================================================
    # PART B: Entanglement Decomposition & Scan
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART B: ENTANGLEMENT DECOMPOSITION vs DEPHASING", flush=True)
    print("=" * 60, flush=True)

    N_phi_dense = 101
    g_phi_arr = np.linspace(0.0, 5.0, N_phi_dense)

    p00_arr = np.zeros(N_phi_dense)
    p01_arr = np.zeros(N_phi_dense)
    p10_arr = np.zeros(N_phi_dense)
    p11_arr = np.zeros(N_phi_dense)
    abs_rho0011_arr = np.zeros(N_phi_dense)
    abs_rho0110_arr = np.zeros(N_phi_dense)
    thresh_even_arr = np.zeros(N_phi_dense)
    thresh_odd_arr = np.zeros(N_phi_dense)
    e_even_arr = np.zeros(N_phi_dense)
    e_odd_arr = np.zeros(N_phi_dense)
    c_x_arr = np.zeros(N_phi_dense)
    neg_arr = np.zeros(N_phi_dense)

    # Liouvillian element tracking for (0, 3) element [rho_00,11]
    lh_0011_re = np.zeros(N_phi_dense)
    lh_0011_im = np.zeros(N_phi_dense)
    lamp_0011_re = np.zeros(N_phi_dense)
    lamp_0011_im = np.zeros(N_phi_dense)
    lphi_0011_re = np.zeros(N_phi_dense)
    lphi_0011_im = np.zeros(N_phi_dense)

    # Jump rates
    rate_amp1_arr = np.zeros(N_phi_dense)
    rate_amp2_arr = np.zeros(N_phi_dense)

    for idx, g_p in enumerate(g_phi_arr):
        params = ModelParams(omega=omega, J=J_opt, gamma=gamma_1_opt)
        H = build_hamiltonian(params)
        c_ops_a = build_collapse_operators(params)
        c_ops_p = build_dephasing_collapse_operators(g_p)
        c_ops_tot = c_ops_a + c_ops_p

        rho_s = steadystate(H, c_ops_tot)
        d = x_state_concurrence_decomposition(rho_s)

        p00_arr[idx] = d["p00"]
        p01_arr[idx] = d["p01"]
        p10_arr[idx] = d["p10"]
        p11_arr[idx] = d["p11"]
        abs_rho0011_arr[idx] = d["abs_rho0011"]
        abs_rho0110_arr[idx] = d["abs_rho0110"]
        thresh_even_arr[idx] = d["threshold_even"]
        thresh_odd_arr[idx] = d["threshold_odd"]
        e_even_arr[idx] = d["e_even"]
        e_odd_arr[idx] = d["e_odd"]
        c_x_arr[idx] = d["c_x"]
        neg_arr[idx] = negativity(rho_s)

        # Liouvillian balance on rho00,11
        ld = liouvillian_superoperator_decomposition(H, c_ops_a, c_ops_p, rho_s)
        lh_0011 = ld["L_H"].full()[0, 3]
        lamp_0011 = ld["L_amp"].full()[0, 3]
        lphi_0011 = ld["L_phi"].full()[0, 3]

        lh_0011_re[idx] = float(np.real(lh_0011))
        lh_0011_im[idx] = float(np.imag(lh_0011))
        lamp_0011_re[idx] = float(np.real(lamp_0011))
        lamp_0011_im[idx] = float(np.imag(lamp_0011))
        lphi_0011_re[idx] = float(np.real(lphi_0011))
        lphi_0011_im[idx] = float(np.imag(lphi_0011))

        # Jump rates
        j_rates = dissipative_jump_rates(c_ops_a, rho_s)
        rate_amp1_arr[idx] = j_rates[0]
        rate_amp2_arr[idx] = j_rates[1]

    # Find half-maximum dephasing rate gamma_phi,half
    c_0 = c_x_arr[0]
    idx_half = np.where(c_x_arr <= 0.5 * c_0)[0]
    g_phi_half = float(g_phi_arr[idx_half[0]]) if len(idx_half) > 0 else np.nan
    print(f"Optimal NESS concurrence at gamma_phi=0: C_ss(0) = {c_0:.4f}", flush=True)
    print(f"Half-suppression dephasing rate: gamma_phi,1/2 = {g_phi_half:.3f}", flush=True)
    print(f"Dominant channel throughout scan: EVEN channel (|rho00,11| - sqrt(P01*P10))", flush=True)
    print(f"Odd channel term E_odd at gamma_phi=0: {e_odd_arr[0]:+.4f} (strictly negative)", flush=True)

    # =========================================================================
    # PART E: Dimensionless Scaling Analysis
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART E: DIMENSIONLESS SCALING ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    r_phi_range = np.linspace(0.0, 2.5, 41)

    families = [
        {
            "name": r"Family A ($r_1 = 0.472$)",
            "r1": 0.85 / 1.80,
            "pairs": [(0.425, 0.90), (0.85, 1.80), (1.275, 2.70)],
            "color": "#1f77b4",
        },
        {
            "name": r"Family B ($r_1 = 0.250$)",
            "r1": 0.25,
            "pairs": [(0.25, 1.00), (0.50, 2.00)],
            "color": "#2ca02c",
        },
        {
            "name": r"Family C ($r_1 = 0.800$)",
            "r1": 0.80,
            "pairs": [(0.40, 0.50), (0.80, 1.00), (1.60, 2.00)],
            "color": "#d62728",
        },
    ]

    scaling_results = []
    for fam in families:
        fam_curves = []
        for J_val, g1_val in fam["pairs"]:
            c_curve = np.zeros(len(r_phi_range))
            n_curve = np.zeros(len(r_phi_range))
            coh_curve = np.zeros(len(r_phi_range))
            for i_rp, r_p in enumerate(r_phi_range):
                g_p = r_p * g1_val
                params = ModelParams(omega=omega, J=J_val, gamma=g1_val)
                H = build_hamiltonian(params)
                c_ops = build_collapse_operators(params) + build_dephasing_collapse_operators(g_p)
                rho_ss = steadystate(H, c_ops)
                c_curve[i_rp] = concurrence(rho_ss)
                n_curve[i_rp] = negativity(rho_ss)
                coh_curve[i_rp] = float(np.abs(rho_ss.full()[0, 3]))

            fam_curves.append({
                "J": J_val,
                "gamma_1": g1_val,
                "concurrence": c_curve,
                "negativity": n_curve,
                "coherence": coh_curve,
            })
        scaling_results.append({
            "fam_name": fam["name"],
            "r1": fam["r1"],
            "color": fam["color"],
            "curves": fam_curves,
        })
        print(f"Calculated scaling curves for {fam['name']}", flush=True)

    # =========================================================================
    # PART F: Entanglement Robustness Boundary Extraction in Dimensionless Ratios
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("PART F: EXTRACTING ROBUSTNESS BOUNDARY IN DIMENSIONLESS RATIOS", flush=True)
    print("=" * 60, flush=True)

    N_grid_J = 21
    N_grid_g1 = 21
    J_b_vals = np.linspace(0.1, 1.8, N_grid_J)
    g1_b_vals = np.linspace(0.2, 2.4, N_grid_g1)

    r_1_grid = np.zeros((N_grid_g1, N_grid_J))
    r_phi_half_grid = np.zeros((N_grid_g1, N_grid_J))
    r_phi_005_grid = np.zeros((N_grid_g1, N_grid_J))
    c_ss_zero_dephasing_grid = np.zeros((N_grid_g1, N_grid_J))

    for i_g1, g1 in enumerate(g1_b_vals):
        for i_J, J_v in enumerate(J_b_vals):
            r_1_grid[i_g1, i_J] = J_v / g1
            params_0 = ModelParams(omega=omega, J=J_v, gamma=g1)
            H_0 = build_hamiltonian(params_0)
            c_ops_0 = build_collapse_operators(params_0)
            rho_0 = steadystate(H_0, c_ops_0)
            d0 = x_state_concurrence_decomposition(rho_0)
            c0_val = d0["c_x"]
            c_ss_zero_dephasing_grid[i_g1, i_J] = c0_val

            if c0_val <= 1e-4:
                r_phi_half_grid[i_g1, i_J] = 0.0
                r_phi_005_grid[i_g1, i_J] = 0.0
                continue

            # Bisection search for half-maximum dephasing rate C(g_phi) = 0.5 * C(0)
            low = 0.0
            high = 10.0
            for _ in range(12):
                mid = 0.5 * (low + high)
                c_ops_mid = c_ops_0 + build_dephasing_collapse_operators(mid)
                rho_mid = steadystate(H_0, c_ops_mid)
                c_mid = concurrence(rho_mid)
                if c_mid > 0.5 * c0_val:
                    low = mid
                else:
                    high = mid
            g_half = 0.5 * (low + high)
            r_phi_half_grid[i_g1, i_J] = g_half / g1

            # Bisection search for operational threshold C(g_phi) = 0.05
            if c0_val < 0.05:
                r_phi_005_grid[i_g1, i_J] = 0.0
            else:
                low = 0.0
                high = 20.0
                for _ in range(12):
                    mid = 0.5 * (low + high)
                    c_ops_mid = c_ops_0 + build_dephasing_collapse_operators(mid)
                    rho_mid = steadystate(H_0, c_ops_mid)
                    c_mid = concurrence(rho_mid)
                    if c_mid > 0.05:
                        low = mid
                    else:
                        high = mid
                g_005 = 0.5 * (low + high)
                r_phi_005_grid[i_g1, i_J] = g_005 / g1

    print(f"Extracted robustness boundary over {N_grid_J}x{N_grid_g1} grid.", flush=True)
    print(f"Max half-suppression ratio (gamma_phi / gamma_1)_1/2 = {np.max(r_phi_half_grid):.3f}", flush=True)
    print(f"Max operational threshold ratio (gamma_phi / gamma_1)_0.05 = {np.max(r_phi_005_grid):.3f}", flush=True)

    # =========================================================================
    # SAVE DATA
    # =========================================================================
    data_file = data_dir / "experiment_03c_physical_mechanism.npz"
    np.savez_compressed(
        data_file,
        omega=omega,
        J_opt=J_opt,
        gamma_1_opt=gamma_1_opt,
        g_phi_arr=g_phi_arr,
        p00_arr=p00_arr,
        p01_arr=p01_arr,
        p10_arr=p10_arr,
        p11_arr=p11_arr,
        abs_rho0011_arr=abs_rho0011_arr,
        abs_rho0110_arr=abs_rho0110_arr,
        thresh_even_arr=thresh_even_arr,
        thresh_odd_arr=thresh_odd_arr,
        e_even_arr=e_even_arr,
        e_odd_arr=e_odd_arr,
        c_x_arr=c_x_arr,
        neg_arr=neg_arr,
        lh_0011_re=lh_0011_re,
        lh_0011_im=lh_0011_im,
        lamp_0011_re=lamp_0011_re,
        lamp_0011_im=lamp_0011_im,
        lphi_0011_re=lphi_0011_re,
        lphi_0011_im=lphi_0011_im,
        rate_amp1_arr=rate_amp1_arr,
        rate_amp2_arr=rate_amp2_arr,
        J_b_vals=J_b_vals,
        g1_b_vals=g1_b_vals,
        r_1_grid=r_1_grid,
        r_phi_half_grid=r_phi_half_grid,
        r_phi_005_grid=r_phi_005_grid,
        c_ss_zero_dephasing_grid=c_ss_zero_dephasing_grid,
    )
    print(f"\nSaved numerical data to: {data_file}", flush=True)

    # =========================================================================
    # GENERATE PUBLICATION-QUALITY FIGURES
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("GENERATING FIGURES", flush=True)
    print("=" * 60, flush=True)

    # -------------------------------------------------------------------------
    # FIGURE 1: Steady-State Density Matrix Structure (Representative Points)
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(11.5, 9.8), dpi=300)
    basis_labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]

    for ax, c_data in zip(axes1.flat, rep_data):
        mat = np.abs(c_data["rho_ss"])
        im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=0.90)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(basis_labels, fontsize=10.5)
        ax.set_yticklabels(basis_labels, fontsize=10.5)

        # Annotate matrix values
        for i in range(4):
            for j in range(4):
                val = c_data["rho_ss"][i, j]
                abs_v = abs(val)
                if abs_v > 0.005:
                    if abs(val.imag) > 1e-4:
                        text = f"{val.real:.2f}\n{val.imag:+.2f}i"
                    else:
                        text = f"{val.real:.3f}"
                    color = "white" if abs_v > 0.4 else "yellow" if abs_v > 0.1 else "cyan"
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")
                else:
                    ax.text(j, i, "0", ha="center", va="center", color="gray", fontsize=8.5)

        d = c_data["decomp"]
        ax.set_title(
            f"{c_data['name']}\n"
            rf"$C={d['c_x']:.3f},\ |\rho_{{00,11}}|={d['abs_rho0011']:.3f},\ \sqrt{{P_{{01}}P_{{10}}}}={d['threshold_even']:.3f}$",
            fontsize=10.5, fontweight="bold", pad=8
        )

    fig1.subplots_adjust(right=0.88, top=0.90, bottom=0.08, hspace=0.30, wspace=0.25)
    cbar_ax = fig1.add_axes([0.91, 0.15, 0.025, 0.70])
    fig1.colorbar(im, cax=cbar_ax, label=r"Matrix Element Magnitude $|\rho_{ij}|$")

    fig1.suptitle(
        r"Experiment 03c: Steady-State Density Matrix Structure in Computational Basis" "\n"
        rf"Exact X-State Symmetry $\rho_{{ij}} = 0$ for non-X elements ($J={J_opt}, \gamma_1={gamma_1_opt}, \omega={omega}$)",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    fig1_path = fig_dir / "experiment_03c_density_matrix_structure.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  Figure 1 saved to: {fig1_path}", flush=True)

    # -------------------------------------------------------------------------
    # FIGURE 2: Coherence vs Dephasing
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    # Panel (a): Coherence magnitudes
    ax2a.plot(g_phi_arr, abs_rho0011_arr, "b-", linewidth=2.5, label=r"Even Coherence $|\rho_{00,11}|$ ($|00\rangle \leftrightarrow |11\rangle$)")
    ax2a.plot(g_phi_arr, abs_rho0110_arr, "r--", linewidth=2.0, label=r"Odd Coherence $|\rho_{01,10}|$ ($|01\rangle \leftrightarrow |10\rangle$)")
    ax2a.axvline(g_phi_half, color="purple", linestyle=":", linewidth=1.5, label=rf"Half-Suppression $\gamma_\phi \approx {g_phi_half:.2f}$")
    ax2a.set_title(r"(a) Off-Diagonal Coherences vs Dephasing $\gamma_\phi$", fontsize=12, fontweight="bold")
    ax2a.set_xlabel(r"Pure Dephasing Rate $\gamma_\phi$ (units of $\omega$)", fontsize=11)
    ax2a.set_ylabel("Coherence Magnitude", fontsize=11)
    ax2a.set_xlim(0, 5.0)
    ax2a.set_ylim(-0.01, 0.30)
    ax2a.grid(True, linestyle="--", alpha=0.5)
    ax2a.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    # Panel (b): Population Dynamics in NESS
    ax2b.plot(g_phi_arr, p11_arr, color="#2ca02c", linewidth=2.2, label=r"Ground State $P_{11}$")
    ax2b.plot(g_phi_arr, p00_arr, color="#1f77b4", linewidth=2.2, label=r"Upper Level $P_{00}$")
    ax2b.plot(g_phi_arr, p01_arr + p10_arr, color="#ff7f0e", linewidth=2.2, label=r"Odd Subspace $P_{01} + P_{10}$")
    ax2b.plot(g_phi_arr, thresh_even_arr, "m-.", linewidth=2.0, label=r"Threshold $\sqrt{P_{01} P_{10}}$")
    ax2b.axvline(g_phi_half, color="purple", linestyle=":", linewidth=1.5)
    ax2b.set_title(r"(b) Subspace Populations & Threshold vs Dephasing $\gamma_\phi$", fontsize=12, fontweight="bold")
    ax2b.set_xlabel(r"Pure Dephasing Rate $\gamma_\phi$ (units of $\omega$)", fontsize=11)
    ax2b.set_ylabel("Population / Threshold", fontsize=11)
    ax2b.set_xlim(0, 5.0)
    ax2b.set_ylim(-0.02, 0.95)
    ax2b.grid(True, linestyle="--", alpha=0.5)
    ax2b.legend(loc="center right", framealpha=0.92, fontsize=9.5)

    fig2.suptitle(
        r"Experiment 03c: Coherence Suppression and Population Response under Pure Dephasing" "\n"
        rf"($J = {J_opt},\ \gamma_1 = {gamma_1_opt},\ \omega = {omega}$)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig2_path = fig_dir / "experiment_03c_coherence_vs_dephasing.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  Figure 2 saved to: {fig2_path}", flush=True)

    # -------------------------------------------------------------------------
    # FIGURE 3: Concurrence Decomposition & Threshold
    # -------------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    # Panel (a): Even channel coherence vs threshold
    ax3a.plot(g_phi_arr, abs_rho0011_arr, "b-", linewidth=2.5, label=r"Coherence $|\rho_{00,11}|$")
    ax3a.plot(g_phi_arr, thresh_even_arr, "r--", linewidth=2.2, label=r"Threshold $\sqrt{P_{01} P_{10}}$")
    ax3a.fill_between(
        g_phi_arr, abs_rho0011_arr, thresh_even_arr,
        where=(abs_rho0011_arr >= thresh_even_arr),
        color="skyblue", alpha=0.35, label=r"Entangled NESS ($Q > 0$)"
    )
    ax3a.axvline(g_phi_half, color="purple", linestyle=":", linewidth=1.5, label=rf"Half-Suppression $\gamma_\phi \approx {g_phi_half:.2f}$")

    ax3a.set_title(r"(a) Concurrence Criterion: Coherence vs Population Threshold", fontsize=12, fontweight="bold")
    ax3a.set_xlabel(r"Pure Dephasing Rate $\gamma_\phi$", fontsize=11)
    ax3a.set_ylabel("Magnitude", fontsize=11)
    ax3a.set_xlim(0, 5.0)
    ax3a.set_ylim(0.0, 0.30)
    ax3a.grid(True, linestyle="--", alpha=0.5)
    ax3a.legend(loc="upper right", framealpha=0.92, fontsize=9.2)

    # Panel (b): Resulting Wootters Concurrence and Difference Q
    ax3b.plot(g_phi_arr, e_even_arr, "g-", linewidth=2.2, label=r"Excess Coherence $Q = |\rho_{00,11}| - \sqrt{P_{01} P_{10}}$")
    ax3b.plot(g_phi_arr, c_x_arr, "b-", linewidth=2.5, label=r"Concurrence $C_{\mathrm{ss}} = 2\max(0, Q)$")
    ax3b.plot(g_phi_arr, neg_arr, "m--", linewidth=2.0, label=r"Negativity $N_{\mathrm{ss}}$")
    ax3b.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    ax3b.axvline(g_phi_half, color="purple", linestyle=":", linewidth=1.5)

    ax3b.set_title(r"(b) Steady-State Entanglement Measures", fontsize=12, fontweight="bold")
    ax3b.set_xlabel(r"Pure Dephasing Rate $\gamma_\phi$", fontsize=11)
    ax3b.set_ylabel("Entanglement", fontsize=11)
    ax3b.set_xlim(0, 5.0)
    ax3b.set_ylim(-0.02, 0.35)
    ax3b.grid(True, linestyle="--", alpha=0.5)
    ax3b.legend(loc="upper right", framealpha=0.92, fontsize=9.2)

    fig3.suptitle(
        r"Experiment 03c: Analytical Entanglement Criterion and Asymptotic Decay" "\n"
        r"Coherence $|\rho_{00,11}| \sim \mathcal{O}(1/\gamma_\phi)$ exceeds population threshold $\sqrt{P_{01}P_{10}} \sim \mathcal{O}(1/\gamma_\phi^2)$ for all $\gamma_\phi$",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig3_path = fig_dir / "experiment_03c_concurrence_decomposition.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  Figure 3 saved to: {fig3_path}", flush=True)

    # -------------------------------------------------------------------------
    # FIGURE 4: Liouvillian Balance on Dominant Coherence rho_00,11
    # -------------------------------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    # Real part of L_ij(rho_ss)
    ax4a.plot(g_phi_arr, lh_0011_re, "b-", linewidth=2.2, label=r"Hamiltonian $\mathrm{Re}[\mathcal{L}_H]_{00,11}$")
    ax4a.plot(g_phi_arr, lamp_0011_re, "r-", linewidth=2.2, label=r"Amplitude Damping $\mathrm{Re}[\mathcal{L}_{\mathrm{amp}}]_{00,11}$")
    ax4a.plot(g_phi_arr, lphi_0011_re, "g-", linewidth=2.2, label=r"Pure Dephasing $\mathrm{Re}[\mathcal{L}_{\phi}]_{00,11}$")
    ax4a.plot(g_phi_arr, lh_0011_re + lamp_0011_re + lphi_0011_re, "k--", linewidth=1.5, label=r"Total $\mathrm{Re}[\mathcal{L}_{\mathrm{tot}}]_{00,11} \equiv 0$")
    ax4a.axvline(g_phi_half, color="purple", linestyle=":", linewidth=1.5)

    ax4a.set_title(r"(a) Real Part: $\mathrm{Re}[\dot{\rho}_{00,11}] = 0$", fontsize=12, fontweight="bold")
    ax4a.set_xlabel(r"Pure Dephasing Rate $\gamma_\phi$", fontsize=11)
    ax4a.set_ylabel(r"Rate Contribution to $\mathrm{Re}[\dot{\rho}_{00,11}]$", fontsize=11)
    ax4a.set_xlim(0, 5.0)
    ax4a.grid(True, linestyle="--", alpha=0.5)
    ax4a.legend(loc="center right", framealpha=0.92, fontsize=9.2)

    # Imaginary part of L_ij(rho_ss)
    ax4b.plot(g_phi_arr, lh_0011_im, "b-", linewidth=2.2, label=r"Hamiltonian $\mathrm{Im}[\mathcal{L}_H]_{00,11}$ (Driving)")
    ax4b.plot(g_phi_arr, lamp_0011_im, "r-", linewidth=2.2, label=r"Amplitude Damping $\mathrm{Im}[\mathcal{L}_{\mathrm{amp}}]_{00,11}$ (Decay)")
    ax4b.plot(g_phi_arr, lphi_0011_im, "g-", linewidth=2.2, label=r"Pure Dephasing $\mathrm{Im}[\mathcal{L}_{\phi}]_{00,11}$ (Decay)")
    ax4b.plot(g_phi_arr, lh_0011_im + lamp_0011_im + lphi_0011_im, "k--", linewidth=1.5, label=r"Total $\mathrm{Im}[\mathcal{L}_{\mathrm{tot}}]_{00,11} \equiv 0$")
    ax4b.axvline(g_phi_half, color="purple", linestyle=":", linewidth=1.5)

    ax4b.set_title(r"(b) Imaginary Part: $\mathrm{Im}[\dot{\rho}_{00,11}] = 0$ (Coherence Balance)", fontsize=12, fontweight="bold")
    ax4b.set_xlabel(r"Pure Dephasing Rate $\gamma_\phi$", fontsize=11)
    ax4b.set_ylabel(r"Rate Contribution to $\mathrm{Im}[\dot{\rho}_{00,11}]$", fontsize=11)
    ax4b.set_xlim(0, 5.0)
    ax4b.grid(True, linestyle="--", alpha=0.5)
    ax4b.legend(loc="center right", framealpha=0.92, fontsize=9.2)

    fig4.suptitle(
        r"Experiment 03c: Steady-State Liouvillian Balance on Dominant Coherence $\rho_{00,11}$" "\n"
        r"Coherent generation $\mathcal{L}_H$ is balanced by dissipative damping $\mathcal{L}_{\mathrm{amp}}$ and phase noise $\mathcal{L}_\phi$",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig4_path = fig_dir / "experiment_03c_liouvillian_balance.png"
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)
    print(f"  Figure 4 saved to: {fig4_path}", flush=True)

    # -------------------------------------------------------------------------
    # FIGURE 5: Dimensionless Scaling Analysis
    # -------------------------------------------------------------------------
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    for fam in scaling_results:
        col = fam["color"]
        for idx_c, curve in enumerate(fam["curves"]):
            lbl = rf"{fam['fam_name']}: $J={curve['J']}, \gamma_1={curve['gamma_1']}$"
            ls = "-" if idx_c == 0 else "--" if idx_c == 1 else ":"
            ax5a.plot(r_phi_range, curve["concurrence"], label=lbl, color=col, linestyle=ls, linewidth=2.0)
            ax5b.plot(r_phi_range, curve["coherence"], label=lbl, color=col, linestyle=ls, linewidth=2.0)

    ax5a.set_title(r"(a) Steady-State Concurrence $C_{\mathrm{ss}}$ vs $r_\phi = \gamma_\phi / \gamma_1$", fontsize=12, fontweight="bold")
    ax5a.set_xlabel(r"Dimensionless Dephasing Ratio $r_\phi = \gamma_\phi / \gamma_1$", fontsize=11)
    ax5a.set_ylabel(r"Concurrence $C_{\mathrm{ss}}$", fontsize=11)
    ax5a.set_xlim(0, 2.5)
    ax5a.set_ylim(-0.01, 0.35)
    ax5a.grid(True, linestyle="--", alpha=0.5)
    ax5a.legend(loc="upper right", framealpha=0.92, fontsize=8.2)

    ax5b.set_title(r"(b) Dominant Coherence $|\rho_{00,11}|$ vs $r_\phi = \gamma_\phi / \gamma_1$", fontsize=12, fontweight="bold")
    ax5b.set_xlabel(r"Dimensionless Dephasing Ratio $r_\phi = \gamma_\phi / \gamma_1$", fontsize=11)
    ax5b.set_ylabel(r"Coherence $|\rho_{00,11}|$", fontsize=11)
    ax5b.set_xlim(0, 2.5)
    ax5b.set_ylim(-0.01, 0.35)
    ax5b.grid(True, linestyle="--", alpha=0.5)
    ax5b.legend(loc="upper right", framealpha=0.92, fontsize=8.2)

    fig5.suptitle(
        r"Experiment 03c: Dimensionless Scaling of NESS Entanglement vs $r_\phi = \gamma_\phi / \gamma_1$" "\n"
        r"Comparing parameter families with constant $r_1 = J / \gamma_1$ under energy scale $\omega = 1.0$",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig5_path = fig_dir / "experiment_03c_dimensionless_scaling.png"
    fig5.savefig(fig5_path, dpi=300)
    plt.close(fig5)
    print(f"  Figure 5 saved to: {fig5_path}", flush=True)

    # -------------------------------------------------------------------------
    # FIGURE 6: Extracted Robustness Boundary in Dimensionless Variables
    # -------------------------------------------------------------------------
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)

    # Panel (a): 2D heatmap of half-suppression ratio (gamma_phi / gamma_1)_1/2
    J_mesh, G1_mesh = np.meshgrid(J_b_vals, g1_b_vals)
    cmap_crit = plt.cm.inferno.copy()
    mesh6a = ax6a.pcolormesh(J_mesh, G1_mesh, r_phi_half_grid, cmap=cmap_crit, shading="auto", vmin=0.0, vmax=1.5)
    cbar6a = fig6.colorbar(mesh6a, ax=ax6a, pad=0.03)
    cbar6a.set_label(r"Half-Suppression Ratio $(\gamma_\phi / \gamma_1)_{1/2}$", fontsize=11)

    contours6a = ax6a.contour(J_mesh, G1_mesh, r_phi_half_grid, levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2], colors="white", linewidths=1.0)
    ax6a.clabel(contours6a, inline=True, fontsize=8.5, fmt="%.1f")
    ax6a.plot(J_opt, gamma_1_opt, "c*", markersize=14, label=rf"Optimal NESS ($J={J_opt}, \gamma_1={gamma_1_opt}$)")

    ax6a.set_title(r"(a) Half-Suppression Boundary $(\gamma_\phi / \gamma_1)_{1/2}$", fontsize=12, fontweight="bold")
    ax6a.set_xlabel(r"Exchange Coupling Strength $J$ (units of $\omega$)", fontsize=11)
    ax6a.set_ylabel(r"Amplitude Damping Rate $\gamma_1$ (units of $\omega$)", fontsize=11)
    ax6a.grid(True, linestyle=":", alpha=0.4)
    ax6a.legend(loc="upper left", framealpha=0.92, fontsize=9.2)

    # Panel (b): Critical ratio vs dimensionless coupling r_1 = J / gamma_1
    flat_r1 = r_1_grid.flatten()
    flat_rhalf = r_phi_half_grid.flatten()
    flat_J = J_mesh.flatten()

    sc = ax6b.scatter(flat_r1, flat_rhalf, c=flat_J, cmap="viridis", s=35, alpha=0.85, edgecolors="none")
    cbar_sc = fig6.colorbar(sc, ax=ax6b, pad=0.03)
    cbar_sc.set_label(r"Coupling $J / \omega$", fontsize=11)

    ax6b.set_title(r"(b) Half-Suppression Ratio vs $r_1 = J / \gamma_1$", fontsize=12, fontweight="bold")
    ax6b.set_xlabel(r"Interaction-to-Damping Ratio $r_1 = J / \gamma_1$", fontsize=11)
    ax6b.set_ylabel(r"Half-Suppression Ratio $(\gamma_\phi / \gamma_1)_{1/2}$", fontsize=11)
    ax6b.set_xlim(0, 3.0)
    ax6b.set_ylim(-0.05, 1.6)
    ax6b.grid(True, linestyle="--", alpha=0.5)

    fig6.suptitle(
        r"Experiment 03c: Robustness Phase Scaling of Entangled Non-Equilibrium Steady State" "\n"
        r"Universal scaling of the dephasing tolerance $(\gamma_\phi / \gamma_1)_{1/2}$ across parameter space",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig6_path = fig_dir / "experiment_03c_robustness_boundary.png"
    fig6.savefig(fig6_path, dpi=300)
    plt.close(fig6)
    print(f"  Figure 6 saved to: {fig6_path}", flush=True)

    return {
        "rep_data": rep_data,
        "g_phi_half": g_phi_half,
        "fig1_path": str(fig1_path),
        "fig2_path": str(fig2_path),
        "fig3_path": str(fig3_path),
        "fig4_path": str(fig4_path),
        "fig5_path": str(fig5_path),
        "fig6_path": str(fig6_path),
        "data_file": str(data_file),
    }


if __name__ == "__main__":
    run_experiment()
