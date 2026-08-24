"""Experiment 03a: Pure Dephasing Baseline.

This experiment isolates local pure dephasing (phase noise) from amplitude relaxation
to establish a clean baseline for bipartite spin entanglement dynamics.

Model:
    H = (omega / 2) * (sigma_z1 + sigma_z2) + J * (sigma_x1 * sigma_x2)  [omega = 1.0, J = 0.5]
    L_phi^(1) = sqrt(gamma_phi / 2) * sigma_z1
    L_phi^(2) = sqrt(gamma_phi / 2) * sigma_z2

Collapse Operator Convention & Analytical Derivation:
    For single-qubit pure dephasing with L_phi = sqrt(gamma_phi / 2) * sigma_z,
    the Lindblad dissipator is:
        D[L_phi] rho = (gamma_phi / 2) * (sigma_z rho sigma_z - rho)
    which yields single-qubit off-diagonal decay:
        d/dt rho_01(t) = -gamma_phi * rho_01(t) => rho_01(t) = rho_01(0) * exp(-gamma_phi * t).

    For the bipartite Bell state |Phi+> = (|00> + |11>) / sqrt(2) at H = 0 under independent local
    dephasing on both qubits, each channel contributes rate gamma_phi to the coherence decay:
        d/dt rho_00,11(t) = -2 * gamma_phi * rho_00,11(t) => rho_00,11(t) = 0.5 * exp(-2 * gamma_phi * t).
    Populations P_00 = P_11 = 0.5, P_01 = P_10 = 0.0 remain strictly conserved for all t.

Investigations:
    1. Analytical benchmark at H = 0 (coherence decay and population conservation).
    2. Case A: Interacting Bell-state dephasing (destruction of pre-existing entanglement).
    3. Case B: Entanglement generation from separable |00> under dephasing.
    4. Entanglement Sudden Death (ESD) analysis.
    5. Asymptotic steady-state entanglement analysis.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend for publication reproducibility
import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj

from bipartit_open_spin.analysis.entanglement import (
    concurrence_trajectory,
    negativity_trajectory,
)
from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import bell_phi_plus, computational_basis
from bipartit_open_spin.dynamics.dissipation import build_dephasing_collapse_operators
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.dynamics.simulation import simulate_dynamics
from bipartit_open_spin.validation.diagnostics import validate_state_trajectory


def run_experiment() -> dict:
    """Execute Experiment 03a, run benchmarks, validate states, and generate all figures."""
    print("=" * 75)
    print("EXPERIMENT 03A: PURE DEPHASING BASELINE")
    print("=" * 75)

    # 1. Model & Simulation Parameters
    omega = 1.0
    J = 0.5
    gamma_phi_list = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
    tlist = np.linspace(0.0, 30.0, 1000)
    config = SimulationConfig(tlist=tlist)

    print(f"\nModel Parameters: omega = {omega:.1f}, J = {J:.1f}")
    print(f"Dephasing Sweep : gamma_phi in {gamma_phi_list}")
    print(f"Time Grid       : t in [0.0, {tlist[-1]:.1f}], {len(tlist)} points")

    fig_dir = Path("results/figures")
    data_dir = Path("results/data")
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Color palette
    colors = {
        0.0: "#1f77b4",
        0.05: "#2ca02c",
        0.10: "#17becf",
        0.25: "#ff7f0e",
        0.50: "#d62728",
        1.00: "#9467bd",
        2.00: "#8c564b",
    }

    # =============================================================
    # 2. MANDATORY ANALYTICAL BENCHMARK (H = 0, Initial Bell |Phi+>)
    # =============================================================
    print("\n--- [1] Mandatory Analytical Benchmark: H = 0 Pure Dephasing ---")
    H0 = Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])
    psi_bell = bell_phi_plus()

    bench_results = {}
    max_coherence_errors = {}
    max_population_drifts = {}

    for g_phi in [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]:
        c_ops = build_dephasing_collapse_operators(g_phi)
        states_bench = simulate_dynamics(H0, psi_bell, c_ops, config)
        diag_b = validate_state_trajectory(states_bench, tol=1e-6)
        if not diag_b["valid"]:
            raise RuntimeError(f"State validation failed in H=0 benchmark for gamma_phi={g_phi}")

        rho_0011_num = np.array([abs(s.full()[0, 3]) for s in states_bench])
        rho_0011_ana = 0.5 * np.exp(-2.0 * g_phi * tlist)
        err_coh = float(np.max(np.abs(rho_0011_num - rho_0011_ana)))
        max_coherence_errors[g_phi] = err_coh

        p00 = np.array([s.full()[0, 0].real for s in states_bench])
        p01 = np.array([s.full()[1, 1].real for s in states_bench])
        p10 = np.array([s.full()[2, 2].real for s in states_bench])
        p11 = np.array([s.full()[3, 3].real for s in states_bench])

        drift = float(max(
            np.max(np.abs(p00 - 0.5)),
            np.max(np.abs(p11 - 0.5)),
            np.max(np.abs(p01 - 0.0)),
            np.max(np.abs(p10 - 0.0)),
        ))
        max_population_drifts[g_phi] = drift

        bench_results[g_phi] = {
            "states": states_bench,
            "rho_0011_num": rho_0011_num,
            "rho_0011_ana": rho_0011_ana,
            "p00": p00, "p01": p01, "p10": p10, "p11": p11,
            "err_coh": err_coh,
            "drift": drift,
        }

        print(f"  gamma_phi = {g_phi:4.2f} | Max Coherence Error: {err_coh:8.2e} | Max Population Drift: {drift:8.2e} | PASSED")

    # =============================================================
    # 3. CASE A: Interacting Bell-State Dephasing (H != 0, Initial |Phi+>)
    # =============================================================
    print("\n--- [2] Case A: Interacting Bell-State Dephasing (H != 0, Initial |Phi+>) ---")
    params = ModelParams(omega=omega, J=J)
    H = build_hamiltonian(params)
    case_a_results = {}

    for g_phi in gamma_phi_list:
        c_ops = build_dephasing_collapse_operators(g_phi) if g_phi > 0 else []
        states_a = simulate_dynamics(H, psi_bell, c_ops, config)
        diag_a = validate_state_trajectory(states_a, tol=1e-6)
        if not diag_a["valid"]:
            raise RuntimeError(f"State validation failed in Case A for gamma_phi={g_phi}")

        c_traj_a = concurrence_trajectory(states_a)
        n_traj_a = negativity_trajectory(states_a)

        # Entanglement Sudden Death (ESD) test (threshold C < 1e-8)
        zero_indices = np.where(c_traj_a < 1e-8)[0]
        t_esd = float(tlist[zero_indices[0]]) if len(zero_indices) > 0 else np.nan

        case_a_results[g_phi] = {
            "states": states_a,
            "c_traj": c_traj_a,
            "n_traj": n_traj_a,
            "c_final": float(c_traj_a[-1]),
            "n_final": float(n_traj_a[-1]),
            "t_esd": t_esd,
            "validation": diag_a,
        }

        print(f"  gamma_phi = {g_phi:4.2f} | C(0) = {c_traj_a[0]:.4f} | C(t_f) = {c_traj_a[-1]:8.2e} | "
              f"N(t_f) = {n_traj_a[-1]:8.2e} | t_ESD = {t_esd} | PASSED")

    # =============================================================
    # 4. CASE B: Entanglement Generation Under Dephasing (H != 0, Initial |00>)
    # =============================================================
    print("\n--- [3] Case B: Entanglement Generation under Dephasing (H != 0, Initial |00>) ---")
    psi00 = computational_basis(0, 0)
    case_b_results = {}

    for g_phi in gamma_phi_list:
        c_ops = build_dephasing_collapse_operators(g_phi) if g_phi > 0 else []
        states_b = simulate_dynamics(H, psi00, c_ops, config)
        diag_b = validate_state_trajectory(states_b, tol=1e-6)
        if not diag_b["valid"]:
            raise RuntimeError(f"State validation failed in Case B for gamma_phi={g_phi}")

        c_traj_b = concurrence_trajectory(states_b)
        n_traj_b = negativity_trajectory(states_b)

        idx_peak = int(np.argmax(c_traj_b))
        c_max = float(c_traj_b[idx_peak])
        t_peak = float(tlist[idx_peak])

        case_b_results[g_phi] = {
            "states": states_b,
            "c_traj": c_traj_b,
            "n_traj": n_traj_b,
            "c_max": c_max,
            "t_peak": t_peak,
            "c_final": float(c_traj_b[-1]),
            "n_final": float(n_traj_b[-1]),
            "validation": diag_b,
        }

        print(f"  gamma_phi = {g_phi:4.2f} | C_max = {c_max:6.4f} at t = {t_peak:5.2f} s | "
              f"C(t_f) = {c_traj_b[-1]:8.2e} | N(t_f) = {n_traj_b[-1]:8.2e} | PASSED")

    # =============================================================
    # 5. Save Machine-Readable Data
    # =============================================================
    data_file = data_dir / "experiment_03a_pure_dephasing.npz"
    np.savez_compressed(
        data_file,
        tlist=tlist,
        gamma_phi_values=np.array(gamma_phi_list),
        case_a_concurrence=np.array([case_a_results[g]["c_traj"] for g in gamma_phi_list]),
        case_a_negativity=np.array([case_a_results[g]["n_traj"] for g in gamma_phi_list]),
        case_b_concurrence=np.array([case_b_results[g]["c_traj"] for g in gamma_phi_list]),
        case_b_negativity=np.array([case_b_results[g]["n_traj"] for g in gamma_phi_list]),
        case_b_c_max=np.array([case_b_results[g]["c_max"] for g in gamma_phi_list]),
        case_b_t_peak=np.array([case_b_results[g]["t_peak"] for g in gamma_phi_list]),
        max_coherence_errors=np.array([max_coherence_errors.get(g, 0.0) for g in gamma_phi_list]),
        max_population_drifts=np.array([max_population_drifts.get(g, 0.0) for g in gamma_phi_list]),
        omega=omega,
        J=J,
    )
    print(f"\nMachine-readable data saved to: {data_file}")

    # =============================================================
    # 6. FIGURE 1: Bell State Dephasing (Case A)
    # =============================================================
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    for g_phi in gamma_phi_list:
        res = case_a_results[g_phi]
        ax1a.plot(tlist, res["c_traj"], label=rf"$\gamma_\phi = {g_phi:.2f}$",
                  color=colors[g_phi], linewidth=2.0 if g_phi > 0 else 1.8,
                  linestyle="-" if g_phi > 0 else "--")
        ax1b.plot(tlist, res["n_traj"], label=rf"$\gamma_\phi = {g_phi:.2f}$",
                  color=colors[g_phi], linewidth=2.0 if g_phi > 0 else 1.8,
                  linestyle="-" if g_phi > 0 else "--")

    ax1a.set_title(r"(a) Concurrence Dynamics $C(t)$", fontsize=12, fontweight="bold")
    ax1a.set_xlabel("Time $t$", fontsize=11)
    ax1a.set_ylabel("Wootters Concurrence $C(t)$", fontsize=11)
    ax1a.set_xlim(0, 30)
    ax1a.set_ylim(-0.02, 1.04)
    ax1a.grid(True, linestyle="--", alpha=0.5)
    ax1a.legend(loc="upper right", framealpha=0.92, fontsize=9.2)

    ax1b.set_title(r"(b) Negativity Dynamics $N(t)$", fontsize=12, fontweight="bold")
    ax1b.set_xlabel("Time $t$", fontsize=11)
    ax1b.set_ylabel("Peres-Horodecki Negativity $N(t)$", fontsize=11)
    ax1b.set_xlim(0, 30)
    ax1b.set_ylim(-0.01, 0.52)
    ax1b.grid(True, linestyle="--", alpha=0.5)
    ax1b.legend(loc="upper right", framealpha=0.92, fontsize=9.2)

    fig1.suptitle(
        r"Experiment 03a: Bell-State Entanglement Destruction under Pure Dephasing" "\n"
        rf"$\omega = {omega:.1f},\ J = {J:.1f},\ |\psi(0)\rangle = |\Phi^+\rangle$",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig1_path = fig_dir / "experiment_03a_bell_state_dephasing.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"Figure 1 saved to: {fig1_path}")

    # =============================================================
    # 7. FIGURE 2: Analytical Coherence Decay Benchmark (H = 0)
    # =============================================================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    for g_phi in [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]:
        res = bench_results[g_phi]
        ax2a.plot(tlist, res["rho_0011_num"], label=rf"Num $\gamma_\phi = {g_phi:.2f}$",
                  color=colors[g_phi], linewidth=2.0)
        ax2a.plot(tlist, res["rho_0011_ana"], ":", label=rf"Ana: $\frac{{1}}{{2}}e^{{-2\gamma_\phi t}}$" if g_phi in [0.05, 0.50] else None,
                  color="black", linewidth=1.5, alpha=0.7)

        # Semi-log plot on right
        ax2b.semilogy(tlist, res["rho_0011_num"], label=rf"$\gamma_\phi = {g_phi:.2f}$",
                      color=colors[g_phi], linewidth=2.0)

    ax2a.set_title(r"(a) Off-Diagonal Coherence $|\rho_{00,11}(t)|$ (Linear)", fontsize=12, fontweight="bold")
    ax2a.set_xlabel("Time $t$", fontsize=11)
    ax2a.set_ylabel(r"Coherence $|\rho_{00,11}(t)|$", fontsize=11)
    ax2a.set_xlim(0, 20)
    ax2a.set_ylim(-0.01, 0.52)
    ax2a.grid(True, linestyle="--", alpha=0.5)
    ax2a.legend(loc="upper right", framealpha=0.92, fontsize=9.0)

    ax2b.set_title(r"(b) Coherence Decay (Log Scale: Slope $= -2\gamma_\phi$)", fontsize=12, fontweight="bold")
    ax2b.set_xlabel("Time $t$", fontsize=11)
    ax2b.set_ylabel(r"$\log_{10} |\rho_{00,11}(t)|$", fontsize=11)
    ax2b.set_xlim(0, 20)
    ax2b.set_ylim(1e-6, 0.6)
    ax2b.grid(True, linestyle="--", alpha=0.5, which="both")
    ax2b.legend(loc="upper right", framealpha=0.92, fontsize=9.0)

    fig2.suptitle(
        r"Experiment 03a: Analytical Coherence Decay Benchmark ($H = 0, |\psi(0)\rangle = |\Phi^+\rangle$)" "\n"
        r"$\dot{\rho}_{00,11} = -2\gamma_\phi \rho_{00,11} \rightarrow \rho_{00,11}(t) = \frac{1}{2}\exp(-2\gamma_\phi t)$",
        fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    fig2_path = fig_dir / "experiment_03a_coherence_decay.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"Figure 2 saved to: {fig2_path}")

    # =============================================================
    # 8. FIGURE 3: Entanglement Generation under Dephasing (Case B)
    # =============================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    for g_phi in gamma_phi_list:
        res_b = case_b_results[g_phi]
        ax3a.plot(tlist, res_b["c_traj"], label=rf"$\gamma_\phi = {g_phi:.2f}$",
                  color=colors[g_phi], linewidth=2.0 if g_phi > 0 else 1.8,
                  linestyle="-" if g_phi > 0 else "--")

    ax3a.set_title(r"(a) Generated Concurrence $C(t)$ from $|\psi(0)\rangle = |00\rangle$", fontsize=12, fontweight="bold")
    ax3a.set_xlabel("Time $t$", fontsize=11)
    ax3a.set_ylabel("Wootters Concurrence $C(t)$", fontsize=11)
    ax3a.set_xlim(0, 30)
    ax3a.set_ylim(-0.02, 0.85)
    ax3a.grid(True, linestyle="--", alpha=0.5)
    ax3a.legend(loc="upper right", framealpha=0.92, fontsize=9.2)

    # Panel (b): Peak generated entanglement C_max vs gamma_phi
    g_arr = np.array(gamma_phi_list)
    c_max_arr = np.array([case_b_results[g]["c_max"] for g in gamma_phi_list])
    n_max_arr = np.array([case_b_results[g]["n_traj"][int(np.argmax(case_b_results[g]["n_traj"]))] for g in gamma_phi_list])

    ax3b.plot(g_arr, c_max_arr, "o-", label=r"Max Concurrence $C_{\max}(\gamma_\phi)$", color="#1f77b4", linewidth=2.2, markersize=7)
    ax3b.plot(g_arr, n_max_arr, "s-", label=r"Max Negativity $N_{\max}(\gamma_\phi)$", color="#ff7f0e", linewidth=2.2, markersize=7)
    ax3b.axhline(0.05, color="#d62728", linestyle=":", linewidth=1.5, label=r"Operational Suppression Threshold ($C_{\max} = 0.05$)")

    ax3b.set_title(r"(b) Peak Generated Entanglement vs Dephasing Rate $\gamma_\phi$", fontsize=12, fontweight="bold")
    ax3b.set_xlabel(r"Dephasing Rate $\gamma_\phi$ (units of $\omega$)", fontsize=11)
    ax3b.set_ylabel("Maximum Entanglement", fontsize=11)
    ax3b.set_xlim(-0.05, 2.1)
    ax3b.set_ylim(-0.01, 0.85)
    ax3b.grid(True, linestyle="--", alpha=0.5)
    ax3b.legend(loc="upper right", framealpha=0.92, fontsize=9.2)

    fig3.suptitle(
        r"Experiment 03a: Entanglement Generation from Separable State under Dephasing" "\n"
        rf"$\omega = {omega:.1f},\ J = {J:.1f},\ |\psi(0)\rangle = |00\rangle$",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig3_path = fig_dir / "experiment_03a_entanglement_generation_dephasing.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"Figure 3 saved to: {fig3_path}")

    # =============================================================
    # 9. FIGURE 4: Population Benchmark (H = 0)
    # =============================================================
    fig4, axes4 = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=300, sharey=True)
    bench_gammas = [0.10, 0.50, 2.00]
    titles4 = [
        r"(a) $\gamma_\phi = 0.10$ (Weak Dephasing)",
        r"(b) $\gamma_\phi = 0.50$ (Intermediate Dephasing)",
        r"(c) $\gamma_\phi = 2.00$ (Strong Dephasing)",
    ]

    for ax, g_phi, t_title in zip(axes4, bench_gammas, titles4):
        res = bench_results[g_phi]
        ax.plot(tlist, res["p00"], label=r"$P_{00}(t)$ (Num)", color="#1f77b4", linewidth=2.2)
        ax.plot(tlist, res["p11"], "--", label=r"$P_{11}(t)$ (Num)", color="#d62728", linewidth=2.0)
        ax.plot(tlist, res["p01"], label=r"$P_{01}(t)$ (Num)", color="#2ca02c", linewidth=1.8)
        ax.plot(tlist, res["p10"], ":", label=r"$P_{10}(t)$ (Num)", color="#ff7f0e", linewidth=1.8)

        ax.axhline(0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.6, label="Exact $P=0.5$" if g_phi == 0.10 else None)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, label="Exact $P=0.0$" if g_phi == 0.10 else None)

        ax.set_title(t_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time $t$", fontsize=10.5)
        ax.set_xlim(0, 30)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        if g_phi == 0.10:
            ax.set_ylabel("Subspace Population", fontsize=11)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9.0)

    fig4.suptitle(
        r"Experiment 03a: Population Conservation Benchmark under Pure Dephasing ($H = 0, |\psi(0)\rangle = |\Phi^+\rangle$)" "\n"
        r"Pure dephasing introduces phase randomization without population relaxation: $P_i(t) = P_i(0)$ for all $t$",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig4_path = fig_dir / "experiment_03a_population_benchmark.png"
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)
    print(f"Figure 4 saved to: {fig4_path}")

    # =============================================================
    # 10. Summary Table Printout
    # =============================================================
    print("\n" + "=" * 95)
    print("EXPERIMENT 03A: NUMERICAL SUMMARY TABLE")
    print("=" * 95)
    print(f"{'gamma_phi':>10} | {'Case A C(t_f)':>14} | {'Case A N(t_f)':>14} | {'Case B C_max':>13} | {'Case B t_peak':>14} | {'Case B C(t_f)':>14}")
    print("-" * 95)
    for g in gamma_phi_list:
        res_a = case_a_results[g]
        res_b = case_b_results[g]
        print(f"{g:10.2f} | {res_a['c_final']:14.2e} | {res_a['n_final']:14.2e} | {res_b['c_max']:13.4f} | {res_b['t_peak']:11.2f} s | {res_b['c_final']:14.2e}")
    print("=" * 95)

    return {
        "bench_results": bench_results,
        "case_a_results": case_a_results,
        "case_b_results": case_b_results,
        "max_coherence_errors": max_coherence_errors,
        "max_population_drifts": max_population_drifts,
        "fig1_path": str(fig1_path),
        "fig2_path": str(fig2_path),
        "fig3_path": str(fig3_path),
        "fig4_path": str(fig4_path),
        "data_file": str(data_file),
    }


if __name__ == "__main__":
    run_experiment()
