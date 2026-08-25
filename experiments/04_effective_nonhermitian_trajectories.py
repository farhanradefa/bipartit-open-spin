"""Experiment 04: Effective Non-Hermitian Dynamics and Quantum Trajectories.

This experiment investigates where non-Hermiticity enters the open-system description:
1. Analytical derivation and matrix representation of H_eff.
2. Full Lindblad master-equation evolution (mesolve).
3. Effective non-Hermitian no-jump evolution (H_eff, P_no_jump, conditional state).
4. Quantum trajectory / Monte Carlo wavefunction evolution (mcsolve).
5. Reconstruction of Lindblad dynamics from stochastic trajectories.
6. Convergence analysis as Ntraj increases.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis, tensor

from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import computational_basis, bell_phi_plus, to_density_matrix
from bipartit_open_spin.core.operators import sigma_x1, sigma_x2, sigma_z1, sigma_z2, sigma_m1, sigma_m2
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian, build_effective_hamiltonian
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators
from bipartit_open_spin.dynamics.simulation import (
    simulate_dynamics,
    simulate_no_jump_dynamics,
    simulate_quantum_trajectories,
)
from bipartit_open_spin.analysis.entanglement import concurrence, negativity
from bipartit_open_spin.validation.diagnostics import (
    validate_state_trajectory,
    check_trace_preservation,
    check_hermiticity,
    check_positivity,
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


def run_analytical_derivation():
    """Derive and display explicit operator structure of H_eff in computational basis."""
    print("=" * 80)
    print("PART 1: ANALYTICAL STRUCTURE OF EFFECTIVE NON-HERMITIAN HAMILTONIAN")
    print("=" * 80)

    omega = 1.0
    J = 0.5
    gamma = 0.5
    params = ModelParams(omega=omega, J=J, gamma=gamma)
    H = build_hamiltonian(params)
    c_ops = build_collapse_operators(params)
    H_eff = build_effective_hamiltonian(params, c_ops)

    H_mat = H.full()
    Heff_mat = H_eff.full()
    decay_mat = (Heff_mat - H_mat) / (-0.5j)

    print("\nHamiltonian H (Hermitian part):")
    print(np.round(H_mat.real, 4))

    print("\nEffective Hamiltonian H_eff = H - (i/2) sum_k L_k^dag L_k:")
    print("Real part:")
    print(np.round(Heff_mat.real, 4))
    print("Imaginary part:")
    print(np.round(Heff_mat.imag, 4))

    print("\nDecay operator sum_k L_k^dag L_k = gamma (n_1 + n_2):")
    print(np.round(decay_mat.real, 4))

    # Explicit basis state decay rates
    basis_labels = ["|00>", "|01>", "|10>", "|11>"]
    print("\nState-dependent decay rates (Gamma_n = 2 * |Im<n|H_eff|n>|):")
    for i, label in enumerate(basis_labels):
        gamma_n = 2.0 * abs(Heff_mat[i, i].imag)
        loss_rate = float(decay_mat[i, i].real)
        print(f"  {label}: Im(H_eff) = {Heff_mat[i, i].imag:+.4f}, Decay Rate Gamma = {gamma_n:.4f}, Jump Rate = {loss_rate:.4f}")

    return H, H_eff, c_ops


def run_comparative_dynamics(tlist, gamma_values, initial_states):
    """Run Lindblad, No-jump, and Quantum Trajectory simulations for all parameter regimes."""
    print("\n" + "=" * 80)
    print("PART 2: COMPARATIVE DYNAMICAL SIMULATION (LINDBLAD vs NO-JUMP vs MONTE CARLO)")
    print("=" * 80)

    omega = 1.0
    J = 0.5
    config = SimulationConfig(tlist=tlist)

    results = {}

    for state_name, psi0 in initial_states.items():
        print(f"\n>>> Initial State: {state_name}")
        results[state_name] = {}

        for gamma in gamma_values:
            print(f"  Simulating gamma = {gamma:.2f}...")
            params = ModelParams(omega=omega, J=J, gamma=gamma)
            H = build_hamiltonian(params)
            c_ops = build_collapse_operators(params)
            H_eff = build_effective_hamiltonian(params, c_ops)

            # ----------------------------------------------------
            # 1. Full Lindblad Master Equation (mesolve)
            # ----------------------------------------------------
            states_lb = simulate_dynamics(H, psi0, c_ops, config)
            val_lb = validate_state_trajectory(states_lb)
            if not val_lb["valid"]:
                print(f"    WARNING: Lindblad trajectory invalid: {val_lb['message']}")

            c_lb = np.array([concurrence(s) for s in states_lb])
            n_lb = np.array([negativity(s) for s in states_lb])
            purity_lb = np.array([float(np.real((s * s).tr())) for s in states_lb])
            trace_lb = np.array([float(np.real(s.tr())) for s in states_lb])

            # Populations
            p00_lb = np.array([float(np.real(s.full()[0, 0])) for s in states_lb])
            p01_lb = np.array([float(np.real(s.full()[1, 1])) for s in states_lb])
            p10_lb = np.array([float(np.real(s.full()[2, 2])) for s in states_lb])
            p11_lb = np.array([float(np.real(s.full()[3, 3])) for s in states_lb])

            # ----------------------------------------------------
            # 2. Non-Hermitian No-Jump Evolution
            # ----------------------------------------------------
            no_jump_dict = simulate_no_jump_dynamics(H_eff, psi0, config, c_ops_for_loss=c_ops)
            p_surv = no_jump_dict["survival_probability"]
            loss_rate = no_jump_dict["theoretical_loss_rate"]
            states_nj_cond = no_jump_dict["conditional_states"]

            # Conditional state entanglement and populations
            dm_nj_cond = [to_density_matrix(s) for s in states_nj_cond]
            c_nj = np.array([concurrence(s) for s in dm_nj_cond])
            n_nj = np.array([negativity(s) for s in dm_nj_cond])

            p00_nj = np.array([float(np.real(s.full()[0, 0])) for s in dm_nj_cond])
            p01_nj = np.array([float(np.real(s.full()[1, 1])) for s in dm_nj_cond])
            p10_nj = np.array([float(np.real(s.full()[2, 2])) for s in dm_nj_cond])
            p11_nj = np.array([float(np.real(s.full()[3, 3])) for s in dm_nj_cond])

            # ----------------------------------------------------
            # 3. Quantum Trajectories / Monte Carlo (mcsolve)
            # ----------------------------------------------------
            ntraj = 500
            mc_res = simulate_quantum_trajectories(H, psi0, c_ops, config, ntraj=ntraj, seed=100)
            states_mc = [to_density_matrix(s) for s in mc_res.states]

            c_mc = np.array([concurrence(s) for s in states_mc])
            n_mc = np.array([negativity(s) for s in states_mc])

            p00_mc = np.array([float(np.real(s.full()[0, 0])) for s in states_mc])
            p01_mc = np.array([float(np.real(s.full()[1, 1])) for s in states_mc])
            p10_mc = np.array([float(np.real(s.full()[2, 2])) for s in states_mc])
            p11_mc = np.array([float(np.real(s.full()[3, 3])) for s in states_mc])

            # Frobenius distance Delta_rho(t) = ||rho_MC(t) - rho_LB(t)||_F
            dist_rho = np.array([
                np.linalg.norm(rho_mc.full() - rho_lb.full())
                for rho_mc, rho_lb in zip(states_mc, states_lb)
            ])

            print(f"    Done. Mean ||rho_MC - rho_LB|| = {np.mean(dist_rho):.4e}, Max P_no_jump(t_end) = {p_surv[-1]:.4e}")

            results[state_name][gamma] = {
                "lindblad": {
                    "concurrence": c_lb,
                    "negativity": n_lb,
                    "purity": purity_lb,
                    "trace": trace_lb,
                    "p00": p00_lb,
                    "p01": p01_lb,
                    "p10": p10_lb,
                    "p11": p11_lb,
                    "states": states_lb,
                },
                "no_jump": {
                    "survival_prob": p_surv,
                    "loss_rate": loss_rate,
                    "concurrence": c_nj,
                    "negativity": n_nj,
                    "p00": p00_nj,
                    "p01": p01_nj,
                    "p10": p10_nj,
                    "p11": p11_nj,
                },
                "monte_carlo": {
                    "concurrence": c_mc,
                    "negativity": n_mc,
                    "p00": p00_mc,
                    "p01": p01_mc,
                    "p10": p10_mc,
                    "p11": p11_mc,
                    "dist_rho": dist_rho,
                    "states": states_mc,
                },
            }

    return results


def run_convergence_analysis(tlist, gamma=0.5):
    """Run Monte Carlo convergence study across multiple Ntraj values."""
    print("\n" + "=" * 80)
    print("PART 3: MONTE CARLO CONVERGENCE SCALING ANALYSIS")
    print("=" * 80)

    omega = 1.0
    J = 0.5
    params = ModelParams(omega=omega, J=J, gamma=gamma)
    H = build_hamiltonian(params)
    c_ops = build_collapse_operators(params)
    config = SimulationConfig(tlist=tlist)
    psi0 = bell_phi_plus()

    # Exact Lindblad solution
    states_lb = simulate_dynamics(H, psi0, c_ops, config)

    ntraj_list = [50, 100, 250, 500, 1000]
    convergence_data = {}

    for ntraj in ntraj_list:
        print(f"  Running Ntraj = {ntraj}...")
        mc_res = simulate_quantum_trajectories(H, psi0, c_ops, config, ntraj=ntraj, seed=42)
        states_mc = [to_density_matrix(s) for s in mc_res.states]

        dist_t = np.array([
            np.linalg.norm(rho_mc.full() - rho_lb.full())
            for rho_mc, rho_lb in zip(states_mc, states_lb)
        ])
        mean_dist = float(np.mean(dist_t))
        max_dist = float(np.max(dist_t))
        print(f"    Ntraj = {ntraj:4d} | Mean Frobenius Distance = {mean_dist:.4e} | Max Distance = {max_dist:.4e}")

        convergence_data[ntraj] = {
            "dist_t": dist_t,
            "mean_dist": mean_dist,
            "max_dist": max_dist,
        }

    return ntraj_list, convergence_data


def run_single_trajectories(tlist, gamma=0.5):
    """Generate individual quantum trajectories to illustrate stochastic jumps."""
    print("\n" + "=" * 80)
    print("PART 4: SINGLE QUANTUM TRAJECTORIES SAMPLING")
    print("=" * 80)

    from qutip import mcsolve
    omega = 1.0
    J = 0.5
    params = ModelParams(omega=omega, J=J, gamma=gamma)
    H = build_hamiltonian(params)
    c_ops = build_collapse_operators(params)
    psi0 = bell_phi_plus()

    # We run mcsolve with 3 trajectories, requesting individual trajectory state histories
    # In QuTiP, passing keep_runs_results=True preserves all individual trajectory wavefunctions
    options = {"progress_bar": None, "keep_runs_results": True}
    res = mcsolve(H, psi0, tlist, c_ops, ntraj=4, options=options, seeds=101)

    single_trajs = []
    # res.runs_states is a list of lists: [run_idx][time_idx]
    if hasattr(res, "runs_states") and res.runs_states is not None:
        for run in res.runs_states:
            p00_t = [float(np.real(to_density_matrix(s).full()[0, 0])) for s in run]
            p11_t = [float(np.real(to_density_matrix(s).full()[3, 3])) for s in run]
            c_t = [concurrence(to_density_matrix(s)) for s in run]
            single_trajs.append({"p00": p00_t, "p11": p11_t, "concurrence": c_t})
    else:
        # Fallback: run 4 individual ntraj=1 simulations with different seeds
        for s in [101, 202, 303, 404]:
            r = mcsolve(H, psi0, tlist, c_ops, ntraj=1, options={"progress_bar": None}, seeds=s)
            run = r.states
            p00_t = [float(np.real(to_density_matrix(st).full()[0, 0])) for st in run]
            p11_t = [float(np.real(to_density_matrix(st).full()[3, 3])) for st in run]
            c_t = [concurrence(to_density_matrix(st)) for st in run]
            single_trajs.append({"p00": p00_t, "p11": p11_t, "concurrence": c_t})

    print(f"  Successfully extracted {len(single_trajs)} stochastic trajectories.")
    return single_trajs


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_figure_1_norm_decay(tlist, sim_results):
    """Figure 1: Non-Hermitian Survival Probability P_no_jump(t) and Loss Rate Identity."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    colors = {0.05: "#1f77b4", 0.5: "#ff7f0e", 2.0: "#d62728"}
    labels = {0.05: r"Weak ($\gamma=0.05$)", 0.5: r"Intermediate ($\gamma=0.5$)", 2.0: r"Strong ($\gamma=2.0$)"}

    # Left: Case I |00>
    ax0 = axes[0]
    for gamma in [0.05, 0.5, 2.0]:
        data = sim_results["|00>"][gamma]["no_jump"]
        p_surv = data["survival_prob"]
        loss_rate = data["loss_rate"]

        lbl = r"$P_{\rm no-jump}(t)$ (" + labels[gamma] + ")"
        ax0.plot(tlist, p_surv, color=colors[gamma], label=lbl)

        # Numerical derivative check overlay
        dt = tlist[1] - tlist[0]
        dp_dt = np.gradient(p_surv, dt)
        ax0.plot(tlist[::12], p_surv[::12], "o", color=colors[gamma], markersize=3.5, fillstyle="none")

    ax0.set_title(r"(a) Initial State $|\psi(0)\rangle = |00\rangle$")
    ax0.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
    ax0.set_ylabel(r"Survival Probability $P_{\rm no\text{-}jump}(t)$")
    ax0.set_ylim(-0.02, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", framealpha=0.9)

    # Right: Case II |Phi+>
    ax1 = axes[1]
    for gamma in [0.05, 0.5, 2.0]:
        data = sim_results["|Phi+>"][gamma]["no_jump"]
        p_surv = data["survival_prob"]
        lbl = r"$P_{\rm no-jump}(t)$ (" + labels[gamma] + ")"
        ax1.plot(tlist, p_surv, color=colors[gamma], label=lbl)
        ax1.plot(tlist[::12], p_surv[::12], "o", color=colors[gamma], markersize=3.5, fillstyle="none")

    ax1.set_title(r"(b) Initial Bell State $|\psi(0)\rangle = |\Phi^+\rangle$")
    ax1.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_04_nonhermitian_norm_decay.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 1 saved to: {fig_path}")


def plot_figure_2_entanglement_comparison(tlist, sim_results):
    """Figure 2: Concurrence Comparison (Lindblad vs No-Jump Conditional vs Monte Carlo)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey="row")

    gamma_list = [0.05, 0.5, 2.0]
    titles_gamma = [r"Weak Dissipation ($\gamma=0.05$)", r"Intermediate Dissipation ($\gamma=0.5$)", r"Strong Dissipation ($\gamma=2.0$)"]

    # Row 0: Case I |00> (Generation)
    for col, gamma in enumerate(gamma_list):
        ax = axes[0, col]
        res = sim_results["|00>"][gamma]
        c_lb = res["lindblad"]["concurrence"]
        c_nj = res["no_jump"]["concurrence"]
        c_mc = res["monte_carlo"]["concurrence"]

        ax.plot(tlist, c_lb, "k-", linewidth=2.0, label="Lindblad (Master Eq.)")
        ax.plot(tlist, c_mc, "b--", linewidth=1.8, label=r"Monte Carlo ($N_{\rm traj}=500$)")
        ax.plot(tlist, c_nj, "r:", linewidth=2.0, label="No-Jump Conditional")

        ax.set_title(titles_gamma[col])
        if col == 0:
            ax.set_ylabel(r"Concurrence $C(t)$" + "\n" + r"[Initial $|\psi(0)\rangle = |00\rangle$]")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    # Row 1: Case II |Phi+> (Decay)
    for col, gamma in enumerate(gamma_list):
        ax = axes[1, col]
        res = sim_results["|Phi+>"][gamma]
        c_lb = res["lindblad"]["concurrence"]
        c_nj = res["no_jump"]["concurrence"]
        c_mc = res["monte_carlo"]["concurrence"]

        ax.plot(tlist, c_lb, "k-", linewidth=2.0, label="Lindblad (Master Eq.)")
        ax.plot(tlist, c_mc, "b--", linewidth=1.8, label=r"Monte Carlo ($N_{\rm traj}=500$)")
        ax.plot(tlist, c_nj, "r:", linewidth=2.0, label="No-Jump Conditional")

        ax.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
        if col == 0:
            ax.set_ylabel(r"Concurrence $C(t)$" + "\n" + r"[Initial $|\psi(0)\rangle = |\Phi^+\rangle$]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_04_entanglement_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 2 saved to: {fig_path}")


def plot_figure_3_population_comparison(tlist, sim_results, gamma=0.5):
    """Figure 3: Population Dynamics (Lindblad vs Conditional No-Jump vs Monte Carlo)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    # Panel A: Case I |00>
    ax0 = axes[0]
    res_00 = sim_results["|00>"][gamma]
    ax0.plot(tlist, res_00["lindblad"]["p00"], "b-", label=r"$P_{00}$ (Lindblad)")
    ax0.plot(tlist, res_00["lindblad"]["p11"], "r-", label=r"$P_{11}$ (Lindblad)")
    ax0.plot(tlist, res_00["lindblad"]["p01"], "g-", label=r"$P_{01}=P_{10}$ (Lindblad)")

    ax0.plot(tlist, res_00["monte_carlo"]["p00"], "b--", alpha=0.7, label=r"$P_{00}$ (MC)")
    ax0.plot(tlist, res_00["monte_carlo"]["p11"], "r--", alpha=0.7, label=r"$P_{11}$ (MC)")

    ax0.plot(tlist, res_00["no_jump"]["p00"], "b:", linewidth=2.2, label=r"$P_{00}^c$ (No-Jump)")
    ax0.plot(tlist, res_00["no_jump"]["p11"], "r:", linewidth=2.2, label=r"$P_{11}^c$ (No-Jump)")

    ax0.set_title(rf"(a) Initial $|\psi(0)\rangle = |00\rangle$ ($\gamma={gamma}$)")
    ax0.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
    ax0.set_ylabel("Subspace Population")
    ax0.set_ylim(-0.02, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="center right", fontsize=8.5, framealpha=0.9)

    # Panel B: Case II |Phi+>
    ax1 = axes[1]
    res_phi = sim_results["|Phi+>"][gamma]
    ax1.plot(tlist, res_phi["lindblad"]["p00"], "b-", label=r"$P_{00}$ (Lindblad)")
    ax1.plot(tlist, res_phi["lindblad"]["p11"], "r-", label=r"$P_{11}$ (Lindblad)")
    ax1.plot(tlist, res_phi["lindblad"]["p01"], "g-", label=r"$P_{01}=P_{10}$ (Lindblad)")

    ax1.plot(tlist, res_phi["monte_carlo"]["p00"], "b--", alpha=0.7, label=r"$P_{00}$ (MC)")
    ax1.plot(tlist, res_phi["monte_carlo"]["p11"], "r--", alpha=0.7, label=r"$P_{11}$ (MC)")

    ax1.plot(tlist, res_phi["no_jump"]["p00"], "b:", linewidth=2.2, label=r"$P_{00}^c$ (No-Jump)")
    ax1.plot(tlist, res_phi["no_jump"]["p11"], "r:", linewidth=2.2, label=r"$P_{11}^c$ (No-Jump)")

    ax1.set_title(rf"(b) Initial $|\psi(0)\rangle = |\Phi^+\rangle$ ($\gamma={gamma}$)")
    ax1.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_04_population_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 3 saved to: {fig_path}")


def plot_figure_4_convergence(tlist, ntraj_list, convergence_data):
    """Figure 4: Monte Carlo Ensemble Convergence ||rho_MC(t) - rho_Lindblad(t)||_F."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: Delta_rho(t) vs time for multiple Ntraj
    ax0 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ntraj_list)))

    for i, ntraj in enumerate(ntraj_list):
        dist_t = convergence_data[ntraj]["dist_t"]
        ax0.plot(tlist, dist_t, color=colors[i], label=rf"$N_{{\rm traj}} = {ntraj}$")

    ax0.set_title(r"(a) Density Matrix Distance $\|\rho_{\rm MC}(t) - \rho_{\rm Lindblad}(t)\|_F$")
    ax0.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
    ax0.set_ylabel(r"Frobenius Distance $\Delta_\rho(t)$")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", framealpha=0.9)

    # Panel B: Time-Averaged Error vs Ntraj (Log-Log)
    ax1 = axes[1]
    mean_errors = [convergence_data[n]["mean_dist"] for n in ntraj_list]
    ax1.loglog(ntraj_list, mean_errors, "s-", color="#1f77b4", linewidth=2.0, markersize=7, label=r"Ensemble Error $\langle \Delta_\rho \rangle_t$")

    # Reference 1/sqrt(N) line
    c_fit = mean_errors[0] * np.sqrt(ntraj_list[0])
    ref_scaling = [c_fit / np.sqrt(n) for n in ntraj_list]
    ax1.loglog(ntraj_list, ref_scaling, "k--", label=r"Ideal $1/\sqrt{N_{\rm traj}}$ Scaling")

    ax1.set_title(r"(b) Statistical Convergence vs Ensemble Size $N_{\rm traj}$")
    ax1.set_xlabel(r"Number of Trajectories $N_{\rm traj}$")
    ax1.set_ylabel(r"Time-Averaged Error $\langle \Delta_\rho \rangle_t$")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_04_density_matrix_convergence.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 4 saved to: {fig_path}")


def plot_figure_5_single_trajectories(tlist, single_trajs, sim_results, gamma=0.5):
    """Figure 5: Single Quantum Trajectories Illustrating Smooth Non-Hermitian Drift & Stochastic Jumps."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.0), sharex=True, sharey=True)

    res_phi = sim_results["|Phi+>"][gamma]
    c_lb = res_phi["lindblad"]["concurrence"]
    c_nj = res_phi["no_jump"]["concurrence"]

    traj_colors = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

    for i in range(4):
        row, col = divmod(i, 2)
        ax = axes[row, col]

        if i < len(single_trajs):
            traj = single_trajs[i]
            ax.plot(tlist, traj["concurrence"], color=traj_colors[i], linewidth=1.8, label=rf"Trajectory #{i+1} (Single Realization)")
            ax.plot(tlist, traj["p00"], color="blue", linestyle=":", alpha=0.7, label=r"$P_{00}(t)$")
            ax.plot(tlist, traj["p11"], color="red", linestyle=":", alpha=0.7, label=r"$P_{11}(t)$")

        ax.plot(tlist, c_lb, "k-", linewidth=1.5, alpha=0.6, label="Lindblad Ensemble Mean")
        ax.plot(tlist, c_nj, "k--", linewidth=1.2, alpha=0.4, label="No-Jump Conditional")

        ax.set_title(rf"Quantum Trajectory Realization #{i+1}")
        if col == 0:
            ax.set_ylabel("Value")
        if row == 1:
            ax.set_xlabel(r"Time $t$ [$\omega^{-1}$]")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_04_single_trajectory_examples.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Figure 5 saved to: {fig_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 04: EFFECTIVE NON-HERMITIAN DYNAMICS & QUANTUM TRAJECTORIES")
    print("=" * 80)

    # 1. Analytical derivation
    H, H_eff, c_ops = run_analytical_derivation()

    # 2. Time grid and parameters
    tlist = np.linspace(0.0, 10.0, 250)
    gamma_values = [0.05, 0.5, 2.0]
    initial_states = {
        "|00>": computational_basis(0, 0),
        "|Phi+>": bell_phi_plus(),
    }

    # 3. Run comparative simulations
    sim_results = run_comparative_dynamics(tlist, gamma_values, initial_states)

    # 4. Run convergence study
    ntraj_list, convergence_data = run_convergence_analysis(tlist, gamma=0.5)

    # 5. Run single trajectory samples
    single_trajs = run_single_trajectories(tlist, gamma=0.5)

    # 6. Save Data
    npz_data = {
        "tlist": tlist,
        "gamma_values": np.array(gamma_values),
        "ntraj_list": np.array(ntraj_list),
    }

    for state_name in ["|00>", "|Phi+>"]:
        key_prefix = "state_00" if state_name == "|00>" else "state_phi"
        for gamma in gamma_values:
            g_str = str(gamma).replace(".", "_")
            r = sim_results[state_name][gamma]
            npz_data[f"{key_prefix}_gamma_{g_str}_lb_concurrence"] = r["lindblad"]["concurrence"]
            npz_data[f"{key_prefix}_gamma_{g_str}_lb_negativity"] = r["lindblad"]["negativity"]
            npz_data[f"{key_prefix}_gamma_{g_str}_lb_p00"] = r["lindblad"]["p00"]
            npz_data[f"{key_prefix}_gamma_{g_str}_lb_p11"] = r["lindblad"]["p11"]

            npz_data[f"{key_prefix}_gamma_{g_str}_nj_survival_prob"] = r["no_jump"]["survival_prob"]
            npz_data[f"{key_prefix}_gamma_{g_str}_nj_concurrence"] = r["no_jump"]["concurrence"]
            npz_data[f"{key_prefix}_gamma_{g_str}_nj_negativity"] = r["no_jump"]["negativity"]

            npz_data[f"{key_prefix}_gamma_{g_str}_mc_concurrence"] = r["monte_carlo"]["concurrence"]
            npz_data[f"{key_prefix}_gamma_{g_str}_mc_negativity"] = r["monte_carlo"]["negativity"]
            npz_data[f"{key_prefix}_gamma_{g_str}_mc_dist_rho"] = r["monte_carlo"]["dist_rho"]

    for ntraj in ntraj_list:
        npz_data[f"convergence_dist_ntraj_{ntraj}"] = convergence_data[ntraj]["dist_t"]
        npz_data[f"convergence_mean_dist_ntraj_{ntraj}"] = convergence_data[ntraj]["mean_dist"]

    data_path = DATA_DIR / "experiment_04_nonhermitian_dynamics.npz"
    np.savez_compressed(data_path, **npz_data)
    print(f"\nSaved numerical dataset to: {data_path}")

    # 7. Generate Figures
    print("\n" + "=" * 80)
    print("PART 5: GENERATING PUBLICATION FIGURES")
    print("=" * 80)
    plot_figure_1_norm_decay(tlist, sim_results)
    plot_figure_2_entanglement_comparison(tlist, sim_results)
    plot_figure_3_population_comparison(tlist, sim_results, gamma=0.5)
    plot_figure_4_convergence(tlist, ntraj_list, convergence_data)
    plot_figure_5_single_trajectories(tlist, single_trajs, sim_results, gamma=0.5)

    print("\n" + "=" * 80)
    print("EXPERIMENT 04 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
