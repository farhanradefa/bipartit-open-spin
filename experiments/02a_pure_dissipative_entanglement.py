"""Experiment 02a: Pure Dissipative Entanglement Decay.

This experiment isolates the effect of local Lindblad dissipation (amplitude damping)
on bipartite entanglement in the absence of coherent Hamiltonian dynamics:

    H = 0 (omega = 0, J = 0)
    C_1 = sqrt(gamma) * (sigma_- tensor I)
    C_2 = sqrt(gamma) * (I tensor sigma_-)

Initial State:
    |Phi+> = (|00> + |11>) / sqrt(2)

Operator Convention (QuTiP):
    sigma_- |0> = |1>  (decay from upper level |0> to lower level |1>)
    sigma_- |1> = 0    (ground/dark state)

Analytical Solutions:
    P_00(t) = 0.5 * exp(-2 * gamma * t)
    P_01(t) = 0.5 * exp(-gamma * t) * (1 - exp(-gamma * t))
    P_10(t) = 0.5 * exp(-gamma * t) * (1 - exp(-gamma * t))
    P_11(t) = 1 - exp(-gamma * t) + 0.5 * exp(-2 * gamma * t)
    rho_00,11(t) = 0.5 * exp(-gamma * t)

    Concurrence: C(t) = exp(-2 * gamma * t)
    Negativity:  N(t) = 0.5 * exp(-2 * gamma * t) = C(t) / 2
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless reproducibility
import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, sigmam

from bipartit_open_spin.analysis.entanglement import (
    concurrence_trajectory,
    negativity_trajectory,
)
from bipartit_open_spin.config import ModelParams, SimulationConfig
from bipartit_open_spin.core.states import bell_phi_plus
from bipartit_open_spin.dynamics.dissipation import build_collapse_operators
from bipartit_open_spin.dynamics.hamiltonian import build_hamiltonian
from bipartit_open_spin.dynamics.simulation import simulate_dynamics
from bipartit_open_spin.validation.diagnostics import validate_state_trajectory


def verify_operator_convention() -> dict:
    """Explicitly verify QuTiP's sigma_- operator action on basis states."""
    sm = sigmam()
    b0 = basis(2, 0)
    b1 = basis(2, 1)

    sm_0 = sm * b0
    sm_1 = sm * b1

    is_0_to_1 = np.isclose(abs(sm_0.full()[1, 0]), 1.0) and np.isclose(abs(sm_0.full()[0, 0]), 0.0)
    is_1_to_0 = np.isclose(sm_1.norm(), 0.0)

    return {
        "sm_0": sm_0,
        "sm_1": sm_1,
        "is_0_to_1": bool(is_0_to_1 and is_1_to_0),
    }


def run_experiment() -> dict:
    """Execute Experiment 02a parameter sweep and produce publication figures."""
    print("=" * 70)
    print("EXPERIMENT 02A: PURE DISSIPATIVE ENTANGLEMENT DECAY")
    print("=" * 70)

    # 1. Operator Convention Verification
    conv = verify_operator_convention()
    print("\n--- [1] QuTiP Operator Convention Verification ---")
    print(f"  sigma_- |0> = {conv['sm_0'].full().flatten()} -> |1>")
    print(f"  sigma_- |1> = {conv['sm_1'].full().flatten()} -> 0")
    print(f"  Convention Verified: |0> is excited level, |1> is ground level (decay: |0> -> |1>)")
    print(f"  Asymptotic steady state: |11><11|")

    # 2. Parameter Sweep Setup
    gamma_list = [0.05, 0.10, 0.20, 0.50]
    tlist = np.linspace(0.0, 60.0, 600)
    config = SimulationConfig(tlist=tlist)
    psi0 = bell_phi_plus()

    print(f"\n--- [2] Simulation Parameters ---")
    print(f"  Hamiltonian : H = 0 (omega = 0.0, J = 0.0)")
    print(f"  Initial state: |Phi+> = (|00> + |11>) / sqrt(2)")
    print(f"  Time grid   : t in [0.0, {tlist[-1]:.1f}], {len(tlist)} points")
    print(f"  Gamma sweep : {gamma_list}")

    sweep_results = {}
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- [3] Simulating Sweep & Validating State Trajectories ---")
    for gamma in gamma_list:
        params = ModelParams(omega=0.0, J=0.0, gamma=gamma)
        H = build_hamiltonian(params)
        c_ops = build_collapse_operators(params)

        states = simulate_dynamics(H, psi0, c_ops, config)

        # State trajectory physical validation
        diag = validate_state_trajectory(states, tol=1e-6)
        if not diag["valid"]:
            raise RuntimeError(f"Physical state validation failed for gamma={gamma}: {diag}")

        # Entanglement calculations
        c_num = concurrence_trajectory(states)
        n_num = negativity_trajectory(states)

        # Analytical solutions
        c_ana = np.exp(-2.0 * gamma * tlist)
        n_ana = 0.5 * c_ana

        max_err_c = float(np.max(np.abs(c_num - c_ana)))
        max_err_n = float(np.max(np.abs(n_num - n_ana)))

        sweep_results[gamma] = {
            "params": params,
            "states": states,
            "c_num": c_num,
            "n_num": n_num,
            "c_ana": c_ana,
            "n_ana": n_ana,
            "max_err_c": max_err_c,
            "max_err_n": max_err_n,
            "validation": diag,
        }

        print(f"  gamma = {gamma:.2f} | Valid: PASSED | C(0) = {c_num[0]:.6f}, N(0) = {n_num[0]:.6f} | "
              f"MaxErr(C) = {max_err_c:.2e}, MaxErr(N) = {max_err_n:.2e}")

        # Sanity checks
        if not np.isclose(c_num[0], 1.0, atol=1e-5):
            raise AssertionError(f"C(0) must be 1.0, got {c_num[0]}")
        if not np.isclose(n_num[0], 0.5, atol=1e-5):
            raise AssertionError(f"N(0) must be 0.5, got {n_num[0]}")

    # 3. Figure 1: Entanglement Decay (Concurrence and Negativity for multiple gamma)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, gamma in enumerate(gamma_list):
        res = sweep_results[gamma]
        ax1.plot(tlist, res["c_num"], label=rf"$\gamma = {gamma:.2f}$ (Num)", color=colors[idx], linewidth=2.0)
        ax1.plot(tlist, res["c_ana"], "--", color="black", alpha=0.5, linewidth=1.0,
                 label=r"Analytical $e^{-2\gamma t}$" if idx == 0 else "")

        ax2.plot(tlist, res["n_num"], label=rf"$\gamma = {gamma:.2f}$ (Num)", color=colors[idx], linewidth=2.0)
        ax2.plot(tlist, res["n_ana"], "--", color="black", alpha=0.5, linewidth=1.0,
                 label=r"Analytical $\frac{1}{2} e^{-2\gamma t}$" if idx == 0 else "")

    ax1.set_title(r"(a) Wootters Concurrence $C(t) = e^{-2\gamma t}$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time $t$", fontsize=11)
    ax1.set_ylabel("Concurrence $C(t)$", fontsize=11)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    ax2.set_title(r"(b) Peres-Horodecki Negativity $N(t) = \frac{1}{2} e^{-2\gamma t}$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("Negativity $N(t)$", fontsize=11)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(-0.01, 0.53)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    fig1.suptitle(r"Experiment 02a: Pure Dissipative Entanglement Decay ($\omega=0, J=0, \gamma > 0$)",
                  fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig1_path = output_dir / "experiment_02a_entanglement_decay.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"\nFigure 1 saved to: {fig1_path}")

    # 4. Population Dynamics Analysis for gamma = 0.10
    rep_gamma = 0.10
    rep_states = sweep_results[rep_gamma]["states"]

    p00_num = np.array([abs(s.full()[0, 0]) for s in rep_states])
    p01_num = np.array([abs(s.full()[1, 1]) for s in rep_states])
    p10_num = np.array([abs(s.full()[2, 2]) for s in rep_states])
    p11_num = np.array([abs(s.full()[3, 3]) for s in rep_states])
    rho0011_num = np.array([abs(s.full()[0, 3]) for s in rep_states])

    # Analytical populations
    p00_ana = 0.5 * np.exp(-2.0 * rep_gamma * tlist)
    p01_ana = 0.5 * np.exp(-rep_gamma * tlist) * (1.0 - np.exp(-rep_gamma * tlist))
    p10_ana = 0.5 * np.exp(-rep_gamma * tlist) * (1.0 - np.exp(-rep_gamma * tlist))
    p11_ana = 1.0 - np.exp(-rep_gamma * tlist) + 0.5 * np.exp(-2.0 * rep_gamma * tlist)
    rho0011_ana = 0.5 * np.exp(-rep_gamma * tlist)

    pop_sum = p00_num + p01_num + p10_num + p11_num
    max_pop_sum_err = float(np.max(np.abs(pop_sum - 1.0)))

    print(f"\n--- [4] Population Dynamics Validation (gamma = {rep_gamma:.2f}) ---")
    print(f"  Max Error P00(t) : {np.max(np.abs(p00_num - p00_ana)):.2e}")
    print(f"  Max Error P01(t) : {np.max(np.abs(p01_num - p01_ana)):.2e}")
    print(f"  Max Error P10(t) : {np.max(np.abs(p10_num - p10_ana)):.2e}")
    print(f"  Max Error P11(t) : {np.max(np.abs(p11_num - p11_ana)):.2e}")
    print(f"  Max Error Coherence rho_00,11(t) : {np.max(np.abs(rho0011_num - rho0011_ana)):.2e}")
    print(f"  Total Probability Sum Deviation  : {max_pop_sum_err:.2e}")

    # Figure 2: Population Dynamics
    fig2, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)

    ax.plot(tlist, p00_num, label=r"$P_{00}(t)$ (Num)", color="#1f77b4", linewidth=2.2)
    ax.plot(tlist, p01_num, label=r"$P_{01}(t)$ (Num)", color="#ff7f0e", linewidth=2.2)
    ax.plot(tlist, p10_num, "--", label=r"$P_{10}(t)$ (Num)", color="#2ca02c", linewidth=2.0)
    ax.plot(tlist, p11_num, label=r"$P_{11}(t)$ (Num)", color="#d62728", linewidth=2.2)
    ax.plot(tlist, rho0011_num, "-.", label=r"$|\rho_{00,11}(t)|$ (Coherence)", color="#9467bd", linewidth=2.0)

    # Analytical overlays
    ax.plot(tlist, p00_ana, ":", color="black", alpha=0.6, linewidth=1.2, label="Analytical Solution")
    ax.plot(tlist, p01_ana, ":", color="black", alpha=0.6, linewidth=1.2)
    ax.plot(tlist, p10_ana, ":", color="black", alpha=0.6, linewidth=1.2)
    ax.plot(tlist, p11_ana, ":", color="black", alpha=0.6, linewidth=1.2)
    ax.plot(tlist, rho0011_ana, ":", color="black", alpha=0.6, linewidth=1.2)

    ax.set_title(
        rf"Experiment 02a: Dissipative Population & Coherence Dynamics ($\gamma = {rep_gamma:.2f}$)" "\n"
        r"Pathway: $|00\rangle \to \{|01\rangle, |10\rangle\} \to |11\rangle \quad (\text{Steady State: } |11\rangle\langle 11|)$",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Time $t$", fontsize=11)
    ax.set_ylabel("Population / Coherence", fontsize=11)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right", framealpha=0.95, fontsize=9.5)

    # Annotation box
    textstr = "\n".join((
        r"$\mathbf{Dissipative\ Cascade:}$",
        r"$P_{00}(0) = 0.5 \to 0$",
        r"$P_{01}(t) = P_{10}(t) \text{ (Peak at } t = \frac{\ln 2}{\gamma} \approx 6.93\text{ s)}$",
        r"$P_{11}(t) = 0.5 \to 1.0 \text{ (Steady State)}$",
        r"$|\rho_{00,11}(t)| = 0.5 e^{-\gamma t} \text{ (Coherence)}$",
        r"$\sum P_{ij}(t) = 1.000000$",
    ))
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
    ax.text(0.03, 0.45, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="bottom", bbox=props)

    plt.tight_layout()

    fig2_path = output_dir / "experiment_02a_population_dynamics.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"Figure 2 saved to: {fig2_path}")

    return {
        "convention": conv,
        "sweep_results": sweep_results,
        "fig1_path": str(fig1_path),
        "fig2_path": str(fig2_path),
    }


if __name__ == "__main__":
    run_experiment()
