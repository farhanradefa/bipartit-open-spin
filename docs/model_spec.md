# Physical Model Specification

This specification defines the baseline model for the dissipative entanglement dynamics of an open bipartite spin system.

---

## 1. System Hamiltonian

The baseline coherent interaction is governed by the two-qubit Hamiltonian ($\hbar = 1$):

$$H = \frac{\omega}{2} (\sigma_{z1} + \sigma_{z2}) + J \sigma_{x1} \sigma_{x2}$$

### Parameters
- $\omega \in \mathbb{R}$: Single-spin transition frequency / detuning.
- $J \in \mathbb{R}$: Transverse exchange coupling strength along the $x$-axis.

---

## 2. Dissipative Environment and Master Equation

The open-system dynamics are modeled by the Markovian Lindblad master equation:

$$\frac{d\rho(t)}{dt} = -i [H, \rho(t)] + \sum_{k=1}^2 \mathcal{D}[C_k]\rho(t)$$

where the Lindblad superoperator dissipator is:

$$\mathcal{D}[C]\rho = C \rho C^\dagger - \frac{1}{2} \{C^\dagger C, \rho\}$$

### Collapse Operators
The baseline dissipation corresponds to independent local spontaneous decay into a zero-temperature Markovian bath:

$$C_1 = \sqrt{\gamma} \sigma_{-1} = \sqrt{\gamma} (\sigma_- \otimes I_2)$$
$$C_2 = \sqrt{\gamma} \sigma_{-2} = \sqrt{\gamma} (I_2 \otimes \sigma_-)$$

- $\gamma \ge 0$: Spontaneous emission decay rate.

---

## 3. Initial State

The baseline research initial condition is the maximally entangled Bell state:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle)$$
$$\rho(0) = |\Phi^+\rangle \langle \Phi^+|$$

---

## 4. Entanglement and Correlation Measures

### Wootters Concurrence $C(\rho)$
For a general two-qubit density matrix $\rho$, the spin-flipped state is:
$$\tilde{\rho} = (\sigma_y \otimes \sigma_y) \rho^* (\sigma_y \otimes \sigma_y)$$
where $\rho^*$ is the element-wise complex conjugate in the standard computational basis. The concurrence is defined as:
$$C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$$
where $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \lambda_4 \ge 0$ are the square roots of the eigenvalues of the non-Hermitian matrix $\rho \tilde{\rho}$.
- $C(|\Phi^+\rangle) = 1$
- $C(\text{separable}) = 0$

### Peres-Horodecki Negativity $\mathcal{N}(\rho)$
Negativity quantifies entanglement via the violation of the positive partial transpose (PPT) criterion:
$$\mathcal{N}(\rho) = \frac{||\rho^{T_A}||_1 - 1}{2} = \sum_{\lambda_i < 0} |\lambda_i|$$
where $\rho^{T_A}$ is the partial transpose of $\rho$ with respect to subsystem $A$, and $\lambda_i$ are its eigenvalues.
- $\mathcal{N}(|\Phi^+\rangle) = 0.5$
- $\mathcal{N}(\text{separable}) = 0$

---

## 5. Analytical Validation Benchmark

For uncoupled spins ($J = 0$), a spin initialized in its excited state $|1\rangle$ evolves under independent decay $\sigma_-$ with excited-state population:

$$P_e(t) = P_e(0) e^{-\gamma t}$$

Numerical solvers must reproduce this exponential decay to within numerical integration tolerance ($< 10^{-3}$).
