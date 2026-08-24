# Mathematical and Software Conventions

This document establishes the physical, mathematical, and software conventions for the `bipartit-open-spin` project.

---

## 1. Hilbert Space Architecture

- **Composite Space**: The system Hilbert space is bipartite:
  $$\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B$$
  where $\dim(\mathcal{H}_A) = 2$ and $\dim(\mathcal{H}_B) = 2$, yielding a 4-dimensional composite space.
- **Subsystem Ordering**: Subsystem $A$ is the first tensor factor (index 1), and subsystem $B$ is the second tensor factor (index 2).
- **QuTiP Subsystem Dimensions**: All composite operators and bipartite density matrices must preserve the explicit bipartite dimensions:
  ```python
  dims = [[2, 2], [2, 2]]
  ```
  Pure state kets have dimensions `[[2, 2], [1, 1]]`.

---

## 2. Basis States and Computational Ordering

- **Single-Qubit Computational Basis**:
  $$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$
  In QuTiP spin/two-level conventions, $|0\rangle$ represents the upper / spin-up state ($\sigma_z = +1$) and $|1\rangle$ represents the lower / spin-down state ($\sigma_z = -1$).

- **Two-Qubit Computational Basis**:
  $$|00\rangle = |0\rangle \otimes |0\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, \quad
    |01\rangle = |0\rangle \otimes |1\rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}$$
  $$|10\rangle = |1\rangle \otimes |0\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, \quad
    |11\rangle = |1\rangle \otimes |1\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$

- **Baseline Initial Bell State**:
  $$|\Phi^+\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle)$$

---

## 3. Operator Conventions

- **Pauli Matrices**:
  $$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
    \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
    \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

- **Lowering Operators**:
  $$\sigma_- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix} = |1\rangle\langle 0|$$
  $\sigma_-$ lowers the upper state $|0\rangle$ into the lower state $|1\rangle$.

- **Two-Qubit Lifted Operators**:
  $$\sigma_{x1} = \sigma_x \otimes I_2, \quad \sigma_{x2} = I_2 \otimes \sigma_x$$
  $$\sigma_{z1} = \sigma_z \otimes I_2, \quad \sigma_{z2} = I_2 \otimes \sigma_z$$
  $$\sigma_{-1} = \sigma_- \otimes I_2, \quad \sigma_{-2} = I_2 \otimes \sigma_-$$

---

## 4. Density Matrices and State Processing

- **Density Matrix Form**: All analysis (entanglement measures) and validation (diagnostics) functions strictly accept density matrices $\rho$.
- **Ket-to-Density-Matrix Coercion**: Any function accepting a general state input must explicitly coerce kets into density matrices using `to_density_matrix(state)` to maintain dimension integrity.
- **Physical Invariants**:
  - $\mathrm{Tr}(\rho) = 1$ (Trace preservation)
  - $\rho = \rho^\dagger$ (Hermiticity)
  - $\lambda_i(\rho) \ge 0$ (Positive semi-definiteness)
