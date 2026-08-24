# Research Agent Protocol

This protocol defines standard operating procedures and development rules for automated agents working within the `bipartit-open-spin` codebase.

---

## 1. Core Principles

1. **Strict Modularity**:
   - Quantum state construction belongs in `core/states.py`.
   - Operators and composite spaces belong in `core/operators.py`.
   - Hamiltonian terms belong in `dynamics/hamiltonian.py`.
   - Dissipation channels belong in `dynamics/dissipation.py`.
   - Time evolution logic belongs in `dynamics/simulation.py`.
   - Entanglement and correlation metrics belong in `analysis/entanglement.py`.
   - Sanity checks, physical validation, and benchmarks belong in `validation/diagnostics.py`.

2. **Subsystem Dimension Preservation**:
   - Never flatten or reshape composite bipartite density matrices into unstructured matrices `[[4], [4]]`.
   - Always preserve `dims = [[2, 2], [2, 2]]` so that partial transpose (`ptrace`), concurrence, and negativity work without dimension mismatch errors.

3. **No Modification to Legacy Experiments**:
   - `experiments/smoke_test.py` serves as the historical baseline and must remain operational and unmodified.

---

## 2. Verification and Diagnostic Protocol

Before accepting any simulation result or committing algorithmic extensions, agents must verify:

1. **Trace Preservation**:
   - Check that $|\mathrm{Tr}(\rho(t)) - 1| < 10^{-6}$ across all time steps.
2. **Hermiticity**:
   - Check that $||\rho(t) - \rho^\dagger(t)||_\infty < 10^{-6}$.
3. **Positivity**:
   - Check that the minimum eigenvalue $\lambda_{\min}(\rho(t)) \ge -10^{-6}$.
4. **Physical Decay Benchmarks**:
   - Verify uncoupled decay rates against analytical solutions ($P_e(t) = P_e(0) e^{-\gamma t}$).

> [!NOTE]
> Open-system dynamics are trace-preserving and completely positive, not unitary. Do not use the term "trajectory unitarity" when describing dissipative evolution.

---

## 3. Testing Rules

- Every new module must have a corresponding test in `tests/test_<module>.py`.
- Tests must be executable using Python's standard `unittest` framework:
  ```powershell
  uv run python -m unittest discover -s tests -p "test_*.py"
  ```
- All tests must pass before considering a research milestone complete.
