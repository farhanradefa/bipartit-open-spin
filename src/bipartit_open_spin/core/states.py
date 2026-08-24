"""Quantum state constructors for bipartite spin systems."""

from qutip import Qobj, basis, ket2dm, tensor


def computational_basis(i: int, j: int) -> Qobj:
    """Generate bipartite computational basis state |ij> in H_A \\otimes H_B.

    Args:
        i: Computational state for subsystem A (0 or 1).
        j: Computational state for subsystem B (0 or 1).

    Returns:
        Qobj ket with dims [[2, 2], [1, 1]].
    """
    if i not in (0, 1) or j not in (0, 1):
        raise ValueError(f"Basis indices must be 0 or 1, got i={i}, j={j}")
    return tensor(basis(2, i), basis(2, j))


def bell_phi_plus() -> Qobj:
    """Generate normalized Bell state |Phi+> = (|00> + |11>) / sqrt(2).

    Returns:
        Qobj ket with dims [[2, 2], [1, 1]].
    """
    state = computational_basis(0, 0) + computational_basis(1, 1)
    return state.unit()


def to_density_matrix(state: Qobj) -> Qobj:
    """Convert a ket or density matrix to a density matrix preserving bipartite dimensions.

    Args:
        state: A Qobj representing either a ket or a density matrix.

    Returns:
        Qobj density matrix with dims [[2, 2], [2, 2]].

    Raises:
        ValueError: If state is not a ket or operator, or if dimensions are incompatible with shape (4, 4).
    """
    if state.isket:
        dm = ket2dm(state)
    elif state.isoper:
        dm = state
    else:
        raise ValueError(f"Expected ket or density matrix operator, got type {state.type}")

    if dm.shape != (4, 4):
        raise ValueError(
            f"Incompatible Hilbert-space dimensions: expected matrix shape (4, 4), got shape {dm.shape}"
        )

    if dm.dims != [[2, 2], [2, 2]]:
        # Enforce bipartite subsystem dimensions
        dm = Qobj(dm.full(), dims=[[2, 2], [2, 2]])
    return dm

