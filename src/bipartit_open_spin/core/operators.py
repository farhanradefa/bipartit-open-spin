"""Composite spin operators on the bipartite Hilbert space H_A \\otimes H_B."""

from qutip import Qobj, qeye, sigmam, sigmax, sigmaz, tensor


def identity_2qubit() -> Qobj:
    """Return the 4x4 identity operator I_A \\otimes I_B."""
    return tensor(qeye(2), qeye(2))


def sigma_x1() -> Qobj:
    """Return sigma_x \\otimes I_B."""
    return tensor(sigmax(), qeye(2))


def sigma_x2() -> Qobj:
    """Return I_A \\otimes sigma_x."""
    return tensor(qeye(2), sigmax())


def sigma_z1() -> Qobj:
    """Return sigma_z \\otimes I_B."""
    return tensor(sigmaz(), qeye(2))


def sigma_z2() -> Qobj:
    """Return I_A \\otimes sigma_z."""
    return tensor(qeye(2), sigmaz())


def sigma_m1() -> Qobj:
    """Return sigma_- \\otimes I_B (lowering operator for subsystem A)."""
    return tensor(sigmam(), qeye(2))


def sigma_m2() -> Qobj:
    """Return I_A \\otimes sigma_- (lowering operator for subsystem B)."""
    return tensor(qeye(2), sigmam())
