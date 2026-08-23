import numpy as np
import matplotlib.pyplot as plt

from qutip import (
    basis,
    tensor,
    qeye,
    sigmax,
    sigmaz,
    sigmam,
    mesolve,
)

# ==========================================
# PARAMETERS
# ==========================================

omega = 1.0
J = 0.5
gamma = 0.1

tlist = np.linspace(0, 20, 400)

# ==========================================
# TWO-QUBIT OPERATORS
# ==========================================

sx1 = tensor(sigmax(), qeye(2))
sx2 = tensor(qeye(2), sigmax())

sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())

sm1 = tensor(sigmam(), qeye(2))
sm2 = tensor(qeye(2), sigmam())

# ==========================================
# HAMILTONIAN
# ==========================================

H = (
    0.5 * omega * (sz1 + sz2)
    + J * sx1 * sx2
)

# ==========================================
# INITIAL BELL STATE
# ==========================================

zero = basis(2, 0)
one = basis(2, 1)

psi0 = (
    tensor(zero, zero)
    + tensor(one, one)
).unit()

# ==========================================
# LINDBLAD COLLAPSE OPERATORS
# ==========================================

c_ops = [
    np.sqrt(gamma) * sm1,
    np.sqrt(gamma) * sm2,
]

# ==========================================
# TIME EVOLUTION
# ==========================================

result = mesolve(
    H,
    psi0,
    tlist,
    c_ops,
)

# ==========================================
# POPULATION
# ==========================================

rho_t = result.states

P00 = [
    abs(state.full()[0, 0])
    for state in rho_t
]

# ==========================================
# PLOT
# ==========================================

plt.plot(tlist, P00)

plt.xlabel("Time")
plt.ylabel("Population |00>")
plt.title("Two-Qubit Open-System Smoke Test")

plt.show()
