import numpy as np
from scipy.linalg import expm
from MatchGates.generators import XY_circuit
from MatchGates.operators import X, Y, Z
from MatchGates.expectations import expectation_from_T
from MatchGates import (
    ProductState,
    Observable,
    MatchGate,
    AppliedMatchGate,
)


def test_XY():
    J = 0.5
    h = 0.23
    dt = 0.1
    n_qubits = 2
    trotter_steps = 1
    circuit = XY_circuit(n_qubits, J, h, dt, trotter_steps)
    state = ProductState.neel(even=True, n_qubits=n_qubits)
    obs_list = []
    obs_list.append(Observable("XY", [0, 1], n_qubits))
    results = circuit.simulate(obs_list, state)

    init_state = np.zeros(4)
    init_state[2] = 1
    RZ = expm(-1j * h * dt / 2 * Z)
    RXXYY = expm(-1j * J * dt * (np.kron(X, X) + np.kron(Y, Y)))
    evolved_state = np.kron(RZ, RZ) @ RXXYY @ np.kron(RZ, RZ) @ init_state
    resXY = evolved_state.conjugate().transpose() @ np.kron(X, Y) @ evolved_state

    U = np.kron(RZ, RZ) @ RXXYY @ np.kron(RZ, RZ)
    mg = MatchGate.from_unitary(U)
    amg = AppliedMatchGate(mg, n_qubits=2, acts_on=[0, 1])
    T = amg.T
    Tres = expectation_from_T([1, 3], T, state)
    assert np.isclose(resXY, -1j * Tres)
    Tcirc = circuit.T
    circ1 = expectation_from_T([1, 3], Tcirc, state)
    assert np.isclose(resXY, -1j * circ1)
    assert np.isclose(resXY, results[("XY", (0, 1))])
