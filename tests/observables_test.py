import numpy as np
import pytest
import scipy.linalg as la
from matchgates import Observable, ProductState, operators as ops


q0 = ops.q0
Z = ops.Z
X = ops.X
Y = ops.Y
Id = ops.Id


def test_obs():
    # Test making an Observable object.
    obs = Observable("XY", [0, 1], 3)
    assert obs.name == "XY"
    assert obs.qubits == (0, 1)
    assert obs.n_qubits == 3
    assert obs.majoranas == [1, 3]
    assert obs.phase == -1j
    with pytest.raises(Exception):
        obs = Observable("ZY", (0, 1), 3)


def test_computation_of_observables():
    obs = Observable("YY", [0, 1], 2)

    theta = 0.13
    phi = 0.45
    gamma = 0.24
    kappa = 0.3
    beta = 0.034
    alpha = np.zeros((4, 4), dtype=complex)
    alpha[1, 2] = theta / 2
    alpha[0, 1] = phi / 2
    alpha[2, 3] = phi / 2
    alpha[0, 3] = -gamma / 2
    alpha[0, 2] = -kappa / 2
    alpha[1, 3] = beta / 2

    alpha[2, 1] = -alpha[1, 2]
    alpha[1, 0] = -alpha[0, 1]
    alpha[3, 2] = -alpha[2, 3]
    alpha[3, 0] = -alpha[0, 3]
    alpha[2, 0] = -alpha[0, 2]
    alpha[3, 1] = -alpha[1, 3]

    T = la.expm(4 * alpha)
    state = ProductState([q0, q0])
    YY_expec = obs.compute_expectation(T, state)
    U1 = la.expm(
        1j
        * (
            theta * np.kron(X, X)
            + phi * np.kron(Z, Id)
            + phi * np.kron(Id, Z)
            + gamma * np.kron(Y, Y)
            + kappa * np.kron(Y, X)
            + beta * np.kron(X, Y)
        )
    )
    s00 = np.array([1, 0, 0, 0], dtype=complex)
    ev = U1 @ s00
    YYs = ev.conj().T @ np.kron(Y, Y) @ ev

    # Test that computation of observable matches matrix-vector multiplication.
    assert np.isclose(YY_expec, YYs)
