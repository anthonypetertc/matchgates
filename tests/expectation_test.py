import numpy as np
import scipy.linalg as la
from matchgates import expectations as expec, operators as ops, ProductState

X = ops.X
Y = ops.Y
Z = ops.Z
Id = ops.Id

q0 = np.array([1, 0], dtype=complex)
q1 = np.array([0, 1], dtype=complex)


def test_minors():
    # Check that minors is correctly computing the determinant of the matrix of minors.
    array = np.random.rand(6, 6)  # random matrix.
    S = [0, 1]
    minors = expec.minors(array, S)

    # Check it works on 0, 1.
    assert np.isclose(minors[(0, 1)], np.linalg.det(array[:2, :2]))

    # Check it works on any 2 indices.
    ll = sorted(np.random.choice(range(0, 6), 2, replace=False))
    i, j = ll[0], ll[1]
    minor_mat = np.array([[array[0, i], array[1, i]], [array[0, j], array[1, j]]])
    assert np.isclose(minors[(i, j)], np.linalg.det(minor_mat))


def test_c_set():
    # Test that we are correctly computing the expectation values of Majorana Fermion
    # Creation operators (c).
    state = ProductState([q0, q0, q0])  # initial state.
    ex = expec.c_set_expectation([1], state)  # c1 - creation op on mode 1.
    assert ex == 0

    ex = expec.c_set_expectation([0, 1], state)  # c0c1 - creation ops on mode 0, 1.
    assert np.isclose(ex, 1j)

    ex = expec.c_set_expectation([4, 3, 2, 5], state)  # multi-mode expectation values.
    assert np.isclose(ex, 1)

    new_state = ProductState([q0, q1, q0])  # Now try it on a different state.
    ex = expec.c_set_expectation([4, 3, 2, 5], new_state)
    assert np.isclose(ex, -1)


def test_exp_from_T():
    # Test code that computes expectation values of multi-mode Majorana Fermion creation operators
    # from the T matrix.
    # Create a Matchgate Unitary and corresponding T matrix.
    theta = 0.2
    phi = 0.3
    gamma = 0.1
    kappa = 0.05
    beta = 0.43
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

    T = la.expm(4 * alpha)  # T-matrix.
    state = ProductState([q0, q0])
    Z0 = -1j * expec.expectation_from_T([0, 1], T, state)  # compute expectation value.

    # Make corresponding Unitary gate.
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
    # Evolution by matrix-vector multiplication.
    ev = U1 @ s00
    Z0s = ev.conj().T @ np.kron(Z, Id) @ ev
    assert np.isclose(
        Z0, Z0s
    ), f"Expected results{Z0s}, MatchGate simulator output {Z0}"
