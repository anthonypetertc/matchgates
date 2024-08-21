import numpy as np
import scipy.linalg as la
from functools import reduce

from MatchGates import (
    operators as ops,
    expectations as expec,
    MatchGate,
    AppliedMatchGate,
    SingleGateNoise,
    ProductState,
)


def test_MG():
    # Test generation of Matchgate from dict specifying the values of parameters for
    # each allowed Pauli generator of the Matchgate.
    param_dict = {}
    param_dict["z1"] = np.random.rand()
    param_dict["z2"] = np.random.rand()
    param_dict["xx"] = np.random.rand()
    param_dict["yy"] = np.random.rand()
    param_dict["xy"] = np.random.rand()
    param_dict["yx"] = np.random.rand()
    mg = MatchGate(param_dict)
    assert mg.param_dict == param_dict

    param_dict = {}
    param_dict["z1"] = 0
    param_dict["z2"] = 0
    param_dict["xx"] = np.random.rand()
    param_dict["yy"] = 0
    param_dict["xy"] = 0
    param_dict["yx"] = 0
    mg = MatchGate(param_dict)
    assert np.allclose(mg.gate, la.expm(1j * param_dict["xx"] * ops.XX))


def test_Unitary():
    # Test generation of Matchgates from a suitable Unitary.
    param_dict = {}
    param_dict["z1"] = np.random.rand()
    param_dict["z2"] = np.random.rand()
    param_dict["xx"] = np.random.rand()
    param_dict["yy"] = np.random.rand()
    param_dict["xy"] = np.random.rand()
    param_dict["yx"] = np.random.rand()
    H = (
        param_dict["z1"] * ops.Z1
        + param_dict["z2"] * ops.Z2
        + param_dict["xx"] * ops.XX
        + param_dict["yy"] * ops.YY
        + param_dict["xy"] * ops.XY
        + param_dict["yx"] * ops.YX
    )
    U = np.round(la.expm(1j * H), 11)
    mg = MatchGate.from_unitary(U)
    for key in mg.param_dict.keys():
        assert np.isclose(mg.param_dict[key], param_dict[key])


def test_fromAB():
    # Test generation of Matchgates from A and B matrices with equal determinant.
    param_dict = {}
    param_dict["z1"] = np.random.rand()
    param_dict["z2"] = np.random.rand()
    param_dict["xx"] = np.random.rand()
    param_dict["yy"] = np.random.rand()
    param_dict["xy"] = np.random.rand()
    param_dict["yx"] = np.random.rand()

    H = (
        param_dict["z1"] * ops.Z1
        + param_dict["z2"] * ops.Z2
        + param_dict["xx"] * ops.XX
        + param_dict["yy"] * ops.YY
        + param_dict["xy"] * ops.XY
        + param_dict["yx"] * ops.YX
    )
    U = np.round(la.expm(1j * H), 11)

    A = np.zeros((2, 2), dtype=complex)  # A should be the four corners of the Unitary.
    A[0, 0] = U[0, 0]
    A[0, 1] = U[0, 3]
    A[1, 0] = U[3, 0]
    A[1, 1] = U[3, 3]

    B = U[1:3, 1:3]  # B should be the center-block of the Unitary.

    mg = MatchGate.from_AB(A, B)
    for key in mg.param_dict.keys():
        assert np.isclose(mg.param_dict[key], param_dict[key])


def test_swap():
    # Test swap method. This should swap direction the matchgate acts from
    # qubits: (0, 1) to qubits: (1, 0)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    param_dict = {}
    param_dict["z1"] = np.random.rand()
    param_dict["z2"] = np.random.rand()
    param_dict["xx"] = np.random.rand()
    param_dict["yy"] = np.random.rand()
    param_dict["xy"] = np.random.rand()
    param_dict["yx"] = np.random.rand()
    sparam_dict = {}
    sparam_dict["z1"] = param_dict["z2"]
    sparam_dict["z2"] = param_dict["z1"]
    sparam_dict["xx"] = param_dict["xx"]
    sparam_dict["yy"] = param_dict["yy"]
    sparam_dict["xy"] = param_dict["yx"]
    sparam_dict["yx"] = param_dict["xy"]
    mg = MatchGate(param_dict)
    U = mg.gate
    smg = mg.swap()
    assert smg.param_dict == sparam_dict
    assert np.allclose(mg.gate, U)
    assert np.allclose(smg.gate, swap @ U @ swap)


def test_applied_mg():
    # Test creating an AppliedMatchGate from a MatchGate.
    mg = MatchGate.random_matchgate()  # Make a random MatchGate.

    # Now create an AppliedMatchGate and use it to compute observables.
    n_qubits = 4
    a = np.random.randint(3)
    acts_on = (a, a + 1)
    amg = AppliedMatchGate(mg, n_qubits, acts_on)
    state = ProductState([ops.q0, ops.q0, ops.q0, ops.q0])
    Z0 = -1j * expec.expectation_from_T([0, 1], amg.T, state)
    Z1 = -1j * expec.expectation_from_T([2, 3], amg.T, state)
    Z2 = -1j * expec.expectation_from_T([4, 5], amg.T, state)
    Z3 = -1j * expec.expectation_from_T([6, 7], amg.T, state)

    # Now do the same by matrix-vector multiplication.
    s0 = np.zeros(16)
    s0[0] = 1
    Id = ops.Id
    Z = ops.Z
    U = amg.convert_to_unitary()
    psi = U @ s0
    z0 = reduce(np.kron, [Z, Id, Id, Id])
    z1 = reduce(np.kron, [Id, Z, Id, Id])
    z2 = reduce(np.kron, [Id, Id, Z, Id])
    z3 = reduce(np.kron, [Id, Id, Id, Z])

    # Check that observables agree.
    assert np.isclose(psi.conj().T @ z0 @ psi, Z0)
    assert np.isclose(psi.conj().T @ z1 @ psi, Z1)
    assert np.isclose(psi.conj().T @ z2 @ psi, Z2)
    assert np.isclose(psi.conj().T @ z3 @ psi, Z3)


def test_single_gate_noise():
    # Test the creation of a noise channel.
    mg_list = []
    for _ in range(4):
        mg_list.append(MatchGate.random_matchgate())
    probabilities = np.random.rand(5)
    probabilities = (probabilities / sum(probabilities)).tolist()
    id_prob = probabilities[0]

    # Make the noise channel.
    sgn = SingleGateNoise(list(zip(mg_list, probabilities[1:])), id_prob)
    assert sgn.probabilities == probabilities[1:]
    assert sgn.scaled_kraus_ops == mg_list
    assert sgn.id_prob == id_prob
