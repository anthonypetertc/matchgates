import numpy as np
import scipy.linalg as la
from functools import reduce
from matchgates import (
    Circuit,
    ProductState,
    Observable,
    AppliedMatchGate,
    MatchGate,
    SingleGateNoise,
    operators as ops,
    expectations as expec,
)


Id = ops.Id
Z = ops.Z
X = ops.X
Y = ops.Y
q0 = ops.q0


def test_circuit():
    # Tests circuit evolution by an individual (pure) matchgate circuit.
    n_qubits = 4
    n_gates = 10
    gate_list = []

    for _ in range(n_gates):
        a = np.random.randint(n_qubits - 1)
        gate_list.append(
            AppliedMatchGate(
                MatchGate.random_matchgate(), n_qubits=n_qubits, acts_on=(a, a + 1)
            )
        )

    circuit = Circuit(n_qubits=n_qubits, gate_list=gate_list)  # make the circuit
    T = circuit.T

    state = ProductState(
        [ops.q0, ops.q0, ops.q0, ops.q0]
    )  # make initial product state |0000>

    # Compute expectation values directly using the exp_from_T function.
    Z0 = -1j * expec.expectation_from_T([0, 1], T, state)
    Z1 = -1j * expec.expectation_from_T([2, 3], T, state)
    Z2 = -1j * expec.expectation_from_T([4, 5], T, state)
    Z3 = -1j * expec.expectation_from_T([6, 7], T, state)

    # Perform the same circuit evolution by matrix-vector multiplication.
    s0 = np.zeros(16)
    s0[0] = 1
    ulist = [amg.convert_to_unitary() for amg in circuit.gate_list]
    U = reduce(np.matmul, reversed(ulist))
    psi = U @ s0
    z0 = reduce(np.kron, [Z, Id, Id, Id])
    z1 = reduce(np.kron, [Id, Z, Id, Id])
    z2 = reduce(np.kron, [Id, Id, Z, Id])
    z3 = reduce(np.kron, [Id, Id, Id, Z])

    # Compare results.
    assert np.isclose(psi.conj().T @ z0 @ psi, Z0)
    assert np.isclose(psi.conj().T @ z1 @ psi, Z1)
    assert np.isclose(psi.conj().T @ z2 @ psi, Z2)
    assert np.isclose(psi.conj().T @ z3 @ psi, Z3)

    # Compute expectation values using built in Observable class.
    z0obs = Observable("Z", [0], 4)
    xx12obs = Observable("XX", [1, 2], 4)
    xy23obs = Observable("XY", [2, 3], 4)
    yx01obs = Observable("YX", [0, 1], 4)
    zz03 = Observable("ZZ", [0, 3], 4)
    obs = [z0obs, xx12obs, xy23obs, yx01obs, zz03]
    results_dict = circuit.simulate(obs, state)
    z0 = reduce(np.kron, [Z, Id, Id, Id])
    xx12 = reduce(np.kron, [Id, X, X, Id])
    xy23 = reduce(np.kron, [Id, Id, X, Y])
    yx01 = reduce(np.kron, [Y, X, Id, Id])
    zz03 = reduce(np.kron, [Z, Id, Id, Z])

    # Compare to results of matrix-vector multiplication.
    assert np.isclose(psi.conj().T @ z0 @ psi, results_dict[("Z", (0,))])
    assert np.isclose(psi.conj().T @ xx12 @ psi, results_dict[("XX", (1, 2))])
    assert np.isclose(psi.conj().T @ xy23 @ psi, results_dict[("XY", (2, 3))])
    assert np.isclose(psi.conj().T @ yx01 @ psi, results_dict[("YX", (0, 1))])
    assert np.isclose(psi.conj().T @ zz03 @ psi, results_dict[("ZZ", (0, 3))])


def test_add_noise():
    # Test that add_uniform_noise to circuit adds correct noise to all qubits.
    n_qubits = 4
    n_gates = 10
    gate_list = []

    # Make the circuit.
    for _ in range(n_gates):
        a = np.random.randint(n_qubits - 1)
        gate_list.append(
            AppliedMatchGate(
                MatchGate.random_matchgate(), n_qubits=n_qubits, acts_on=(a, a + 1)
            )
        )
    circuit = Circuit(n_qubits=n_qubits, gate_list=gate_list)

    # Make the noise model.
    mg_list = []
    for _ in range(4):
        mg_list.append(MatchGate.random_matchgate())
    probabilities = np.random.rand(5)
    probabilities = (probabilities / sum(probabilities)).tolist()
    id_prob = probabilities[0]
    sgn = SingleGateNoise(list(zip(mg_list, probabilities[1:])), id_prob)

    noisy_circuit = circuit.add_uniform_noise_to_circuit(
        sgn
    )  # Add noise to the circuit.

    # Check that noisy_circuit has all correc gates in the right order and
    # has added correct noise to the gates.
    for gate, noisy_gate in zip(circuit.gate_list, noisy_circuit.noisy_gate_list):
        assert noisy_gate.applied_match_gate == gate
        assert noisy_gate.noise == sgn


def test_noisy_circuit():
    # Generate a gate on 2-qubits and add noise to that gate. Check that the noisy simulation
    # Gives correct result up to a generous margin of error.

    theta = 0.2
    phi = 0.3
    gamma = 0.1
    kappa = 0.05
    beta = 0.43

    state = ProductState([q0, q0])  # Make state.

    # Make Unitary.
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

    n_qubits = 2
    gate_list = []
    gate_list.append(
        AppliedMatchGate(MatchGate.from_unitary(U1), n_qubits=n_qubits, acts_on=(0, 1))
    )
    circuit = Circuit(n_qubits=n_qubits, gate_list=gate_list)  # Make circuit.

    # Create noise model.
    mg_list = [MatchGate.from_unitary(U1)]
    probabilities = [0.5, 0.5]
    id_prob = probabilities[0]
    sgn = SingleGateNoise(list(zip(mg_list, probabilities[1:])), id_prob)
    noisy_circuit = circuit.add_uniform_noise_to_circuit(sgn)  # Add noise to circuit.

    # Perform circuit evolution.
    vec = np.zeros(2**2)
    vec[0] = 1
    dm = np.outer(vec, vec)
    for noisy_gate in noisy_circuit.noisy_gate_list:
        amg = noisy_gate.applied_match_gate
        noise = noisy_gate.noise
        kops = [
            AppliedMatchGate(op, amg.n_qubits, amg.acts_on).convert_to_unitary()
            for op in noise.scaled_kraus_ops
        ]
        probs = noisy_gate.noise.probabilities
        gate = amg.convert_to_unitary()
        dm = gate @ dm @ gate.conj().transpose()
        dm = sum(
            [(1 - sum(probs)) * dm]
            + [
                prob * kraus @ dm @ kraus.conj().transpose()
                for prob, kraus in zip(probs, kops)
            ]
        )
    z0 = reduce(np.kron, [Z, Id])
    z0_exp = np.trace(z0 @ dm)

    Z0 = Observable("Z", qubits=[0], n_qubits=2)
    res_dict = noisy_circuit.simulate_noisy(16, [Z0], 10000, state)

    # Compare observables to exact results.
    assert np.isclose(res_dict["Z" + str((0,))], z0_exp, atol=0.02)
    # \    #f"Result of density matrix simulation: {z0_exp}, and noisy matchagate simulator{res_dict["Z" + str((0,))]}"
