"""Module with classes for defining matchgates."""

from functools import reduce
from numbers import Real
import numpy as np
import scipy.linalg as scipy_la

import jax.scipy.linalg as jax_la

from matchgates.operators import X, Y, Id, Z, Z1, Z2, XX, YY, XY, YX


class MatchGate:
    """
    Class defining a matchgate.

    Attributes:
    param_dict: dictionary of parameters for Pauli generators of the matchgate.
    gate: matrix representation of matchgate.
    Name (optional): Name of the object

    Methods:
    fromUnitary: create a MatchGate object from a unitary matrix.
    fromAB: create a MatchGate object from two 2x2 matrices.
    swap: swap qubits 1 and 2.
    randomMatchGate: create a random MatchGate object.
    empty_param_dict: create an empty parameter dictionary.
    is_single_qubit: check if the matchgate is tensor product of single qubit gates.
    """

    def __init__(self, param_dict, name: str=""):
        """
        Args:
        param_dict: dictionary of parameters for Pauli generators of the matchgate.

        Returns:
        MatchGate object.
        """
        assert param_dict.keys() == set(["z1", "z2", "xx", "yy", "xy", "yx"])
        self.param_dict = param_dict

        H = (
            param_dict["z1"] * Z1
            + param_dict["z2"] * Z2
            + param_dict["xx"] * XX
            + param_dict["yy"] * YY
            + param_dict["xy"] * XY
            + param_dict["yx"] * YX
        )
        self.gate = np.round(jax_la.expm(1j * H), 11)
        self.name = name

    @classmethod
    def from_unitary(cls, U: np.ndarray, name: str = ""):
        """
        Define a matchgate from a suitable 4x4 Unitary matrix.
        A Unitary is suitable if it can be written in the form:

        U = a00 0 0 a01
            0 b00 b01 0
            0 b10 b11 0
            a10 0 0 a11
        with det(A) = det(B).

        Args:
        U: 4x4 unitary matrix.

        Returns:
        MatchGate object.
        """
        U = np.round(U, 11)
        assert U.shape == (4, 4)
        assert np.allclose(U[0, 1:3], [0, 0])
        assert np.allclose(U[1:3, 0], [0, 0])
        assert np.allclose(U[3, 1:3], [0, 0])
        assert np.allclose(U[1:3, 3], [0, 0])
        A = np.zeros((2, 2), dtype=complex)
        A[0, 0] = U[0, 0]
        A[0, 1] = U[0, 3]
        A[1, 0] = U[3, 0]
        A[1, 1] = U[3, 3]
        B = U[1:3, 1:3]
        assert np.isclose(jax_la.det(A), jax_la.det(B))

        H = -1j * scipy_la.logm(U)
        assert np.allclose(H.conj().transpose(), H)
        param_dict = {}
        param_dict["z1"] = 0.25 * np.trace(Z1 @ H)
        param_dict["z2"] = 0.25 * np.trace(Z2 @ H)
        param_dict["xx"] = 0.25 * np.trace(XX @ H)
        param_dict["yy"] = 0.25 * np.trace(YY @ H)
        param_dict["xy"] = 0.25 * np.trace(XY @ H)
        param_dict["yx"] = 0.25 * np.trace(YX @ H)

        return MatchGate(param_dict, name=name)

    @classmethod
    def from_AB(self, A: np.ndarray, B: np.ndarray):
        """
        Define a matchgate from two 2x2 matrices A and B. With det(A) = det(B).

        Args:
        A: 2x2 np.ndarray.
        B: 2x2 np.ndarray.

        Returns:
        MatchGate object.
        """
        assert A.shape == (2, 2)
        assert B.shape == (2, 2)
        assert np.isclose(jax_la.det(A), jax_la.det(B))
        U = np.zeros((4, 4), dtype=complex)
        U[0, 0] = A[0, 0]
        U[0, 3] = A[0, 1]
        U[3, 0] = A[1, 0]
        U[3, 3] = A[1, 1]
        U[1, 1] = B[0, 0]
        U[1, 2] = B[0, 1]
        U[2, 1] = B[1, 0]
        U[2, 2] = B[1, 1]
        return MatchGate.from_unitary(U)

    def swap(self):
        """
        Takes a MatchGate acting on qubits (0, 1) and
        returns the same MatchGate acting on qubits (1, 0).

        Returns:
        MatchGate object with qubits 1 and 2 swapped.
        """
        param_dict = self.param_dict
        new_param_dict = {}
        new_param_dict["z1"] = param_dict["z2"]
        new_param_dict["z2"] = param_dict["z1"]
        new_param_dict["xx"] = param_dict["xx"]
        new_param_dict["yy"] = param_dict["yy"]
        new_param_dict["xy"] = param_dict["yx"]
        new_param_dict["yx"] = param_dict["xy"]
        return MatchGate(new_param_dict)

    @classmethod
    def random_matchgate(cls, seed=None):
        """
        Create a random MatchGate object.

        Args:
        seed: seed for random number generator.

        Returns:
        MatchGate object.
        """

        param_dict = {}
        if seed is not None:
            np.random.seed(seed)
        random_numbers = np.random.rand(6)
        param_dict["z1"] = random_numbers[0]
        param_dict["z2"] = random_numbers[1]
        param_dict["xx"] = random_numbers[2]
        param_dict["yy"] = random_numbers[3]
        param_dict["xy"] = random_numbers[4]
        param_dict["yx"] = random_numbers[5]
        return MatchGate(param_dict)

    @classmethod
    def empty_param_dict(cls):
        """
        Create an empty parameter dictionary.

        Returns:
        dict: dictionary with all parameters set to 0.
        """
        return {"z1": 0, "z2": 0, "xx": 0, "yy": 0, "xy": 0, "yx": 0}

    def is_single_qubit(self):
        """
        Check if the matchgate is a tensor product of single qubit gates.

        Returns:
        bool: True if matchgate is single qubit, False otherwise.
        """
        xx = self.param_dict["xx"]
        yy = self.param_dict["yy"]
        xy = self.param_dict["xy"]
        yx = self.param_dict["yx"]
        if np.allclose([xx, yy, xy, yx], 0, atol=1e-11):
            return True
        else:
            return False


class AppliedMatchGate:
    """
    Class defining a matchgate acting on qubits.

    Attributes:
    mg: MatchGate object.
    n_qubits: number of qubits the gate acts on.
    acts_on: qubits the gate acts on.
    T: matrix representation of the gate acting on all qubits.

    Methods:
    convert_to_unitary: convert the gate to a unitary acting on all qubits.
    """

    def __init__(self, mg: MatchGate, n_qubits: int, acts_on: list):
        assert len(acts_on) == 2
        if acts_on[1] < acts_on[0]:
            mg = mg.swap()
            acts_on = [acts_on[1], acts_on[0]]
        elif acts_on[1] == acts_on[0]:
            raise ValueError("Gate must act on two distinct qubits.")
        assert acts_on[1] == acts_on[0] + 1, "Matchgates must be nearest neighbour."
        assert acts_on[1] < n_qubits
        self.mg = mg
        self.n_qubits = n_qubits
        self.acts_on = acts_on
        k = acts_on[0]
        alpha = np.zeros((4, 4), dtype=complex)
        alpha[1, 2] = mg.param_dict["xx"] / 2
        alpha[0, 1] = mg.param_dict["z1"] / 2
        alpha[2, 3] = mg.param_dict["z2"] / 2
        alpha[0, 3] = -mg.param_dict["yy"] / 2
        alpha[0, 2] = -mg.param_dict["yx"] / 2
        alpha[1, 3] = mg.param_dict["xy"] / 2

        alpha[2, 1] = -alpha[1, 2]
        alpha[1, 0] = -alpha[0, 1]
        alpha[3, 2] = -alpha[2, 3]
        alpha[3, 0] = -alpha[0, 3]
        alpha[2, 0] = -alpha[0, 2]
        alpha[3, 1] = -alpha[1, 3]
        h = np.zeros((2 * n_qubits, 2 * n_qubits), dtype=complex)
        h[2 * k : 2 * k + 4, 2 * k : 2 * k + 4] = alpha

        T = jax_la.expm(4 * h)
        self.T = T

    def convert_to_unitary(self):
        """
        Convert the gate to a unitary acting on all qubits.

        Returns:
        np.ndarray: unitary matrix representation of the gate.
        """
        n = self.n_qubits
        (a, b) = self.acts_on
        U = self.mg.gate
        startlist = [Id] * a
        endlist = [Id] * (n - b - 1)
        return reduce(np.kron, startlist + [U] + endlist)


class SingleGateNoise:
    """
    Class defining noise on a single qubit gate. Kraus operators must
    be scaled matchgates.

    Attributes:
    probabilities: list of probabilities of each noise.
    ops: list of MatchGate objects for the scaled Kraus operators.
    id_prob: probability of identity noise.

    Methods:
    matchgate_depolarizing: create a depolarizing noise model.
    trivial_noise: create a noiseless model.
    """

    def __init__(self, noise_list: list[tuple[MatchGate, Real]], id_prob: Real):
        """
        Args:
        noise_list: list of tuples with MatchGate objects and probabilities.
        id_prob: probability of identity noise.

        Returns:
        SingleGateNoise object.
        """
        probabilities = [noise[1] for noise in noise_list]
        kraus_ops = [noise[0] for noise in noise_list]
        assert np.isclose(
            sum(probabilities) + id_prob, 1
        ), "Probabilities must sum to 1."
        for op in kraus_ops:
            assert isinstance(
                op, MatchGate
            ), "Kraus operators must be scaled Matchgates."
        self.probabilities = probabilities
        self.scaled_kraus_ops = kraus_ops
        self.id_prob = id_prob

    @classmethod
    def matchgate_depolarizing(cls, p):
        """
        Create a matchgate analogue of depolarizing noise.
        All Pauli operators that are also matchgates have equal probability
        of perturbing the state.

        Args:
        p: probability of noise.

        Returns:
        SingleGateNoise object.
        """

        assert p < 1
        kraus_ops = [
            ("XX", np.kron(X, X)),
            ("YY", np.kron(Y, Y)),
            ("ZI", np.kron(Z, Id)),
            ("IZ", np.kron(Id, Z)),
            ("ZZ", np.kron(Z, Z)),
            ("XY", np.kron(X, Y)),
            ("YX", np.kron(Y, X)),
        ]
        mg_list = []
        for name, kraus in kraus_ops:
            mg_list.append(MatchGate.from_unitary(kraus, name=name))
        return SingleGateNoise([(mg, p / 7) for mg in mg_list], 1 - p)

    @classmethod
    def trivial_noise(cls):
        """
        Returns:
        SingleGateNoise object with no noise.
        """
        kraus = np.kron(X, X)
        mg = MatchGate.from_unitary(kraus)
        return SingleGateNoise([(mg, 0)], 1)


class NoisyAppliedMatchGate:
    """
    A matchgate acting on specific qubits with a noise model for that gate.

    Attributes:
    applied_match_gate: AppliedMatchGate object.
    noise: SingleGateNoise object.
    """

    def __init__(self, applied_match_gate: AppliedMatchGate, noise: SingleGateNoise):
        self.applied_match_gate = applied_match_gate
        self.noise = noise
