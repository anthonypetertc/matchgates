import numpy as np
from MatchGates.operators import ProductState
from MatchGates.expectations import exp_from_T


class Observable:
    def __init__(self, name: str, qubits: list[int], n_qubits: int):
        assert name in set(
            ["Z", "ZZ", "XX", "XY", "YY", "YX"]
        ), "Observable not implemented,\
              only allowed observables are Z, ZZ, XX, XY, YX, YY"
        if name == "Z":
            assert len(qubits) == 1
        elif name == "ZZ":
            assert len(qubits) == 2
        else:
            assert len(qubits) == 2
            assert (
                qubits[1] - qubits[0] == 1
            ), "XX, YY, XY, YX observables must be nearest neighbour."
            assert (qubits[0] < n_qubits) and (
                qubits[1] < n_qubits
            ), f"Observable must apply to a qubit in a chain of length {n_qubits}"
        self.name = name
        self.qubits = tuple(qubits)
        self.n_qubits = n_qubits
        self.majoranas, self.phase = self.get_majoranas()

    def get_majoranas(self):
        if self.name == "Z":
            qubit = self.qubits[0]
            majoranas = [2 * qubit, 2 * qubit + 1]
            phase = -1j
        else:
            q0 = self.qubits[0]
            q1 = self.qubits[1]

        if self.name == "ZZ":
            majoranas = [2 * q0, 2 * q0 + 1, 2 * q1, 2 * q1 + 1]
            phase = -1
        elif self.name == "XX":
            majoranas = [2 * q0 + 1, 2 * q0 + 2]
            phase = -1j
        elif self.name == "YY":
            majoranas = [2 * q0, 2 * q0 + 3]
            phase = 1j
        elif self.name == "XY":
            majoranas = [2 * q0 + 1, 2 * q0 + 3]
            phase = -1j
        elif self.name == "YX":
            majoranas = [2 * q0, 2 * q0 + 2]
            phase = 1j
        elif self.name == "Z":
            pass
        else:
            raise ValueError(
                "Observable not implemented, \
                only allowed observables are Z, ZZ, XX, XY, YX, YY"
            )
        return majoranas, phase

    def compute_expectation(self, T: np.ndarray, state: ProductState):
        S = self.majoranas
        return (self.phase * exp_from_T(S, T, state)).real
