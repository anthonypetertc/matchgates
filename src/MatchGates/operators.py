import numpy as np
from functools import reduce

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
I = np.array([[1, 0], [0, 1]])

Z1 = np.kron(Z, I)
Z2 = np.kron(I, Z)
XX = np.kron(X, X)
YY = np.kron(Y, Y)
XY = np.kron(X, Y)
YX = np.kron(Y, X)

q0 = np.array([1, 0], dtype=complex)
q1 = np.array([0, 1], dtype=complex)


class OpList:
    def __init__(self, op_list):
        if isinstance(op_list, list):
            op_array = np.zeros((len(op_list), 2, 2), dtype=complex)
            for i, op in enumerate(op_list):
                op_array[i] = op
            self.op_list = op_array
        elif isinstance(op_list, np.ndarray):
            assert len(op_list.shape) == 3
            assert op_list.shape[1] == 2
            assert op_list.shape[2] == 2
            self.op_list = op_list
        else:
            raise ValueError

    def mult(self, other):
        return OpList(self.op_list @ other.op_list)

    def expec(self, state):
        assert len(state.state_list) == len(self.op_list)
        assert state.ket == True
        bra = state.makebra()
        return np.prod(bra.state_list @ self.op_list @ state.state_list).item()


class MajoranaOp(OpList):
    def __init__(self, k: int, n: int):
        assert k >= 0, "invalid"
        assert 2 * n >= k, "invalid"
        op_list = np.zeros((n, 2, 2), dtype=complex)
        if k % 2 == 0:
            site = int(k / 2)
            op = X
        else:
            site = int((k - 1) / 2)
            op = Y
        for i in range(n):
            if i < site:
                op_list[i] = Z
            elif i == site:
                op_list[i] = op
            else:
                op_list[i] = I
        self.op_list = op_list


class ProductState:
    def __init__(self, state_list):
        if isinstance(state_list, list):
            state_array = np.zeros((len(state_list), 2, 1), dtype=complex)
            for i, state in enumerate(state_list):
                state_array[i] = np.reshape(state, (2, 1))
            state_list = state_array
        assert len(state_list.shape) == 3
        self.state_list = state_list
        if state_list.shape[1] == 2 and state_list.shape[2] == 1:
            self.ket = True
            self.bra = False
        elif state_list.shape[1] == 1 and state_list.shape[2] == 2:
            self.ket = False
            self.bra = True
        else:
            raise ValueError

    def makebra(self):
        return ProductState(self.state_list.conj().transpose([0, 2, 1]))

    @classmethod
    def uniform(self, spin: {0, 1}, n_qubits: int):
        if spin == 0:
            return ProductState([q0] * n_qubits)
        elif spin == 1:
            return ProductState([q1] * n_qubits)
        else:
            raise ValueError("spin must be 0 or 1.")

    @classmethod
    def Neel(self, even: bool, n_qubits: int):
        assert n_qubits % 2 == 0, "Not implemented for odd numbers of qubits."
        if even:
            return ProductState([q1, q0] * int(n_qubits / 2))
        else:
            return ProductState([q0, q1] * int(n_qubits / 2))


def mpower(M, k):  # Do I need this?
    if k == 0:
        return I
    else:
        return reduce(lambda a, b: a @ b, [M] * k)
