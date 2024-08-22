"""
Module for defining operators and states, for matchgate simulations.
"""

from typing_extensions import Self
import numpy as np

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Id = np.array([[1, 0], [0, 1]])

Z1 = np.kron(Z, Id)
Z2 = np.kron(Id, Z)
XX = np.kron(X, X)
YY = np.kron(Y, Y)
XY = np.kron(X, Y)
YX = np.kron(Y, X)

q0 = np.array([1, 0], dtype=complex)
q1 = np.array([0, 1], dtype=complex)


class ProductState:
    """
    ProductState class for representing a product of states.

    Args:
    state_list: list of states to be multiplied.

    Attributes:
    state_list: numpy array of states.
    ket: True if the state is a ket, False if the state is a bra.
    bra: True if the state is a bra, False if the state is a ket.

    Methods:
    makebra: convert a ket to a bra.
    uniform: create a uniform product state.
    Neel: create a Neel state.
    """

    def __init__(self, state_list):
        """
        Args:
        state_list: list of states to be multiplied.

        Returns:
        ProductState object.
        """
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
        """
        Convert a ket to a bra.

        Returns:
        ProductState object.
        """
        if self.ket is True:
            return ProductState(self.state_list.conj().transpose([0, 2, 1]))
        else:
            return self

    @classmethod
    def uniform(cls, spin, n_qubits: int):
        """
        Create a uniform product state.

        Args:
        spin: 0 or 1.
        n_qubits: number of qubits in the state.

        Returns:
        ProductState object.
        """
        if spin == 0:
            return ProductState([q0] * n_qubits)
        elif spin == 1:
            return ProductState([q1] * n_qubits)
        else:
            raise ValueError("spin must be 0 or 1.")

    @classmethod
    def neel(cls, even: bool, n_qubits: int):
        """
        Create a Neel state.

        Args:
        even: True if want 1's on even sites. 1010101010...
              False if want 1's on odd sites. 0101010101...
        n_qubits: number of qubits in the state.

        Returns:
        ProductState object.
        """
        assert n_qubits % 2 == 0, "Not implemented for odd numbers of qubits."
        if even:
            return ProductState([q1, q0] * int(n_qubits / 2))
        else:
            return ProductState([q0, q1] * int(n_qubits / 2))


class ProductOperator:
    """
    ProductOperator class for representing a product of operators.

    Args:
    op_list: list of operators to be multiplied.

    Attributes:
    op_list: numpy array of operators.

    Methods:
    mult: multiply two ProductOperator objects.
    expectation: calculate expectation value of a ProductOperator object on a ProductState object.
    """

    def __init__(self, op_list: list | np.ndarray):
        """
        Args:
        op_list: list of operators to be multiplied.

        Returns:
        ProductOperator object.
        """
        if isinstance(
            op_list, list
        ):  # If op_list is a list of operators, turn it into an array.
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
            raise ValueError("Invalid input.")

    def mult(self, other: Self):
        """
        Multiply two ProductOperator objects.

        Args:
        other: ProductOperator object to be multiplied.

        Returns:
        ProductOperator object.
        """

        return ProductOperator(self.op_list @ other.op_list)

    def expectation(self, state: ProductState):
        """
        Calculate expectation value of a ProductOperator object on a ProductState object.

        Args:
        state: ProductState object.

        Returns:
        Expectation value of ProductOperator object on ProductState object.
        """
        assert len(state.state_list) == len(self.op_list)
        assert state.ket is True
        bra = state.makebra()
        return np.prod(bra.state_list @ self.op_list @ state.state_list).item()


class MajoranaOperator(ProductOperator):
    """
    MajoranaOperator class for representing a Majorana operator.

    Args:
    k: index of Majorana operator.
    n: number of qubits in the state.

    Attributes:
    op_list: numpy array of Majorana operator.

    Methods:
    mult: multiply two MajoranaOperator objects.
    expectation: calculate expectation value of a MajoranaOperator object on a ProductState object.
    """

    def __init__(self, k: int, n: int):
        """
        Args:
        k: index of Majorana operator.
        n: number of qubits in the state.

        Returns:
        MajoranaOperator object.
        """
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
                op_list[i] = Id
        super().__init__(op_list)
