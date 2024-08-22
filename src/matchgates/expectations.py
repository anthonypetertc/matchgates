"""Module with functions for calculating expectations of operators."""

import itertools
import numpy as np
import matchgates.operators as ops


def minors(T: np.ndarray, S: list):
    """
    Computes the determinant of the matrix of minors of T with respect to the set indices S.
    Args:
        T (np.ndarray): matrix.
        S (list): list of indices.
    Returns:
        dict: dictionary of minors.
    """
    shape = np.shape(T)
    assert len(shape) == 2
    assert shape[0] == shape[1]
    length = len(S)
    assert length <= shape[0]
    all_inds = list(range(0, shape[0]))
    S_comp = [ind for ind in all_inds if ind not in S]
    indices = list(itertools.combinations(range(shape[0]), length))
    dict_of_minors = {}
    for index in indices:
        complement = [ind for ind in all_inds if ind not in index]
        minor = np.delete(np.delete(T, S_comp, axis=0), complement, axis=1)
        dict_of_minors[index] = np.linalg.det(minor)
    return dict_of_minors


def c_set_expectation(indices: tuple, state: ops.ProductState):
    """
    Compute the expectation value of a product of Majorana operators on a state.
    Args:
        indices (tuple): tuple of indices for the Majorana Fermion creation ops.
        state (ProductState): state.

    Returns:
        float: expectation value.
    """

    N = len(state.state_list)
    current_op = ops.MajoranaOperator(indices[0], N)
    for ind in indices[1:]:
        current_op = current_op.mult(ops.MajoranaOperator(ind, N))
    return current_op.expectation(state)


def expectation_from_T(S: list, T: np.ndarray, state: ops.ProductState):
    """
    Compute the expectation value of a T matrix on a state.
    Args:
        S (list): list of indices.
        T (np.ndarray): matrix.
        state (ProductState): state.

    Returns:
        float: expectation value.
    """

    dminors = minors(T, S)
    exp = 0
    for key, item in zip(dminors.keys(), dminors.values()):
        exp += item * c_set_expectation(key, state)
    return exp
