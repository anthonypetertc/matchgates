import numpy as np
import itertools
import MatchGates.operators as ops



def minors(T: np.ndarray, S: list):
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


def c_set_exp(indices: tuple, state: ops.ProductState):
    N = len(state.state_list)
    current_op = ops.MajoranaOp(indices[0], N)
    for ind in indices[1:]:
        current_op = current_op.mult(ops.MajoranaOp(ind, N))
    return current_op.expec(state)

def exp_from_T(S: list, T: np.ndarray, state: ops.ProductState):
    dminors = minors(T, S)
    exp = 0
    for key in dminors.keys():
        exp += dminors[key] * c_set_exp(key, state)
    return exp


