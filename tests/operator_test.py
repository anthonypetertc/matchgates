import numpy as np
from MatchGates import operators as ops

X = ops.X
Y = ops.Y
Z = ops.Z
I = ops.I
q0 = np.array([1, 0])
q1 = np.array([0, 1])


def test_oplist():
    # Test OpList object.
    ol = ops.OpList([X, X, X])
    ol2 = ops.OpList([X, X, X])
    new_ol = ol.mult(ol2)  # multiplication of OpList objects.
    for o in new_ol.op_list:
        assert np.allclose(o, I)
    ol3 = ops.OpList([Y, Z, I])
    new_ol = ol.mult(ol3)
    for i, o in enumerate(new_ol.op_list):
        assert np.allclose(o, ol.op_list[i] @ ol3.op_list[i])


def test_cops():
    # Test Majorana Fermion Creation operators.
    N = 10
    c5 = ops.MajoranaOp(5, 10)
    assert np.allclose(c5.op_list[0], Z)  # Starts with Z.
    assert len(c5.op_list) == 10  # Correct length.
    assert np.allclose(c5.op_list[3], c5.op_list[9])
    assert np.allclose(c5.op_list[3], I)  # Ends with I.

    N = 5
    c3 = ops.MajoranaOp(3, 5)
    c6 = ops.MajoranaOp(6, 5)
    op_list = c6.mult(c3).op_list  # Multiplication of Majorana operators.
    assert np.allclose(op_list[0], I)
    assert np.allclose(op_list[1], Z @ Y)
    assert np.allclose(op_list[2], Z)
    assert np.allclose(op_list[3], X)
    assert np.allclose(op_list[4], I)


def test_expec():
    # Test expectation values of OpList objects starting from a product state.
    op_list = ops.OpList([X, X, X])
    state = ops.ProductState([q0, q0, q0])
    expec = op_list.expec(state)
    assert expec == 0

    op_list = ops.OpList([Z, Z, Z])
    state = ops.ProductState([q1, q0, q0])
    assert op_list.expec(state) == -1
