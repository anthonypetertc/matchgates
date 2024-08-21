import numpy as np
from MatchGates import operators as ops

X = ops.X
Y = ops.Y
Z = ops.Z
Id = ops.Id
q0 = np.array([1, 0])
q1 = np.array([0, 1])


def test_oplist():
    # Test OpList object.
    ol = ops.ProductOperator([X, X, X])
    ol2 = ops.ProductOperator([X, X, X])
    new_ol = ol.mult(ol2)  # multiplication of OpList objects.
    for o in new_ol.op_list:
        assert np.allclose(o, Id)
    ol3 = ops.ProductOperator([Y, Z, Id])
    new_ol = ol.mult(ol3)
    for i, o in enumerate(new_ol.op_list):
        assert np.allclose(o, ol.op_list[i] @ ol3.op_list[i])


def test_cops():
    # Test Majorana Fermion Creation operators.
    N = 10
    c5 = ops.MajoranaOperator(5, N)
    assert np.allclose(c5.op_list[0], Z)  # Starts with Z.
    assert len(c5.op_list) == 10  # Correct length.
    assert np.allclose(c5.op_list[3], c5.op_list[9])
    assert np.allclose(c5.op_list[3], Id)  # Ends with I.

    N = 5
    c3 = ops.MajoranaOperator(3, N)
    c6 = ops.MajoranaOperator(6, N)
    op_list = c6.mult(c3).op_list  # Multiplication of Majorana operators.
    assert np.allclose(op_list[0], Id)
    assert np.allclose(op_list[1], Z @ Y)
    assert np.allclose(op_list[2], Z)
    assert np.allclose(op_list[3], X)
    assert np.allclose(op_list[4], Id)


def test_expec():
    # Test expectation values of OpList objects starting from a product state.
    op_list = ops.ProductOperator([X, X, X])
    state = ops.ProductState([q0, q0, q0])
    expec = op_list.expectation(state)
    assert expec == 0

    op_list = ops.ProductOperator([Z, Z, Z])
    state = ops.ProductState([q1, q0, q0])
    assert op_list.expectation(state) == -1
