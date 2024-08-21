import numpy as np
from MatchGates.circuit import Circuit
from MatchGates.matchgate import MatchGate, AppliedMatchGate 


def XX_YY(J, dt):
    param_dict = MatchGate.empty_param_dict()
    param_dict["xx"] = -J * dt
    param_dict["yy"] = -J * dt
    return MatchGate(param_dict)

def Z(h, dt):
    param_dict = MatchGate.empty_param_dict()
    param_dict["z1"] = -h * dt /2
    param_dict["z2"] = -h * dt / 2
    return MatchGate(param_dict)


def XY_circuit(n_qubits, J, h, dt, trotter_steps):
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even.")

    Uxxyy = XX_YY(J, dt)
    Uz1z2 = Z(h, dt)
    gate_list = []
    qubits = list(range(n_qubits))
    for step in range(trotter_steps):
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uz1z2, n_qubits, (i, j))
            gate_list.append(amg)
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uxxyy, n_qubits, (i, j))
            gate_list.append(amg)
        for i, j in zip(qubits[1::2], qubits[2::2]):
            amg = AppliedMatchGate(Uxxyy, n_qubits, (i, i+1))
            gate_list.append(amg)
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uz1z2, n_qubits, (i, j))
            gate_list.append(amg)
    return Circuit(n_qubits, gate_list)


def XX(J, dt):
    param_dict = MatchGate.empty_param_dict()
    param_dict["xx"] = -J * dt
    return MatchGate(param_dict)

def Ising_circuit(n_qubits, J, h, dt, trotter_steps):
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even.")
    
    Uxx = XX(J, dt)
    Uz1z2 = Z(h, dt)
    gate_list = []
    qubits = list(range(n_qubits))
    for step in range(trotter_steps):
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uz1z2, n_qubits, (i, j))
            gate_list.append(amg)
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uxx, n_qubits, (i, j))
            gate_list.append(amg)
        for i, j in zip(qubits[1::2], qubits[2::2]):
            amg = AppliedMatchGate(Uxx, n_qubits, (i, i+1))
            gate_list.append(amg)
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uz1z2, n_qubits, (i, j))
            gate_list.append(amg)
    return Circuit(n_qubits, gate_list)


