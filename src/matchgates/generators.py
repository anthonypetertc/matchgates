"""
Create generators for MatchGate circuits.
"""
from time import perf_counter

from matchgates.circuit import Circuit
from matchgates.matchgate import MatchGate, AppliedMatchGate


def XX_YY(J: float, dt: float):
    """
    Time evolution matchgate with Hamiltonian XX + YY
        Params:
            J (float): coupling strength.
            dt (float): time interval.
        Returns:
            MatchGate object: exp(-i J dt (XX + YY))
    """
    param_dict = MatchGate.empty_param_dict()
    param_dict["xx"] = -J * dt
    param_dict["yy"] = -J * dt
    return MatchGate(param_dict)


def Z(h, dt):
    """
    Time evolution matchgate with Hamiltonian Z
        Params:
            h (float): Transverse perturbation.
            dt (float): time interval.
        Returns:
            MatchGate object: exp(-i h dt Z)
    """
    param_dict = MatchGate.empty_param_dict()
    param_dict["z1"] = -h * dt / 2
    param_dict["z2"] = -h * dt / 2
    return MatchGate(param_dict)


def XY_circuit(n_qubits: int, J: float, h: float, dt: float, trotter_steps: int):
    """
    Generate a circuit for Heisenberg XY time evolution.
        Params:
            n_qubits (int): number of qubits.
            J (float): Coupling strength in Hamiltonian.
            h (float): Transverse perturbation.
            dt (float): time interval per Trotter step.
            trotter_steps (int): number of Trotter steps.
        Returns:
            Circuit object for the time evolution.
    """
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even.")

    Uxxyy = XX_YY(J, dt)
    Uz1z2 = Z(h, dt)

    avg_gate_creation_time = 0.
    count = 0

    gate_list = []
    qubits = list(range(n_qubits))
    for step in range(trotter_steps):
        print(f"Step {step + 1} / {trotter_steps}")
        start = perf_counter()
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uz1z2, n_qubits, [i, j])
            gate_list.append(amg)
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uxxyy, n_qubits, [i, j])
            gate_list.append(amg)
        for i, j in zip(qubits[1::2], qubits[2::2]):
            amg = AppliedMatchGate(Uxxyy, n_qubits, [i, i + 1])
            gate_list.append(amg)
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uz1z2, n_qubits, [i, j])
            gate_list.append(amg)

        avg_gate_creation_time += perf_counter() - start
        count += 1

        print(f"Avg. gate creation time per layer: {avg_gate_creation_time / count}")


    return Circuit(n_qubits, gate_list)


def XX(J, dt):
    """
    Time evolution matchgate with Hamiltonian XX
        Params:
            J (float): coupling strength.
            dt (float): time interval.
        Returns:
            MatchGate object: exp(-i J dt XX )
    """
    param_dict = MatchGate.empty_param_dict()
    param_dict["xx"] = -J * dt
    return MatchGate(param_dict)


def Ising_circuit(n_qubits, J, h, dt, trotter_steps):
    """
    Generate a circuit for Ising time evolution.
        Params:
            n_qubits (int): number of qubits.
            J (float): Coupling strength in Hamiltonian.
            h (float): Transverse perturbation.
            dt (float): time interval per Trotter step.
            trotter_steps (int): number of Trotter steps.
        Returns:
            Circuit object for the time evolution.
    """
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even.")

    Uxx = XX(J, dt)
    Uz1z2 = Z(h, dt)
    gate_list = []
    qubits = list(range(n_qubits))
    for _ in range(trotter_steps):
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uz1z2, n_qubits, (i, j))
            gate_list.append(amg)
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uxx, n_qubits, (i, j))
            gate_list.append(amg)
        for i, j in zip(qubits[1::2], qubits[2::2]):
            amg = AppliedMatchGate(Uxx, n_qubits, (i, i + 1))
            gate_list.append(amg)
        for i, j in zip(qubits[::2], qubits[1::2]):
            amg = AppliedMatchGate(Uz1z2, n_qubits, (i, j))
            gate_list.append(amg)
    return Circuit(n_qubits, gate_list)
