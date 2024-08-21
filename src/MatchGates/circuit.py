"Module for Circuit classes."

from multiprocessing import Pool
import numpy as np
import MatchGates.matchgate as MG
from MatchGates.observables import Observable
from MatchGates.operators import ProductState
import os


class Circuit:
    """ """

    def __init__(self, n_qubits: int, gate_list: list[MG.AppliedMatchGate]):
        self.n_qubits = n_qubits
        self.gate_list = gate_list
        T = gate_list[0].T
        for gate in gate_list[1:]:
            assert (
                gate.n_qubits == n_qubits
            ), "Invalid gate list, gates don't act on same number qubits as circuit."
            T = gate.T @ T
        self.T = T

    def add_uniform_noise_to_circuit(self, noise: MG.SingleGateNoise):
        noisy_gate_list = []
        for gate in self.gate_list:
            noisy_gate_list.append(MG.NoisyAppliedMatchGate(gate, noise))
        return NoisyCircuit(self.n_qubits, noisy_gate_list)

    def add_two_qubit_uniform_noise(self, noise: MG.SingleGateNoise):
        noisy_gate_list = []
        trivial_noise = MG.SingleGateNoise.trivial_noise()
        for gate in self.gate_list:
            if not gate.mg.is_single_qubit():
                noisy_gate_list.append(MG.NoisyAppliedMatchGate(gate, noise))
            elif gate.mg.is_single_qubit():
                noisy_gate_list.append(MG.NoisyAppliedMatchGate(gate, trivial_noise))
            else:
                raise ValueError("MatchGate.is_single_qubit() method has failed.")
        return NoisyCircuit(self.n_qubits, noisy_gate_list)

    def simulate(self, observables: list[Observable], state: ProductState):
        results_dict = {}
        for obs in observables:
            results_dict[(obs.name, obs.qubits)] = obs.compute_expectation(
                self.T, state
            )
        return results_dict


class NoisyCircuit(Circuit):
    def __init__(self, n_qubits: int, gate_list: list[MG.NoisyAppliedMatchGate]):
        self.n_qubits = n_qubits
        self.noisy_gate_list = gate_list

    def simulate_noisy(
        self, n_jobs: int, observables: list[Observable], reps: int, state: ProductState
    ):
        results_dict = {}
        for obs in observables:
            results_dict[obs.name + str(obs.qubits)] = 0
        reps_per_job = reps // n_jobs
        args = [[observables, reps_per_job, state] for i in range(n_jobs)]
        with Pool(n_jobs) as p:
            results = p.starmap(self.single_run, args)
            for result in results:
                for i, obs in enumerate(observables):
                    results_dict[obs.name + str(obs.qubits)] += (
                        1 / n_jobs * result[obs.name + str(obs.qubits)]
                    )
        return results_dict

    def single_run(self, observables: list[Observable], reps: int, state: ProductState):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
        n_qubits = self.n_qubits
        gate_list = self.noisy_gate_list
        results_dict = {}
        for obs in observables:
            results_dict[obs.name + str(obs.qubits)] = 0
        for rep in range(reps):
            T = self.make_noisy_T(gate_list[0])
            for gate in gate_list[1:]:
                assert (
                    gate.amg.n_qubits == n_qubits
                ), "Invalid gate list, \
                    gates don't act on same number qubits as circuit."
                noisy_T = self.make_noisy_T(gate)
                T = noisy_T @ T
            for obs in observables:
                results_dict[obs.name + str(obs.qubits)] += (
                    1 / reps * obs.compute_expectation(T, state)
                )
        return results_dict

    def make_noisy_T(self, gate: MG.NoisyAppliedMatchGate):
        noise = gate.noise
        amg = gate.amg
        T = amg.T
        probs = noise.probabilities + [noise.id_prob]
        kops = gate.noise.ops + ["I"]
        gate_noise = np.random.choice(kops, p=probs)
        if gate_noise != "I":
            applied_noise = MG.AppliedMatchGate(gate_noise, amg.n_qubits, amg.acts_on)
            T = applied_noise.T @ T
        return T
