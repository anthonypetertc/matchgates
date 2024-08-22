"Module for Circuit classes."

from multiprocessing import Pool
import os
import numpy as np
import matchgates.matchgate as MG
from matchgates.observables import Observable
from matchgates.operators import ProductState


class Circuit:
    """Class for noiseless matchgate circuits.

    Attributes
    ----------
    n_qubits (int): number of qubits circuit acts on.
    gate_list (list[AppliedMatchGate]): list of gates in circuit.
    T (np.ndarray): matrix used for calculating observables.

    Methods
    -------
    add_uniform_noise_to_circuit(noise: SingleGateNoise) -> NoisyCircuit:
        Adds noise to all gates in circuit.
    add_two_qubit_uniform_noise(noise: SingleGateNoise) -> NoisyCircuit:
        Adds noise to two qubit gates in circuit.
    simulate(observables: list[Observable], state: ProductState) -> dict:
        Simulates circuit and returns expectation values of
        observables.
    """

    def __init__(self, n_qubits: int, gate_list: list[MG.AppliedMatchGate]):
        """Initializes Circuit object.

        Parameters
        ----------
        n_qubits: int
            Number of qubits circuit acts on.
            gate_list: list[AppliedMatchGate]
            List of gates in circuit.

        Returns
        -------
        None
        """
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
        """Adds noise to all gates in circuit.

        Parameters
        ----------
        noise: SingleGateNoise
            Noise to be added to gates.

        Returns
        -------
        NoisyCircuit
            Noisy circuit with noise added to all gates."""

        noisy_gate_list = []
        for gate in self.gate_list:
            noisy_gate_list.append(MG.NoisyAppliedMatchGate(gate, noise))
        return NoisyCircuit(self.n_qubits, noisy_gate_list)

    def add_two_qubit_uniform_noise(self, noise: MG.SingleGateNoise):
        """Adds noise to two qubit gates in circuit.

        Parameters
        ----------
        noise: SingleGateNoise
            Noise to be added to gates.

        Returns
        -------
        NoisyCircuit
            Noisy circuit with noise added to two qubit gates."""

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
        """Simulates circuit and returns expectation values of observables.

        Parameters
        ----------
        observables: list[Observable]
            List of observables to compute expectation values of.
            state: ProductState
            Initial state of the system.

        Returns
        -------
        dict
        Dictionary of expectation values of observables."""

        results_dict = {}
        for obs in observables:
            results_dict[(obs.name, obs.qubits)] = obs.compute_expectation(
                self.T, state
            ).astype(float)
        return results_dict


class NoisyCircuit:
    """Class for noisy matchgate circuits.

    Attributes
    ----------
    n_qubits: number of qubits circuit acts on.
    noisy_gate_list: list of noisy gates in circuit.

    Methods
    -------
    simulate_noisy(n_jobs: int, observables: list[Observable], reps: int, state: ProductState) -> dict:
        Simulates noisy circuit and returns expectation values of observables.
    single_run(observables: list[Observable], reps: int, state: ProductState) -> dict:
        Simulates noisy circuit for a single run and returns expectation values of observables.
    make_noisy_T(gate: NoisyAppliedMatchGate) -> np.ndarray:
        Adds noise to a gate and returns the matrix representation of the noisy gate.
    """

    def __init__(self, n_qubits: int, gate_list: list[MG.NoisyAppliedMatchGate]):
        """Initializes NoisyCircuit object.

        Parameters
        ----------
        n_qubits: int
            Number of qubits circuit acts on.
            gate_list: list[NoisyAppliedMatchGate]
            List of noisy gates in circuit.

        Returns
        -------
        None
        """
        self.n_qubits = n_qubits
        self.noisy_gate_list = gate_list

    def simulate_noisy(
        self, n_jobs: int, observables: list[Observable], reps: int, state: ProductState
    ):
        """Simulates noisy circuit and returns expectation values of observables.

        Parameters
        ----------
        n_jobs: int
            Number of jobs to run in parallel.
            observables: list[Observable]
            List of observables to compute expectation values of.
            reps: int
            Number of repetitions for each job.
            state: ProductState
            Initial state of the system.

        Returns
        -------
        dict
        Dictionary of expectation values of observables."""
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
        """Simulates noisy circuit for a single run and returns expectation values of observables.

        Parameters
        ----------
        observables: list[Observable]
            List of observables to compute expectation values of.
        reps: int
            Number of repetitions for each job.
        state: ProductState
            Initial state of the system.

        Returns
        -------
        dict
            Dictionary of expectation values of observables."""
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
        n_qubits = self.n_qubits
        gate_list = self.noisy_gate_list
        results_dict = {}
        for obs in observables:
            results_dict[obs.name + str(obs.qubits)] = 0
        for _ in range(reps):
            T = self.make_noisy_T(gate_list[0])
            for gate in gate_list[1:]:
                assert (
                    gate.applied_match_gate.n_qubits == n_qubits
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
        """Adds noise to a gate and returns the matrix representation of the noisy gate.

        Parameters
        ----------
        gate: NoisyAppliedMatchGate
            Gate to add noise to.

        Returns
        -------
        np.ndarray
            Matrix representation of noisy gate."""
        noise = gate.noise
        amg = gate.applied_match_gate
        T = amg.T
        probs = noise.probabilities + [noise.id_prob]
        kops = gate.noise.scaled_kraus_ops + ["I"]
        gate_noise = np.random.choice(kops, p=probs)
        if gate_noise != "I":
            applied_noise = MG.AppliedMatchGate(gate_noise, amg.n_qubits, amg.acts_on)
            T = applied_noise.T @ T
        return T
