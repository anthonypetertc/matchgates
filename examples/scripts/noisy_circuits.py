from time import perf_counter

from matchgates.generators import XY_circuit
from matchgates import Observable, ProductState
from matchgates import SingleGateNoise

n_qubits = 60  # number of qubits
dt = 0.1  # trotter step time interval
J = 0.5  # coupling strength
h = 0.23  # tranverse field
trotter_steps = 30  # number of trotter steps

p = 0.002  # Noise strength

n_jobs = 1  # Number of things to run in parallel
repetitions = 100  # Number of repetitions (per exp. val) 


# generate the XY circuit.
xy = XY_circuit(n_qubits=n_qubits, dt=dt, J=J, h=h, trotter_steps=trotter_steps)

# Make observables
obsz = Observable(name="Z", qubits=[8], n_qubits=n_qubits)
obsyx = Observable(name="YX", qubits=[9, 10], n_qubits=n_qubits)

# prepare initial product state.
initial_state = ProductState.neel(even=True, n_qubits=n_qubits)

noise_channel = SingleGateNoise.matchgate_depolarizing(p=p)
# matchgate depolarizing is equal probability of perturbation by 
# any matchgate Pauli (Z, ZZ, XY, YX, YY, XX). 

noisy_xy = xy.add_two_qubit_uniform_noise(noise_channel)  # Add noise to circuit.

# Simulate
start = perf_counter()
results, hash_counts = noisy_xy.simulate_noisy(n_jobs, [obsz, obsyx], repetitions, initial_state)
end = perf_counter()

print("Results for XY circuit with noise:", results)

# Post-process
hash_counts_sorted_by_freq = sorted(list(hash_counts.values()), reverse=True)
trajectories_sorted_by_freq = sorted([(k, v,) for k, v in hash_counts.items()], key=lambda x: x[1], reverse=True)
all_I_string = "I" * len(list(hash_counts.keys())[0])
print("Sorted hash counts:")
print(hash_counts_sorted_by_freq)
print(f"Frequency of all-identity trajectory: {hash_counts[all_I_string]}")
print(f"Most common string: {trajectories_sorted_by_freq[0][0]}")
print(f"with frequency {hash_counts[trajectories_sorted_by_freq[0][0]]}")

print(f"Avg. time per run: {(end - start) / repetitions}s")