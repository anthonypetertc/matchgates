from time import perf_counter
import pandas as pd

from matchgates.generators import XY_circuit
from matchgates import Observable, ProductState
from matchgates import SingleGateNoise


if __name__=="__main__":

    n_qubits = 60  # number of qubits
    dt = 0.1  # trotter step time interval
    J = 0.5  # coupling strength
    h = 0.23  # tranverse field
    trotter_steps = 30  # number of trotter steps

    qubit_for_observables = n_qubits // 2

    p = 0.002  # Noise strength

    # Total number of runs will be repetitions * iterations
    n_jobs = 32  # Number of cores to use
    repetitions = 960  # Number of repetitions
    iterations = 100  # Number of iterations per repetition

    # generate the XY circuit.
    xy = XY_circuit(n_qubits=n_qubits, dt=dt, J=J, h=h, trotter_steps=trotter_steps)

    # Make observables
    obsz = Observable(name="Z", qubits=[qubit_for_observables], n_qubits=n_qubits)
    obsyx = Observable(name="YX", qubits=[qubit_for_observables, qubit_for_observables + 1], n_qubits=n_qubits)

    # Prepare initial product state.
    initial_state = ProductState.neel(even=True, n_qubits=n_qubits)

    # Create noise channel
    noise_channel = SingleGateNoise.matchgate_depolarizing(p=p)
    # matchgate depolarizing is equal probability of perturbation by 
    # any matchgate Pauli (Z, ZZ, XY, YX, YY, XX). 

    # Add noise to circuit
    noisy_xy = xy.add_two_qubit_uniform_noise(noise_channel) 

    # Filename 
    filename = f"./XY_p{p}_N{n_qubits}_J{J}_h{h}_steps{trotter_steps}_rep{repetitions}.csv"

    print(f"Running {repetitions} over {n_jobs} cores for {iterations} iterations.")
    print(f"Saving data to {filename}")
    print()

    # Run (distributed) noisy emulations for each iteration
    start_start = perf_counter()
    for i in range(iterations):
        start = perf_counter()
        results, _ = noisy_xy.simulate_noisy(n_jobs, [obsz, obsyx], repetitions, initial_state)
        end = perf_counter()

        print(f"Finished {i+1} / {iterations} iterations  |  Total time: {end - start}s  |  Avg. time per run: {(end - start) / repetitions}s")

        new_results = {}
        for key in results.keys():
            new_results[key] = [results[key]]
        df = pd.DataFrame(new_results)

        if i == 0:
            df.to_csv(
                filename,
                mode="w",
                header=True,
                index=False,
            )
        else:
            df.to_csv(
                filename,
                mode="a",
                header=False,
                index=False,
            )

    print(f"All finished in {perf_counter() - start_start}s.")
