# matchgates


## Introduction
This is a pure python implementation of matchgate simulation via mapping to free fermions. It includes functionality for simulation of pure matchgate circuits, and also noisy matchgate circuits, with matchgate noise by sampling over trajectories. No attempt at optimization has been made and there are no guarantees of performance or functionality of the package.


## Installation
Clone the repository:

    git clone git@github.com:anthonypetertc/matchgates.git

Navigate into the cloned directory:

    cd matchgates

Install dependencies:

    pip install -r requirements.txt

Install the package:

    pip install .
## Tests
To check that installation has worked correctly, run the unit tests:

    pytest

## Usage
This package includes two pre-built matchgate circuits: the Ising model and the XY model. It also includes a noise model that can be applied to these circuits: a channel obtained from sampling with equal probability from the matchgate Pauli's on two qubits (IZ, ZI, ZZ, XY, YX, YY, XX).

See the examples: `examples/noiseless_circuits.ipynb` and `examples/noisy_circuits.ipynb` for examples demonstrating how to simulate these pre-built circuits with and without noise.

It is also possible to use this package to build new circuits out of matchgates, by creating a new Circuit object. For an example demonstrating how this is done see  `examples/custom_circuits.ipynb` . Furthermore, it is also possible to build and simulate custom noise models for matchgate circuits using scaled matchgates as the kraus channels. An example showing this can be found in `examples/custom_noise_models.ipynb`
