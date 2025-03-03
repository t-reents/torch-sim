# TorchSim

[![CI](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml/badge.svg)](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/radical-ai/torch-sim/branch/main/graph/badge.svg)](https://codecov.io/gh/radical-ai/torch-sim)

TorchSim is an open-source simulation engine in PyTorch focused on atomistic simulations. It provides a easy to use and efficient interface for running simulations on both CPU and GPU. Being built on PyTorch, it allows for automatic differentiation of the simulation equations, making it easy to compute quantities of interest.

## Installation

```bash
git clone https://github.com/radical-ai/torch-sim.git
cd torch-sim
pip install .
```

## Core modules

TorchSim is built around the following core modules:

- `torchsim.integrators`: Provides batched molecular dynamics integrators for simulating the time evolution of molecular systems.

- `torchsim.optimizers`: Provides optimization algorithms for molecular systems, including gradient descent and the FIRE algorithm.

- `torchsim.unbatched_integrators`: Provides unbatched molecular dynamics integrators for simulating the time evolution of molecular systems.

- `torchsim.unbatched_optimizers`: Provides unbatched optimization algorithms for molecular systems, including gradient descent and the FIRE algorithm.

- `torchsim.monte_carlo`: Contains functions for performing Monte Carlo simulations, including swap-based Monte Carlo.

- `torchsim.neighbors`: Contains functions for constructing neighbor lists.

- `torchsim.quantities`: Functions for computing physical quantities, such as kinetic energy, from simulation data.

- `torchsim.runners`: Provides functions for running molecular dynamics simulations and optimizations, handling simulation state and conversions between different representations.

- `torchsim.state`: Defines the `BaseState` class, a dataclass for representing the state of molecular systems.

- `torchsim.trajectory`: Implements classes for handling trajectory data, allowing for reading and writing of simulation data to HDF5 files.

- `torchsim.transforms`: Functions for handling coordinate transformations and periodic boundary conditions in molecular simulations.

- `torchsim.elastic`: Contains classes and functions for calculating crystal elasticity, including the representation of elastic states and computation of elastic tensors.

- `torchsim.utils`: Utility functions.

- `torchsim.units`: Unit system and conversion factors

- `torchsim.worlflows`: Utility functions for running workflows.

Each module is designed to work seamlessly with PyTorch, enabling efficient and flexible molecular simulations.
## API

### State and Parameters

The simulation engine uses two main objects:

| Temperature Profile | RDF Comparison |
| ------------------ | -------------- |
| ![Temperature Profile](https://github.com/user-attachments/assets/4d87444f-751d-49f4-ada2-c4578abe0a18) | ![RDF Comparison](https://github.com/user-attachments/assets/f84b4b3f-5b09-4cf5-9eda-b0f766be93fc) |

### State (S)

- Positions (p)
- Velocities (q)
- Mass (m)
- Other system properties

### Parameters (P)

- Temperature (T)
- Pressure (P)
- Timestep (dt)
- Other simulation parameters

The integrator/optimizer maps: S, U(S,P) → S

### Energy Styles

TorchSim implements various classical interaction potentials including:

#### Soft Sphere

- Finite-range repulsive potential
- Parameters: sigma (diameter), epsilon (energy scale), alpha (stiffness)
- Suitable for modeling excluded volume interactions

#### Lennard-Jones

- Standard 12-6 potential combining repulsion and attraction
- Parameters: sigma (minimum distance), epsilon (well depth)
- Widely used for van der Waals interactions

#### Morse Potential

- Anharmonic potential for chemical bonding
- Parameters: sigma (equilibrium distance), epsilon (well depth), alpha (well width)
- Good for modeling diatomic molecules

#### Stillinger-Weber

- Many-body potential originally developed for silicon
- Combines two-body and three-body terms
- Parameters: sigma (length scale), epsilon (energy scale), various angular terms

## Example: Melt Quenching Simulation

TorchSim can be used to simulate complex processes like melt quenching of silicon:

1. Heat system to melting temperature
2. Equilibrate liquid phase
3. Rapidly cool to create amorphous structure
4. Analyze resulting structure via RDF

The simulation results can be analyzed through:

- Temperature profiles
- Radial distribution functions (RDF)
- Structure visualization
- Property calculations

## Citation

If you use TorchSim in your research, please cite:

```bib
@repository{gangan-2025-torchsim,
  ...
}
```
