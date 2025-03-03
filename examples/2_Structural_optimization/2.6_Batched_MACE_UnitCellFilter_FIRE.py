import numpy as np
import torch
from ase.build import bulk

from torchsim.optimizers import batched_unit_cell_fire
from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.optimizers import batched_unit_cell_fire

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
PERIODIC = True

rng = np.random.default_rng()

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.2).repeat((4, 4, 4))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)
si_dc_small = bulk("Si", "diamond", a=5.2).repeat((3, 3, 3))
si_dc_small.positions += 0.2 * rng.standard_normal(si_dc_small.positions.shape)
atomic_numbers = si_dc.get_atomic_numbers()
atomic_numbers_small = si_dc_small.get_atomic_numbers()

batched_model = MaceModel(
    model=torch.load(MODEL_PATH, map_location=device),
    atomic_numbers_list=[atomic_numbers, atomic_numbers, atomic_numbers_small],
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)
pos_1 = torch.tensor(si_dc.positions, device=device, dtype=dtype)
pos_2 = torch.tensor(si_dc.positions, device=device, dtype=dtype)
pos_3 = torch.tensor(si_dc_small.positions, device=device, dtype=dtype)

cell_1 = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
cell_2 = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
cell_3 = torch.tensor(si_dc_small.cell.array, device=device, dtype=dtype)

masses_1 = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)
masses_2 = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)
masses_3 = torch.tensor(si_dc_small.get_masses(), device=device, dtype=dtype)

positions_list = [pos_1, pos_2, pos_3]
cell_list = [cell_1, cell_2, cell_3]
masses_list = [masses_1, masses_2, masses_3]
pbc_list = [True, True, True]

print(f"Positions: {positions_list}")
print(f"Cell: {cell_list}")

results = batched_model(positions_list, cell_list)
print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
print(f"Stress: {results['stress']}")

# Check if the energy, forces, and stress are the same for the Si system across the batch
print(torch.max(torch.abs(results["energy"][0] - results["energy"][1])))
print(torch.max(torch.abs(results["forces"][0] - results["forces"][1])))
print(torch.max(torch.abs(results["stress"][0] - results["stress"][1])))

print(results["energy"].shape)
print(results["forces"][0].shape, results["forces"][1].shape, results["forces"][2].shape)
print(results["stress"].shape)

# Create batch indices tensor for scatter operations
n_atoms_per_batch = [pos.shape[0] for pos in positions_list]  # List of atoms per batch
batch_indices = torch.arange(len(positions_list), device=device).repeat_interleave(
    torch.tensor(n_atoms_per_batch, device=device)
)

# Use different learning rates for each batch
learning_rate = 0.01
learning_rates = torch.tensor(
    [learning_rate, learning_rate, learning_rate], device=device, dtype=dtype
)

# Initialize unit cell gradient descent optimizer
batch_state, fire_update = batched_unit_cell_fire(
    model=batched_model,
    positions_list=positions_list,
    masses_list=masses_list,
    cell_list=cell_list,
    batch_indices=batch_indices,
    cell_factor=None,
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
)
# Run optimization for a few steps
for step in range(500):
    atomic_positions = batch_state.positions[: 2 * n_atoms_per_batch].reshape(
        2, n_atoms_per_batch, 3
    )
    results = batched_model(atomic_positions, batch_state.cell)
    print(
        f"{step=}, Energy: {batch_state.energy}, Pressure: sys 1: {torch.trace(results['stress'][0]) * 160.21766208 / 3} eV/A^3, sys 2: {torch.trace(results['stress'][1]) * 160.21766208 / 3} eV/A^3"
    )
    batch_state = fire_update(batch_state)

print(cell_list[0])
print(batch_state.cell[0])
print(cell_list[1])
print(batch_state.cell[1])
