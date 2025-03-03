import numpy as np
import torch
from ase.build import bulk

from torchsim.optimizers import batched_unit_cell_gradient_descent
from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.optimizers import batched_unit_cell_gradient_descent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
PERIODIC = True

rng = np.random.default_rng()

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.21).repeat((5, 5, 5))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((6, 6, 6))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((6, 6, 6))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

print(si_dc.positions.shape)
print(cu_dc.positions.shape)
print(fe_dc.positions.shape)

atomic_numbers_1 = si_dc.get_atomic_numbers()
atomic_numbers_2 = cu_dc.get_atomic_numbers()
atomic_numbers_3 = fe_dc.get_atomic_numbers()

# Create lists of 100 structures by repeating the original 3 structures
atomic_numbers_list = [atomic_numbers_1, atomic_numbers_2, atomic_numbers_3] * 23

batched_model = MaceModel(
    model=torch.load(MODEL_PATH, map_location=device),
    atomic_numbers_list=atomic_numbers_list,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=True,
)

pos_1 = torch.tensor(si_dc.positions, device=device, dtype=dtype)
pos_2 = torch.tensor(cu_dc.positions, device=device, dtype=dtype)
pos_3 = torch.tensor(fe_dc.positions, device=device, dtype=dtype)

cell_1 = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
cell_2 = torch.tensor(cu_dc.cell.array, device=device, dtype=dtype)
cell_3 = torch.tensor(fe_dc.cell.array, device=device, dtype=dtype)

masses_1 = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)
masses_2 = torch.tensor(cu_dc.get_masses(), device=device, dtype=dtype)
masses_3 = torch.tensor(fe_dc.get_masses(), device=device, dtype=dtype)

# Create lists of 100 structures
positions_list = [pos_1, pos_2, pos_3] * 23
cell_list = [cell_1, cell_2, cell_3] * 23
masses_list = [masses_1, masses_2, masses_3] * 23
pbc_list = [True] * 23

print([pos.shape for pos in positions_list])
print([cell.shape for cell in cell_list])
print([masses.shape for masses in masses_list])
print(list(pbc_list))

results = batched_model(positions_list, cell_list)
print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
print(f"Stress: {results['stress']}")

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
learning_rates = torch.full((69,), learning_rate, device=device, dtype=dtype)
# Initialize unit cell gradient descent optimizer
batch_state, gd_update = batched_unit_cell_gradient_descent(
    model=batched_model,
    positions_list=positions_list,
    masses_list=masses_list,
    cell_list=cell_list,
    batch_indices=batch_indices,
    learning_rates=learning_rates,
    cell_factor=None,  # Will default to n_atoms_per_batch
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
)
# Run optimization for a few steps
for step in range(500):
    # Split positions back into list for model input
    positions_split = torch.split(
        batch_state.positions[: sum(n_atoms_per_batch)], n_atoms_per_batch
    )
    positions_list = [pos.clone() for pos in positions_split]
    cell_list = [batch_state.cell[i].clone() for i in range(len(n_atoms_per_batch))]
    results = batched_model(positions_list, cell_list)
    B1 = torch.trace(results["stress"][0]) * 160.21766208 / 3
    B2 = torch.trace(results["stress"][1]) * 160.21766208 / 3
    B3 = torch.trace(results["stress"][2]) * 160.21766208 / 3
    print(
        f"{step=}, E: {batch_state.energy}, P: {batch_state.pressure}, "
        f"{B1=} eV/A^3, {B2=} eV/A^3, {B3=} eV/A^3"
    )
    batch_state = gd_update(batch_state)
