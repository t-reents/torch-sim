"""Implementations of NVT integrators."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from torch_sim.integrators.md import (
    MDState,
    NoseHooverChain,
    NoseHooverChainFns,
    calculate_momenta,
    construct_nose_hoover_chain,
    momentum_step,
    position_step,
    velocity_verlet,
)
from torch_sim.quantities import calc_kinetic_energy
from torch_sim.state import SimState
from torch_sim.typing import StateDict


def nvt_langevin(  # noqa: C901
    model: torch.nn.Module,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    gamma: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor], MDState],
    Callable[[MDState, torch.Tensor], MDState],
]:
    """Initialize and return an NVT (canonical) integrator using Langevin dynamics.

    This function sets up integration in the NVT ensemble, where particle number (N),
    volume (V), and temperature (T) are conserved. It returns both an initial state
    and an update function for time evolution.

    It uses Langevin dynamics with stochastic noise and friction to maintain constant
    temperature. The integration scheme combines deterministic velocity Verlet steps with
    stochastic Ornstein-Uhlenbeck processes following the BAOAB splitting scheme.

    Args:
        model (torch.nn.Module): Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_systems]
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            with shape [n_systems]
        gamma (torch.Tensor, optional): Friction coefficient for Langevin thermostat,
            either scalar or with shape [n_systems]. Defaults to 1/(100*dt).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - callable: Function to initialize the MDState from input data
              with signature: init_fn(state, kT=kT, seed=seed) -> MDState
            - callable: Update function that evolves system by one timestep
              with signature: update_fn(state, dt=dt, kT=kT, gamma=gamma) -> MDState

    Notes:
        - Uses BAOAB splitting scheme for Langevin dynamics
        - Preserves detailed balance for correct NVT sampling
        - Handles periodic boundary conditions if enabled in state
        - Friction coefficient gamma controls the thermostat coupling strength
        - Weak coupling (small gamma) preserves dynamics but with slower thermalization
        - Strong coupling (large gamma) faster thermalization but may distort dynamics
    """
    device, dtype = model.device, model.dtype

    if gamma is None:
        gamma = 1 / (100 * dt)

    if isinstance(gamma, float):
        gamma = torch.tensor(gamma, device=device, dtype=dtype)

    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)

    def ou_step(
        state: MDState,
        dt: torch.Tensor,
        kT: torch.Tensor,
        gamma: torch.Tensor,
    ) -> MDState:
        """Apply stochastic noise and friction for Langevin dynamics.

        This function implements the Ornstein-Uhlenbeck process for Langevin dynamics,
        applying random noise and friction forces to particle momenta. The noise amplitude
        is chosen to satisfy the fluctuation-dissipation theorem, ensuring proper
        sampling of the canonical ensemble at temperature kT.

        Args:
            state (MDState): Current system state containing positions, momenta, etc.
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                with shape [n_systems]
            gamma (torch.Tensor): Friction coefficient controlling noise strength,
                either scalar or with shape [n_systems]

        Returns:
            MDState: Updated state with new momenta after stochastic step

        Notes:
            - Implements the "O" step in the BAOAB Langevin integration scheme
            - Uses Ornstein-Uhlenbeck process for correct thermal sampling
            - Noise amplitude scales with sqrt(mass) for equipartition
            - Preserves detailed balance through fluctuation-dissipation relation
            - The equation implemented is:
              p(t+dt) = c1*p(t) + c2*sqrt(m)*N(0,1)
              where c1 = exp(-gamma*dt) and c2 = sqrt(kT*(1-c1²))
        """
        c1 = torch.exp(-gamma * dt)

        if isinstance(kT, torch.Tensor) and len(kT.shape) > 0:
            # kT is a tensor with shape (n_systems,)
            kT = kT[state.system_idx]

        # Index c1 and c2 with state.system_idx to align shapes with state.momenta
        if isinstance(c1, torch.Tensor) and len(c1.shape) > 0:
            c1 = c1[state.system_idx]

        c2 = torch.sqrt(kT * (1 - c1**2)).unsqueeze(-1)

        # Generate random noise from normal distribution
        noise = torch.randn_like(state.momenta, device=state.device, dtype=state.dtype)
        new_momenta = (
            c1.unsqueeze(-1) * state.momenta
            + c2 * torch.sqrt(state.masses).unsqueeze(-1) * noise
        )
        state.momenta = new_momenta
        return state

    def langevin_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        seed: int | None = seed,
    ) -> MDState:
        """Initialize an NVT state from input data for Langevin dynamics.

        Creates an initial state for NVT molecular dynamics by computing initial
        energies and forces, and sampling momenta from a Maxwell-Boltzmann distribution
        at the specified temperature.

        Args:
            state (SimState | StateDict): Either a SimState object or a dictionary
                containing positions, masses, cell, pbc, and other required state vars
            kT (torch.Tensor): Temperature in energy units for initializing momenta,
                either scalar or with shape [n_systems]
            seed (int, optional): Random seed for reproducibility

        Returns:
            MDState: Initialized state for NVT integration containing positions,
                momenta, forces, energy, and other required attributes

        Notes:
            The initial momenta are sampled from a Maxwell-Boltzmann distribution
            at the specified temperature. This provides a proper thermal initial
            state for the subsequent Langevin dynamics.
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        model_output = model(state)

        momenta = getattr(
            state,
            "momenta",
            calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
        )

        initial_state = MDState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            system_idx=state.system_idx,
            atomic_numbers=state.atomic_numbers,
        )
        return initial_state  # noqa: RET504

    def langevin_update(
        state: MDState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
        gamma: torch.Tensor = gamma,
    ) -> MDState:
        """Perform one complete Langevin dynamics integration step.

        This function implements the BAOAB splitting scheme for Langevin dynamics,
        which provides accurate sampling of the canonical ensemble. The integration
        sequence is:
        1. Half momentum update using forces (B step)
        2. Half position update using updated momenta (A step)
        3. Full stochastic update with noise and friction (O step)
        4. Half position update using updated momenta (A step)
        5. Half momentum update using new forces (B step)

        Args:
            state (MDState): Current system state containing positions, momenta, forces
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                with shape [n_systems]
            gamma (torch.Tensor): Friction coefficient for Langevin thermostat,
                either scalar or with shape [n_systems]

        Returns:
            MDState: Updated state after one complete Langevin step with new positions,
                momenta, forces, and energy
        """
        if isinstance(gamma, float):
            gamma = torch.tensor(gamma, device=device, dtype=dtype)

        if isinstance(dt, float):
            dt = torch.tensor(dt, device=device, dtype=dtype)

        state = momentum_step(state, dt / 2)
        state = position_step(state, dt / 2)
        state = ou_step(state, dt, kT, gamma)
        state = position_step(state, dt / 2)

        model_output = model(state)
        state.energy = model_output["energy"]
        state.forces = model_output["forces"]

        return momentum_step(state, dt / 2)

    return langevin_init, langevin_update


@dataclass
class NVTNoseHooverState(MDState):
    """State information for an NVT system with a Nose-Hoover chain thermostat.

    This class represents the complete state of a molecular system being integrated
    in the NVT (constant particle number, volume, temperature) ensemble using a
    Nose-Hoover chain thermostat. The thermostat maintains constant temperature
    through a deterministic extended system approach.

    Attributes:
        positions: Particle positions with shape [n_particles, n_dimensions]
        momenta: Particle momenta with shape [n_particles, n_dimensions]
        energy: Energy of the system
        forces: Forces on particles with shape [n_particles, n_dimensions]
        masses: Particle masses with shape [n_particles]
        cell: Simulation cell matrix with shape [n_dimensions, n_dimensions]
        pbc: Whether to use periodic boundary conditions
        chain: State variables for the Nose-Hoover chain thermostat

    Properties:
        velocities: Particle velocities computed as momenta/masses
            Has shape [n_particles, n_dimensions]

    Notes:
        - The Nose-Hoover chain provides deterministic temperature control
        - Extended system approach conserves an extended energy quantity
        - Chain variables evolve to maintain target temperature
        - Time-reversible when integrated with appropriate algorithms
    """

    chain: NoseHooverChain
    _chain_fns: NoseHooverChainFns

    @property
    def velocities(self) -> torch.Tensor:
        """Velocities calculated from momenta and masses with shape
        [n_particles, n_dimensions].
        """
        return self.momenta / self.masses.unsqueeze(-1)


def nvt_nose_hoover(
    *,
    model: torch.nn.Module,
    dt: torch.Tensor,
    kT: torch.Tensor,
    chain_length: int = 3,
    chain_steps: int = 3,
    sy_steps: int = 3,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor, int | None, Any], NVTNoseHooverState],
    Callable[[NVTNoseHooverState, torch.Tensor], NVTNoseHooverState],
]:
    """Initialize NVT Nose-Hoover chain thermostat integration.

    This function sets up integration of an NVT system using a Nose-Hoover chain
    thermostat. The Nose-Hoover chain provides deterministic temperature control by
    coupling the system to a chain of thermostats. The integration scheme is
    time-reversible and conserves an extended energy quantity.

    Args:
        model: Neural network model that computes energies and forces
        dt: Integration timestep
        kT: Target temperature in energy units
        chain_length: Number of thermostats in Nose-Hoover chain (default: 3)
        chain_steps: Number of chain integration substeps (default: 3)
        sy_steps: Number of Suzuki-Yoshida steps - must be 1, 3, 5, or 7 (default: 3)

    Returns:
        tuple containing:
        - Initialization function that takes a state and returns NVTNoseHooverState
        - Update function that performs one complete integration step

    Notes:
        The initialization function accepts:
        - state: Initial system state (SimState or dict)
        - kT: Target temperature (optional, defaults to constructor value)
        - tau: Thermostat relaxation time (optional, defaults to 100*dt)
        - seed: Random seed for momenta initialization (optional)
        - **kwargs: Additional state variables

        The update function accepts:
        - state: Current NVTNoseHooverState
        - dt: Integration timestep (optional, defaults to constructor value)
        - kT: Target temperature (optional, defaults to constructor value)

        The integration sequence is:
        1. Update chain masses
        2. First half-step of chain evolution
        3. Full velocity Verlet step
        4. Update chain kinetic energy
        5. Second half-step of chain evolution
    """

    def nvt_nose_hoover_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        tau: torch.Tensor | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> NVTNoseHooverState:
        """Initialize the NVT Nose-Hoover state.

        Args:
            state: Initial system state as SimState or dict
            kT: Target temperature in energy units
            tau: Thermostat relaxation time (defaults to 100*dt)
            seed: Random seed for momenta initialization
            **kwargs: Additional state variables

        Returns:
            Initialized NVTNoseHooverState with positions, momenta, forces,
            and thermostat chain variables
        """
        # Set default tau if not provided
        if tau is None:
            tau = dt * 100.0

        # Create thermostat functions
        chain_fns = construct_nose_hoover_chain(
            dt, chain_length, chain_steps, sy_steps, tau
        )

        if not isinstance(state, SimState):
            state = SimState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        model_output = model(state)
        momenta = kwargs.get(
            "momenta",
            calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
        )

        # Calculate initial kinetic energy per system
        KE = calc_kinetic_energy(momenta, state.masses, system_idx=state.system_idx)

        # Calculate degrees of freedom per system
        n_atoms_per_system = torch.bincount(state.system_idx)
        dof_per_system = (
            n_atoms_per_system * state.positions.shape[-1]
        )  # n_atoms * n_dimensions

        # For now, sum the per-system DOF as chain expects a single int
        # This is a limitation that should be addressed in the chain implementation
        total_dof = int(dof_per_system.sum().item())

        # Initialize state
        state = NVTNoseHooverState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
            chain=chain_fns.initialize(total_dof, KE, kT),
            _chain_fns=chain_fns,  # Store the chain functions
        )
        return state  # noqa: RET504

    def nvt_nose_hoover_update(
        state: NVTNoseHooverState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
    ) -> NVTNoseHooverState:
        """Perform one complete Nose-Hoover chain integration step.

        Args:
            state: Current system state containing positions, momenta, forces, and chain
            dt: Integration timestep
            kT: Target temperature in energy units

        Returns:
            Updated state after one complete Nose-Hoover step

        Notes:
            Integration sequence:
            1. Update chain masses based on target temperature
            2. First half-step of chain evolution
            3. Full velocity Verlet step
            4. Update chain kinetic energy
            5. Second half-step of chain evolution
        """
        # Get chain functions from state
        chain_fns = state._chain_fns  # noqa: SLF001
        chain = state.chain

        # Update chain masses based on target temperature
        chain = chain_fns.update_mass(chain, kT)

        # First half-step of chain evolution
        momenta, chain = chain_fns.half_step(state.momenta, chain, kT)
        state.momenta = momenta

        # Full velocity Verlet step
        state = velocity_verlet(state=state, dt=dt, model=model)

        # Update chain kinetic energy per system
        KE = calc_kinetic_energy(state.momenta, state.masses, system_idx=state.system_idx)
        chain.kinetic_energy = KE

        # Second half-step of chain evolution
        momenta, chain = chain_fns.half_step(state.momenta, chain, kT)
        state.momenta = momenta
        state.chain = chain

        return state

    return nvt_nose_hoover_init, nvt_nose_hoover_update


def nvt_nose_hoover_invariant(
    state: NVTNoseHooverState,
    kT: torch.Tensor,
) -> torch.Tensor:
    """Calculate the conserved quantity for NVT ensemble with Nose-Hoover thermostat.

    This function computes the conserved Hamiltonian of the extended system for
    NVT dynamics with a Nose-Hoover chain thermostat. The invariant includes:
    1. System potential energy
    2. System kinetic energy
    3. Chain thermostat energy terms

    This quantity should remain approximately constant during simulation and is
    useful for validating the thermostat implementation.

    Args:
        energy_fn: Function that computes system potential energy given positions
        state: Current state of the system including chain variables
        kT: Target temperature in energy units

    Returns:
        torch.Tensor: The conserved Hamiltonian of the extended NVT dynamics

    Notes:
        - Conservation indicates correct thermostat implementation
        - Drift in this quantity suggests numerical instability
        - Includes both physical and thermostat degrees of freedom
        - Useful for debugging thermostat behavior
    """
    # Calculate system energy terms per system
    e_pot = state.energy
    e_kin = calc_kinetic_energy(state.momenta, state.masses, system_idx=state.system_idx)

    # Get system degrees of freedom per system
    n_atoms_per_system = torch.bincount(state.system_idx)
    dof = n_atoms_per_system * state.positions.shape[-1]  # n_atoms * n_dimensions

    # Start with system energy
    e_tot = e_pot + e_kin

    # Add first thermostat term
    c = state.chain
    # Ensure chain momenta and masses broadcast correctly with batch dimensions
    chain_ke_0 = c.momenta[0] ** 2 / (2 * c.masses[0])
    chain_pe_0 = dof * kT * c.positions[0]

    # If chain variables are scalars but we have batches, broadcast them
    if chain_ke_0.numel() == 1 and e_tot.numel() > 1:
        chain_ke_0 = chain_ke_0.expand_as(e_tot)
    if chain_pe_0.numel() == 1 and e_tot.numel() > 1:
        chain_pe_0 = chain_pe_0.expand_as(e_tot)

    e_tot = e_tot + chain_ke_0 + chain_pe_0

    # Add remaining chain terms
    for pos, momentum, mass in zip(
        c.positions[1:], c.momenta[1:], c.masses[1:], strict=True
    ):
        chain_ke = momentum**2 / (2 * mass)
        chain_pe = kT * pos

        # Ensure proper broadcasting for batch dimensions
        if chain_ke.numel() == 1 and e_tot.numel() > 1:
            chain_ke = chain_ke.expand_as(e_tot)
        if chain_pe.numel() == 1 and e_tot.numel() > 1:
            chain_pe = chain_pe.expand_as(e_tot)

        e_tot = e_tot + chain_ke + chain_pe

    return e_tot
