from dataclasses import asdict
from typing import Any

import pytest
import torch

from torchsim.state import BaseState
from torchsim.unbatched_optimizers import fire, gradient_descent


pytest.skip(
    "Unbatched optimizers deprecated, skipping optimizer tests", allow_module_level=True
)


def test_fire_optimizer(si_base_state: BaseState, lj_calculator: Any) -> None:
    """Test FIRE optimization of Si structure."""

    # perturb the structure
    si_base_state.positions[1][0] = 1.0

    # Initialize optimizer
    state, update_fn = fire(
        model=lj_calculator,
        **asdict(si_base_state),
        dt_start=0.01,
        dt_max=0.1,
        n_min=5,
        f_inc=1.1,
        f_dec=0.5,
        alpha_start=0.1,
    )

    # Run a few optimization steps
    initial_force_norm = torch.norm(state.forces)
    for _ in range(10):
        state = update_fn(state)

    # Check that forces are being minimized
    assert torch.norm(state.forces) < initial_force_norm, (
        "Forces should decrease during optimization"
    )

    # Check state properties
    assert hasattr(state, "dt"), "FIRE state should have timestep"
    assert hasattr(state, "alpha"), "FIRE state should have mixing parameter"
    assert hasattr(state, "n_pos"), "FIRE state should have step counter"
    assert state.positions.shape == si_base_state.positions.shape


def test_gradient_descent_optimizer(si_base_state: BaseState, lj_calculator: Any) -> None:
    """Test gradient descent optimization of Si structure."""

    # perturb the structure
    si_base_state.positions[1][0] = 1.0

    # Initialize optimizer
    state, update_fn = gradient_descent(
        model=lj_calculator,
        **asdict(si_base_state),
        learning_rate=0.01,
    )

    # Run a few optimization steps
    initial_energy = state.energy
    for _ in range(10):
        state = update_fn(state)

    # Check that energy is decreasing
    assert state.energy < initial_energy, "Energy should decrease during optimization"

    # Check state properties
    assert hasattr(state, "lr"), "GD state should have learning rate"
    assert state.positions.shape == si_base_state.positions.shape


def test_optimizer_convergence(si_base_state: BaseState, lj_calculator: Any) -> None:
    """Test that both optimizers can reach similar final states."""

    # perturb the structure
    si_base_state.positions[1][0] = 1.0

    # Run FIRE
    fire_state, fire_update = fire(
        model=lj_calculator,
        **asdict(si_base_state),
        dt_start=0.01,
    )

    # Run GD
    gd_state, gd_update = gradient_descent(
        **asdict(si_base_state),
        model=lj_calculator,
        learning_rate=0.01,
    )

    # Optimize both for more steps
    for _ in range(50):
        fire_state = fire_update(fire_state)
        gd_state = gd_update(gd_state)

    # Check that both methods reach similar energies
    assert torch.allclose(
        fire_state.energy,
        gd_state.energy,
        rtol=1e-1,  # 10% tolerance
    ), "Both optimizers should converge to similar energies"
