"""Stage-wise spherical Cartoon evolution."""

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.spherical_symmetry.reconstruction import (
    fill_cartoon_ghosts,
    project_cartoon_rhs,
    reconstruct_cartoon_support,
)
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs,
    enforce_algebraic_constraints,
)


@jit
def compute_cartoon_rhs(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Evaluate the Cartesian BSSN RHS and project it onto compact storage."""

    support_vars = reconstruct_cartoon_support(vars, params)
    support_rhs = compute_bssn_rhs(support_vars, params)
    return project_cartoon_rhs(support_rhs)


def _add_scaled(vars, rhs, scale):
    return BSSNVariables(
        *(field + scale * derivative for field, derivative in zip(vars, rhs))
    )


def _prepare_stage(vars):
    # Algebraic projection acts pointwise. Refreshing the ghosts afterwards
    # makes the reflected half-cell data authoritative for reconstruction.
    vars = enforce_algebraic_constraints(vars)
    return fill_cartoon_ghosts(vars)


@jit
def cartoon_rk4_step(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Advance one compact spherical Cartoon step with classical RK4."""

    dt = params.dt
    vars = _prepare_stage(vars)

    k1 = compute_cartoon_rhs(vars, params)

    midpoint = _prepare_stage(_add_scaled(vars, k1, 0.5 * dt))
    k2 = compute_cartoon_rhs(midpoint, params)

    midpoint = _prepare_stage(_add_scaled(vars, k2, 0.5 * dt))
    k3 = compute_cartoon_rhs(midpoint, params)

    endpoint = _prepare_stage(_add_scaled(vars, k3, dt))
    k4 = compute_cartoon_rhs(endpoint, params)

    new_vars = BSSNVariables(
        *(
            field
            + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
            for field, d1, d2, d3, d4 in zip(vars, k1, k2, k3, k4)
        )
    )

    return _prepare_stage(new_vars)
