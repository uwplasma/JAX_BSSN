"""
Boundary filters for BSSN evolution variables.

The derivative operators in this codebase remain periodic. These routines apply
an additional state filter near selected faces, damping evolved fields toward
the flat-space BSSN solution after periodic RK stage construction.
"""

import jax
import jax.numpy as jnp
from jax import jit

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables


PERIODIC_BC = 0
SUPERGAUSSIAN_BC = 1


def _side_weight(shape, axis, side_code, left_side, params, dtype):
    """Return one finite-support super-Gaussian face weight."""

    n = shape[axis]
    indices = jnp.arange(n, dtype=dtype)

    if left_side:
        distance = indices
    else:
        distance = (n - 1) - indices

    width = jnp.maximum(jnp.asarray(params.bc_width, dtype=dtype), 1.0)
    order = jnp.maximum(jnp.asarray(params.bc_order, dtype=dtype), 1.0)
    strength = jnp.clip(jnp.asarray(params.bc_strength, dtype=dtype), 0.0, 1.0)

    face_weight = jnp.exp(-jnp.power(distance / width, order))
    face_weight = jnp.where(distance < width, face_weight, 0.0)

    active = jnp.asarray(side_code == SUPERGAUSSIAN_BC, dtype=dtype)
    face_weight = active * strength * face_weight

    broadcast_shape = [1, 1, 1]
    broadcast_shape[axis] = n

    return face_weight.reshape(tuple(broadcast_shape))


@jit
def supergaussian_boundary_weight(vars: BSSNVariables, params: BSSNParameters):
    """Build the combined boundary weight for all active faces."""

    shape = vars.conformal_factor.shape
    dtype = vars.conformal_factor.dtype

    one_minus_weight = jnp.ones(shape, dtype=dtype)

    for axis, left_code, right_code in (
        (0, params.xl_bc, params.xr_bc),
        (1, params.yl_bc, params.yr_bc),
        (2, params.zl_bc, params.zr_bc),
    ):
        left_weight = _side_weight(shape, axis, left_code, True, params, dtype)
        right_weight = _side_weight(shape, axis, right_code, False, params, dtype)
        one_minus_weight = one_minus_weight * (1.0 - left_weight)
        one_minus_weight = one_minus_weight * (1.0 - right_weight)

    return 1.0 - one_minus_weight


def _blend_to_flat(field, flat_target, weight):
    """Blend a field toward its flat-space target with the supplied weight."""

    return flat_target + (field - flat_target) * (1.0 - weight)


def _apply_active_supergaussian_boundaries(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """
    Dampen active boundary layers toward the flat-space BSSN solution.

    Metric diagonal components are driven to 1 while off-diagonal components are
    driven to 0. The remaining dynamical fields use their flat-space values.
    """

    weight = supergaussian_boundary_weight(vars, params)
    tensor_weight = weight[None, None, ...]
    vector_weight = weight[None, ...]

    flat_metric = (
        jnp.eye(3, dtype=vars.conformal_metric.dtype)[:, :, None, None, None]
        * jnp.ones_like(vars.conformal_metric)
    )

    return BSSNVariables(
        conformal_metric=_blend_to_flat(
            vars.conformal_metric, flat_metric, tensor_weight
        ),
        conformal_factor=_blend_to_flat(
            vars.conformal_factor, jnp.ones_like(vars.conformal_factor), weight
        ),
        traceless_K=_blend_to_flat(
            vars.traceless_K, jnp.zeros_like(vars.traceless_K), tensor_weight
        ),
        trace_K=_blend_to_flat(
            vars.trace_K, jnp.zeros_like(vars.trace_K), weight
        ),
        conformal_connection=_blend_to_flat(
            vars.conformal_connection,
            jnp.zeros_like(vars.conformal_connection),
            vector_weight,
        ),
        lapse=_blend_to_flat(vars.lapse, jnp.ones_like(vars.lapse), weight),
        shift=_blend_to_flat(
            vars.shift, jnp.zeros_like(vars.shift), vector_weight
        ),
    )


@jit
def apply_supergaussian_boundaries(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Apply active super-Gaussian boundary filters, or leave periodic runs unchanged."""

    has_supergaussian_bc = (
        (params.xl_bc == SUPERGAUSSIAN_BC)
        | (params.xr_bc == SUPERGAUSSIAN_BC)
        | (params.yl_bc == SUPERGAUSSIAN_BC)
        | (params.yr_bc == SUPERGAUSSIAN_BC)
        | (params.zl_bc == SUPERGAUSSIAN_BC)
        | (params.zr_bc == SUPERGAUSSIAN_BC)
    )

    return jax.lax.cond(
        has_supergaussian_bc,
        lambda _: _apply_active_supergaussian_boundaries(vars, params),
        lambda _: vars,
        operand=None,
    )
