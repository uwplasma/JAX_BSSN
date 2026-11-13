"""Boundary conditions for BSSN evolution variables."""

import jax
import jax.numpy as jnp

from JAX_BSSN.evolution.spatial_derivatives import (
    diff1_physical,
)
from JAX_BSSN.evolution.coordinates import grid_coordinates
from jax import jit
from functools import partial

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables


PERIODIC_BC = 0
SOMMERFELD_BC = 1


def _radial_coordinates(shape, params, dtype):
    """Return broadcast coordinate factors and radius for the spatial grid."""

    X, Y, Z = grid_coordinates(shape, params, dtype)
    r = jnp.sqrt(X**2 + Y**2 + Z**2)

    return X, Y, Z, r


@jit
def radial_derivative(field: jnp.ndarray, params: BSSNParameters) -> jnp.ndarray:
    """Compute the Cartesian contraction ``(x^i / r) partial_i field``."""

    spatial_start = field.ndim - 3
    dfdx = diff1_physical(field, spatial_start, params)
    dfdy = diff1_physical(field, spatial_start + 1, params)
    dfdz = diff1_physical(field, spatial_start + 2, params)

    X, Y, Z, r = _radial_coordinates(field.shape[-3:], params, field.dtype)
    r_safe = jnp.where(r > 0.0, r, 1.0)

    return (X * dfdx + Y * dfdy + Z * dfdz) / r_safe


@partial(jit, static_argnames=["shape"])
def sommerfeld_boundary_mask(shape, params: BSSNParameters) -> jnp.ndarray:
    """Return the union of all grid planes configured as Sommerfeld faces."""

    nx, ny, nz = shape
    i = jnp.arange(nx)[:, None, None]
    j = jnp.arange(ny)[None, :, None]
    k = jnp.arange(nz)[None, None, :]

    start = params.guard_cells
    counts = shape if params.owned_shape is None else params.owned_shape
    codes = (
        params.xl_bc, params.xr_bc, params.yl_bc,
        params.yr_bc, params.zl_bc, params.zr_bc,
    )
    mask = jnp.zeros((nx, ny, nz), dtype=bool)
    owned = jnp.ones((nx, ny, nz), dtype=bool)
    for d, index in enumerate((i, j, k)):
        end = start + counts[d]
        lower = (codes[2*d] == SOMMERFELD_BC) & params.owned_faces[2*d]
        upper = (codes[2*d + 1] == SOMMERFELD_BC) & params.owned_faces[2*d + 1]
        if params.owned_shape is not None:
            lower = lower & (counts[d] > 1)
            upper = upper & (counts[d] > 1)
        mask = mask | (lower & (index == start)) | (upper & (index == end - 1))
        owned = owned & (index >= start) & (index < end)

    return mask & owned


@jit
def sommerfeld(
    field: jnp.ndarray,
    rhs: jnp.ndarray,
    asymptotic_value: jnp.ndarray,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Replace the RHS on active faces with the radial Sommerfeld equation."""

    _, _, _, r = _radial_coordinates(field.shape[-3:], params, field.dtype)
    r_safe = jnp.where(r > 0.0, r, 1.0)
    sommerfeld_rhs = -(
        radial_derivative(field, params) + (field - asymptotic_value) / r_safe
    )

    mask = sommerfeld_boundary_mask(field.shape[-3:], params)
    broadcast_shape = (1,) * (field.ndim - 3) + mask.shape
    mask = mask.reshape(broadcast_shape)

    return jnp.where(mask, sommerfeld_rhs, rhs)


def _apply_active_sommerfeld_boundaries(
    vars: BSSNVariables,
    rhs: BSSNVariables,
    params: BSSNParameters,
) -> BSSNVariables:
    """Apply the Sommerfeld RHS to every BSSN evolution variable."""

    flat_metric = jnp.eye(3, dtype=vars.conformal_metric.dtype)[
        :, :, None, None, None
    ]

    return BSSNVariables(
        conformal_metric=sommerfeld(
            vars.conformal_metric, rhs.conformal_metric, flat_metric, params
        ),
        conformal_factor=sommerfeld(
            vars.conformal_factor,
            rhs.conformal_factor,
            jnp.asarray(1.0, dtype=vars.conformal_factor.dtype),
            params,
        ),
        traceless_K=sommerfeld(
            vars.traceless_K,
            rhs.traceless_K,
            jnp.asarray(0.0, dtype=vars.traceless_K.dtype),
            params,
        ),
        trace_K=sommerfeld(
            vars.trace_K,
            rhs.trace_K,
            jnp.asarray(0.0, dtype=vars.trace_K.dtype),
            params,
        ),
        conformal_connection=sommerfeld(
            vars.conformal_connection,
            rhs.conformal_connection,
            jnp.asarray(0.0, dtype=vars.conformal_connection.dtype),
            params,
        ),
        lapse=sommerfeld(
            vars.lapse,
            rhs.lapse,
            jnp.asarray(1.0, dtype=vars.lapse.dtype),
            params,
        ),
        shift=sommerfeld(
            vars.shift,
            rhs.shift,
            jnp.asarray(0.0, dtype=vars.shift.dtype),
            params,
        ),
    )


@jit
def apply_sommerfeld_boundaries(
    vars: BSSNVariables,
    rhs: BSSNVariables,
    params: BSSNParameters,
) -> BSSNVariables:
    """Apply Sommerfeld RHS replacement, or leave non-Sommerfeld runs unchanged."""

    has_sommerfeld_bc = (
        (params.xl_bc == SOMMERFELD_BC)
        | (params.xr_bc == SOMMERFELD_BC)
        | (params.yl_bc == SOMMERFELD_BC)
        | (params.yr_bc == SOMMERFELD_BC)
        | (params.zl_bc == SOMMERFELD_BC)
        | (params.zr_bc == SOMMERFELD_BC)
    )

    return jax.lax.cond(
        has_sommerfeld_bc,
        lambda _: _apply_active_sommerfeld_boundaries(vars, rhs, params),
        lambda _: rhs,
        operand=None,
    )
