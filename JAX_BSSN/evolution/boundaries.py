"""Boundary conditions for BSSN evolution variables."""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.derivatives import diff1_field


PERIODIC_BC = 0
SOMMERFELD_BC = 1


def _radial_coordinates(shape, params, dtype):
    """Return broadcast coordinate factors and radius for the spatial grid."""

    nx, ny, nz = shape
    dx = jnp.asarray(params.dx, dtype=dtype)
    x = jnp.asarray(params.x_min, dtype=dtype) + dx * jnp.arange(nx, dtype=dtype)
    y = jnp.asarray(params.y_min, dtype=dtype) + dx * jnp.arange(ny, dtype=dtype)
    z = jnp.asarray(params.z_min, dtype=dtype) + dx * jnp.arange(nz, dtype=dtype)

    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]
    r = jnp.sqrt(X**2 + Y**2 + Z**2)

    return X, Y, Z, r


@jit
def radial_derivative(field: jnp.ndarray, params: BSSNParameters) -> jnp.ndarray:
    """Compute the Cartesian contraction ``(x^i / r) partial_i field``."""

    spatial_start = field.ndim - 3
    dfdx = diff1_field(
        field,
        spatial_start,
        params.dx,
        params.xl_bc,
        params.xr_bc,
        params.mad_q,
    )
    dfdy = diff1_field(
        field,
        spatial_start + 1,
        params.dx,
        params.yl_bc,
        params.yr_bc,
        params.mad_q,
    )
    dfdz = diff1_field(
        field,
        spatial_start + 2,
        params.dx,
        params.zl_bc,
        params.zr_bc,
        params.mad_q,
    )

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

    mask = jnp.zeros((nx, ny, nz), dtype=bool)
    mask = mask | ((params.xl_bc == SOMMERFELD_BC) & (i == 0))
    mask = mask | ((params.xr_bc == SOMMERFELD_BC) & (i == nx - 1))
    mask = mask | ((params.yl_bc == SOMMERFELD_BC) & (j == 0))
    mask = mask | ((params.yr_bc == SOMMERFELD_BC) & (j == ny - 1))
    mask = mask | ((params.zl_bc == SOMMERFELD_BC) & (k == 0))
    mask = mask | ((params.zr_bc == SOMMERFELD_BC) & (k == nz - 1))

    return mask


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
