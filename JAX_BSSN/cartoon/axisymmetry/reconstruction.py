"""Compact z-axis axisymmetric Cartoon reconstruction.

The evolved state lives on the ``y = 0`` half-plane.  Four negative-``x``
entries are parity ghosts, while the remaining samples are positive radial
half-cell centres.  Reconstruction produces the nine Cartesian ``y`` planes
needed by the existing fourth-order derivative operators, by interpolating each
profile along the positive radius and rotating it about the z axis.
"""

import math

import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.cartoon.symmetry import (
    SCALAR_PARITY,
    map_bssn_fields,
    reflect,
    tensor_parity,
    vector_parity,
)
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC


AXISYMMETRIC_GHOST_CELLS = 4
AXISYMMETRIC_SUPPORT_SIZE = 9
AXISYMMETRIC_CENTER = 4
AXISYMMETRIC_OUTER_BUFFER_CELLS = 3

VECTOR_RADIAL_REFLECTION_PARITY = (-1.0, -1.0, 1.0)


def _map_fields(vars, transform):
    return map_bssn_fields(vars, transform, VECTOR_RADIAL_REFLECTION_PARITY)


def _compact_field(positive_plane, parity):
    """Attach the parity ghosts and restore the singleton y plane."""

    ghosted = reflect(
        positive_plane, parity, axis=-2, count=AXISYMMETRIC_GHOST_CELLS
    )
    return jnp.expand_dims(ghosted, axis=-2)


def _expanded_field(positive_plane, parity):
    """Mirror the whole positive radius onto the complete signed x-z plane."""

    return jnp.expand_dims(reflect(positive_plane, parity, axis=-2), axis=-2)


def _reference_plane(field):
    """Remove the singleton or reconstructed y dimension."""

    return field[..., :, 0, :]


def _positive_plane(field):
    """Return the authoritative positive-rho half of the reference plane."""

    return _reference_plane(field)[..., AXISYMMETRIC_GHOST_CELLS:, :]


def validate_axisymmetric_grid(
    vars: BSSNVariables, params: BSSNParameters
) -> None:
    """Validate compact storage, support geometry, and boundary choices."""

    nx_compact, ny, nz = vars.conformal_factor.shape
    num_radial_points = nx_compact - AXISYMMETRIC_GHOST_CELLS
    spatial_shape = (nx_compact, ny, nz)

    if ny != 1:
        raise ValueError("axisymmetric Cartoon state requires a singleton y plane")
    if num_radial_points < 6:
        raise ValueError("axisymmetric Cartoon mode requires at least six radial points")
    if nz < 9:
        raise ValueError("axisymmetric Cartoon mode requires at least nine z points")

    expected_shapes = (
        (vars.conformal_metric, (3, 3) + spatial_shape, "conformal_metric"),
        (vars.conformal_factor, spatial_shape, "conformal_factor"),
        (vars.traceless_K, (3, 3) + spatial_shape, "traceless_K"),
        (vars.trace_K, spatial_shape, "trace_K"),
        (vars.conformal_connection, (3,) + spatial_shape, "conformal_connection"),
        (vars.lapse, spatial_shape, "lapse"),
        (vars.shift, (3,) + spatial_shape, "shift"),
    )
    for field, shape, name in expected_shapes:
        if field.shape != shape:
            raise ValueError(f"{name} has shape {field.shape}; expected {shape}")

    if params.xl_bc != PERIODIC_BC:
        raise ValueError("axisymmetric x-left is a parity ghost region")
    if params.xr_bc != SOMMERFELD_BC:
        raise ValueError("axisymmetric x-right must use the Sommerfeld boundary")
    if params.yl_bc != PERIODIC_BC or params.yr_bc != PERIODIC_BC:
        raise ValueError("axisymmetric reconstructed y faces must be periodic")
    valid_z_codes = (PERIODIC_BC, SOMMERFELD_BC)
    if params.zl_bc not in valid_z_codes or params.zr_bc not in valid_z_codes:
        raise ValueError("axisymmetric z faces must be periodic or Sommerfeld")
    if params.mad_q != 1:
        raise ValueError("axisymmetric Cartoon requires mad_q=1")

    dx = float(params.dx)
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("axisymmetric Cartoon requires finite dx > 0")
    if not jnp.isclose(params.x_min, -3.5 * dx):
        raise ValueError("axisymmetric Cartoon requires x_min=-3.5*dx")
    if not jnp.isclose(params.y_min, -4.0 * params.dx):
        raise ValueError("axisymmetric Cartoon requires y_min=-4*dx")


def compact_axisymmetric_state(vars: BSSNVariables) -> BSSNVariables:
    """Compact a complete signed ``x-z`` plane into positive-rho storage."""

    nx, ny, _ = vars.conformal_factor.shape
    if ny != 1 or nx % 2:
        raise ValueError("input must be an even-length signed x plane with y=1")
    positive = slice(nx // 2, None)

    return _map_fields(
        vars,
        lambda field, parity: _compact_field(
            _reference_plane(field)[..., positive, :], parity
        ),
    )


def fill_axisymmetric_ghosts(vars: BSSNVariables) -> BSSNVariables:
    """Replace the four negative-rho entries using rotation-by-pi parity.

    Unlike spherical symmetry, axisymmetry about z forbids no component on the
    reference plane: ``gamma_xz`` is physical for a deformed spacetime, and the
    y components carry the rotational twist.  Parity is the whole of the
    symmetry condition here.
    """

    return _map_fields(
        vars,
        lambda field, parity: _compact_field(_positive_plane(field), parity),
    )


def _outer_buffer(reference, asymptotic, params):
    """Append three fixed-z samples with physical 1/r falloff."""

    nx, nz = reference.shape[-2:]
    num_radial_points = nx - AXISYMMETRIC_GHOST_CELLS
    dtype = reference.dtype
    dx = jnp.asarray(params.dx, dtype=dtype)
    rho_edge = (num_radial_points - 0.5) * dx
    z = jnp.asarray(params.z_min, dtype=dtype) + dx * jnp.arange(nz, dtype=dtype)
    rho_buffer = rho_edge + dx * jnp.arange(
        1, AXISYMMETRIC_OUTER_BUFFER_CELLS + 1, dtype=dtype
    )
    r_edge = jnp.sqrt(rho_edge**2 + z**2)
    r_buffer = jnp.sqrt(rho_buffer[:, None] ** 2 + z[None, :] ** 2)
    ratio = r_edge[None, :] / r_buffer

    asymptotic = jnp.asarray(asymptotic, dtype=dtype)
    leading = reference.ndim - 2
    asymptotic = asymptotic.reshape(asymptotic.shape + (1, 1))
    ratio = ratio.reshape((1,) * leading + ratio.shape)
    buffer = asymptotic + (reference[..., -1:, :] - asymptotic) * ratio
    return jnp.concatenate((reference, buffer), axis=-2)


def _support_geometry(num_x, params, dtype):
    """Return the support cylindrical radius and its x and y coordinates."""

    dx = jnp.asarray(params.dx, dtype=dtype)
    x = (jnp.arange(num_x, dtype=dtype) - 3.5) * dx
    y = (jnp.arange(AXISYMMETRIC_SUPPORT_SIZE, dtype=dtype) - 4.0) * dx
    X, Y = jnp.broadcast_arrays(x[:, None], y[None, :])

    return jnp.sqrt(X**2 + Y**2), X, Y


def radial_interpolate(
    positive_plane: jnp.ndarray,
    rho: jnp.ndarray,
    params: BSSNParameters,
    parity: float = SCALAR_PARITY,
    asymptotic_value: float = 0.0,
) -> jnp.ndarray:
    """Interpolate a positive-rho reference-plane profile at arbitrary radii.

    The positive half-plane is the only authoritative data.  Samples through
    the axis come from its rotation-by-pi parity reflection, and samples past
    the outer face come from a fixed-z 1/r continuation, so no target ever
    extrapolates.  ``parity`` carries the component axes, so a whole vector or
    tensor is interpolated in one call.
    """

    plane = reflect(
        positive_plane, parity, axis=-2, count=AXISYMMETRIC_GHOST_CELLS
    )
    source = _outer_buffer(plane, asymptotic_value, params)
    q = rho / jnp.asarray(params.dx, dtype=rho.dtype) + AXISYMMETRIC_GHOST_CELLS - 0.5

    return lagrange6_nonperiodic(source, q, axis=-2)


def transformation_matrix(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Return Lambda^i_jbar from the reference half-plane to the point (x, y).

    The reference frame sits at ybar = 0, so the columns of Lambda are the
    cylindrical radial, azimuthal and axial unit vectors in Cartesian
    components -- that is, a rotation by phi about z.  It transforms
    contravariant and covariant indices alike: ``v^i = Lambda^i_abar v^abar``
    and ``T_ij = Lambda_iabar Lambda_jbbar T_abarbbar``.

    The support grid never samples rho = 0, since its x coordinates are
    half-cell offset, but the rho = 0 branch keeps a stray call from injecting
    NaN into a jitted RHS.
    """

    x, y = jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y))

    rho = jnp.sqrt(x**2 + y**2)
    on_axis = rho <= 0.0
    safe_rho = jnp.where(on_axis, jnp.ones_like(rho), rho)
    cosine = jnp.where(on_axis, jnp.ones_like(x), x / safe_rho)
    sine = jnp.where(on_axis, jnp.zeros_like(x), y / safe_rho)

    zero = jnp.zeros_like(x)
    one = jnp.ones_like(x)

    return jnp.stack(
        (
            jnp.stack((cosine, sine, zero)),
            jnp.stack((-sine, cosine, zero)),
            jnp.stack((zero, zero, one)),
        ),
        axis=1,
    )


def _rotate_vector(rotation, reference_vector):
    return jnp.einsum("ij...,j...->i...", rotation, reference_vector)


def _rotate_tensor(rotation, reference_tensor):
    return jnp.einsum(
        "ia...,jb...,ab...->ij...", rotation, rotation, reference_tensor
    )


def reconstruct_axisymmetric_support(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Interpolate the reference plane and rotate it onto nine y planes."""

    dtype = vars.lapse.dtype
    num_x = vars.conformal_factor.shape[0]
    rho, X, Y = _support_geometry(num_x, params, dtype)
    # The rotation is independent of z; the trailing axis lets it broadcast
    # against the interpolated (..., nx, 9, nz) support.
    rotation = transformation_matrix(X[..., None], Y[..., None])

    flat_metric = jnp.eye(3, dtype=vars.conformal_metric.dtype)
    vector = vector_parity(VECTOR_RADIAL_REFLECTION_PARITY, dtype)
    tensor = tensor_parity(VECTOR_RADIAL_REFLECTION_PARITY, dtype)

    def profile(field, parity, asymptotic_value):
        return radial_interpolate(
            _positive_plane(field), rho, params, parity, asymptotic_value
        )

    return BSSNVariables(
        conformal_metric=_rotate_tensor(
            rotation, profile(vars.conformal_metric, tensor, flat_metric)
        ),
        conformal_factor=profile(vars.conformal_factor, SCALAR_PARITY, 1.0),
        traceless_K=_rotate_tensor(
            rotation, profile(vars.traceless_K, tensor, jnp.zeros((3, 3)))
        ),
        trace_K=profile(vars.trace_K, SCALAR_PARITY, 0.0),
        conformal_connection=_rotate_vector(
            rotation,
            profile(vars.conformal_connection, vector, jnp.zeros(3)),
        ),
        lapse=profile(vars.lapse, SCALAR_PARITY, 1.0),
        shift=_rotate_vector(
            rotation, profile(vars.shift, vector, jnp.zeros(3))
        ),
    )


def _project_support_field(field, parity):
    positive = field[..., AXISYMMETRIC_GHOST_CELLS:, AXISYMMETRIC_CENTER, :]
    return _compact_field(positive, parity)


def project_axisymmetric_rhs(rhs: BSSNVariables) -> BSSNVariables:
    """Retain positive-rho reference-plane RHS values and rebuild ghosts."""

    return _map_fields(rhs, _project_support_field)


def project_axisymmetric_scalar(field: jnp.ndarray) -> jnp.ndarray:
    """Project a scalar support field into compact parity-filled storage."""

    return _project_support_field(field, SCALAR_PARITY)


def project_axisymmetric_vector(field: jnp.ndarray) -> jnp.ndarray:
    """Project a Cartesian vector support field into compact storage."""

    return _project_support_field(
        field, vector_parity(VECTOR_RADIAL_REFLECTION_PARITY, field.dtype)
    )


def expand_axisymmetric_plane(vars: BSSNVariables) -> BSSNVariables:
    """Expand compact fields to the complete signed ``x-z`` plane."""

    return _map_fields(
        vars,
        lambda field, parity: _expanded_field(_positive_plane(field), parity),
    )


def expand_axisymmetric_scalar(field: jnp.ndarray) -> jnp.ndarray:
    """Expand a compact even scalar onto the complete signed ``x-z`` plane."""

    return _expanded_field(_positive_plane(field), SCALAR_PARITY)


def expand_axisymmetric_vector(field: jnp.ndarray) -> jnp.ndarray:
    """Expand a compact Cartesian vector onto the signed ``x-z`` plane."""

    return _expanded_field(
        _positive_plane(field),
        vector_parity(VECTOR_RADIAL_REFLECTION_PARITY, field.dtype),
    )


__all__ = [
    "AXISYMMETRIC_CENTER",
    "AXISYMMETRIC_GHOST_CELLS",
    "AXISYMMETRIC_OUTER_BUFFER_CELLS",
    "AXISYMMETRIC_SUPPORT_SIZE",
    "compact_axisymmetric_state",
    "expand_axisymmetric_plane",
    "expand_axisymmetric_scalar",
    "expand_axisymmetric_vector",
    "fill_axisymmetric_ghosts",
    "project_axisymmetric_rhs",
    "project_axisymmetric_scalar",
    "project_axisymmetric_vector",
    "radial_interpolate",
    "reconstruct_axisymmetric_support",
    "transformation_matrix",
    "validate_axisymmetric_grid",
]
