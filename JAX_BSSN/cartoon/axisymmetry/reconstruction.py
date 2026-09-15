"""Compact z-axis axisymmetric Cartoon reconstruction.

The evolved state lives on the ``y = 0`` half-plane.  Four negative-``x``
entries are parity ghosts, while the remaining samples are positive radial
half-cell centres.  Reconstruction produces the nine Cartesian ``y`` planes
needed by the existing fourth-order derivative operators.
"""

import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC


AXISYMMETRIC_GHOST_CELLS = 4
AXISYMMETRIC_SUPPORT_SIZE = 9
AXISYMMETRIC_CENTER = 4
AXISYMMETRIC_OUTER_BUFFER_CELLS = 3

VECTOR_RADIAL_REFLECTION_PARITY = (-1.0, -1.0, 1.0)


def _vector_parity(dtype):
    return jnp.asarray(VECTOR_RADIAL_REFLECTION_PARITY, dtype=dtype)


def _tensor_parity(dtype):
    parity = _vector_parity(dtype)
    return parity[:, None] * parity[None, :]


def _reflect_inner_ghosts(positive_plane, parity):
    parity = jnp.asarray(parity, dtype=positive_plane.dtype)
    parity = parity.reshape(parity.shape + (1, 1))
    ghosts = jnp.flip(
        positive_plane[..., :AXISYMMETRIC_GHOST_CELLS, :], axis=-2
    ) * parity
    return jnp.concatenate((ghosts, positive_plane), axis=-2)


def _reflect_full_plane(positive_plane, parity):
    parity = jnp.asarray(parity, dtype=positive_plane.dtype)
    parity = parity.reshape(parity.shape + (1, 1))
    negative = jnp.flip(positive_plane, axis=-2) * parity
    return jnp.concatenate((negative, positive_plane), axis=-2)


def _compact_field(positive_plane, parity):
    return jnp.expand_dims(_reflect_inner_ghosts(positive_plane, parity), axis=-2)


def _expanded_field(positive_plane, parity):
    return jnp.expand_dims(_reflect_full_plane(positive_plane, parity), axis=-2)


def _reference_plane(field):
    """Remove the singleton or reconstructed y dimension."""

    return field[..., :, 0, :]


def _positive_plane(field):
    return _reference_plane(field)[..., AXISYMMETRIC_GHOST_CELLS :, :]


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
    if dx <= 0.0:
        raise ValueError("axisymmetric Cartoon requires dx > 0")
    if not jnp.isclose(params.x_min, -3.5 * dx):
        raise ValueError("axisymmetric Cartoon requires x_min=-3.5*dx")
    if not jnp.isclose(params.y_min, -4.0 * dx):
        raise ValueError("axisymmetric Cartoon requires y_min=-4*dx")


def compact_axisymmetric_state(vars: BSSNVariables) -> BSSNVariables:
    """Compact a complete signed ``x-z`` plane into positive-rho storage."""

    nx, ny, _ = vars.conformal_factor.shape
    if ny != 1 or nx % 2:
        raise ValueError("input must be an even-length signed x plane with y=1")
    positive = slice(nx // 2, None)

    def scalar(field):
        return _compact_field(field[:, 0, :][positive, :], 1.0)

    def vector(field):
        return _compact_field(
            field[:, :, 0, :][:, positive, :], _vector_parity(field.dtype)
        )

    def tensor(field):
        return _compact_field(
            field[:, :, :, 0, :][:, :, positive, :], _tensor_parity(field.dtype)
        )

    return BSSNVariables(
        conformal_metric=tensor(vars.conformal_metric),
        conformal_factor=scalar(vars.conformal_factor),
        traceless_K=tensor(vars.traceless_K),
        trace_K=scalar(vars.trace_K),
        conformal_connection=vector(vars.conformal_connection),
        lapse=scalar(vars.lapse),
        shift=vector(vars.shift),
    )


def fill_axisymmetric_ghosts(vars: BSSNVariables) -> BSSNVariables:
    """Replace the four negative-rho entries using rotation-by-pi parity."""

    def scalar(field):
        return _compact_field(_positive_plane(field), 1.0)

    def vector(field):
        return _compact_field(_positive_plane(field), _vector_parity(field.dtype))

    def tensor(field):
        return _compact_field(_positive_plane(field), _tensor_parity(field.dtype))

    return BSSNVariables(
        conformal_metric=tensor(vars.conformal_metric),
        conformal_factor=scalar(vars.conformal_factor),
        traceless_K=tensor(vars.traceless_K),
        trace_K=scalar(vars.trace_K),
        conformal_connection=vector(vars.conformal_connection),
        lapse=scalar(vars.lapse),
        shift=vector(vars.shift),
    )


def _outer_buffer(reference, asymptotic, params):
    """Append three fixed-z samples with physical 1/r falloff."""

    nx, nz = reference.shape[-2:]
    num_radial_points = nx - AXISYMMETRIC_GHOST_CELLS
    dtype = reference.dtype
    dx = jnp.asarray(params.dx, dtype=dtype)
    rho_edge = (num_radial_points - 0.5) * dx
    z = jnp.asarray(params.z_min, dtype=dtype) + dx * jnp.arange(nz, dtype=dtype)
    r_edge = jnp.sqrt(rho_edge**2 + z**2)
    rho_buffer = rho_edge + dx * jnp.arange(
        1, AXISYMMETRIC_OUTER_BUFFER_CELLS + 1, dtype=dtype
    )
    r_buffer = jnp.sqrt(rho_buffer[:, None] ** 2 + z[None, :] ** 2)
    ratio = r_edge[None, :] / r_buffer

    asymptotic = jnp.asarray(asymptotic, dtype=dtype)
    leading = reference.ndim - 2
    asymptotic = asymptotic.reshape(asymptotic.shape + (1, 1))
    ratio = ratio.reshape((1,) * leading + ratio.shape)
    buffer = asymptotic + (reference[..., -1:, :] - asymptotic) * ratio
    return jnp.concatenate((reference, buffer), axis=-2)


def _support_geometry(reference, params):
    nx = reference.shape[-2]
    dtype = reference.dtype
    dx = jnp.asarray(params.dx, dtype=dtype)
    x = (jnp.arange(nx, dtype=dtype) - 3.5) * dx
    y = (jnp.arange(AXISYMMETRIC_SUPPORT_SIZE, dtype=dtype) - 4.0) * dx
    X = x[:, None]
    Y = y[None, :]
    rho = jnp.sqrt(X**2 + Y**2)
    q = rho / dx + 3.5
    cosine = X / rho
    sine = Y / rho
    return q, cosine, sine


def _rotation_matrix(cosine, sine, dtype):
    zeros = jnp.zeros_like(cosine)
    ones = jnp.ones_like(cosine)
    rows = (
        jnp.stack((cosine, -sine, zeros), axis=0),
        jnp.stack((sine, cosine, zeros), axis=0),
        jnp.stack((zeros, zeros, ones), axis=0),
    )
    return jnp.stack(rows, axis=0).astype(dtype)[..., None]


def _interpolate_scalar(field, asymptotic, params):
    reference = _reference_plane(field)
    source = _outer_buffer(reference, asymptotic, params)
    q, _, _ = _support_geometry(reference, params)
    return lagrange6_nonperiodic(source, q, axis=-2)


def _interpolate_vector(field, asymptotic, params):
    reference = _reference_plane(field)
    source = _outer_buffer(reference, asymptotic, params)
    q, cosine, sine = _support_geometry(reference, params)
    radial_values = lagrange6_nonperiodic(source, q, axis=-2)
    rotation = _rotation_matrix(cosine, sine, field.dtype)
    return jnp.einsum("ijxyz,jxyz->ixyz", rotation, radial_values)


def _interpolate_tensor(field, asymptotic, params):
    reference = _reference_plane(field)
    source = _outer_buffer(reference, asymptotic, params)
    q, cosine, sine = _support_geometry(reference, params)
    radial_values = lagrange6_nonperiodic(source, q, axis=-2)
    rotation = _rotation_matrix(cosine, sine, field.dtype)
    return jnp.einsum(
        "ikxyz,klxyz,jlxyz->ijxyz", rotation, radial_values, rotation
    )


def reconstruct_axisymmetric_support(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Interpolate the reference plane and rotate it onto nine y planes."""

    flat_metric = jnp.eye(3, dtype=vars.conformal_metric.dtype)
    return BSSNVariables(
        conformal_metric=_interpolate_tensor(
            vars.conformal_metric, flat_metric, params
        ),
        conformal_factor=_interpolate_scalar(vars.conformal_factor, 1.0, params),
        traceless_K=_interpolate_tensor(vars.traceless_K, jnp.zeros((3, 3)), params),
        trace_K=_interpolate_scalar(vars.trace_K, 0.0, params),
        conformal_connection=_interpolate_vector(
            vars.conformal_connection, jnp.zeros(3), params
        ),
        lapse=_interpolate_scalar(vars.lapse, 1.0, params),
        shift=_interpolate_vector(vars.shift, jnp.zeros(3), params),
    )


def _project_scalar(field):
    positive = field[AXISYMMETRIC_GHOST_CELLS :, AXISYMMETRIC_CENTER, :]
    return _compact_field(positive, 1.0)


def _project_vector(field):
    positive = field[:, AXISYMMETRIC_GHOST_CELLS :, AXISYMMETRIC_CENTER, :]
    return _compact_field(positive, _vector_parity(field.dtype))


def _project_tensor(field):
    positive = field[:, :, AXISYMMETRIC_GHOST_CELLS :, AXISYMMETRIC_CENTER, :]
    return _compact_field(positive, _tensor_parity(field.dtype))


def project_axisymmetric_rhs(rhs: BSSNVariables) -> BSSNVariables:
    """Retain positive-rho reference-plane RHS values and rebuild ghosts."""

    return BSSNVariables(
        conformal_metric=_project_tensor(rhs.conformal_metric),
        conformal_factor=_project_scalar(rhs.conformal_factor),
        traceless_K=_project_tensor(rhs.traceless_K),
        trace_K=_project_scalar(rhs.trace_K),
        conformal_connection=_project_vector(rhs.conformal_connection),
        lapse=_project_scalar(rhs.lapse),
        shift=_project_vector(rhs.shift),
    )


def expand_axisymmetric_plane(vars: BSSNVariables) -> BSSNVariables:
    """Expand compact fields to the complete signed ``x-z`` plane."""

    def scalar(field):
        return _expanded_field(_positive_plane(field), 1.0)

    def vector(field):
        return _expanded_field(_positive_plane(field), _vector_parity(field.dtype))

    def tensor(field):
        return _expanded_field(_positive_plane(field), _tensor_parity(field.dtype))

    return BSSNVariables(
        conformal_metric=tensor(vars.conformal_metric),
        conformal_factor=scalar(vars.conformal_factor),
        traceless_K=tensor(vars.traceless_K),
        trace_K=scalar(vars.trace_K),
        conformal_connection=vector(vars.conformal_connection),
        lapse=scalar(vars.lapse),
        shift=vector(vars.shift),
    )


def _expand_axisymmetric_scalar(field):
    return _expanded_field(_positive_plane(field), 1.0)


def _expand_axisymmetric_vector(field):
    return _expanded_field(_positive_plane(field), _vector_parity(field.dtype))


__all__ = [
    "AXISYMMETRIC_CENTER",
    "AXISYMMETRIC_GHOST_CELLS",
    "AXISYMMETRIC_OUTER_BUFFER_CELLS",
    "AXISYMMETRIC_SUPPORT_SIZE",
    "compact_axisymmetric_state",
    "expand_axisymmetric_plane",
    "fill_axisymmetric_ghosts",
    "project_axisymmetric_rhs",
    "reconstruct_axisymmetric_support",
    "validate_axisymmetric_grid",
]
