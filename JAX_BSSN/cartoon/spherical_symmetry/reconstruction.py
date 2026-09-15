"""Compact storage and Cartesian support reconstruction for spherical Cartoon."""

import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC


CARTOON_GHOST_CELLS = 4
CARTOON_SUPPORT_SIZE = 2 * CARTOON_GHOST_CELLS + 1
CARTOON_CENTER = CARTOON_GHOST_CELLS
CARTOON_OUTER_BUFFER_CELLS = 3

VECTOR_X_REFLECTION_PARITY = (-1.0, 1.0, 1.0)


def _vector_parity(dtype):
    return jnp.asarray(VECTOR_X_REFLECTION_PARITY, dtype=dtype)


def _tensor_parity(dtype):
    vector_parity = _vector_parity(dtype)
    return vector_parity[:, None] * vector_parity[None, :]


def _reflect_inner_ghosts(positive_axis, parity):
    """Prepend the four reflected half-cell samples nearest the origin."""

    parity = jnp.asarray(parity, dtype=positive_axis.dtype)
    parity = parity.reshape(parity.shape + (1,))
    ghosts = jnp.flip(
        positive_axis[..., :CARTOON_GHOST_CELLS], axis=-1
    ) * parity
    return jnp.concatenate((ghosts, positive_axis), axis=-1)


def _reflect_full_axis(positive_axis, parity):
    """Reflect every positive sample to form the complete signed axis."""

    parity = jnp.asarray(parity, dtype=positive_axis.dtype)
    parity = parity.reshape(parity.shape + (1,))
    negative_axis = jnp.flip(positive_axis, axis=-1) * parity
    return jnp.concatenate((negative_axis, positive_axis), axis=-1)


def _compact_field_from_positive_axis(positive_axis, parity):
    compact_axis = _reflect_inner_ghosts(positive_axis, parity)
    return compact_axis[..., :, None, None]


def _full_field_from_positive_axis(positive_axis, parity):
    full_axis = _reflect_full_axis(positive_axis, parity)
    return full_axis[..., :, None, None]


def validate_cartoon_grid(
    vars: BSSNVariables, params: BSSNParameters
) -> None:
    """Validate the compact radial state and temporary support geometry."""

    nx_compact, ny, nz = vars.conformal_factor.shape
    num_radial_points = nx_compact - CARTOON_GHOST_CELLS

    if ny != 1 or nz != 1:
        raise ValueError(
            "spherical Cartoon state requires an (Nr + 4) x 1 x 1 grid"
        )
    if num_radial_points < 6:
        raise ValueError(
            "spherical Cartoon mode requires at least six radial points"
        )
    if params.xl_bc != PERIODIC_BC:
        raise ValueError(
            "Cartoon x-left is a parity ghost region, not a physical boundary"
        )
    if params.xr_bc != SOMMERFELD_BC:
        raise ValueError("Cartoon x-right must use the Sommerfeld boundary")
    if (
        params.yl_bc != PERIODIC_BC
        or params.yr_bc != PERIODIC_BC
        or params.zl_bc != PERIODIC_BC
        or params.zr_bc != PERIODIC_BC
    ):
        raise ValueError("Cartoon y/z support faces must be periodic")
    if float(params.mad_q) != 1.0:
        raise ValueError(
            "Cartoon reconstruction requires mad_q=1 for its 9 x 9 support"
        )

    expected_minima = (
        -(CARTOON_GHOST_CELLS - 0.5) * params.dx,
        -CARTOON_GHOST_CELLS * params.dx,
        -CARTOON_GHOST_CELLS * params.dx,
    )
    actual_minima = (params.x_min, params.y_min, params.z_min)
    if any(
        abs(float(actual) - float(expected))
        > 1.0e-12 * max(1.0, abs(float(expected)))
        for actual, expected in zip(actual_minima, expected_minima)
    ):
        raise ValueError(
            "Cartoon coordinate minima must be (-3.5dx, -4dx, -4dx)"
        )


def cartoon_centerline(field: jnp.ndarray) -> jnp.ndarray:
    """Return the x-axis from compact storage or reconstructed support."""

    if field.shape[-2:] == (1, 1):
        return field[..., :, 0, 0]
    return field[..., :, CARTOON_CENTER, CARTOON_CENTER]


def cartoon_positive_radius(field: jnp.ndarray) -> jnp.ndarray:
    """Return the authoritative positive radial half-axis."""

    return cartoon_centerline(field)[..., CARTOON_GHOST_CELLS:]


def compact_cartoon_state(vars: BSSNVariables) -> BSSNVariables:
    """Convert an even, full signed x-axis state to compact storage."""

    full_nx, ny, nz = vars.conformal_factor.shape
    if ny != 1 or nz != 1 or full_nx % 2:
        raise ValueError(
            "Cartoon initialization requires an even Nx x 1 x 1 signed axis"
        )

    positive_slice = slice(full_nx // 2, None)
    scalar_parity = jnp.asarray(1.0, dtype=vars.lapse.dtype)
    vector_parity = _vector_parity(vars.shift.dtype)
    tensor_parity = _tensor_parity(vars.conformal_metric.dtype)

    return BSSNVariables(
        conformal_metric=_compact_field_from_positive_axis(
            vars.conformal_metric[..., positive_slice, 0, 0], tensor_parity
        ),
        conformal_factor=_compact_field_from_positive_axis(
            vars.conformal_factor[positive_slice, 0, 0], scalar_parity
        ),
        traceless_K=_compact_field_from_positive_axis(
            vars.traceless_K[..., positive_slice, 0, 0], tensor_parity
        ),
        trace_K=_compact_field_from_positive_axis(
            vars.trace_K[positive_slice, 0, 0], scalar_parity
        ),
        conformal_connection=_compact_field_from_positive_axis(
            vars.conformal_connection[..., positive_slice, 0, 0],
            vector_parity,
        ),
        lapse=_compact_field_from_positive_axis(
            vars.lapse[positive_slice, 0, 0], scalar_parity
        ),
        shift=_compact_field_from_positive_axis(
            vars.shift[..., positive_slice, 0, 0], vector_parity
        ),
    )


def fill_cartoon_ghosts(vars: BSSNVariables) -> BSSNVariables:
    """Overwrite compact negative-x ghosts from the positive radial axis."""

    scalar_parity = jnp.asarray(1.0, dtype=vars.lapse.dtype)
    vector_parity = _vector_parity(vars.shift.dtype)
    tensor_parity = _tensor_parity(vars.conformal_metric.dtype)

    return BSSNVariables(
        conformal_metric=_compact_field_from_positive_axis(
            cartoon_positive_radius(vars.conformal_metric), tensor_parity
        ),
        conformal_factor=_compact_field_from_positive_axis(
            cartoon_positive_radius(vars.conformal_factor), scalar_parity
        ),
        traceless_K=_compact_field_from_positive_axis(
            cartoon_positive_radius(vars.traceless_K), tensor_parity
        ),
        trace_K=_compact_field_from_positive_axis(
            cartoon_positive_radius(vars.trace_K), scalar_parity
        ),
        conformal_connection=_compact_field_from_positive_axis(
            cartoon_positive_radius(vars.conformal_connection), vector_parity
        ),
        lapse=_compact_field_from_positive_axis(
            cartoon_positive_radius(vars.lapse), scalar_parity
        ),
        shift=_compact_field_from_positive_axis(
            cartoon_positive_radius(vars.shift), vector_parity
        ),
    )


def _cartoon_geometry(num_x, dx, dtype):
    spacing = jnp.asarray(dx, dtype=dtype)
    x = (
        jnp.arange(num_x, dtype=dtype)
        - CARTOON_GHOST_CELLS
        + 0.5
    ) * spacing
    transverse = (
        jnp.arange(CARTOON_SUPPORT_SIZE, dtype=dtype) - CARTOON_CENTER
    ) * spacing

    X = x[:, None, None]
    Y = transverse[None, :, None]
    Z = transverse[None, None, :]
    shape = (num_x, CARTOON_SUPPORT_SIZE, CARTOON_SUPPORT_SIZE)
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    direction = jnp.stack(
        [
            jnp.broadcast_to(X / r, shape),
            jnp.broadcast_to(Y / r, shape),
            jnp.broadcast_to(Z / r, shape),
        ]
    )

    return r, direction


def _sommerfeld_outer_buffer(axis, asymptotic_value, dx):
    """Append temporary samples with Sommerfeld's radial 1/r falloff.

    The compact state's last sample remains the physical outer face.  These
    three reconstruction-only samples prevent off-axis radii from requesting
    polynomial extrapolation beyond that face.
    """

    spacing = jnp.asarray(dx, dtype=axis.dtype)
    asymptotic = jnp.asarray(asymptotic_value, dtype=axis.dtype)
    asymptotic = jnp.broadcast_to(asymptotic, axis.shape[:-1])[..., None]
    outer_radius = (
        axis.shape[-1] - CARTOON_GHOST_CELLS - 0.5
    ) * spacing
    buffer_radius = outer_radius + spacing * jnp.arange(
        1, CARTOON_OUTER_BUFFER_CELLS + 1, dtype=axis.dtype
    )
    buffer = asymptotic + (axis[..., -1:] - asymptotic) * (
        outer_radius / buffer_radius
    )
    return jnp.concatenate((axis, buffer), axis=-1)


def _interpolate_axis(field, r, dx, asymptotic_value):
    """Interpolate the full parity-filled signed axis at positive radius."""

    axis = cartoon_centerline(field)
    axis = _sommerfeld_outer_buffer(axis, asymptotic_value, dx)
    q = (
        r / jnp.asarray(dx, dtype=r.dtype)
        + CARTOON_GHOST_CELLS
        - 0.5
    )
    return lagrange6_nonperiodic(axis, q, axis=-1)


def _reconstruct_tensor(radial, tangential, direction):
    identity = jnp.eye(3, dtype=radial.dtype)[:, :, None, None, None]

    return (
        tangential[None, None, ...] * identity
        + (radial - tangential)[None, None, ...]
        * jnp.einsum("i...,j...->ij...", direction, direction)
    )


def reconstruct_cartoon_support(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Build Cartesian support by signed-axis interpolation and rotation."""

    vars = fill_cartoon_ghosts(vars)
    num_x = vars.conformal_factor.shape[0]
    r, direction = _cartoon_geometry(num_x, params.dx, vars.lapse.dtype)

    conformal_factor = _interpolate_axis(
        vars.conformal_factor, r, params.dx, 1.0
    )
    trace_K = _interpolate_axis(vars.trace_K, r, params.dx, 0.0)
    lapse = _interpolate_axis(vars.lapse, r, params.dx, 1.0)

    shift_radial = _interpolate_axis(vars.shift[0], r, params.dx, 0.0)
    connection_radial = _interpolate_axis(
        vars.conformal_connection[0], r, params.dx, 0.0
    )
    shift = shift_radial[None, ...] * direction
    conformal_connection = connection_radial[None, ...] * direction

    metric_radial = _interpolate_axis(
        vars.conformal_metric[0, 0], r, params.dx, 1.0
    )
    metric_tangential = _interpolate_axis(
        vars.conformal_metric[1, 1], r, params.dx, 1.0
    )
    A_radial = _interpolate_axis(
        vars.traceless_K[0, 0], r, params.dx, 0.0
    )
    A_tangential = _interpolate_axis(
        vars.traceless_K[1, 1], r, params.dx, 0.0
    )

    return BSSNVariables(
        conformal_metric=_reconstruct_tensor(
            metric_radial,
            metric_tangential,
            direction,
        ),
        conformal_factor=conformal_factor,
        traceless_K=_reconstruct_tensor(
            A_radial,
            A_tangential,
            direction,
        ),
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
    )


def _project_support_field(field, parity):
    positive_axis = field[
        ..., CARTOON_GHOST_CELLS:, CARTOON_CENTER, CARTOON_CENTER
    ]
    return _compact_field_from_positive_axis(positive_axis, parity)


def project_cartoon_rhs(rhs: BSSNVariables) -> BSSNVariables:
    """Project a support RHS onto the independent compact radial state."""

    scalar_parity = jnp.asarray(1.0, dtype=rhs.lapse.dtype)
    vector_parity = _vector_parity(rhs.shift.dtype)
    tensor_parity = _tensor_parity(rhs.conformal_metric.dtype)

    return BSSNVariables(
        conformal_metric=_project_support_field(
            rhs.conformal_metric, tensor_parity
        ),
        conformal_factor=_project_support_field(
            rhs.conformal_factor, scalar_parity
        ),
        traceless_K=_project_support_field(rhs.traceless_K, tensor_parity),
        trace_K=_project_support_field(rhs.trace_K, scalar_parity),
        conformal_connection=_project_support_field(
            rhs.conformal_connection, vector_parity
        ),
        lapse=_project_support_field(rhs.lapse, scalar_parity),
        shift=_project_support_field(rhs.shift, vector_parity),
    )


def project_cartoon_scalar(field: jnp.ndarray) -> jnp.ndarray:
    """Project a scalar support field into compact parity-filled storage."""

    return _project_support_field(
        field, jnp.asarray(1.0, dtype=field.dtype)
    )


def project_cartoon_vector(field: jnp.ndarray) -> jnp.ndarray:
    """Project a Cartesian vector support field into compact storage."""

    return _project_support_field(field, _vector_parity(field.dtype))


def expand_cartoon_axis(vars: BSSNVariables) -> BSSNVariables:
    """Expand compact BSSN variables onto the complete signed x-axis."""

    scalar_parity = jnp.asarray(1.0, dtype=vars.lapse.dtype)
    vector_parity = _vector_parity(vars.shift.dtype)
    tensor_parity = _tensor_parity(vars.conformal_metric.dtype)

    return BSSNVariables(
        conformal_metric=_full_field_from_positive_axis(
            cartoon_positive_radius(vars.conformal_metric), tensor_parity
        ),
        conformal_factor=_full_field_from_positive_axis(
            cartoon_positive_radius(vars.conformal_factor), scalar_parity
        ),
        traceless_K=_full_field_from_positive_axis(
            cartoon_positive_radius(vars.traceless_K), tensor_parity
        ),
        trace_K=_full_field_from_positive_axis(
            cartoon_positive_radius(vars.trace_K), scalar_parity
        ),
        conformal_connection=_full_field_from_positive_axis(
            cartoon_positive_radius(vars.conformal_connection), vector_parity
        ),
        lapse=_full_field_from_positive_axis(
            cartoon_positive_radius(vars.lapse), scalar_parity
        ),
        shift=_full_field_from_positive_axis(
            cartoon_positive_radius(vars.shift), vector_parity
        ),
    )


def expand_cartoon_scalar(field: jnp.ndarray) -> jnp.ndarray:
    """Expand a compact even scalar onto the complete signed x-axis."""

    return _full_field_from_positive_axis(
        cartoon_positive_radius(field), jnp.asarray(1.0, dtype=field.dtype)
    )


def expand_cartoon_vector(field: jnp.ndarray) -> jnp.ndarray:
    """Expand a compact Cartesian vector onto the complete signed x-axis."""

    return _full_field_from_positive_axis(
        cartoon_positive_radius(field), _vector_parity(field.dtype)
    )
