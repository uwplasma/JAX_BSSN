"""Compact storage and Cartesian support reconstruction for spherical Cartoon.

Two steps carry the evolved state from its compact radial storage onto the
Cartesian support the fourth-order derivative operators need:

``radial_interpolate``
    Evaluate one positive-radius profile at arbitrary radii, taking the
    negative side from parity and the outside from a Sommerfeld continuation.
``transformation_matrix``
    Rotate on-axis frame components onto the Cartesian axes of an off-axis
    support point.

``enforce_spherical_symmetry`` is the matching projection in the other
direction.  Reconstruction does not need it -- it reads only the independent
profiles -- but the RK4 stepper applies it to the state itself so that the
components spherical symmetry forbids cannot accumulate in storage and leak
back through the algebraic constraint projection.
"""

import math

import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.cartoon.symmetry import (
    SCALAR_PARITY,
    map_bssn_fields,
    reflect,
    vector_parity,
)
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC


CARTOON_GHOST_CELLS = 4
CARTOON_SUPPORT_SIZE = 2 * CARTOON_GHOST_CELLS + 1
CARTOON_CENTER = CARTOON_GHOST_CELLS
CARTOON_OUTER_BUFFER_CELLS = 3

VECTOR_X_REFLECTION_PARITY = (-1.0, 1.0, 1.0)
RADIAL_VECTOR_PARITY = VECTOR_X_REFLECTION_PARITY[0]


def _map_fields(vars, transform):
    return map_bssn_fields(vars, transform, VECTOR_X_REFLECTION_PARITY)


def _compact(positive_axis, parity):
    """Attach the parity ghosts and restore the singleton y and z axes."""

    ghosted = reflect(positive_axis, parity, axis=-1, count=CARTOON_GHOST_CELLS)
    return ghosted[..., :, None, None]


def _expanded(positive_axis, parity):
    """Mirror the whole positive axis onto the complete signed x axis."""

    return reflect(positive_axis, parity, axis=-1)[..., :, None, None]


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

    if not math.isfinite(float(params.dx)) or float(params.dx) <= 0.0:
        raise ValueError("spherical Cartoon requires finite dx > 0")

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

    return _map_fields(
        vars,
        lambda field, parity: _compact(
            field[..., positive_slice, 0, 0], parity
        ),
    )


def _project_onto_symmetric_subspace(positive_axis):
    """Keep only the components spherical symmetry allows on the +x axis.

    On the reference axis a spherically symmetric vector points along x and a
    spherically symmetric tensor is ``diag(radial, tangential, tangential)``.
    The yy entry is the authoritative tangential value; zz follows it.
    """

    rank = positive_axis.ndim - 1
    if rank == 0:
        return positive_axis

    zero = jnp.zeros_like(positive_axis[(0,) * rank])
    if rank == 1:
        return jnp.stack((positive_axis[0], zero, zero))
    if rank == 2:
        tangential = positive_axis[1, 1]
        return jnp.stack(
            (
                jnp.stack((positive_axis[0, 0], zero, zero)),
                jnp.stack((zero, tangential, zero)),
                jnp.stack((zero, zero, tangential)),
            )
        )
    raise ValueError(f"unsupported tensor rank {rank} for spherical symmetry")


def enforce_spherical_symmetry(vars: BSSNVariables) -> BSSNVariables:
    """Project the compact state onto the spherically symmetric subspace.

    Components forbidden by spherical symmetry -- the tensor off-diagonals, the
    transverse vector components, and any difference between the yy and zz
    tangential entries -- are discarded on the positive radius, which is then
    reflected across the origin into the four negative-x parity ghosts.
    """

    def transform(field, parity):
        positive = _project_onto_symmetric_subspace(
            cartoon_positive_radius(field)
        )
        return _compact(positive, parity)

    return _map_fields(vars, transform)


def _cartoon_geometry(num_x, dx, dtype):
    """Return the support radius and its broadcast Cartesian coordinates."""

    spacing = jnp.asarray(dx, dtype=dtype)
    x = (
        jnp.arange(num_x, dtype=dtype)
        - CARTOON_GHOST_CELLS
        + 0.5
    ) * spacing
    transverse = (
        jnp.arange(CARTOON_SUPPORT_SIZE, dtype=dtype) - CARTOON_CENTER
    ) * spacing

    X, Y, Z = jnp.broadcast_arrays(
        x[:, None, None], transverse[None, :, None], transverse[None, None, :]
    )

    return jnp.sqrt(X**2 + Y**2 + Z**2), X, Y, Z


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


def radial_interpolate(
    positive_axis: jnp.ndarray,
    radius: jnp.ndarray,
    dx: float,
    parity: float = SCALAR_PARITY,
    asymptotic_value: float = 0.0,
) -> jnp.ndarray:
    """Interpolate one positive-x-axis profile at arbitrary radii.

    The positive half-axis is the only authoritative data.  Samples through the
    origin come from its parity reflection, and samples past the outer face come
    from a Sommerfeld 1/r continuation, so no target ever extrapolates.

    ``parity`` is the sign the profile picks up under ``x -> -x``: ``+1`` for a
    scalar or for the rr and tangential tensor entries, ``-1`` for the radial
    component of a vector.
    """

    axis = reflect(positive_axis, parity, axis=-1, count=CARTOON_GHOST_CELLS)
    axis = _sommerfeld_outer_buffer(axis, asymptotic_value, dx)

    q = (
        radius / jnp.asarray(dx, dtype=radius.dtype)
        + CARTOON_GHOST_CELLS
        - 0.5
    )

    return lagrange6_nonperiodic(axis, q, axis=-1)


def transformation_matrix(
    x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
) -> jnp.ndarray:
    """Return Lambda^i_jbar from the on-axis frame to the point (x, y, z).

    The on-axis frame sits at xbar = (r, 0, 0), so the columns of Lambda are the
    radial, azimuthal and polar unit vectors in Cartesian components.  Lambda is
    a rotation, so it transforms contravariant and covariant indices alike:
    ``v^i = Lambda^i_abar v^abar`` and ``T_ij = Lambda_iabar Lambda_jbbar
    T_abarbbar``.

    The azimuthal and polar columns are fixed only up to a rotation about the
    radius, which is harmless here: they ever multiply a degenerate tangential
    pair or a zero.  The support grid never samples the polar axis, since its x
    coordinates are half-cell offset, but the rho = 0 branch keeps a stray call
    from injecting NaN into a jitted RHS.
    """

    x, y, z = jnp.broadcast_arrays(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)
    )

    r = jnp.sqrt(x**2 + y**2 + z**2)
    rho = jnp.sqrt(x**2 + y**2)
    safe_rho = jnp.where(rho > 0.0, rho, jnp.ones_like(rho))

    zero = jnp.zeros_like(x)
    one = jnp.ones_like(x)
    off_polar_axis = (rho > 0.0)[None]

    radial = jnp.stack((x / r, y / r, z / r))
    azimuthal = jnp.where(
        off_polar_axis,
        jnp.stack((-y / safe_rho, x / safe_rho, zero)),
        jnp.stack((zero, one, zero)),
    )
    polar = jnp.where(
        off_polar_axis,
        jnp.stack(
            (-x * z / (r * safe_rho), -y * z / (r * safe_rho), rho / r)
        ),
        jnp.stack((-z / r, zero, zero)),
    )

    return jnp.stack((radial, azimuthal, polar), axis=1)


def _rotate_vector(rotation, radial_profile):
    """Rotate the on-axis vector (v_r, 0, 0) onto the Cartesian axes."""

    zero = jnp.zeros_like(radial_profile)
    on_axis = jnp.stack((radial_profile, zero, zero))

    return jnp.einsum("ij...,j...->i...", rotation, on_axis)


def _rotate_tensor(rotation, radial_profile, tangential_profile):
    """Rotate diag(T_rr, T_tt, T_tt) onto the Cartesian axes."""

    zero = jnp.zeros_like(radial_profile)
    on_axis = jnp.stack(
        (
            jnp.stack((radial_profile, zero, zero)),
            jnp.stack((zero, tangential_profile, zero)),
            jnp.stack((zero, zero, tangential_profile)),
        )
    )

    return jnp.einsum("ia...,jb...,ab...->ij...", rotation, rotation, on_axis)


def reconstruct_cartoon_support(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Build Cartesian support by radial interpolation and frame rotation.

    Only the four profiles spherical symmetry leaves independent are read --
    the rr and tangential tensor entries and the radial vector component -- so
    no explicit symmetry projection is needed here.  The forbidden components
    of ``vars`` are ignored rather than discarded, and the parity ghosts are
    rebuilt from the positive radius by ``radial_interpolate``.
    """

    num_x = vars.conformal_factor.shape[0]

    radius, X, Y, Z = _cartoon_geometry(num_x, params.dx, vars.lapse.dtype)
    rotation = transformation_matrix(X, Y, Z)

    def profile(field, asymptotic_value, parity=SCALAR_PARITY):
        return radial_interpolate(
            cartoon_positive_radius(field),
            radius,
            params.dx,
            parity,
            asymptotic_value,
        )

    return BSSNVariables(
        conformal_metric=_rotate_tensor(
            rotation,
            profile(vars.conformal_metric[0, 0], 1.0),
            profile(vars.conformal_metric[1, 1], 1.0),
        ),
        conformal_factor=profile(vars.conformal_factor, 1.0),
        traceless_K=_rotate_tensor(
            rotation,
            profile(vars.traceless_K[0, 0], 0.0),
            profile(vars.traceless_K[1, 1], 0.0),
        ),
        trace_K=profile(vars.trace_K, 0.0),
        conformal_connection=_rotate_vector(
            rotation,
            profile(
                vars.conformal_connection[0], 0.0, RADIAL_VECTOR_PARITY
            ),
        ),
        lapse=profile(vars.lapse, 1.0),
        shift=_rotate_vector(
            rotation, profile(vars.shift[0], 0.0, RADIAL_VECTOR_PARITY)
        ),
    )


def _project_support_field(field, parity):
    positive_axis = field[
        ..., CARTOON_GHOST_CELLS:, CARTOON_CENTER, CARTOON_CENTER
    ]
    return _compact(positive_axis, parity)


def project_cartoon_rhs(rhs: BSSNVariables) -> BSSNVariables:
    """Project a support RHS onto the independent compact radial state."""

    return _map_fields(rhs, _project_support_field)


def project_cartoon_scalar(field: jnp.ndarray) -> jnp.ndarray:
    """Project a scalar support field into compact parity-filled storage."""

    return _project_support_field(field, SCALAR_PARITY)


def project_cartoon_vector(field: jnp.ndarray) -> jnp.ndarray:
    """Project a Cartesian vector support field into compact storage."""

    return _project_support_field(
        field, vector_parity(VECTOR_X_REFLECTION_PARITY, field.dtype)
    )


def expand_cartoon_axis(vars: BSSNVariables) -> BSSNVariables:
    """Expand compact BSSN variables onto the complete signed x-axis."""

    return _map_fields(
        vars,
        lambda field, parity: _expanded(
            cartoon_positive_radius(field), parity
        ),
    )


def expand_cartoon_scalar(field: jnp.ndarray) -> jnp.ndarray:
    """Expand a compact even scalar onto the complete signed x-axis."""

    return _expanded(cartoon_positive_radius(field), SCALAR_PARITY)


def expand_cartoon_vector(field: jnp.ndarray) -> jnp.ndarray:
    """Expand a compact Cartesian vector onto the complete signed x-axis."""

    return _expanded(
        cartoon_positive_radius(field),
        vector_parity(VECTOR_X_REFLECTION_PARITY, field.dtype),
    )
