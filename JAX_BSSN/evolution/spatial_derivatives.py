"""Uniform Cartesian derivatives with parameter-aware boundaries and MAD stencils."""

import jax.numpy as jnp

from JAX_BSSN.evolution.derivatives import (
    SOMMERFELD_BC,
    diff1_field,
    diff2_field,
    diff1_upwind_field,
    diff6_field,
)


def _direction(field, direction):
    return direction % field.ndim - (field.ndim - 3)


def _boundary_data(field, direction, params):
    """Local closure indices and face codes, retaining the global MAD policy."""

    # Import lazily: bssn.__init__ itself imports these derivative wrappers.
    from JAX_BSSN.bssn.variables import get_boundary_codes

    left_bc, right_bc = get_boundary_codes(params, direction)
    mad_q = jnp.where(
        (left_bc == SOMMERFELD_BC) | (right_bc == SOMMERFELD_BC),
        1.0, params.mad_q,
    )
    left_bc = jnp.where(params.owned_faces[2 * direction], left_bc, 0)
    right_bc = jnp.where(params.owned_faces[2 * direction + 1], right_bc, 0)
    start = params.guard_cells
    count = (
        field.shape[field.ndim - 3 + direction]
        if params.owned_shape is None else params.owned_shape[direction]
    )
    reduced = False
    if params.reduced_axes is not None:
        reduced = params.reduced_axes[direction]
    elif params.owned_shape is not None:
        reduced = count == 1
    return left_bc, right_bc, start, start + count, mad_q, reduced


def diff1_physical(field, direction, params):
    d = _direction(field, direction)
    left_bc, right_bc, start, end, mad_q, reduced = _boundary_data(field, d, params)
    derivative = diff1_field(
        field, direction, params.dx, left_bc, right_bc, mad_q=mad_q,
        interior_start=start, interior_end=end,
    )
    return jnp.where(reduced, 0.0, derivative)


def diff2_physical(field, direction, params):
    d = _direction(field, direction)
    left_bc, right_bc, start, end, mad_q, reduced = _boundary_data(field, d, params)
    derivative = diff2_field(
        field, direction, params.dx, left_bc, right_bc, mad_q=mad_q,
        interior_start=start, interior_end=end,
    )
    return jnp.where(reduced, 0.0, derivative)


def diff1_upwind_physical(field, rhs_coefficient, direction, params):
    d = _direction(field, direction)
    left_bc, right_bc, start, end, mad_q, reduced = _boundary_data(field, d, params)
    derivative = diff1_upwind_field(
        field,
        rhs_coefficient,
        direction,
        params.dx,
        left_bc,
        right_bc,
        mad_q=mad_q,
        interior_start=start,
        interior_end=end,
    )
    return jnp.where(reduced, 0.0, derivative)


def ko_dissipation(field, params):
    """Sixth-difference Kreiss–Oliger filter on the uniform grid."""
    terms = []
    for d in range(3):
        h = params.dx
        left_bc, right_bc, start, end, _, reduced = _boundary_data(field, d, params)
        term = h**5 * diff6_field(
            field, field.ndim - 3 + d, h, left_bc, right_bc,
            interior_start=start, interior_end=end,
        )
        terms.append(jnp.where(reduced, 0.0, term))
    return params.nu / 64 * sum(terms)
