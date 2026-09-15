"""
Finite difference derivatives with JAX JIT compilation.

This module provides finite difference operators for computing spatial derivatives
in numerical relativity simulations. All functions are JIT-compiled for performance.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, Union
import numpy as np
from functools import partial

# NOTE: FULLY TESTED AND FUNCTIONAL AS OF DEC 3RD 2025

# Keep this module independent of the BSSN parameter and boundary modules.
# Boundary code 1 selects the nonperiodic Sommerfeld closures.
SOMMERFELD_BC = 1

@jit
def periodic_indexing(idx: int, size: int) -> int:
    """Handle periodic boundary conditions for array indexing."""
    return jnp.where(idx < 0, idx + size, jnp.where(idx >= size, idx - size, idx))

@jit
def get_stencil_indices(i: int, j: int, k: int, direction: int, offset: int, 
                       shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get indices for finite difference stencil with periodic boundaries."""
    ni, nj, nk = shape
    
    if direction == 0:  # x-direction
        new_i = periodic_indexing(i + offset, ni)
        return new_i, j, k
    elif direction == 1:  # y-direction
        new_j = periodic_indexing(j + offset, nj)
        return i, new_j, k
    else:  # z-direction
        new_k = periodic_indexing(k + offset, nk)
        return i, j, new_k

@partial(jit, static_argnames=['direction'])
def diff1_field(
    field: jnp.ndarray,
    direction: int,
    dx: float,
    left_bc: int = 0,
    right_bc: int = 0,
    mad_q: float = 1.0,
) -> jnp.ndarray:
    """
    Compute first derivative of entire field in given direction.
    
    Args:
        field: 3D array containing the field values
        direction: Array axis along which to differentiate
        dx: Grid spacing
        left_bc: Boundary code for the left face on this spatial axis
        right_bc: Boundary code for the right face on this spatial axis
        
    Returns:
        3D array containing the derivative
    """

    # Blend D4 with the centered, sixth-order-accurate first derivative.  The
    # default q=1 is exactly the historical D4 operator.

    field_forward1 = jnp.roll(field, -1, axis=direction)
    field_forward2 = jnp.roll(field, -2, axis=direction)
    field_backward1 = jnp.roll(field, 1, axis=direction)
    field_backward2 = jnp.roll(field, 2, axis=direction)
    field_forward3 = jnp.roll(field, -3, axis=direction)
    field_backward3 = jnp.roll(field, 3, axis=direction)

    d4fdx = 2/3 * (field_forward1 - field_backward1) / dx + \
            -1/12 * (field_forward2 - field_backward2) / dx
    d6fdx = (
        3/4 * (field_forward1 - field_backward1)
        - 3/20 * (field_forward2 - field_backward2)
        + 1/60 * (field_forward3 - field_backward3)
    ) / dx
    dfdx = jax.lax.cond(
        mad_q == 1.0,
        lambda _: d4fdx,
        lambda _: mad_q * d4fdx + (1.0 - mad_q) * d6fdx,
        operand=None,
    )

    field_axis_last = jnp.moveaxis(field, direction, -1)
    derivative_axis_last = jnp.moveaxis(dfdx, direction, -1)

    def replace_left_boundary(derivative):
        left_0 = (
            -25.0 / 12.0 * field_axis_last[..., 0]
            + 4.0 * field_axis_last[..., 1]
            - 3.0 * field_axis_last[..., 2]
            + 4.0 / 3.0 * field_axis_last[..., 3]
            - 1.0 / 4.0 * field_axis_last[..., 4]
        ) / dx
        left_1 = (
            -1.0 / 4.0 * field_axis_last[..., 0]
            - 5.0 / 6.0 * field_axis_last[..., 1]
            + 3.0 / 2.0 * field_axis_last[..., 2]
            - 1.0 / 2.0 * field_axis_last[..., 3]
            + 1.0 / 12.0 * field_axis_last[..., 4]
        ) / dx

        derivative = derivative.at[..., 0].set(left_0)
        derivative = derivative.at[..., 1].set(left_1)
        return derivative

    # MAD's +/-3 stencil needs additional nonperiodic closures.  Until such
    # closures are supplied, deliberately retain D4 on physical Sommerfeld
    # axes; FMR currently uses periodic physical boundaries.
    derivative_axis_last = jax.lax.cond(
        (left_bc == SOMMERFELD_BC) | (right_bc == SOMMERFELD_BC),
        lambda _: jnp.moveaxis(d4fdx, direction, -1),
        lambda _: derivative_axis_last,
        operand=None,
    )
    derivative_axis_last = jax.lax.cond(
        left_bc == SOMMERFELD_BC,
        replace_left_boundary,
        lambda derivative: derivative,
        derivative_axis_last,
    )

    def replace_right_boundary(derivative):
        right_1 = (
            -1.0 / 12.0 * field_axis_last[..., -5]
            + 1.0 / 2.0 * field_axis_last[..., -4]
            - 3.0 / 2.0 * field_axis_last[..., -3]
            + 5.0 / 6.0 * field_axis_last[..., -2]
            + 1.0 / 4.0 * field_axis_last[..., -1]
        ) / dx
        right_0 = (
            1.0 / 4.0 * field_axis_last[..., -5]
            - 4.0 / 3.0 * field_axis_last[..., -4]
            + 3.0 * field_axis_last[..., -3]
            - 4.0 * field_axis_last[..., -2]
            + 25.0 / 12.0 * field_axis_last[..., -1]
        ) / dx

        derivative = derivative.at[..., -2].set(right_1)
        derivative = derivative.at[..., -1].set(right_0)
        return derivative

    derivative_axis_last = jax.lax.cond(
        right_bc == SOMMERFELD_BC,
        replace_right_boundary,
        lambda derivative: derivative,
        derivative_axis_last,
    )

    return jnp.moveaxis(derivative_axis_last, -1, direction)
    # Using 4th-order central difference


@partial(jit, static_argnames=["direction"])
def diff1_upwind_field(
    field: jnp.ndarray,
    rhs_coefficient: jnp.ndarray,
    direction: int,
    dx: float,
    left_bc: int = 0,
    right_bc: int = 0,
    mad_q: float = 1.0,
) -> jnp.ndarray:
    """Compute a shift-advection first derivative along one array axis.

    ``rhs_coefficient`` is the coefficient ``c`` in a right-hand-side term
    ``+ c * partial_i(field)``.  This convention matters: positive ``c``
    selects the forward-biased derivative because the equivalent transport
    velocity in ``partial_t(field) + v * partial_i(field) = ...`` is
    ``v = -c``.  Negative coefficients select the mirrored backward-biased
    derivative, while a zero coefficient selects the centered derivative.

    The fourth-order biased stencil is blended with its sixth-order counterpart
    using the same MAD weight as :func:`diff1_field`.  A physical Sommerfeld
    face forces fourth order on the whole axis.  At each such face the outer
    three points are replaced by the existing boundary-aware centered D4
    derivative, preventing periodic ``roll`` values from entering the biased
    stencil.

    ``direction`` is an absolute array axis.  A coefficient containing only
    the trailing spatial dimensions therefore broadcasts naturally over any
    leading tensor-component dimensions of ``field``.

    Args:
        field: Field values, optionally with leading component dimensions.
        rhs_coefficient: Coefficient of this derivative in the evolution RHS.
        direction: Absolute array axis along which to differentiate.
        dx: Grid spacing along ``direction``.
        left_bc: Boundary code for the left face on this spatial axis.
        right_bc: Boundary code for the right face on this spatial axis.
        mad_q: D4 weight in the D4/D6 MAD blend.

    Returns:
        An array with the same shape as ``field`` containing the selected
        derivative (without multiplication by ``rhs_coefficient``).
    """

    forward1 = jnp.roll(field, -1, axis=direction)
    forward2 = jnp.roll(field, -2, axis=direction)
    forward3 = jnp.roll(field, -3, axis=direction)
    backward1 = jnp.roll(field, 1, axis=direction)
    backward2 = jnp.roll(field, 2, axis=direction)
    backward3 = jnp.roll(field, 3, axis=direction)

    d4_forward = (
        -3.0 * backward1
        - 10.0 * field
        + 18.0 * forward1
        - 6.0 * forward2
        + forward3
    ) / (12.0 * dx)
    d4_backward = (
        -backward3
        + 6.0 * backward2
        - 18.0 * backward1
        + 10.0 * field
        + 3.0 * forward1
    ) / (12.0 * dx)
    d4_centered = (
        2.0 / 3.0 * (forward1 - backward1)
        - 1.0 / 12.0 * (forward2 - backward2)
    ) / dx

    physical_axis = (left_bc == SOMMERFELD_BC) | (
        right_bc == SOMMERFELD_BC
    )

    def d4_stencils(_):
        return d4_forward, d4_backward, d4_centered

    def mad_stencils(_):
        forward4 = jnp.roll(field, -4, axis=direction)
        backward4 = jnp.roll(field, 4, axis=direction)

        d6_forward = (
            2.0 * backward2
            - 24.0 * backward1
            - 35.0 * field
            + 80.0 * forward1
            - 30.0 * forward2
            + 8.0 * forward3
            - forward4
        ) / (60.0 * dx)
        d6_backward = (
            backward4
            - 8.0 * backward3
            + 30.0 * backward2
            - 80.0 * backward1
            + 35.0 * field
            + 24.0 * forward1
            - 2.0 * forward2
        ) / (60.0 * dx)
        d6_centered = (
            3.0 / 4.0 * (forward1 - backward1)
            - 3.0 / 20.0 * (forward2 - backward2)
            + 1.0 / 60.0 * (forward3 - backward3)
        ) / dx

        return (
            mad_q * d4_forward + (1.0 - mad_q) * d6_forward,
            mad_q * d4_backward + (1.0 - mad_q) * d6_backward,
            mad_q * d4_centered + (1.0 - mad_q) * d6_centered,
        )

    forward, backward, centered = jax.lax.cond(
        (mad_q == 1.0) | physical_axis,
        d4_stencils,
        mad_stencils,
        operand=None,
    )
    derivative = jnp.where(
        rhs_coefficient > 0.0,
        forward,
        jnp.where(rhs_coefficient < 0.0, backward, centered),
    )

    def replace_physical_boundaries(selected):
        centered_d4 = diff1_field(
            field,
            direction,
            dx,
            left_bc,
            right_bc,
            mad_q=1.0,
        )
        selected_axis_last = jnp.moveaxis(selected, direction, -1)
        centered_axis_last = jnp.moveaxis(centered_d4, direction, -1)

        selected_axis_last = jax.lax.cond(
            left_bc == SOMMERFELD_BC,
            lambda derivative_axis_last: derivative_axis_last.at[..., :3].set(
                centered_axis_last[..., :3]
            ),
            lambda derivative_axis_last: derivative_axis_last,
            selected_axis_last,
        )
        selected_axis_last = jax.lax.cond(
            right_bc == SOMMERFELD_BC,
            lambda derivative_axis_last: derivative_axis_last.at[..., -3:].set(
                centered_axis_last[..., -3:]
            ),
            lambda derivative_axis_last: derivative_axis_last,
            selected_axis_last,
        )
        return jnp.moveaxis(selected_axis_last, -1, direction)

    return jax.lax.cond(
        physical_axis,
        replace_physical_boundaries,
        lambda selected: selected,
        derivative,
    )


@partial(jit, static_argnames=['direction'])
def diff2_field(
    field: jnp.ndarray,
    direction: int,
    dx: float,
    left_bc: int = 0,
    right_bc: int = 0,
    mad_q: float = 1.0,
) -> jnp.ndarray:
    """Compute a pure second derivative along one array axis.

    Periodic interiors blend the centered fourth- and sixth-order-accurate
    second-derivative stencils using the same MAD weight as ``diff1_field``.
    Physical Sommerfeld axes deliberately retain the fourth-order stencil and
    use fourth-order one-sided closures at their outer two points.
    """

    forward1 = jnp.roll(field, -1, axis=direction)
    forward2 = jnp.roll(field, -2, axis=direction)
    forward3 = jnp.roll(field, -3, axis=direction)
    backward1 = jnp.roll(field, 1, axis=direction)
    backward2 = jnp.roll(field, 2, axis=direction)
    backward3 = jnp.roll(field, 3, axis=direction)

    d4fdx2 = (
        -forward2
        + 16.0 * forward1
        - 30.0 * field
        + 16.0 * backward1
        - backward2
    ) / (12.0 * dx**2)
    d6fdx2 = (
        1.0 / 90.0 * forward3
        - 3.0 / 20.0 * forward2
        + 3.0 / 2.0 * forward1
        - 49.0 / 18.0 * field
        + 3.0 / 2.0 * backward1
        - 3.0 / 20.0 * backward2
        + 1.0 / 90.0 * backward3
    ) / dx**2
    d2fdx2 = jax.lax.cond(
        mad_q == 1.0,
        lambda _: d4fdx2,
        lambda _: mad_q * d4fdx2 + (1.0 - mad_q) * d6fdx2,
        operand=None,
    )

    field_axis_last = jnp.moveaxis(field, direction, -1)
    derivative_axis_last = jnp.moveaxis(d2fdx2, direction, -1)

    def replace_left_boundary(derivative):
        left_0 = (
            15.0 / 4.0 * field_axis_last[..., 0]
            - 77.0 / 6.0 * field_axis_last[..., 1]
            + 107.0 / 6.0 * field_axis_last[..., 2]
            - 13.0 * field_axis_last[..., 3]
            + 61.0 / 12.0 * field_axis_last[..., 4]
            - 5.0 / 6.0 * field_axis_last[..., 5]
        ) / dx**2
        left_1 = (
            5.0 / 6.0 * field_axis_last[..., 0]
            - 5.0 / 4.0 * field_axis_last[..., 1]
            - 1.0 / 3.0 * field_axis_last[..., 2]
            + 7.0 / 6.0 * field_axis_last[..., 3]
            - 1.0 / 2.0 * field_axis_last[..., 4]
            + 1.0 / 12.0 * field_axis_last[..., 5]
        ) / dx**2
        derivative = derivative.at[..., 0].set(left_0)
        derivative = derivative.at[..., 1].set(left_1)
        return derivative

    # The sixth-order centered stencil has no physical boundary closures yet.
    derivative_axis_last = jax.lax.cond(
        (left_bc == SOMMERFELD_BC) | (right_bc == SOMMERFELD_BC),
        lambda _: jnp.moveaxis(d4fdx2, direction, -1),
        lambda _: derivative_axis_last,
        operand=None,
    )
    derivative_axis_last = jax.lax.cond(
        left_bc == SOMMERFELD_BC,
        replace_left_boundary,
        lambda derivative: derivative,
        derivative_axis_last,
    )

    def replace_right_boundary(derivative):
        right_1 = (
            1.0 / 12.0 * field_axis_last[..., -6]
            - 1.0 / 2.0 * field_axis_last[..., -5]
            + 7.0 / 6.0 * field_axis_last[..., -4]
            - 1.0 / 3.0 * field_axis_last[..., -3]
            - 5.0 / 4.0 * field_axis_last[..., -2]
            + 5.0 / 6.0 * field_axis_last[..., -1]
        ) / dx**2
        right_0 = (
            -5.0 / 6.0 * field_axis_last[..., -6]
            + 61.0 / 12.0 * field_axis_last[..., -5]
            - 13.0 * field_axis_last[..., -4]
            + 107.0 / 6.0 * field_axis_last[..., -3]
            - 77.0 / 6.0 * field_axis_last[..., -2]
            + 15.0 / 4.0 * field_axis_last[..., -1]
        ) / dx**2
        derivative = derivative.at[..., -2].set(right_1)
        derivative = derivative.at[..., -1].set(right_0)
        return derivative

    derivative_axis_last = jax.lax.cond(
        right_bc == SOMMERFELD_BC,
        replace_right_boundary,
        lambda derivative: derivative,
        derivative_axis_last,
    )

    return jnp.moveaxis(derivative_axis_last, -1, direction)

@partial(jit, static_argnames=['direction'])
def diff6_field(
    field: jnp.ndarray,
    direction: int,
    dx: float,
    left_bc: int = 0,
    right_bc: int = 0,
) -> jnp.ndarray:
    """
    Compute 6th-order derivative of entire field in given direction.
    
    Uses 7-point stencil for 6th-order accuracy.
    
    Args:
        field: 3D array containing the field values
        direction: Array axis along which to differentiate
        dx: Grid spacing
        left_bc: Boundary code for the left face on this spatial axis
        right_bc: Boundary code for the right face on this spatial axis
    Returns:
        3D array containing the 6th derivative
    """

    # 6th order finite difference coefficients
    # [1, -6, 15, -20, 15, -6, 1] / (dx^6)

    forward3 = jnp.roll(field, -3, axis=direction)
    forward2 = jnp.roll(field, -2, axis=direction)
    forward1 = jnp.roll(field, -1, axis=direction)
    backward1 = jnp.roll(field, 1, axis=direction)
    backward2 = jnp.roll(field, 2, axis=direction)
    backward3 = jnp.roll(field, 3, axis=direction)
    # shift the field to get stencil points

    d6fdx6 = ( forward3 + (-6)*forward2 + 15*forward1 + (-20)*field +
               15*backward1 + (-6)*backward2 + backward3 ) / ( dx**6 )
    # Apply the finite difference formula directly

    field_axis_last = jnp.moveaxis(field, direction, -1)
    derivative_axis_last = jnp.moveaxis(d6fdx6, direction, -1)

    def replace_left_boundary(derivative):
        left_0 = (
            4.0 * field_axis_last[..., 0]
            - 27.0 * field_axis_last[..., 1]
            + 78.0 * field_axis_last[..., 2]
            - 125.0 * field_axis_last[..., 3]
            + 120.0 * field_axis_last[..., 4]
            - 69.0 * field_axis_last[..., 5]
            + 22.0 * field_axis_last[..., 6]
            - 3.0 * field_axis_last[..., 7]
        ) / dx**6
        left_1 = (
            3.0 * field_axis_last[..., 0]
            - 20.0 * field_axis_last[..., 1]
            + 57.0 * field_axis_last[..., 2]
            - 90.0 * field_axis_last[..., 3]
            + 85.0 * field_axis_last[..., 4]
            - 48.0 * field_axis_last[..., 5]
            + 15.0 * field_axis_last[..., 6]
            - 2.0 * field_axis_last[..., 7]
        ) / dx**6
        left_2 = (
            2.0 * field_axis_last[..., 0]
            - 13.0 * field_axis_last[..., 1]
            + 36.0 * field_axis_last[..., 2]
            - 55.0 * field_axis_last[..., 3]
            + 50.0 * field_axis_last[..., 4]
            - 27.0 * field_axis_last[..., 5]
            + 8.0 * field_axis_last[..., 6]
            - field_axis_last[..., 7]
        ) / dx**6

        derivative = derivative.at[..., 0].set(left_0)
        derivative = derivative.at[..., 1].set(left_1)
        derivative = derivative.at[..., 2].set(left_2)
        return derivative

    derivative_axis_last = jax.lax.cond(
        left_bc == SOMMERFELD_BC,
        replace_left_boundary,
        lambda derivative: derivative,
        derivative_axis_last,
    )

    def replace_right_boundary(derivative):
        right_2 = (
            -field_axis_last[..., -8]
            + 8.0 * field_axis_last[..., -7]
            - 27.0 * field_axis_last[..., -6]
            + 50.0 * field_axis_last[..., -5]
            - 55.0 * field_axis_last[..., -4]
            + 36.0 * field_axis_last[..., -3]
            - 13.0 * field_axis_last[..., -2]
            + 2.0 * field_axis_last[..., -1]
        ) / dx**6
        right_1 = (
            -2.0 * field_axis_last[..., -8]
            + 15.0 * field_axis_last[..., -7]
            - 48.0 * field_axis_last[..., -6]
            + 85.0 * field_axis_last[..., -5]
            - 90.0 * field_axis_last[..., -4]
            + 57.0 * field_axis_last[..., -3]
            - 20.0 * field_axis_last[..., -2]
            + 3.0 * field_axis_last[..., -1]
        ) / dx**6
        right_0 = (
            -3.0 * field_axis_last[..., -8]
            + 22.0 * field_axis_last[..., -7]
            - 69.0 * field_axis_last[..., -6]
            + 120.0 * field_axis_last[..., -5]
            - 125.0 * field_axis_last[..., -4]
            + 78.0 * field_axis_last[..., -3]
            - 27.0 * field_axis_last[..., -2]
            + 4.0 * field_axis_last[..., -1]
        ) / dx**6

        derivative = derivative.at[..., -3].set(right_2)
        derivative = derivative.at[..., -2].set(right_1)
        derivative = derivative.at[..., -1].set(right_0)
        return derivative

    derivative_axis_last = jax.lax.cond(
        right_bc == SOMMERFELD_BC,
        replace_right_boundary,
        lambda derivative: derivative,
        derivative_axis_last,
    )

    return jnp.moveaxis(derivative_axis_last, -1, direction)

@jit
def compute_all_derivatives(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute all first derivatives of a field.
    
    Args:
        field: 3D array containing the field values
        dx: Grid spacing
        
    Returns:
        4D array with shape (3, ni, nj, nk) containing derivatives in each direction
    """

    x1_derivative = diff1_field(field, 0, dx)
    y1_derivative = diff1_field(field, 1, dx)
    z1_derivative = diff1_field(field, 2, dx)
    # compute the first derivatives in all three directions

    derivatives = jnp.stack([x1_derivative, y1_derivative, z1_derivative], axis=0)
    # combine them into a single array with shape (3, ni, nj, nk)
    
    return derivatives

@jit
def gradient_3d(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute 3D gradient using 4th-order finite differences.
    
    Args:
        field: 3D scalar field
        dx: Grid spacing (assumed uniform)
        
    Returns:
        4D array with shape (3, ni, nj, nk) containing gradient components
    """
    return compute_all_derivatives(field, dx)

@jit  
def divergence_3d(vector_field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute divergence of a 3D vector field.
    
    Args:
        vector_field: 4D array with shape (3, ni, nj, nk)
        dx: Grid spacing
        
    Returns:
        3D array containing the divergence
    """

    df1dx1 = diff1_field(vector_field[0], 0, dx)
    df2dx2 = diff1_field(vector_field[1], 1, dx)
    df3dx3 = diff1_field(vector_field[2], 2, dx)
    # compute the partial derivatives

    div = df1dx1 + df2dx2 + df3dx3
    # sum them to get the divergence

    return div

@jit
def laplacian_3d(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute 3D Laplacian using finite differences.
    
    Args:
        field: 3D scalar field
        dx: Grid spacing
        
    Returns:
        3D array containing the Laplacian
    """

    d2fdx1dx1 = diff2_field(field, 0, dx)
    d2fdx2dx2 = diff2_field(field, 1, dx)
    d2fdx3dx3 = diff2_field(field, 2, dx)

    lapl = d2fdx1dx1 + d2fdx2dx2 + d2fdx3dx3
    # sum them to get the Laplacian
    
    return lapl
