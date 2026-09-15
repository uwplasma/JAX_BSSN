"""Lapse and shift evolution equations."""

import jax
import jax.numpy as jnp
from jax import jit

from JAX_BSSN.evolution.derivatives import (
    diff1_field,
    diff1_upwind_field,
    diff6_field,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables, get_boundary_codes


@jit
def compute_shift_derivatives(
    shift: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """
    Compute spatial derivatives of the shift vector.

    The returned tensor is indexed as d_beta[i, j] = partial_j beta^i.
    Keeping the component and derivative axes separate is important for the
    weighted Lie derivative terms in the nonzero-shift BSSN equations.
    """

    d_beta = jnp.stack(
        [
            jnp.stack(
                [
                    diff1_field(
                        shift[i, ...],
                        j,
                        params.dx,
                        *get_boundary_codes(params, j), mad_q=params.mad_q,
                    )
                    for j in range(3)
                ],
                axis=0,
            )
            for i in range(3)
        ],
        axis=0,
    )

    return d_beta


@jit
def compute_shift_advection(
    field: jnp.ndarray, shift: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """Return the upwinded shift-advection term ``beta^i partial_i field``.

    ``diff1_upwind_field`` interprets its coefficient with the sign it has on
    the right-hand side. Passing ``shift[i]`` therefore selects the stable
    lop-sided stencil for the BSSN convention
    ``partial_t field = ... + beta^i partial_i field``. Only transport terms
    use this helper; derivatives of the shift in Lie and geometric terms must
    continue to use :func:`compute_shift_derivatives`.
    """

    spatial_start = field.ndim - 3
    directional_terms = [
        shift[direction]
        * diff1_upwind_field(
            field,
            shift[direction],
            spatial_start + direction,
            params.dx,
            *get_boundary_codes(params, direction),
            mad_q=params.mad_q,
        )
        for direction in range(3)
    ]
    return sum(directional_terms)




@jit
def evolve_lapse(vars: BSSNVariables, params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve lapse function.

    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Time derivative of lapse
    """
    def harmonic_slicing(_):
        return -jnp.power(vars.lapse, 2) * vars.trace_K

    def one_plus_log_slicing(_):
        return -2.0 * vars.lapse * vars.trace_K

    slicing_term = jax.lax.cond(
        params.gauge == 0,
        harmonic_slicing,
        one_plus_log_slicing,
        operand=None,
    )
    # choose the lapse source term without leaving JIT-compatible control flow

    advection_term = compute_shift_advection(vars.lapse, vars.shift, params)
    # advect the lapse with the shift using the production upwind operator

    dalpha_dx1 = diff6_field(
        vars.lapse, 0, params.dx, *get_boundary_codes(params, 0)
    )
    dalpha_dx2 = diff6_field(
        vars.lapse, 1, params.dx, *get_boundary_codes(params, 1)
    )
    dalpha_dx3 = diff6_field(
        vars.lapse, 2, params.dx, *get_boundary_codes(params, 2)
    )
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dalpha_dx1 + dalpha_dx2 + dalpha_dx3)
    # compute dissipation term

    return slicing_term + advection_term + dissipation_term


@jit
def evolve_shift(vars: BSSNVariables, params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve shift vector β^i using Gamma driver.


    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Time derivative of shift
    """
    shift = vars.shift
    # unpack the shift vector

    advection_term = compute_shift_advection(shift, shift, params)
    # beta^j partial_j beta^i, with only this transport derivative upwinded

    gamma_driver_term = params.g * vars.conformal_connection
    # single-variable Gamma-driver source for beta^i

    damping_term = -params.eta * shift
    # linear damping of the shift

    dt_beta = gamma_driver_term + advection_term + damping_term

    dbeta_dx1 = diff6_field(
        shift, 1, params.dx, *get_boundary_codes(params, 0)
    )
    dbeta_dx2 = diff6_field(
        shift, 2, params.dx, *get_boundary_codes(params, 1)
    )
    dbeta_dx3 = diff6_field(
        shift, 3, params.dx, *get_boundary_codes(params, 2)
    )
    # beta is shape (3, ni, nj, nk)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (
        dbeta_dx1 + dbeta_dx2 + dbeta_dx3
    )
    # compute dissipation term

    return dt_beta + dissipation_term
