"""Conformal spatial-metric and conformal-factor evolution equations."""

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.evolution.derivatives import diff6_field
from JAX_BSSN.bssn.shift_and_lapse import (
    compute_shift_advection,
    compute_shift_derivatives,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables, get_boundary_codes


@jit
def evolve_conformal_metric(vars: BSSNVariables,
                           params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal metric γ_ij.

    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Time derivative of conformal metric
    """
    shift = vars.shift
    # unpack the shift vector

    grad_shift = compute_shift_derivatives(shift, params)
    # grad_shift[i, j] = partial_j beta^i

    first_term = compute_shift_advection(
        vars.conformal_metric, shift, params
    )
    # compute the upwinded advection term due to shift

    second_term = jnp.einsum("mi...,mj...->ij...", vars.conformal_metric, grad_shift)
    # compute the term due to the gradient of the shift

    third_term = jnp.einsum("mj...,mi...->ij...", vars.conformal_metric, grad_shift)
    # compute the term due to the gradient of the shift

    div_shift = jnp.einsum("ii...->...", grad_shift)
    # compute divergence of the shift vector

    fourth_term = -2.0/3.0 * vars.conformal_metric * div_shift
    # compute the term due to the divergence of the shift

    fifth_term = -2.0 * vars.lapse * vars.traceless_K
    # compute the term due to the traceless extrinsic curvature

    dt_gamma = first_term + second_term + third_term + fourth_term + fifth_term
    # compute dt_gamma

    # Kreiss-Oliger dissipation can be added here if desired
    dgamma_dx1 = diff6_field(
        vars.conformal_metric, 2, params.dx, *get_boundary_codes(params, 0)
    )
    dgamma_dx2 = diff6_field(
        vars.conformal_metric, 3, params.dx, *get_boundary_codes(params, 1)
    )
    dgamma_dx3 = diff6_field(
        vars.conformal_metric, 4, params.dx, *get_boundary_codes(params, 2)
    )
    # gamma is shape (3, 3, ni, nj, nk)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dgamma_dx1 + dgamma_dx2 + dgamma_dx3)
    # compute dissipation term

    return dt_gamma + dissipation_term


@jit
def evolve_conformal_factor(vars: BSSNVariables,
                           params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal factor W.

    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Time derivative of conformal factor
    """

    shift = vars.shift
    # unpack the shift vector

    # First term: advection due to shift: beta^i ∂_i W
    first_term = compute_shift_advection(
        vars.conformal_factor, shift, params
    )
    # compute the upwinded advection term due to shift

    # Second term: (1/3) α W K
    second_term = (1.0/3.0) * vars.lapse * vars.conformal_factor * vars.trace_K

    grad_shift = compute_shift_derivatives(shift, params)
    # grad_shift[i, j] = partial_j beta^i

    div_shift = jnp.einsum("ii...->...", grad_shift)
    # compute the divergence of the shift vector

    third_term = -(1.0/3.0) * vars.conformal_factor * div_shift
    # compute the term due to divergence of shift

    dW_dx1 = diff6_field(
        vars.conformal_factor, 0, params.dx, *get_boundary_codes(params, 0)
    )
    dW_dx2 = diff6_field(
        vars.conformal_factor, 1, params.dx, *get_boundary_codes(params, 1)
    )
    dW_dx3 = diff6_field(
        vars.conformal_factor, 2, params.dx, *get_boundary_codes(params, 2)
    )
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dW_dx1 + dW_dx2 + dW_dx3)
    # compute dissipation term


    return first_term + second_term + third_term + dissipation_term
