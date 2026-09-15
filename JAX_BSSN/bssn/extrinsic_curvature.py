"""Trace and traceless extrinsic-curvature evolution equations."""

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.evolution.derivatives import diff1_field, diff6_field
from JAX_BSSN.bssn.constraints import compute_momentum_constraint_and_derivative
from JAX_BSSN.bssn.geometry import compute_W2_covariant_lapse_hessian, compute_W2_ricci
from JAX_BSSN.bssn.shift_and_lapse import (
    compute_shift_advection,
    compute_shift_derivatives,
)
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
    traceless_part,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables, get_boundary_codes


@jit
def evolve_trace_extrinsic_curvature(vars: BSSNVariables,
                                    params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve trace of extrinsic curvature K.

    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Time derivative of trace K
    """

    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    W2_DiDj_alpha = compute_W2_covariant_lapse_hessian(vars, params)

    first_term = -jnp.einsum(
        'ij...,ij...->...', inv_gamma, W2_DiDj_alpha
    )
    # -W**2 gamma_tilde^ij D_i D_j alpha

    second_term = alpha * jnp.einsum('ij...,kl...,ik...,jl...->...', inv_gamma, inv_gamma, A_ij, A_ij)
    # second term

    third_term = alpha * K**2 / 3.0
    # third term


    shift = vars.shift
    # unpack the shift vector

    fourth_term = compute_shift_advection(K, shift, params)
    # compute the upwinded advection term due to shift


    dt_K = first_term + second_term + third_term + fourth_term
    # compute dt_K

    dK_dx1 = diff6_field(
        vars.trace_K, 0, params.dx, *get_boundary_codes(params, 0)
    )
    dK_dx2 = diff6_field(
        vars.trace_K, 1, params.dx, *get_boundary_codes(params, 1)
    )
    dK_dx3 = diff6_field(
        vars.trace_K, 2, params.dx, *get_boundary_codes(params, 2)
    )
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dK_dx1 + dK_dx2 + dK_dx3)
    # compute dissipation term

    return dt_K + dissipation_term




@jit
def evolve_traceless_extrinsic_curvature(vars: BSSNVariables,
                                        params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve traceless extrinsic curvature A_ij.

    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Time derivative of traceless extrinsic curvature
    """

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    metric_derivs = jnp.stack(
        [
            diff1_field(
                gamma, d + 2, dx, *get_boundary_codes(params, d), mad_q=params.mad_q
            )
            for d in range(3)
        ],
        axis=0,
    )
    christoffel_second = christoffel_symbols_second_kind(
        inv_gamma, metric_derivs
    )


    first_term = alpha * K * A_ij
    # first term

    second_term = -2 * alpha * jnp.einsum('ik...,kl...,lj...->ij...', A_ij, inv_gamma, A_ij)
    # second term

    W2_DiDj_alpha = compute_W2_covariant_lapse_hessian(vars, params)
    W2_ricci = compute_W2_ricci(vars, params)

    third_term = alpha * W2_ricci - W2_DiDj_alpha
    third_term = traceless_part(third_term, vars.conformal_metric, inv_gamma)
    # [alpha W**2 R_ij - W**2 D_i D_j alpha]^TF

    shift = vars.shift
    # unpack the shift vector

    grad_shift = compute_shift_derivatives(shift, params)
    # compute the gradient of the shift vector

    fourth_term = compute_shift_advection(A_ij, shift, params)
    # compute the upwinded advection term due to shift

    fifth_term = (
        jnp.einsum('mi...,mj...->ij...', A_ij, grad_shift)
        + jnp.einsum('mj...,mi...->ij...', A_ij, grad_shift)
    )
    # compute the term due to the gradient of the shift)

    div_shift = jnp.einsum("ii...->...", grad_shift)
    # compute the divergence of the shift vector

    sixth_term = -2.0/3.0 * A_ij * div_shift
    # compute the term due to divergence of shift


    dt_A = first_term + second_term + third_term + fourth_term + fifth_term + sixth_term
    # compute dt_A

    dA_dx1 = diff6_field(
        vars.traceless_K, 2, params.dx, *get_boundary_codes(params, 0)
    )
    dA_dx2 = diff6_field(
        vars.traceless_K, 3, params.dx, *get_boundary_codes(params, 1)
    )
    dA_dx3 = diff6_field(
        vars.traceless_K, 4, params.dx, *get_boundary_codes(params, 2)
    )
    # A_ij is shape (3, 3, ni, nj, nk)
    # compute the 6th derivative in each direction

    M, dMidj = compute_momentum_constraint_and_derivative(vars, params)
    # Momentum constraint and its product-rule spatial derivative

    kappa = params.kappa
    # constraint damping parameter

    DjMi = dMidj - jnp.einsum('kij...,k...->ij...', christoffel_second, M)
    DiMj = jnp.swapaxes(DjMi, 0, 1)
    # compute covariant derivatives of M_i
    seventh_term = kappa/2 * alpha * (DjMi + DiMj)
    # seventh term

    dissipation_term = params.nu / 64 * params.dx**5 * (dA_dx1 + dA_dx2 + dA_dx3)
    # compute dissipation term

    return dt_A + seventh_term + dissipation_term
