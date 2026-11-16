"""Trace and traceless extrinsic-curvature evolution equations."""

import jax.numpy as jnp

from JAX_BSSN.evolution.spatial_derivatives import (
    diff1_physical, ko_dissipation,
)
from jax import jit

from JAX_BSSN.bssn.constraints import (
    compute_momentum_constraint_and_derivative,
    compute_momentum_constraint_and_derivative_with_matter,
)
from JAX_BSSN.bssn.geometry import (
    W_FLOOR_VALUE,
    compute_W2_covariant_lapse_hessian,
    compute_W2_ricci,
)
from JAX_BSSN.bssn.shift_and_lapse import (
    compute_shift_advection,
    compute_shift_derivatives,
)
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
    traceless_part,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables


def _evolve_trace_extrinsic_curvature(
    vars: BSSNVariables,
    params: BSSNParameters,
    energy_density=None,
    spatial_stress=None,
) -> jnp.ndarray:
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


    matter_term = 4.0 * jnp.pi * alpha * params.rho
    # static matter source 4*pi*alpha*(rho + S) with S = 0 (pressureless dust)

    dt_K = first_term + second_term + third_term + fourth_term + matter_term
    # compute dt_K

    if energy_density is not None:
        W_floor = jnp.maximum(vars.conformal_factor, W_FLOOR_VALUE)
        physical_inverse_metric = W_floor**2 * inv_gamma
        stress_trace = jnp.einsum(
            'ij...,ij...->...', physical_inverse_metric, spatial_stress
        )
        dt_K = dt_K + 4.0 * jnp.pi * alpha * (
            energy_density + stress_trace
        )
        # +4 pi alpha (rho + S), with S = gamma^ij S_ij

    dissipation_term = ko_dissipation(vars.trace_K, params)
    # compute dissipation term

    return dt_K + dissipation_term


@jit
def evolve_trace_extrinsic_curvature(
    vars: BSSNVariables,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Evolve the vacuum trace of the extrinsic curvature."""

    return _evolve_trace_extrinsic_curvature(vars, params)


@jit
def evolve_trace_extrinsic_curvature_with_matter(
    vars: BSSNVariables,
    params: BSSNParameters,
    energy_density: jnp.ndarray,
    spatial_stress: jnp.ndarray,
) -> jnp.ndarray:
    """Evolve ``K`` with physical Eulerian matter sources."""

    return _evolve_trace_extrinsic_curvature(
        vars, params, energy_density, spatial_stress
    )


def _evolve_traceless_extrinsic_curvature(
    vars: BSSNVariables,
    params: BSSNParameters,
    momentum_density=None,
    spatial_stress=None,
) -> jnp.ndarray:
    """
    Evolve traceless extrinsic curvature A_ij.

    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Time derivative of traceless extrinsic curvature
    """

    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    metric_derivs = jnp.stack(
        [
            diff1_physical(gamma, d + 2, params)
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
    if spatial_stress is not None:
        W_floor = jnp.maximum(vars.conformal_factor, W_FLOOR_VALUE)
        third_term = (
            third_term
            - 8.0
            * jnp.pi
            * alpha
            * W_floor**2
            * spatial_stress
        )
    third_term = traceless_part(third_term, vars.conformal_metric, inv_gamma)
    # [alpha W**2 (R_ij - 8 pi S_ij) - W**2 D_i D_j alpha]^TF

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

    # A_ij is shape (3, 3, ni, nj, nk)

    if momentum_density is None:
        M, dMidj = compute_momentum_constraint_and_derivative(vars, params)
    else:
        M, dMidj = compute_momentum_constraint_and_derivative_with_matter(
            vars, params, momentum_density
        )
    # Momentum constraint and its product-rule spatial derivative

    kappa = params.kappa
    # constraint damping parameter

    DjMi = dMidj - jnp.einsum('kij...,k...->ij...', christoffel_second, M)
    DiMj = jnp.swapaxes(DjMi, 0, 1)
    # compute covariant derivatives of M_i
    seventh_term = kappa/2 * alpha * (DjMi + DiMj)
    # seventh term

    dissipation_term = ko_dissipation(vars.traceless_K, params)
    # compute dissipation term

    return dt_A + seventh_term + dissipation_term


@jit
def evolve_traceless_extrinsic_curvature(
    vars: BSSNVariables,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Evolve the vacuum conformal traceless extrinsic curvature."""

    return _evolve_traceless_extrinsic_curvature(vars, params)


@jit
def evolve_traceless_extrinsic_curvature_with_matter(
    vars: BSSNVariables,
    params: BSSNParameters,
    momentum_density: jnp.ndarray,
    spatial_stress: jnp.ndarray,
) -> jnp.ndarray:
    """Evolve ``A_tilde_ij`` with physical Eulerian matter sources."""

    return _evolve_traceless_extrinsic_curvature(
        vars, params, momentum_density, spatial_stress
    )
