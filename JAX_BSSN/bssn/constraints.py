"""Raw mathematical constraint fields for the Cartesian BSSN system."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.evolution.derivatives import diff1_field, diff2_field
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE, compute_W2_ricci
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    determinant_3x3_metric,
    invert_3x3_metric,
    trace_tensor,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables, get_boundary_codes


@jit
def compute_momentum_constraint(vars: BSSNVariables,
                               params: BSSNParameters) -> jnp.ndarray:
    """
    Compute momentum constraint M_i.

    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Momentum constraint vector M_i
    """

    dx = params.dx
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dKdi = jnp.stack(
        [
            diff1_field(K, d, dx, *get_boundary_codes(params, d), mad_q=params.mad_q)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of K

    dWdi = jnp.stack(
        [
            diff1_field(W, d, dx, *get_boundary_codes(params, d), mad_q=params.mad_q)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of W

    A_i_up_j = jnp.einsum('jk...,ik...->ij...', inv_gamma, A_ij)
    # raise the second index in A_i^j

    dA_i_up_j_dk = jnp.stack(
        [
            diff1_field(
                A_i_up_j, d + 2, dx, *get_boundary_codes(params, d), mad_q=params.mad_q
            )
            for d in range(3)
        ],
        axis=0,
    )
    # derivatives of A_i^j

    dA_ij_dk = jnp.stack(
        [
            diff1_field(A_ij, d + 2, dx, *get_boundary_codes(params, d), mad_q=params.mad_q)
            for d in range(3)
        ],
        axis=0,
    )
    # derivatives of A_ij

    first_term = jnp.einsum('jij...->i...', dA_i_up_j_dk)
    # first term

    second_term = -0.5 * jnp.einsum('jk...,ijk...->i...', inv_gamma, dA_ij_dk)
    # second term

    third_term = -3 * jnp.einsum('ij...,j...->i...', A_i_up_j, dWdi) / W_floor
    # third term

    fourth_term = -2.0/3.0 * dKdi
    # fourth term

    M_i = first_term + second_term + third_term + fourth_term
    # compute momentum constraint

    return M_i


@jit
def compute_momentum_constraint_and_derivative(
    vars: BSSNVariables,
    params: BSSNParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``M_i`` and ``partial_l M_i`` without composed pure D1 stencils.

    The momentum-constraint derivative is expanded with the continuum product
    rule. Pure second partials use ``diff2_field`` while genuine mixed partials
    retain ordered applications of ``diff1_field``.
    """

    dx = params.dx
    A_ij = vars.traceless_K
    W = vars.conformal_factor
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
    K = vars.trace_K
    inv_gamma = invert_3x3_metric(vars.conformal_metric)
    B_i_up_j = jnp.einsum('jk...,ik...->ij...', inv_gamma, A_ij)
    B_over_W = B_i_up_j / W_floor

    M_i = compute_momentum_constraint(vars, params)

    dB = jnp.stack(
        [
            diff1_field(
                B_i_up_j,
                direction + 2,
                dx,
                *get_boundary_codes(params, direction), mad_q=params.mad_q,
            )
            for direction in range(3)
        ],
        axis=0,
    )
    dA = jnp.stack(
        [
            diff1_field(
                A_ij,
                direction + 2,
                dx,
                *get_boundary_codes(params, direction), mad_q=params.mad_q,
            )
            for direction in range(3)
        ],
        axis=0,
    )
    d_inv_gamma = jnp.stack(
        [
            diff1_field(
                inv_gamma,
                direction + 2,
                dx,
                *get_boundary_codes(params, direction), mad_q=params.mad_q,
            )
            for direction in range(3)
        ],
        axis=0,
    )
    dB_over_W = jnp.stack(
        [
            diff1_field(
                B_over_W,
                direction + 2,
                dx,
                *get_boundary_codes(params, direction), mad_q=params.mad_q,
            )
            for direction in range(3)
        ],
        axis=0,
    )
    dW = jnp.stack(
        [
            diff1_field(
                W,
                direction,
                dx,
                *get_boundary_codes(params, direction), mad_q=params.mad_q,
            )
            for direction in range(3)
        ],
        axis=0,
    )
    dK = jnp.stack(
        [
            diff1_field(
                K,
                direction,
                dx,
                *get_boundary_codes(params, direction), mad_q=params.mad_q,
            )
            for direction in range(3)
        ],
        axis=0,
    )

    d2B = jnp.zeros((3, 3) + B_i_up_j.shape, dtype=B_i_up_j.dtype)
    d2A = jnp.zeros((3, 3) + A_ij.shape, dtype=A_ij.dtype)
    d2W = jnp.zeros((3, 3) + W.shape, dtype=W.dtype)
    d2K = jnp.zeros((3, 3) + K.shape, dtype=K.dtype)
    for l in range(3):
        for inner in range(3):
            if l == inner:
                second_B = diff2_field(
                    B_i_up_j,
                    l + 2,
                    dx,
                    *get_boundary_codes(params, l), mad_q=params.mad_q,
                )
                second_A = diff2_field(
                    A_ij,
                    l + 2,
                    dx,
                    *get_boundary_codes(params, l), mad_q=params.mad_q,
                )
                second_W = diff2_field(
                    W,
                    l,
                    dx,
                    *get_boundary_codes(params, l), mad_q=params.mad_q,
                )
                second_K = diff2_field(
                    K,
                    l,
                    dx,
                    *get_boundary_codes(params, l), mad_q=params.mad_q,
                )
            else:
                second_B = diff1_field(
                    dB[inner],
                    l + 2,
                    dx,
                    *get_boundary_codes(params, l), mad_q=params.mad_q,
                )
                second_A = diff1_field(
                    dA[inner],
                    l + 2,
                    dx,
                    *get_boundary_codes(params, l), mad_q=params.mad_q,
                )
                second_W = diff1_field(
                    dW[inner],
                    l,
                    dx,
                    *get_boundary_codes(params, l), mad_q=params.mad_q,
                )
                second_K = diff1_field(
                    dK[inner],
                    l,
                    dx,
                    *get_boundary_codes(params, l), mad_q=params.mad_q,
                )
            d2B = d2B.at[l, inner].set(second_B)
            d2A = d2A.at[l, inner].set(second_A)
            d2W = d2W.at[l, inner].set(second_W)
            d2K = d2K.at[l, inner].set(second_K)

    dM = jnp.zeros((3, 3) + K.shape, dtype=K.dtype)
    for i in range(3):
        for l in range(3):
            divergence_derivative = jnp.zeros_like(K)
            conformal_factor_derivative = jnp.zeros_like(K)
            for j in range(3):
                divergence_derivative = divergence_derivative + d2B[l, j, i, j]
                conformal_factor_derivative = conformal_factor_derivative + (
                    dB_over_W[l, i, j] * dW[j]
                    + B_over_W[i, j] * d2W[l, j]
                )

            metric_derivative_term = jnp.einsum(
                'jk...,jk...->...', d_inv_gamma[l], dA[i]
            )
            metric_second_derivative_term = jnp.einsum(
                'jk...,jk...->...', inv_gamma, d2A[l, i]
            )
            dM_il = (
                divergence_derivative
                - 0.5 * metric_derivative_term
                - 0.5 * metric_second_derivative_term
                - 3.0 * conformal_factor_derivative
                - 2.0 / 3.0 * d2K[l, i]
            )
            dM = dM.at[i, l].set(dM_il)

    return M_i, dM




class ConstraintViolations(NamedTuple):
    """Container for constraint violation measures."""
    hamiltonian: jnp.ndarray      # Hamiltonian constraint violation
    momentum: jnp.ndarray         # Momentum constraint violation (3-vector)
    det_gamma: jnp.ndarray        # det(γ) = 1 violation
    trace_A: jnp.ndarray          # tr(A) = 0 violation
    gamma_condition: jnp.ndarray   # Gamma constraint violation


@jit
def compute_hamiltonian_constraint(vars: BSSNVariables,
                                  params: BSSNParameters) -> jnp.ndarray:
    """
    Compute Hamiltonian constraint violation.

    where R is the 3D Ricci scalar.

    Args:
        vars: BSSN variables
        params: Evolution parameters

    Returns:
        Hamiltonian constraint violation H
    """
    conformal_metric = vars.conformal_metric

    inv_metric = invert_3x3_metric(conformal_metric)

    W2_ricci = compute_W2_ricci(vars, params)
    ricci_scalar = trace_tensor(W2_ricci, inv_metric)
    # R = gamma_tilde^ij (W**2 R_ij), evaluated without inverse powers of W.

    K_squared = vars.trace_K**2

    A_squared = jnp.einsum('ik...,jl...,ij...,kl...->...', inv_metric, inv_metric,
                            vars.traceless_K, vars.traceless_K)


    # Hamiltonian constraint
    hamiltonian = ricci_scalar + 2/3 * K_squared - A_squared

    return hamiltonian


@jit
def compute_det_gamma_violation(vars: BSSNVariables) -> jnp.ndarray:
    """
    Compute violation of det(γ) = 1 condition.

    In BSSN, the conformal metric should satisfy det(γ) = 1.

    Args:
        vars: BSSN variables

    Returns:
        det(γ) - 1
    """
    det_gamma = determinant_3x3_metric(vars.conformal_metric)
    return det_gamma - 1.0


@jit
def compute_trace_A_violation(vars: BSSNVariables) -> jnp.ndarray:
    """
    Compute violation of tr(A) = 0 condition.

    The traceless extrinsic curvature should be traceless.

    Args:
        vars: BSSN variables

    Returns:
        tr(A) = γ^ij A_ij
    """
    inv_metric = invert_3x3_metric(vars.conformal_metric)
    trace_A = trace_tensor(vars.traceless_K, inv_metric)
    return trace_A


@jit
def compute_gamma_constraint(vars: BSSNVariables,
                            params: BSSNParameters) -> jnp.ndarray:
    """
    Compute Gamma constraint violation.

    The Gamma constraint relates the conformal connection to metric derivatives:
    Γ^i = γ^jk Γ^i_jk

    Args:
        vars: BSSN variables
        params: Evolution parameters

    Returns:
        Gamma constraint violation
    """

    metric_derivatives = jnp.stack(
        [
            diff1_field(
                vars.conformal_metric,
                direction + 2,
                params.dx,
                *get_boundary_codes(params, direction),
                mad_q=params.mad_q,
            )
            for direction in range(3)
        ],
        axis=0,
    )
    inverse_metric = invert_3x3_metric(vars.conformal_metric)
    christoffel = christoffel_symbols_second_kind(
        inverse_metric, metric_derivatives
    )
    contracted_christoffel = jnp.einsum(
        'jk...,ijk...->i...', inverse_metric, christoffel
    )
    return vars.conformal_connection - contracted_christoffel


@jit
def compute_all_constraints(vars: BSSNVariables,
                           params: BSSNParameters) -> ConstraintViolations:
    """
    Compute all constraint violations.

    Args:
        vars: BSSN variables
        params: Evolution parameters

    Returns:
        All constraint violations
    """
    hamiltonian = compute_hamiltonian_constraint(vars, params)
    momentum = compute_momentum_constraint(vars, params)
    det_gamma = compute_det_gamma_violation(vars)
    trace_A = compute_trace_A_violation(vars)
    gamma_condition = compute_gamma_constraint(vars, params)

    return ConstraintViolations(
        hamiltonian=hamiltonian,
        momentum=momentum,
        det_gamma=det_gamma,
        trace_A=trace_A,
        gamma_condition=gamma_condition
    )
