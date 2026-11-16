"""Raw mathematical constraint fields for the Cartesian BSSN system."""

from typing import NamedTuple

import jax.numpy as jnp

from JAX_BSSN.evolution.spatial_derivatives import (
    diff1_physical, diff2_physical,
)
from jax import jit

from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE, compute_W2_ricci
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    determinant_3x3_metric,
    invert_3x3_metric,
    trace_tensor,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables


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

    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dKdi = jnp.stack(
        [
            diff1_physical(K, d, params)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of K

    dWdi = jnp.stack(
        [
            diff1_physical(W, d, params)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of W

    A_i_up_j = jnp.einsum('jk...,ik...->ij...', inv_gamma, A_ij)
    # raise the second index in A_i^j

    dA_i_up_j_dk = jnp.stack(
        [
            diff1_physical(A_i_up_j, d + 2, params)
            for d in range(3)
        ],
        axis=0,
    )
    # derivatives of A_i^j

    dA_ij_dk = jnp.stack(
        [
            diff1_physical(A_ij, d + 2, params)
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
            diff1_physical(B_i_up_j, direction + 2, params)
            for direction in range(3)
        ],
        axis=0,
    )
    dA = jnp.stack(
        [
            diff1_physical(A_ij, direction + 2, params)
            for direction in range(3)
        ],
        axis=0,
    )
    d_inv_gamma = jnp.stack(
        [
            diff1_physical(inv_gamma, direction + 2, params)
            for direction in range(3)
        ],
        axis=0,
    )
    dB_over_W = jnp.stack(
        [
            diff1_physical(B_over_W, direction + 2, params)
            for direction in range(3)
        ],
        axis=0,
    )
    dW = jnp.stack(
        [
            diff1_physical(W, direction, params)
            for direction in range(3)
        ],
        axis=0,
    )
    dK = jnp.stack(
        [
            diff1_physical(K, direction, params)
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
                second_B = diff2_physical(B_i_up_j, l + 2, params)
                second_A = diff2_physical(A_ij, l + 2, params)
                second_W = diff2_physical(W, l, params)
                second_K = diff2_physical(K, l, params)
            else:
                second_B = diff1_physical(dB[inner], l + 2, params)
                second_A = diff1_physical(dA[inner], l + 2, params)
                second_W = diff1_physical(dW[inner], l, params)
                second_K = diff1_physical(dK[inner], l, params)
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


@jit
def compute_momentum_constraint_with_matter(
    vars: BSSNVariables,
    params: BSSNParameters,
    momentum_density: jnp.ndarray,
) -> jnp.ndarray:
    """Return the momentum constraint including the physical source ``S_i``.

    ``momentum_density`` is the covariant Eulerian momentum density on the
    Cartesian grid, with shape ``(3, nx, ny, nz)``.
    """

    geometric_momentum = compute_momentum_constraint(vars, params)
    return geometric_momentum - 8.0 * jnp.pi * momentum_density


@jit
def compute_momentum_constraint_and_derivative_with_matter(
    vars: BSSNVariables,
    params: BSSNParameters,
    momentum_density: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return matter-coupled ``M_i`` and its coordinate derivative.

    The derivative layout is ``partial_l M_i -> (i, l, nx, ny, nz)``, matching
    :func:`compute_momentum_constraint_and_derivative`.
    """

    geometric_momentum, geometric_derivative = (
        compute_momentum_constraint_and_derivative(vars, params)
    )
    momentum_derivative = jnp.stack(
        [
            diff1_physical(momentum_density, direction + 1, params)
            for direction in range(3)
        ],
        axis=1,
    )

    momentum = geometric_momentum - 8.0 * jnp.pi * momentum_density
    momentum_derivative = (
        geometric_derivative - 8.0 * jnp.pi * momentum_derivative
    )
    return momentum, momentum_derivative


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
    hamiltonian = ricci_scalar + 2/3 * K_squared - A_squared - 16.0 * jnp.pi * params.rho

    return hamiltonian


@jit
def compute_hamiltonian_constraint_with_matter(
    vars: BSSNVariables,
    params: BSSNParameters,
    energy_density: jnp.ndarray,
) -> jnp.ndarray:
    """Return the Hamiltonian constraint including physical energy density."""

    geometric_hamiltonian = compute_hamiltonian_constraint(vars, params)
    return geometric_hamiltonian - 16.0 * jnp.pi * energy_density


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
            diff1_physical(vars.conformal_metric, direction + 2, params)
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


@jit
def compute_all_constraints_with_matter(
    vars: BSSNVariables,
    params: BSSNParameters,
    energy_density: jnp.ndarray,
    momentum_density: jnp.ndarray,
) -> ConstraintViolations:
    """Compute the BSSN constraints with physical matter sources."""

    hamiltonian = compute_hamiltonian_constraint_with_matter(
        vars, params, energy_density
    )
    momentum = compute_momentum_constraint_with_matter(
        vars, params, momentum_density
    )
    det_gamma = compute_det_gamma_violation(vars)
    trace_A = compute_trace_A_violation(vars)
    gamma_condition = compute_gamma_constraint(vars, params)

    return ConstraintViolations(
        hamiltonian=hamiltonian,
        momentum=momentum,
        det_gamma=det_gamma,
        trace_A=trace_A,
        gamma_condition=gamma_condition,
    )
