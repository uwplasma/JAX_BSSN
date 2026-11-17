"""Geometric quantities shared by the Cartesian BSSN equations."""

import jax.numpy as jnp

from JAX_BSSN.evolution.spatial_derivatives import (
    diff1_physical, diff2_physical,
)
from jax import jit

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_first_kind,
    christoffel_symbols_second_kind,
    invert_3x3_metric,
)


# Keep inverse powers of W finite without clamping the evolved conformal factor.
W_FLOOR_VALUE = 1.0e-12

@jit
def compute_W2_ricci(vars: BSSNVariables,
                     params: BSSNParameters) -> jnp.ndarray:
    """Compute the denominator-free scaled Ricci tensor ``W**2 R_ij``."""

    conformal_metric = vars.conformal_metric
    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    conformal_connection = vars.conformal_connection
    W = vars.conformal_factor

    metric_derivs = jnp.stack(
        [
            diff1_physical(conformal_metric, d + 2, params)
            for d in range(3)
        ],
        axis=0,
    )
    # shape (3, 3, 3, ni, nj, nk)

    # Compute Christoffel symbols
    christoffel_first  = christoffel_symbols_first_kind(metric_derivs)
    christoffel_second = christoffel_symbols_second_kind(inv_conformal_metric, metric_derivs)

    # Contract each second derivative as it is formed without materializing
    # the full tensor. Pure derivatives use a dedicated D2 stencil; only
    # derivatives across distinct axes compose the centered D1 operator.
    term_1 = jnp.zeros_like(conformal_metric)
    for m in range(3):
        for n in range(3):
            if m == n:
                metric_second_derivative = diff2_physical(conformal_metric, n + 2, params)
            else:
                metric_second_derivative = diff1_physical(metric_derivs[m, ...], n + 2, params)
            term_1 = term_1 - 0.5 * (
                inv_conformal_metric[m, n] * metric_second_derivative
            )
    # compute first term of Ricci tensor

    connection_derivs = jnp.stack(
        [
            diff1_physical(conformal_connection, d + 1, params)
            for d in range(3)
        ],
        axis=0,
    )
    # shape (3, 3, ni, nj, nk)

    term_2 = (jnp.einsum('mi...,jm...->ij...', conformal_metric, connection_derivs) + jnp.einsum('mj...,im...->ij...', conformal_metric, connection_derivs)) / 2.0
    # compute second term of Ricci tensor

    term_3 = ( jnp.einsum('m...,ijm...->ij...', conformal_connection, christoffel_first) + jnp.einsum('m...,jim...->ij...', conformal_connection, christoffel_first) ) / 2.0
    # compute third term of Ricci tensor

    term_4 = jnp.einsum('mn...,kmi...,jkn...->ij...', inv_conformal_metric, christoffel_second, christoffel_first) + \
    jnp.einsum('mn...,kmj...,ikn...->ij...', inv_conformal_metric, christoffel_second, christoffel_first)
    # compute fourth term of Ricci tensor

    term_5 = jnp.einsum('mn...,kim...,kjn...->ij...', inv_conformal_metric, christoffel_second, christoffel_first)
    # compute fifth term of Ricci tensor

    conformal_ricci = term_1 + term_2 + term_3 + term_4 + term_5
    # conformal Ricci tensor without conformal factor terms

    dWdi = jnp.stack(
        [
            diff1_physical(W, d, params)
            for d in range(3)
        ],
        axis=0,
    )

    dWdij = jnp.zeros((3, 3) + W.shape, dtype=W.dtype)
    for i in range(3):
        for j in range(3):
            if i == j:
                second_derivative = diff2_physical(W, j, params)
            else:
                second_derivative = diff1_physical(dWdi[i], j, params)
            dWdij = dWdij.at[i, j].set(second_derivative)
    # second derivatives of W


    DiDj_W = dWdij  - jnp.einsum('kij...,k...->ij...', christoffel_second, dWdi)

    conformal_laplacian_W = jnp.einsum(
        'mn...,mn...->...', inv_conformal_metric, DiDj_W
    )
    gradient_W_squared = jnp.einsum(
        'mn...,m...,n...->...', inv_conformal_metric, dWdi, dWdi
    )

    W2_R_ij_W = W * (
        DiDj_W + conformal_metric * conformal_laplacian_W
    ) - 2.0 * conformal_metric * gradient_W_squared
    # Compute W**2 R^W_ij directly so no inverse power of W is formed.

    W2_R_ij = W**2 * conformal_ricci + W2_R_ij_W

    return W2_R_ij


@jit
def compute_W2_covariant_lapse_hessian(
    vars: BSSNVariables,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Compute the denominator-free tensor ``W**2 D_i D_j alpha``."""

    alpha = vars.lapse
    W = vars.conformal_factor
    conformal_metric = vars.conformal_metric
    inv_conformal_metric = invert_3x3_metric(conformal_metric)

    dalphadi = jnp.stack(
        [
            diff1_physical(alpha, d, params)
            for d in range(3)
        ],
        axis=0,
    )
    dalphadij = jnp.zeros((3, 3) + alpha.shape, dtype=alpha.dtype)
    for i in range(3):
        for j in range(3):
            if i == j:
                second_derivative = diff2_physical(alpha, j, params)
            else:
                second_derivative = diff1_physical(dalphadi[i], j, params)
            dalphadij = dalphadij.at[i, j].set(second_derivative)

    dWdi = jnp.stack(
        [
            diff1_physical(W, d, params)
            for d in range(3)
        ],
        axis=0,
    )

    metric_derivs = jnp.stack(
        [
            diff1_physical(conformal_metric, d + 2, params)
            for d in range(3)
        ],
        axis=0,
    )
    christoffel_second = christoffel_symbols_second_kind(
        inv_conformal_metric, metric_derivs
    )

    conformal_hessian_alpha = dalphadij - jnp.einsum(
        'kij...,k...->ij...', christoffel_second, dalphadi
    )
    gradient_terms = (
        jnp.einsum('i...,j...->ij...', dWdi, dalphadi)
        + jnp.einsum('j...,i...->ij...', dWdi, dalphadi)
        - jnp.einsum(
            'ij...,mn...,m...,n...->ij...',
            conformal_metric,
            inv_conformal_metric,
            dWdi,
            dalphadi,
        )
    )

    W2_DiDj_alpha = W**2 * conformal_hessian_alpha + W * gradient_terms

    return W2_DiDj_alpha
