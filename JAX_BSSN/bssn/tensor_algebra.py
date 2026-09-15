"""
Tensor algebra operations for numerical relativity.

This module provides tensor operations commonly used in numerical relativity,
including Christoffel symbols, Riemann tensor components, and other geometric
quantities. All operations are JIT-compiled with JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
import numpy as np

from JAX_BSSN.evolution.derivatives import diff1_field, compute_all_derivatives

# NOTE: FULLY TESTED AND FUNCTIONAL AS OF DEC 3RD 2025


@jit
def invert_3x3_metric(metric: jnp.ndarray) -> jnp.ndarray:
    """
    Invert a symmetric positive-definite 3x3 metric tensor field.
    
    Args:
        metric: Array with shape (3, 3, ni, nj, nk) containing metric components
        
    Returns:
        Inverse metric with same shape
    """
    g00 = metric[0, 0]
    g01 = metric[0, 1]
    g02 = metric[0, 2]
    g11 = metric[1, 1]
    g12 = metric[1, 2]
    g22 = metric[2, 2]

    # Cholesky factor metric = L L^T, written explicitly for the fixed
    # three-dimensional tensor axes. The conformal metric is SPD by contract.
    L00 = jnp.sqrt(g00)
    L10 = g01 / L00
    L20 = g02 / L00
    L11 = jnp.sqrt(g11 - L10 * L10)
    L21 = (g12 - L20 * L10) / L11
    L22 = jnp.sqrt(g22 - L20 * L20 - L21 * L21)

    # Invert the lower-triangular factor, then form L^{-T} L^{-1}.
    M00 = 1.0 / L00
    M10 = -L10 / (L00 * L11)
    M11 = 1.0 / L11
    M20 = (L10 * L21 - L20 * L11) / (L00 * L11 * L22)
    M21 = -L21 / (L11 * L22)
    M22 = 1.0 / L22

    inv00 = M00 * M00 + M10 * M10 + M20 * M20
    inv01 = M10 * M11 + M20 * M21
    inv02 = M20 * M22
    inv11 = M11 * M11 + M21 * M21
    inv12 = M21 * M22
    inv22 = M22 * M22

    inv_metric = jnp.stack(
        (
            jnp.stack((inv00, inv01, inv02), axis=0),
            jnp.stack((inv01, inv11, inv12), axis=0),
            jnp.stack((inv02, inv12, inv22), axis=0),
        ),
        axis=0,
    )

    return inv_metric


@jit
def determinant_3x3_metric(metric: jnp.ndarray) -> jnp.ndarray:
    """
    Compute determinant of 3x3 metric tensor field.
    
    Args:
        metric: Array with shape (3, 3, ni, nj, nk) containing metric components
        
    Returns:
        3D array containing determinant at each grid point
    """
    ni, nj, nk = metric.shape[2:]

    g_flatten = metric.reshape(3, 3, -1)
    # flatten the last three dimensions for vectorized determinant
    vectorized_det = jax.vmap(jnp.linalg.det, in_axes=2, out_axes=0)
    det_flat = vectorized_det(g_flatten)
    # vmap over the last dimension to compute determinant of each 3x3 matrix

    det = det_flat.reshape(ni, nj, nk)
    # Reshape back to original grid shape
    
    return det


@jit
def raise_index(tensor: jnp.ndarray, inverse_metric: jnp.ndarray, 
                index_position: int) -> jnp.ndarray:
    """
    Raise an index of a rank-2 tensor using the inverse metric.
    
    Args:
        tensor: Rank-2 tensor with shape (3, 3, ni, nj, nk)
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        index_position: Which index to raise (0 for first, 1 for second)
        
    Returns:
        Tensor with raised index
    """

    raised = jax.lax.cond(
                index_position == 0,
                lambda _: jnp.einsum('ik...,kj...->ij...', inverse_metric, tensor),
                lambda _: jnp.einsum('ik...,jk...->ij...', tensor, inverse_metric),
                operand=None)
                # T^i_j = g^ik T_kj or T_i^j = T_i^k g^jk
    
    return raised


@jit
def lower_index(tensor: jnp.ndarray, metric: jnp.ndarray, 
                index_position: int) -> jnp.ndarray:
    """
    Lower an index of a rank-2 tensor using the metric.
    
    Args:
        tensor: Rank-2 tensor with shape (3, 3, ni, nj, nk)
        metric: Metric with shape (3, 3, ni, nj, nk)
        index_position: Which index to lower (0 for first, 1 for second)
        
    Returns:
        Tensor with lowered index
    """
    
    lowered = jax.lax.cond(
                index_position == 0,
                lambda _: jnp.einsum('ik...,kj...->ij...', metric, tensor),
                lambda _: jnp.einsum('ik...,jk...->ij...', tensor, metric),
                operand=None)
                # T_ij = g_ik T^k_j or T_i_j = T_i^k g_kj
    return lowered


@jit
def christoffel_symbols_first_kind(metric_derivatives: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Christoffel symbols of the first kind.
    
    Γ_ijk = (1/2) * (∂g_ij/∂x^k + ∂g_ik/∂x^j - ∂g_jk/∂x^i)
    
    Args:
        metric_derivatives: Array with shape (3, 3, 3, ni, nj, nk)
                           where first index is derivative direction
        
    Returns:
        Christoffel symbols of first kind with shape (3, 3, 3, ni, nj, nk)
    """

    christoffel_1 = 0.5 * (
        jnp.einsum('kij...->ijk...', metric_derivatives) +
        jnp.einsum('jik...->ijk...', metric_derivatives) -
        jnp.einsum('ijk...->ijk...', metric_derivatives)
    )  # Vectorized computation using einsum

    return christoffel_1


@jit
def christoffel_symbols_second_kind(inverse_metric: jnp.ndarray,
                                   metric_derivatives: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Christoffel symbols of the second kind.
    
    Γ^i_jk = g^im * Γ_mjk
    
    Args:
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        metric_derivatives: Metric derivatives with shape (3, 3, 3, ni, nj, nk)
        
    Returns:
        Christoffel symbols of second kind with shape (3, 3, 3, ni, nj, nk)
    """
    # First compute first kind
    christoffel_1 = christoffel_symbols_first_kind(metric_derivatives)

    christoffel_2 = jnp.einsum('im...,mjk...->ijk...', inverse_metric, christoffel_1)
    # Vectorized computation using einsum
    
    return christoffel_2


@jit
def riemann_tensor(christoffel: jnp.ndarray, christoffel_derivatives: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Riemann curvature tensor.
    
    R^l_ijk = ∂Γ^l_ik/∂x^j - ∂Γ^l_ij/∂x^k + Γ^l_mj Γ^m_ik - Γ^l_mk Γ^m_ij
    
    Args:
        christoffel: Christoffel symbols with shape (3, 3, 3, ni, nj, nk)
        christoffel_derivatives: Derivatives of Christoffel symbols
        
    Returns:
        Riemann tensor with shape (3, 3, 3, 3, ni, nj, nk)
    """

    riemann = (
        jnp.einsum('jlik...->lijk...', christoffel_derivatives) -
        jnp.einsum('klij...->lijk...', christoffel_derivatives) +
        jnp.einsum('lmj...,mik...->lijk...', christoffel, christoffel) -
        jnp.einsum('lmk...,mij...->lijk...', christoffel, christoffel)
    )
    # Vectorized computation using einsum

    return riemann


@jit
def ricci_tensor(riemann: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Ricci tensor from the Riemann tensor.
    
    R_ij = R^k_ikj
    
    Args:
        riemann: Riemann tensor with shape (3, 3, 3, 3, ni, nj, nk)
        
    Returns:
        Ricci tensor with shape (3, 3, ni, nj, nk)
    """

    ricci = jnp.einsum('kikj...->ij...', riemann)
    # Vectorized computation using einsum
    
    return ricci


@jit
def ricci_scalar(ricci: jnp.ndarray, inverse_metric: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Ricci scalar from the Ricci tensor.
    
    R = g^ij R_ij
    
    Args:
        ricci: Ricci tensor with shape (3, 3, ni, nj, nk)
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        
    Returns:
        Ricci scalar with shape (ni, nj, nk)
    """

    scalar = trace_tensor(ricci, inverse_metric)
    # Vectorized computation using trace_tensor

    return scalar


@jit
def lie_derivative_metric(vector: jnp.ndarray, metric: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute Lie derivative of metric tensor with respect to vector field.
    
    £_v g_ij = v^k ∂g_ij/∂x^k + g_kj ∂v^k/∂x^i + g_ik ∂v^k/∂x^j
    
    Args:
        vector: Vector field with shape (3, ni, nj, nk)
        metric: Metric tensor with shape (3, 3, ni, nj, nk)
        dx: Grid spacing
        
    Returns:
        Lie derivative with shape (3, 3, ni, nj, nk)
    """

    # Compute metric derivatives: ∂g_ij/∂x^k
    metric_derivs = jnp.zeros((3, 3, 3) + metric.shape[2:])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                metric_derivs = metric_derivs.at[k, i, j].set(
                    diff1_field(metric[i, j], k, dx)
                )

    # Compute vector derivatives: ∂v^k/∂x^i
    vector_derivs = jnp.zeros((3, 3) + vector.shape[1:])
    for k in range(3):
        for i in range(3):
            vector_derivs = vector_derivs.at[i, k].set(
                diff1_field(vector[k], i, dx)
            )
    
    # Compute the three terms of the Lie derivative
    # £_v g_ij = v^k ∂g_ij/∂x^k + g_kj ∂v^k/∂x^i + g_ik ∂v^k/∂x^j
    term1 = jnp.einsum('k...,kij...->ij...', vector, metric_derivs)
    term2 = jnp.einsum('kj...,ik...->ij...', metric, vector_derivs)  # g_kj * ∂v^k/∂x^i  
    term3 = jnp.einsum('ik...,jk...->ij...', metric, vector_derivs)  # g_ik * ∂v^k/∂x^j

    lie_deriv = term1 + term2 + term3

    return lie_deriv


@jit
def lie_derivative_conformal_metric(vector: jnp.ndarray, metric: jnp.ndarray, 
                                   dx: float) -> jnp.ndarray:
    """
    Compute conformal Lie derivative with weight -2/3.
    
    £_v γ_ij = v^k ∂γ_ij/∂x^k + γ_kj ∂v^k/∂x^i + γ_ik ∂v^k/∂x^j - (2/3) γ_ij ∂v^k/∂x^k
    
    Args:
        vector: Vector field with shape (3, ni, nj, nk)
        metric: Conformal metric with shape (3, 3, ni, nj, nk)
        dx: Grid spacing
        
    Returns:
        Conformal Lie derivative with shape (3, 3, ni, nj, nk)
    """
    # Start with regular Lie derivative
    lie_deriv = lie_derivative_metric(vector, metric, dx)

    # Compute divergence: ∂v^k/∂x^k
    div_v = jnp.zeros(vector.shape[1:])
    for k in range(3):
        div_v += diff1_field(vector[k,...], k, dx)

    metric_div_v = jnp.einsum('ij..., ...->ij...', metric, div_v)
    lie_deriv = lie_deriv - (2.0/3.0) * metric_div_v

    return lie_deriv


@jit
def trace_tensor(tensor: jnp.ndarray, inverse_metric: jnp.ndarray) -> jnp.ndarray:
    """
    Compute trace of a rank-2 tensor.
    
    T = g^ij T_ij
    
    Args:
        tensor: Rank-2 tensor with shape (3, 3, ni, nj, nk)
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        
    Returns:
        Trace with shape (ni, nj, nk)
    """

    trace = jnp.einsum('ij...,ij...->...', inverse_metric, tensor)
    # Vectorized computation using einsum
    

    return trace


@jit
def traceless_part(tensor: jnp.ndarray, metric: jnp.ndarray, 
                   inverse_metric: jnp.ndarray) -> jnp.ndarray:
    """
    Extract traceless part of a rank-2 tensor.
    
    A_ij = T_ij - (1/3) γ_ij T
    
    Args:
        tensor: Input tensor with shape (3, 3, ni, nj, nk)
        metric: Metric tensor with shape (3, 3, ni, nj, nk)
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        
    Returns:
        Traceless tensor with shape (3, 3, ni, nj, nk)
    """
    trace = trace_tensor(tensor, inverse_metric)
    # compute trace

    traceless = tensor - (1.0/3.0) * metric * trace
    # Vectorized computation using broadcasting
    
    return traceless
