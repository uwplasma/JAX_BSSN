"""
BSSN (Baumgarte-Shapiro-Shibata-Nakamura) evolution equations.

This module implements the BSSN formulation for 3+1 numerical relativity.
The BSSN variables are:
- conformal metric γ_ij 
- conformal factor W (or φ)
- traceless extrinsic curvature A_ij
- trace of extrinsic curvature K
- conformal connection functions Γ^i

All evolution equations are JIT-compiled with JAX for performance.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, NamedTuple
import numpy as np

from JAX_BSSN.derivatives import (diff1_field, compute_all_derivatives, 
                        laplacian_3d, divergence_3d, diff6_field)
from JAX_BSSN.tensor_algebra import (invert_3x3_metric, determinant_3x3_metric,
                           christoffel_symbols_second_kind,
                           christoffel_symbols_first_kind,
                           ricci_tensor, ricci_scalar, trace_tensor,
                           traceless_part, lie_derivative_conformal_metric,
                           raise_index, lower_index)


# NOTE: FULLY TESTED AND FUNCTIONAL AS OF DEC 3RD 2025


class BSSNVariables(NamedTuple):
    """Container for BSSN evolution variables."""
    conformal_metric: jnp.ndarray      # γ_ij (3x3 symmetric)
    conformal_factor: jnp.ndarray      # W or φ
    traceless_K: jnp.ndarray          # A_ij (3x3 traceless)
    trace_K: jnp.ndarray              # K (scalar)
    conformal_connection: jnp.ndarray  # Γ^i (3-vector)
    lapse: jnp.ndarray                # α (scalar)
    shift: jnp.ndarray                # β^i (3-vector)


class BSSNParameters(NamedTuple):
    """Parameters for BSSN evolution."""
    eta: float = 2.0          # Damping parameter for Γ^i evolution
    kappa: float = 0.0        # Constraint damping parameter
    nu: float = 0.25          # Kreiss-Oliger dissipation coefficient
    f: float = 1.0            # Multiple of 1+log slicing
    g: float = 0.75           # Gamma driver shift parameter
    dx: float = 0.1           # Grid spacing
    dt: float = 0.001         # Time step


@jit
def pack_symmetric_3x3(tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Pack symmetric 3x3 tensor into 6-component array.
    Order: [00, 01, 02, 11, 12, 22]
    
    Args:
        tensor: Array with shape (3, 3, ni, nj, nk)
        
    Returns:
        Packed array with shape (6, ni, nj, nk)
    """
    shape = tensor.shape[2:]
    packed = jnp.zeros((6,) + shape)
    
    # Pack symmetric components
    packed = packed.at[0].set(tensor[0, 0])  # γ_xx
    packed = packed.at[1].set(tensor[0, 1])  # γ_xy  
    packed = packed.at[2].set(tensor[0, 2])  # γ_xz
    packed = packed.at[3].set(tensor[1, 1])  # γ_yy
    packed = packed.at[4].set(tensor[1, 2])  # γ_yz
    packed = packed.at[5].set(tensor[2, 2])  # γ_zz
    
    return packed


@jit
def unpack_symmetric_3x3(packed: jnp.ndarray) -> jnp.ndarray:
    """
    Unpack 6-component array into symmetric 3x3 tensor.
    
    Args:
        packed: Array with shape (6, ni, nj, nk)
        
    Returns:
        Tensor with shape (3, 3, ni, nj, nk)
    """
    shape = packed.shape[1:]
    tensor = jnp.zeros((3, 3) + shape)
    
    # Unpack symmetric components
    tensor = tensor.at[0, 0].set(packed[0])  # γ_xx
    tensor = tensor.at[0, 1].set(packed[1])  # γ_xy
    tensor = tensor.at[1, 0].set(packed[1])  # γ_xy  
    tensor = tensor.at[0, 2].set(packed[2])  # γ_xz
    tensor = tensor.at[2, 0].set(packed[2])  # γ_xz
    tensor = tensor.at[1, 1].set(packed[3])  # γ_yy
    tensor = tensor.at[1, 2].set(packed[4])  # γ_yz
    tensor = tensor.at[2, 1].set(packed[4])  # γ_yz
    tensor = tensor.at[2, 2].set(packed[5])  # γ_zz
    
    return tensor


@jit
def compute_physical_metric(conformal_metric: jnp.ndarray, 
                           conformal_factor: jnp.ndarray) -> jnp.ndarray:
    """
    Compute physical metric from conformal metric and conformal factor.
    
    g_ij = W^-2 * γ_ij  (W-formulation)
    
    Args:
        conformal_metric: Conformal metric γ_ij with shape (3, 3, ni, nj, nk)
        conformal_factor: Conformal factor W with shape (ni, nj, nk)
        
    Returns:
        Physical metric with shape (3, 3, ni, nj, nk)
    """

    physical_metric = conformal_metric / jnp.power(conformal_factor, 2)
    # compute physical metric by scaling conformal metric with W^-2
    
    return physical_metric

@jit
def compute_ricci(vars: BSSNVariables,
                             params: BSSNParameters) -> jnp.ndarray:
    """
    Compute full Ricci tensor including conformal factor contributions.
    
    R_ij = R̃_ij + R_ij^φ
    
    where R_ij^φ contains terms from the conformal factor.
    
    Args:
        conformal_ricci: Conformal Ricci tensor R̃_ij
        conformal_metric: γ_ij  
        conformal_factor: W
        params: BSSN parameters
    """

    dx = params.dx
    alpha = vars.lapse
    conformal_metric = vars.conformal_metric
    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    conformal_connection = vars.conformal_connection
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    shape = conformal_metric.shape[2:]

    metric_derivs = jnp.stack( [diff1_field(conformal_metric, d+2, dx) for d in range(3)], axis=0) 
    # shape (3, 3, 3, ni, nj, nk)

    # Compute inverse conformal metric
    inv_metric = invert_3x3_metric(conformal_metric)

    # Compute Christoffel symbols
    christoffel_first  = christoffel_symbols_first_kind(metric_derivs)
    christoffel_second = christoffel_symbols_second_kind(inv_metric, metric_derivs)

    mixed_derivatives = jnp.zeros((3, 3, 3, 3) + shape)
    for m in range(3):
        for n in range(3):
            mixed_derivatives = mixed_derivatives.at[m, n, ...].set(
            diff1_field( metric_derivs[m, ...], n+2, dx))
        
    term_1 = -0.5 * jnp.einsum('mn...,mnij...->ij...', inv_metric, mixed_derivatives)
    # compute first term of Ricci tensor

    connection_derivs = jnp.stack( [diff1_field(conformal_connection, d+1, dx) for d in range(3)], axis=0)
    # shape (3, 3, ni, nj, nk)

    term_2 = (jnp.einsum('mi...,jm...->ij...', conformal_metric, connection_derivs) + jnp.einsum('mj...,im...->ij...', conformal_metric, connection_derivs)) / 2.0
    # compute second term of Ricci tensor

    term_3 = ( jnp.einsum('m...,ijm...->ij...', conformal_connection, christoffel_first) + jnp.einsum('m...,jim...->ij...', conformal_connection, christoffel_first) ) / 2.0
    # compute third term of Ricci tensor

    term_4 = jnp.einsum('mn...,kmi...,jkn...->ij...', inv_metric, christoffel_second, christoffel_first) + \
    jnp.einsum('mn...,kmj...,ikn...->ij...', inv_metric, christoffel_second, christoffel_first)
    # compute fourth term of Ricci tensor

    term_5 = jnp.einsum('mn...,kim...,kjn...->ij...', inv_metric, christoffel_second, christoffel_first)
    # compute fifth term of Ricci tensor

    conformal_ricci = term_1 + term_2 + term_3 + term_4 + term_5
    # conformal Ricci tensor without conformal factor terms

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)

    dWdij = jnp.zeros((3,3) + W.shape)
    for i in range(3):
        for j in range(3):
            dWdij = dWdij.at[i,j].set( diff1_field( dWdi[i], j, dx) )
    # second derivatives of W


    metric_derivs = jnp.stack( [diff1_field(conformal_metric, d+2, dx) for d in range(3)], axis=0) 
    # shape (3, 3, 3, ni, nj, nk)
    christoffel_second = christoffel_symbols_second_kind(inv_conformal_metric, metric_derivs)
    # Compute Christoffel symbols
    DiDj_W = dWdij  - jnp.einsum('kij...,k...->ij...', christoffel_second, dWdi)


    first_term = DiDj_W / W
    # first term

    second_term = jnp.einsum('ij...,mn...,nm...->ij...', conformal_metric, inv_conformal_metric, DiDj_W) / W
    # second term

    third_term = -2 * jnp.einsum('ij...,mn...,m...,n...->ij...', conformal_metric, inv_conformal_metric, dWdi, dWdi) / jnp.power(W, 2)
    # third term

    R_ij_W = first_term + second_term + third_term
    # conformal factor contribution to Ricci tensor

    R_ij = conformal_ricci + R_ij_W
    # full Ricci tensor

    return R_ij


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
    # Note: this is only valid in the harmonic gauge where shift is 0
    
    # First term: -2α A_ij
    dt_gamma = -2.0 * vars.lapse * vars.traceless_K

    # Kreiss-Oliger dissipation can be added here if desired
    dgamma_dx1 = diff6_field(vars.conformal_metric, 2, params.dx)
    dgamma_dx2 = diff6_field(vars.conformal_metric, 3, params.dx)
    dgamma_dx3 = diff6_field(vars.conformal_metric, 4, params.dx)
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
    
    # Note: this is only valid in the harmonic gauge where shift is 0
    
    # First term: (1/3) α W K
    dt_W = (1.0/3.0) * vars.lapse * vars.conformal_factor * vars.trace_K

    dW_dx1 = diff6_field(vars.conformal_factor, 0, params.dx)
    dW_dx2 = diff6_field(vars.conformal_factor, 1, params.dx)
    dW_dx3 = diff6_field(vars.conformal_factor, 2, params.dx)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dW_dx1 + dW_dx2 + dW_dx3)
    # compute dissipation term

    
    return dt_W + dissipation_term

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

    # Note: this is only valid in the harmonic gauge where shift is 0

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)


    dalphadi = jnp.stack( [diff1_field(alpha, d, dx) for d in range(3)], axis=0)
    dalphadij = jnp.zeros((3,3) + alpha.shape)
    for i in range(3):
        for j in range(3):
            dalphadij = dalphadij.at[i,j].set( diff1_field( dalphadi[i], j, dx) )
    # second derivatives of alpha

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    metric_derivs = jnp.stack( [diff1_field(gamma, d+2, dx) for d in range(3)], axis=0) 
    # shape (3, 3, 3, ni, nj, nk)
    christoffel_second = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
    # Compute Christoffel symbols

    DiDj_alpha = dalphadij  - jnp.einsum('kij...,k...->ij...', christoffel_second, dalphadi)
    DiDj_alpha = DiDj_alpha + 1/W * jnp.einsum('i...,j...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha + 1/W * jnp.einsum('j...,i...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha - 1/W * jnp.einsum('ij...,mn...,m...,n...->ij...', gamma, inv_gamma, dWdi, dalphadi)
    # full covariant second derivative of alpha

    first_term = -1 * W**2 * jnp.einsum('ij...,ij...->...', inv_gamma, DiDj_alpha)
    # first term

    second_term = alpha * jnp.einsum('ij...,kl...,ik...,jl...->...', inv_gamma, inv_gamma, A_ij, A_ij)
    # second term

    third_term = alpha * K**2 / 3.0
    # third term

    dt_K = first_term + second_term + third_term
    # compute dt_K

    dK_dx1 = diff6_field(vars.trace_K, 0, params.dx)
    dK_dx2 = diff6_field(vars.trace_K, 1, params.dx)
    dK_dx3 = diff6_field(vars.trace_K, 2, params.dx)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dK_dx1 + dK_dx2 + dK_dx3)
    # compute dissipation term
    
    return dt_K + dissipation_term


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
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dKdi = jnp.stack( [diff1_field(K, d, dx) for d in range(3)], axis=0)
    # first derivatives of K

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    dA_ij_dk = jnp.stack( [diff1_field(A_ij, d+2, dx) for d in range(3)], axis=0)
    # derivatives of A_ij

    first_term = jnp.einsum('jl...,lij...->i...', inv_gamma, dA_ij_dk)
    # first term

    second_term = -0.5 * jnp.einsum('jk...,ijk...->i...', inv_gamma, dA_ij_dk)
    # second term

    third_term = -3 * jnp.einsum('kj...,k...,ij...->i...', inv_gamma, dWdi, A_ij) / W
    # third term

    fourth_term = -2.0/3.0 * dKdi
    # fourth term

    M_i = first_term + second_term + third_term + fourth_term
    # compute momentum constraint

    return M_i


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

    # Note this is only valid in holonomic gauge (shift=0)

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)


    first_term = alpha * K * A_ij
    # first term

    second_term = -2 * alpha * jnp.einsum('ik...,kl...,lj...->ij...', A_ij, inv_gamma, A_ij)
    # second term

    dalphadi = jnp.stack( [diff1_field(alpha, d, dx) for d in range(3)], axis=0)
    dalphadij = jnp.zeros((3,3) + alpha.shape)
    for i in range(3):
        for j in range(3):
            dalphadij = dalphadij.at[i,j].set( diff1_field( dalphadi[i], j, dx) )
    # second derivatives of alpha

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    metric_derivs = jnp.stack( [diff1_field(gamma, d+2, dx) for d in range(3)], axis=0) 
    # shape (3, 3, 3, ni, nj, nk)
    christoffel_second = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
    # Compute Christoffel symbols

    DiDj_alpha = dalphadij  - jnp.einsum('kij...,k...->ij...', christoffel_second, dalphadi)
    DiDj_alpha = DiDj_alpha + 1/W * jnp.einsum('i...,j...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha + 1/W * jnp.einsum('j...,i...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha - 1/W * jnp.einsum('ij...,mn...,m...,n...->ij...', gamma, inv_gamma, dWdi, dalphadi)
    # full covariant second derivative of alpha
        
    # Compute full Ricci tensor
    ricci = compute_ricci(vars, params)


    third_term = jnp.power(W, 2) * (alpha * ricci - DiDj_alpha)
    third_term = traceless_part(third_term, vars.conformal_metric, inv_gamma)
    # third term

    M = compute_momentum_constraint(vars, params)
    # Momentum constraint term

    kappa = params.kappa
    # constraint damping parameter

    dMidj = jnp.zeros((3,) + M.shape)

    for i in range(3):
        for j in range(3):
            dMidj = dMidj.at[i,j].set( diff1_field( M[i,...], j, dx) ) 
        
    DjMi = dMidj - jnp.einsum('kij...,k...->ij...', christoffel_second, M)
    DiMj = jnp.swapaxes(DjMi, 0, 1)
    # compute covariant derivatives of M_i

    fourth_term = kappa/2 * alpha * (DjMi + DiMj)
    # fourth term

    dt_A = first_term + second_term + third_term #+ fourth_term
    # compute dt_A

    dA_dx1 = diff6_field(vars.traceless_K, 2, params.dx)
    dA_dx2 = diff6_field(vars.traceless_K, 3, params.dx)
    dA_dx3 = diff6_field(vars.traceless_K, 4, params.dx)
    # A_ij is shape (3, 3, ni, nj, nk)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dA_dx1 + dA_dx2 + dA_dx3)
    # compute dissipation term

    return dt_A + dissipation_term


@jit
def evolve_conformal_connection(vars: BSSNVariables,
                               params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal connection functions Γ̃^i.

    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of conformal connection
    """

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    dalphadi = jnp.stack( [diff1_field(alpha, d, dx) for d in range(3)], axis=0)
    # first derivatives of alpha

    dKdi = jnp.stack( [diff1_field(K, d, dx) for d in range(3)], axis=0)
    # first derivatives of K

    first_term = -4/3 * alpha * jnp.einsum('ij...,j...->i...', inv_gamma, dKdi)
    # first term


    metric_derivs = jnp.stack( [diff1_field(gamma, d+2, dx) for d in range(3)], axis=0) 
    # shape (3, 3, 3, ni, nj, nk)
    christoffel_second = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
    # compute Christoffel symbols of the second kind

    A_ij_raised = jnp.einsum('ik...,jl...,kl...->ij...', inv_gamma, inv_gamma, A_ij)
    # raise indices of A_ij
    second_term = 2 * alpha * jnp.einsum('ijk...,jk...->i...', christoffel_second, A_ij_raised)
    # second term

    third_term = -6 * alpha / W * jnp.einsum('ij...,j...->i...', A_ij_raised, dWdi)
    # third term

    fourth_term = -2 * jnp.einsum('ij...,j...->i...', A_ij_raised, dalphadi)
    # fourth term

    dt_Gamma = first_term + second_term + third_term + fourth_term
    # compute dt_Gamma

    dGamma_dx1 = diff6_field(vars.conformal_connection, 1, params.dx)
    dGamma_dx2 = diff6_field(vars.conformal_connection, 2, params.dx)
    dGamma_dx3 = diff6_field(vars.conformal_connection, 3, params.dx)
    # Gamma is shape (3, ni, nj, nk)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dGamma_dx1 + dGamma_dx2 + dGamma_dx3)
    # compute dissipation term

    return dt_Gamma + dissipation_term


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
    dx = params.dx
    f = params.f
    
    # harmonic slicing evolution
    dt_alpha = -f * jnp.power(vars.lapse, 2) * vars.trace_K

    dalpha_dx1 = diff6_field(vars.lapse, 0, params.dx)
    dalpha_dx2 = diff6_field(vars.lapse, 1, params.dx)
    dalpha_dx3 = diff6_field(vars.lapse, 2, params.dx)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dalpha_dx1 + dalpha_dx2 + dalpha_dx3)
    # compute dissipation term
    
    return dt_alpha + dissipation_term


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
    # in Harmonic gauge with 0 shift, the shift does not evolve

    dt_beta = 0.0 * vars.conformal_connection  # Placeholder for no evolution
    
    return dt_beta
