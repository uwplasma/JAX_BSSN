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


# Keep inverse powers of W finite without clamping the evolved conformal factor.
W_FLOOR_VALUE = 1.0e-12


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
    rho: jnp.ndarray = 0.0            # Energy density ρ = n_a n_b T^{ab} (defaults to 0)
    S_ij: jnp.ndarray = jnp.zeros((3, 3, 1, 1, 1))           # Stress tensor S_ij (3x3 symmetric)
    momentum_density: jnp.ndarray = jnp.zeros((3, 1, 1, 1))  # Momentum density J^i


class BSSNParameters(NamedTuple):
    """Parameters for BSSN evolution."""
    eta: float = 2.0          # Damping parameter for Γ^i evolution
    kappa: float = 0.0        # Constraint damping parameter
    nu: float = 0.25          # Kreiss-Oliger dissipation coefficient
    g: float = 0.75           # Gamma driver shift parameter
    dx: float = 0.1           # Grid spacing
    dt: float = 0.001         # Time step
    zero_shift: int = 0       # If 1, hold the shift fixed during RK stages
    gauge: int = 0            # 0 = harmonic slicing, 1 = 1+log slicing
    xl_bc: int = 0            # x-left boundary code: 0 periodic, 1 super-Gaussian
    xr_bc: int = 0            # x-right boundary code: 0 periodic, 1 super-Gaussian
    yl_bc: int = 0            # y-left boundary code: 0 periodic, 1 super-Gaussian
    yr_bc: int = 0            # y-right boundary code: 0 periodic, 1 super-Gaussian
    zl_bc: int = 0            # z-left boundary code: 0 periodic, 1 super-Gaussian
    zr_bc: int = 0            # z-right boundary code: 0 periodic, 1 super-Gaussian
    bc_width: float = 8.0     # Super-Gaussian layer width in grid cells
    bc_order: float = 4.0     # Super-Gaussian exponent
    bc_strength: float = 1.0  # Boundary blend strength


@jit
def compute_shift_derivatives(shift: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute spatial derivatives of the shift vector.

    The returned tensor is indexed as d_beta[i, j] = partial_j beta^i.
    Keeping the component and derivative axes separate is important for the
    weighted Lie derivative terms in the nonzero-shift BSSN equations.
    """

    d_beta = jnp.stack(
        [
            jnp.stack(
                [diff1_field(shift[i, ...], j, dx) for j in range(3)],
                axis=0,
            )
            for i in range(3)
        ],
        axis=0,
    )

    return d_beta


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

    W = jnp.maximum(conformal_factor, W_FLOOR_VALUE)

    physical_metric = conformal_metric / jnp.power(W, 2)
    # compute physical metric by scaling conformal metric with W^-2
    
    return physical_metric


@jit
def compute_em_sources(conformal_metric: jnp.ndarray,
                        conformal_factor: jnp.ndarray,
                        E_flat: jnp.ndarray,
                        B_flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute electromagnetic energy density and stress tensor."""
    physical_metric = compute_physical_metric(conformal_metric, conformal_factor)
    inv_physical_metric = invert_3x3_metric(physical_metric)

    E_raised = jnp.einsum('ij...,j...->i...', inv_physical_metric, E_flat)
    B_raised = jnp.einsum('ij...,j...->i...', inv_physical_metric, B_flat)

    E_sq = jnp.einsum('i...,i...->...', E_raised, E_flat)
    B_sq = jnp.einsum('i...,i...->...', B_raised, B_flat)

    rho = (E_sq + B_sq) / (8.0 * jnp.pi)

    stress_part = -jnp.einsum('i...,j...->ij...', E_flat, E_flat) - jnp.einsum('i...,j...->ij...', B_flat, B_flat)
    trace_part = 0.5 * physical_metric * (E_sq + B_sq)
    S_ij = (stress_part + trace_part) / (4.0 * jnp.pi)

    return rho, S_ij


@jit
def compute_em_momentum_density(conformal_metric: jnp.ndarray,
                                 conformal_factor: jnp.ndarray,
                                 E_flat: jnp.ndarray,
                                 B_flat: jnp.ndarray) -> jnp.ndarray:
    """Compute electromagnetic momentum density J^i (Poynting vector)."""
    physical_metric = compute_physical_metric(conformal_metric, conformal_factor)
    inv_physical_metric = invert_3x3_metric(physical_metric)

    J_flat = jnp.cross(E_flat, B_flat, axis=0) / (4.0 * jnp.pi)
    J_raised = jnp.einsum('ij...,j...->i...', inv_physical_metric, J_flat)

    return J_raised

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
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
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


    first_term = DiDj_W / W_floor
    # first term

    second_term = jnp.einsum('ij...,mn...,nm...->ij...', conformal_metric, inv_conformal_metric, DiDj_W) / W_floor
    # second term

    third_term = -2 * jnp.einsum('ij...,mn...,m...,n...->ij...', conformal_metric, inv_conformal_metric, dWdi, dWdi) / jnp.power(W_floor, 2)
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
    grad_gamma = jnp.stack( [diff1_field(vars.conformal_metric, d+2, params.dx) for d in range(3)], axis=0)
    # compute the gradient of the conformal metric

    shift = vars.shift
    # unpack the shift vector

    grad_shift = compute_shift_derivatives(shift, params.dx)
    # grad_shift[i, j] = partial_j beta^i

    first_term = jnp.einsum("m...,mij...->ij...", shift, grad_gamma)
    # compute the advection term due to shift

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
    
    shift = vars.shift
    # unpack the shift vector

    grad_W = jnp.stack( [diff1_field(vars.conformal_factor, d, params.dx) for d in range(3)], axis=0)
    # compute the gradient of the conformal factor

    # First term: advection due to shift: beta^i ∂_i W
    first_term = jnp.einsum('i...,i...->...', shift, grad_W)
    # compute the advection term due to shift

    # Second term: (1/3) α W K
    second_term = (1.0/3.0) * vars.lapse * vars.conformal_factor * vars.trace_K

    grad_shift = compute_shift_derivatives(shift, params.dx)
    # grad_shift[i, j] = partial_j beta^i

    div_shift = jnp.einsum("ii...->...", grad_shift)
    # compute the divergence of the shift vector

    third_term = -(1.0/3.0) * vars.conformal_factor * div_shift
    # compute the term due to divergence of shift

    dW_dx1 = diff6_field(vars.conformal_factor, 0, params.dx)
    dW_dx2 = diff6_field(vars.conformal_factor, 1, params.dx)
    dW_dx3 = diff6_field(vars.conformal_factor, 2, params.dx)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dW_dx1 + dW_dx2 + dW_dx3)
    # compute dissipation term

    
    return first_term + second_term + third_term + dissipation_term

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

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
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
    DiDj_alpha = DiDj_alpha + 1/W_floor * jnp.einsum('i...,j...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha + 1/W_floor * jnp.einsum('j...,i...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha - 1/W_floor * jnp.einsum('ij...,mn...,m...,n...->ij...', gamma, inv_gamma, dWdi, dalphadi)
    # full covariant second derivative of alpha

    first_term = -1 * W**2 * jnp.einsum('ij...,ij...->...', inv_gamma, DiDj_alpha)
    # first term

    second_term = alpha * jnp.einsum('ij...,kl...,ik...,jl...->...', inv_gamma, inv_gamma, A_ij, A_ij)
    # second term

    third_term = alpha * K**2 / 3.0
    # third term

    physical_metric = compute_physical_metric(gamma, W)
    inv_physical_metric = invert_3x3_metric(physical_metric)
    S = jnp.einsum('ij...,ij...->...', inv_physical_metric, vars.S_ij)

    fourth_term = 4 * jnp.pi * alpha * (vars.rho + S)


    shift = vars.shift
    # unpack the shift vector

    grad_K = jnp.stack( [diff1_field(K, d, params.dx) for d in range(3)], axis=0)
    # compute the gradient of K

    fourth_term = jnp.einsum('i...,i...->...', shift, grad_K)
    # compute the advection term due to shift


    dt_K = first_term + second_term + third_term + fourth_term
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
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dKdi = jnp.stack( [diff1_field(K, d, dx) for d in range(3)], axis=0)
    # first derivatives of K

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    A_i_up_j = jnp.einsum('jk...,ik...->ij...', inv_gamma, A_ij)
    # raise the second index in A_i^j

    dA_i_up_j_dk = jnp.stack( [diff1_field(A_i_up_j, d+2, dx) for d in range(3)], axis=0)
    # derivatives of A_i^j

    dA_ij_dk = jnp.stack( [diff1_field(A_ij, d+2, dx) for d in range(3)], axis=0)
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
    W    = vars.conformal_factor
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
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
    DiDj_alpha = DiDj_alpha + 1/W_floor * jnp.einsum('i...,j...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha + 1/W_floor * jnp.einsum('j...,i...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha - 1/W_floor * jnp.einsum('ij...,mn...,m...,n...->ij...', gamma, inv_gamma, dWdi, dalphadi)
    # full covariant second derivative of alpha
        
    # Compute full Ricci tensor
    ricci = compute_ricci(vars, params)


    third_term = jnp.power(W, 2) * (alpha * ricci - DiDj_alpha)
    third_term = traceless_part(third_term, vars.conformal_metric, inv_gamma)
    # third term

    shift = vars.shift
    # unpack the shift vector

    grad_shift = compute_shift_derivatives(shift, params.dx)
    # compute the gradient of the shift vector

    grad_A = jnp.stack( [diff1_field(A_ij, d+2, params.dx) for d in range(3)], axis=0)
    # compute the gradient of A_ij

    fourth_term = jnp.einsum('m...,mij...->ij...', shift, grad_A)
    # compute the advection term due to shift

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

    dA_dx1 = diff6_field(vars.traceless_K, 2, params.dx)
    dA_dx2 = diff6_field(vars.traceless_K, 3, params.dx)
    dA_dx3 = diff6_field(vars.traceless_K, 4, params.dx)
    # A_ij is shape (3, 3, ni, nj, nk)
    # compute the 6th derivative in each direction

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
    seventh_term = kappa/2 * alpha * (DjMi + DiMj)
    # seventh term

    dissipation_term = params.nu / 64 * params.dx**5 * (dA_dx1 + dA_dx2 + dA_dx3)
    # compute dissipation term

    return dt_A + seventh_term + dissipation_term


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
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    dalphadi = jnp.stack( [diff1_field(alpha, d, dx) for d in range(3)], axis=0)
    # first derivatives of alpha

    dKdi = jnp.stack( [diff1_field(K, d, dx) for d in range(3)], axis=0)
    # first derivatives of K

    shift = vars.shift
    # unpack the shift vector

    d_shift = compute_shift_derivatives(shift, dx)
    # d_shift[i, j] = partial_j beta^i

    div_shift = jnp.einsum("ii...->...", d_shift)
    # divergence of the shift vector

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

    third_term = -6 * alpha / W_floor * jnp.einsum('ij...,j...->i...', A_ij_raised, dWdi)
    # third term

    fourth_term = -2 * jnp.einsum('ij...,j...->i...', A_ij_raised, dalphadi)
    # fourth term

    fifth_term = -16.0 * jnp.pi * alpha * vars.momentum_density

    grad_Gamma = jnp.stack(
        [diff1_field(vars.conformal_connection, m + 1, dx) for m in range(3)],
        axis=0,
    )
    # grad_Gamma[m, i] = partial_m Gamma^i

    fifth_term = jnp.einsum('m...,mi...->i...', shift, grad_Gamma)
    # advection of the conformal connection by the shift

    sixth_term = (2.0 / 3.0) * vars.conformal_connection * div_shift
    # conformal-weight correction from div(beta)

    seventh_term = -jnp.einsum('m...,im...->i...', vars.conformal_connection, d_shift)
    # -Gamma^m partial_m beta^i

    d2_shift = jnp.zeros((3, 3, 3) + shift.shape[1:])
    for i in range(3):
        for m in range(3):
            for n in range(3):
                d2_shift = d2_shift.at[i, m, n].set(
                    diff1_field(d_shift[i, n], m, dx)
                )
    # d2_shift[i, m, n] = partial_m partial_n beta^i

    eighth_term = jnp.einsum('mn...,imn...->i...', inv_gamma, d2_shift)
    # gamma^mn partial_m partial_n beta^i

    div_shift_deriv = jnp.stack(
        [diff1_field(div_shift, m, dx) for m in range(3)],
        axis=0,
    )
    # partial_m partial_n beta^n = partial_m div(beta)

    ninth_term = (1.0 / 3.0) * jnp.einsum(
        'im...,m...->i...', inv_gamma, div_shift_deriv
    )
    # 1/3 gamma^im partial_m partial_n beta^n

    dt_Gamma = (
        first_term
        + second_term
        + third_term
        + fourth_term
        + fifth_term
        + sixth_term
        + seventh_term
        + eighth_term
        + ninth_term
    )
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

    grad_alpha = jnp.stack(
        [diff1_field(vars.lapse, d, dx) for d in range(3)],
        axis=0,
    )
    # first derivatives of the lapse

    advection_term = jnp.einsum('m...,m...->...', vars.shift, grad_alpha)
    # advect the lapse with the shift

    dalpha_dx1 = diff6_field(vars.lapse, 0, params.dx)
    dalpha_dx2 = diff6_field(vars.lapse, 1, params.dx)
    dalpha_dx3 = diff6_field(vars.lapse, 2, params.dx)
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

    grad_shift = compute_shift_derivatives(shift, params.dx)
    # grad_shift[i, j] = partial_j beta^i

    advection_term = jnp.einsum('j...,ij...->i...', shift, grad_shift)
    # beta^j partial_j beta^i

    gamma_driver_term = params.g * vars.conformal_connection
    # single-variable Gamma-driver source for beta^i

    damping_term = -params.eta * shift
    # linear damping of the shift

    dt_beta = gamma_driver_term + advection_term + damping_term

    dbeta_dx1 = diff6_field(shift, 1, params.dx)
    dbeta_dx2 = diff6_field(shift, 2, params.dx)
    dbeta_dx3 = diff6_field(shift, 3, params.dx)
    # beta is shape (3, ni, nj, nk)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (
        dbeta_dx1 + dbeta_dx2 + dbeta_dx3
    )
    # compute dissipation term
    
    return dt_beta + dissipation_term