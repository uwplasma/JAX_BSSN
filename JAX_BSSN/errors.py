"""
Error analysis and constraint monitoring for BSSN evolution.

This module provides functions to compute and monitor various constraint
violations and convergence measures. All functions are JIT-compiled with JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, NamedTuple
import numpy as np

from JAX_BSSN.bssn import BSSNVariables, BSSNParameters, compute_ricci, compute_momentum_constraint
from JAX_BSSN.derivatives import diff1_field, divergence_3d, compute_all_derivatives
from JAX_BSSN.tensor_algebra import (invert_3x3_metric, determinant_3x3_metric,
                           christoffel_symbols_second_kind, ricci_tensor,
                           ricci_scalar, trace_tensor)


# NOTE: HAMILTONIAN AND MOMENTUM CONSTRAINT METHODS HAVE BEEN TESTED AND VERIFIED AS OF DEC 3RD 2025
# THE OTHERS ARE PARTIALLY IMPLEMENTED AND NEED FURTHER TESTING


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
    dx = params.dx
    conformal_metric = vars.conformal_metric
    conformal_connection = vars.conformal_connection
    W = vars.conformal_factor

    metric_derivs = jnp.stack( [diff1_field(conformal_metric, d+2, dx) for d in range(3)], axis=0) 

    # Compute physical Ricci scalar (simplified calculation)
    inv_metric = invert_3x3_metric(conformal_metric)
    
    ricci_tensor = compute_ricci(vars, params)
    ricci_scalar = trace_tensor(ricci_tensor, inv_metric)
    # compute the ricci scalar from the conformal Ricci tensor

    K_squared = vars.trace_K**2
    
    A_squared = jnp.einsum('ik...,jl...,ij...,kl...->...', inv_metric, inv_metric,
                            vars.traceless_K, vars.traceless_K)


    # Hamiltonian constraint with matter source term:
    # R + 2/3 K^2 - A_ij A^ij = 16 pi rho
    hamiltonian = W**2 * ricci_scalar + 2/3 * K_squared - A_squared - 16.0 * jnp.pi * vars.rho

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


# @jit
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

    # raise NotImplementedError("Gamma constraint computation not implemented")
    return jnp.zeros_like(vars.lapse)
    # dx = params.dx
    # shape = vars.conformal_metric.shape[2:]
    
    # # Compute metric derivatives
    # metric_derivs = jnp.zeros((3, 3, 3) + shape)
    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             metric_derivs = metric_derivs.at[k, i, j].set(
    #                 diff1_field(vars.conformal_metric[i, j], k, dx))
    
    # # Compute inverse metric
    # inv_metric = invert_3x3_metric(vars.conformal_metric)
    
    # # Compute Christoffel symbols
    # christoffel = christoffel_symbols_second_kind(inv_metric, metric_derivs)
    
    # # Compute γ^jk Γ^i_jk
    # gamma_from_christoffel = jnp.zeros((3,) + shape)
    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             gamma_from_christoffel = gamma_from_christoffel.at[i].add(
    #                 inv_metric[j, k] * christoffel[i, j, k])
    
    # # Constraint violation
    # gamma_violation = jnp.zeros((3,) + shape)
    # for i in range(3):
    #     gamma_violation = gamma_violation.at[i].set(
    #         vars.conformal_connection[i] - gamma_from_christoffel[i])
    
    # return gamma_violation


# @jit
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
def compute_constraint_norms(violations: ConstraintViolations) -> dict:
    """
    Compute various norms of constraint violations.
    
    Args:
        violations: Constraint violations
        
    Returns:
        Dictionary of constraint norms
    """
    norms = {}
    
    # L2 norms
    norms['hamiltonian_l2'] = jnp.sqrt(jnp.mean(violations.hamiltonian**2))
    norms['momentum_l2'] = jnp.sqrt(jnp.mean(violations.momentum**2))
    norms['det_gamma_l2'] = jnp.sqrt(jnp.mean(violations.det_gamma**2))
    norms['trace_A_l2'] = jnp.sqrt(jnp.mean(violations.trace_A**2))
    norms['gamma_l2'] = jnp.sqrt(jnp.mean(violations.gamma_condition**2))
    
    # L∞ norms (maximum values)
    norms['hamiltonian_linf'] = jnp.max(jnp.abs(violations.hamiltonian))
    norms['momentum_linf'] = jnp.max(jnp.abs(violations.momentum))
    norms['det_gamma_linf'] = jnp.max(jnp.abs(violations.det_gamma))
    norms['trace_A_linf'] = jnp.max(jnp.abs(violations.trace_A))
    norms['gamma_linf'] = jnp.max(jnp.abs(violations.gamma_condition))
    
    return norms

def print_constraint_summary(violations: ConstraintViolations, time: float):
    """
    Print summary of constraint violations (not JIT-compiled).
    
    Args:
        violations: Constraint violations
        time: Current simulation time
    """
    norms = compute_constraint_norms(violations)
    
    print(f"Time: {time:.4f}")
    print(f"  Hamiltonian L2:  {norms['hamiltonian_l2']:.2e}")
    print(f"  Momentum L2:     {norms['momentum_l2']:.2e}")
    print(f"  det(γ)-1 L2:     {norms['det_gamma_l2']:.2e}")
    print(f"  tr(A) L2:        {norms['trace_A_l2']:.2e}")
    print(f"  Γ constraint L2: {norms['gamma_l2']:.2e}")
    print()


def monitor_simulation_health(vars: BSSNVariables, params: BSSNParameters,
                             time: float, max_constraint_violation: float = 1e-2) -> bool:
    """
    Monitor simulation health and return whether to continue.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters  
        time: Current time
        max_constraint_violation: Maximum allowed constraint violation
        
    Returns:
        True if simulation should continue, False if it should stop
    """
    # Check for NaN or infinite values
    for field in [vars.conformal_metric, vars.conformal_factor, vars.traceless_K,
                  vars.trace_K, vars.conformal_connection, vars.lapse, vars.shift]:
        if jnp.any(jnp.isnan(field)) or jnp.any(jnp.isinf(field)):
            print(f"ERROR: NaN or Inf detected at time {time}")
            return False
    
    # Check constraint violations
    violations = compute_all_constraints(vars, params)
    norms = compute_constraint_norms(violations)
    
    max_violation = max(norms['hamiltonian_l2'], norms['momentum_l2'], 
                       norms['det_gamma_l2'], norms['trace_A_l2'])
    
    if max_violation > max_constraint_violation:
        print(f"ERROR: Constraint violation too large at time {time}: {max_violation}")
        return False
    
    return True
