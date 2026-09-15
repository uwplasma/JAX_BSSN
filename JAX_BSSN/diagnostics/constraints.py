"""Constraint reductions and simulation-health reporting."""

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.bssn.constraints import ConstraintViolations, compute_all_constraints
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables


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
