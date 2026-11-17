"""Constraint reductions."""

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.bssn.constraints import ConstraintViolations


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
