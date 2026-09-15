"""Constraints and output mappings for compact axisymmetric Cartoon states."""

from functools import partial

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.bssn.constraints import ConstraintViolations, compute_all_constraints
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
    AXISYMMETRIC_GHOST_CELLS,
    _expand_axisymmetric_scalar,
    _expand_axisymmetric_vector,
    _project_scalar,
    _project_vector,
    expand_axisymmetric_plane,
    reconstruct_axisymmetric_support,
)


@jit
def compute_axisymmetric_constraints(
    vars: BSSNVariables, params: BSSNParameters
) -> ConstraintViolations:
    """Reconstruct, evaluate Cartesian constraints, and compact the result."""

    support_vars = reconstruct_axisymmetric_support(vars, params)
    support = compute_all_constraints(support_vars, params)
    return ConstraintViolations(
        hamiltonian=_project_scalar(support.hamiltonian),
        momentum=_project_vector(support.momentum),
        det_gamma=_project_scalar(support.det_gamma),
        trace_A=_project_scalar(support.trace_A),
        gamma_condition=_project_vector(support.gamma_condition),
    )


@partial(jit, static_argnames=("exclude_outer_rho", "exclude_z"))
def compute_axisymmetric_constraint_norms(
    violations: ConstraintViolations,
    exclude_outer_rho: int = 4,
    exclude_z: int = 4,
) -> dict:
    """Return normalized cylindrical L2 norms and ordinary Linf norms."""

    def physical(field):
        field = field[..., AXISYMMETRIC_GHOST_CELLS :, 0, :]
        if exclude_outer_rho:
            field = field[..., :-exclude_outer_rho, :]
        if exclude_z:
            field = field[..., exclude_z:-exclude_z]
        return field

    fields = {
        "hamiltonian": physical(violations.hamiltonian),
        "momentum": physical(violations.momentum),
        "det_gamma": physical(violations.det_gamma),
        "trace_A": physical(violations.trace_A),
        "gamma": physical(violations.gamma_condition),
    }

    norms = {}
    for name, field in fields.items():
        nrho, nz = field.shape[-2:]
        rho = jnp.arange(nrho, dtype=field.dtype) + 0.5
        weight = rho.reshape((1,) * (field.ndim - 2) + (nrho, 1))
        component_count = 1
        for size in field.shape[:-2]:
            component_count *= size
        normalization = jnp.sum(rho) * nz * component_count
        norms[f"{name}_l2"] = jnp.sqrt(jnp.sum(weight * field**2) / normalization)
        norms[f"{name}_linf"] = jnp.max(jnp.abs(field))
    return norms


def axisymmetric_plane_output_fields(
    vars: BSSNVariables,
    hamiltonian: jnp.ndarray,
    momentum: jnp.ndarray,
    det_gamma: jnp.ndarray | None = None,
    trace_A: jnp.ndarray | None = None,
    gamma_condition: jnp.ndarray | None = None,
) -> dict:
    """Return the complete state and constraints on a signed ``x-z`` plane."""

    full_vars = expand_axisymmetric_plane(vars)
    full_hamiltonian = _expand_axisymmetric_scalar(hamiltonian)
    full_momentum = _expand_axisymmetric_vector(momentum)
    fields = {
        "W": full_vars.conformal_factor,
        "K": full_vars.trace_K,
        "lapse": full_vars.lapse,
        "shift": tuple(full_vars.shift[i] for i in range(3)),
        "conformal_connection": tuple(
            full_vars.conformal_connection[i] for i in range(3)
        ),
        "hamiltonian_constraint": full_hamiltonian,
        "momentum_constraint": tuple(full_momentum[i] for i in range(3)),
    }

    if det_gamma is not None:
        fields["det_gamma_constraint"] = _expand_axisymmetric_scalar(det_gamma)
    if trace_A is not None:
        fields["trace_A_constraint"] = _expand_axisymmetric_scalar(trace_A)
    if gamma_condition is not None:
        full_gamma = _expand_axisymmetric_vector(gamma_condition)
        fields["gamma_constraint"] = tuple(full_gamma[i] for i in range(3))

    for name, tensor in (
        ("conformal_metric", full_vars.conformal_metric),
        ("traceless_K", full_vars.traceless_K),
    ):
        fields[f"{name}_xx"] = tensor[0, 0]
        fields[f"{name}_xy"] = tensor[0, 1]
        fields[f"{name}_xz"] = tensor[0, 2]
        fields[f"{name}_yy"] = tensor[1, 1]
        fields[f"{name}_yz"] = tensor[1, 2]
        fields[f"{name}_zz"] = tensor[2, 2]
    return fields


__all__ = [
    "axisymmetric_plane_output_fields",
    "compute_axisymmetric_constraint_norms",
    "compute_axisymmetric_constraints",
]
