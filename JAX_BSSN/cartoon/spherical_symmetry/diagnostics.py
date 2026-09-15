"""Constraints and output mappings for compact spherical Cartoon states."""

from functools import partial

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.bssn.constraints import ConstraintViolations, compute_all_constraints
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.spherical_symmetry.reconstruction import (
    cartoon_positive_radius,
    expand_cartoon_axis,
    expand_cartoon_scalar,
    expand_cartoon_vector,
    project_cartoon_scalar,
    project_cartoon_vector,
    reconstruct_cartoon_support,
)


@jit
def compute_cartoon_constraints(
    vars: BSSNVariables, params: BSSNParameters
) -> ConstraintViolations:
    """Compute constraints on Cartesian support and return compact fields."""

    support_vars = reconstruct_cartoon_support(vars, params)
    support = compute_all_constraints(support_vars, params)

    return ConstraintViolations(
        hamiltonian=project_cartoon_scalar(support.hamiltonian),
        momentum=project_cartoon_vector(support.momentum),
        det_gamma=project_cartoon_scalar(support.det_gamma),
        trace_A=project_cartoon_scalar(support.trace_A),
        gamma_condition=project_cartoon_vector(support.gamma_condition),
    )


@partial(jit, static_argnames=("exclude_outer",))
def compute_cartoon_constraint_norms(
    violations: ConstraintViolations, exclude_outer: int = 4
) -> dict:
    """Compute norms over independent positive radii away from the outer edge."""

    def physical_radius(field):
        radius = cartoon_positive_radius(field)
        if exclude_outer:
            radius = radius[..., :-exclude_outer]
        return radius

    fields = {
        "hamiltonian": physical_radius(violations.hamiltonian),
        "momentum": physical_radius(violations.momentum),
        "det_gamma": physical_radius(violations.det_gamma),
        "trace_A": physical_radius(violations.trace_A),
        "gamma": physical_radius(violations.gamma_condition),
    }

    norms = {}
    for name, field in fields.items():
        norms[f"{name}_l2"] = jnp.sqrt(jnp.mean(field**2))
        norms[f"{name}_linf"] = jnp.max(jnp.abs(field))

    return norms


@partial(jit, static_argnames=("exclude_outer",))
def compute_spherical_symmetry_norms(
    vars: BSSNVariables, exclude_outer: int = 0
) -> dict:
    """Measure reference-axis components forbidden by spherical symmetry."""

    def physical_radius(field):
        radius = cartoon_positive_radius(field)
        if exclude_outer:
            radius = radius[..., :-exclude_outer]
        return radius

    errors = {
        "metric_tangential": vars.conformal_metric[1, 1]
        - vars.conformal_metric[2, 2],
        "A_tangential": vars.traceless_K[1, 1]
        - vars.traceless_K[2, 2],
        "metric_xy": vars.conformal_metric[0, 1],
        "metric_xz": vars.conformal_metric[0, 2],
        "metric_yz": vars.conformal_metric[1, 2],
        "A_xy": vars.traceless_K[0, 1],
        "A_xz": vars.traceless_K[0, 2],
        "A_yz": vars.traceless_K[1, 2],
        "shift_y": vars.shift[1],
        "shift_z": vars.shift[2],
        "connection_y": vars.conformal_connection[1],
        "connection_z": vars.conformal_connection[2],
    }

    norms = {}
    for name, error in errors.items():
        error = physical_radius(error)
        norms[f"{name}_l2"] = jnp.sqrt(jnp.mean(error**2))
        norms[f"{name}_linf"] = jnp.max(jnp.abs(error))
    return norms


def cartoon_axis_output_fields(
    vars: BSSNVariables,
    hamiltonian: jnp.ndarray,
    momentum: jnp.ndarray,
    det_gamma: jnp.ndarray | None = None,
    trace_A: jnp.ndarray | None = None,
    gamma_condition: jnp.ndarray | None = None,
) -> dict:
    """Return the complete BSSN state and constraints on the signed x-axis."""

    full_vars = expand_cartoon_axis(vars)
    full_hamiltonian = expand_cartoon_scalar(hamiltonian)
    full_momentum = expand_cartoon_vector(momentum)

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
        fields["det_gamma_constraint"] = expand_cartoon_scalar(det_gamma)
    if trace_A is not None:
        fields["trace_A_constraint"] = expand_cartoon_scalar(trace_A)
    if gamma_condition is not None:
        full_gamma_condition = expand_cartoon_vector(gamma_condition)
        fields["gamma_constraint"] = tuple(
            full_gamma_condition[i] for i in range(3)
        )

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
