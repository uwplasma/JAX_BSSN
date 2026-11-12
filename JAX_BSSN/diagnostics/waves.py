"""Physical wave fields and composite diagnostics for Cartesian BSSN patches."""

import math

import jax.numpy as jnp

from JAX_BSSN.bssn.constraints import compute_hamiltonian_constraint, compute_momentum_constraint
from JAX_BSSN.fmr.refinement import fine_active_view, hierarchy_parents


def h_plus(variables):
    """Return the physical plus-polarized metric perturbation."""

    physical_metric_yy = (
        variables.conformal_metric[1, 1] / variables.conformal_factor**2
    )
    physical_metric_zz = (
        variables.conformal_metric[2, 2] / variables.conformal_factor**2
    )
    return 0.5 * (physical_metric_yy - physical_metric_zz)


def diagnostic_fields(variables, parameters):
    """Compute the wave and BSSN constraint fields for one grid level."""

    hamiltonian = compute_hamiltonian_constraint(variables, parameters)
    momentum = compute_momentum_constraint(variables, parameters)

    return {
        "h_plus": h_plus(variables),
        "lapse": variables.lapse,
        "shift": tuple(variables.shift[i] for i in range(3)),
        "K": variables.trace_K,
        "W": variables.conformal_factor,
        "hamiltonian_constraint": hamiltonian,
        "momentum_constraint": tuple(momentum[i] for i in range(3)),
    }


def composite_wave_diagnostics(level_fields, level_coordinates, hierarchy,
                                time, amplitude, wavelength, *, active_fields=False):
    """Point-count wave/constraint norms on uncovered vertices of a patch tree."""
    parents = hierarchy_parents(hierarchy)
    error_sum = 0.0
    point_count = 0
    maximum = 0.0
    hamiltonian_sum = 0.0
    momentum_sum = 0.0
    for level, (fields, coordinates) in enumerate(
        zip(level_fields, level_coordinates)
    ):
        if level and not active_fields:
            fields = {name: fine_active_view(value, hierarchy.patches[level - 1])
                      if not isinstance(value, tuple) else tuple(
                          fine_active_view(component, hierarchy.patches[level - 1])
                          for component in value)
                      for name, value in fields.items()}
            coordinates = fine_active_view(coordinates,
                                           hierarchy.patches[level - 1])
        mask = jnp.ones(fields["h_plus"].shape, dtype=bool)
        for child_index, parent in enumerate(parents[1:], start=1):
            if parent == level:
                child = hierarchy.patches[child_index - 1]
                covered = tuple(slice(lo, hi + 1)
                                for lo, hi in zip(child.coarse_lo, child.coarse_hi))
                mask = mask.at[covered].set(False)
        exact = amplitude * jnp.sin(
            2.0 * jnp.pi * (coordinates - time) / wavelength
        )
        error = fields["h_plus"] - exact
        momentum = jnp.stack(fields["momentum_constraint"])
        count = jnp.sum(mask)
        error_sum += float(jnp.sum(jnp.where(mask, error**2, 0.0)))
        hamiltonian_sum += float(jnp.sum(jnp.where(
            mask, fields["hamiltonian_constraint"]**2, 0.0
        )))
        # Preserve the wave demo's vector magnitude norm per vertex, rather
        # than averaging the three momentum components as separate samples.
        momentum_norm = jnp.sum(momentum**2, axis=0)
        momentum_sum += float(jnp.sum(jnp.where(mask, momentum_norm, 0.0)))
        maximum = max(maximum, float(jnp.max(jnp.where(mask, jnp.abs(error), 0.0))))
        point_count += int(count)
    return {
        "composite_rms": math.sqrt(error_sum / point_count),
        "composite_max": maximum,
        "composite_hamiltonian_l2": math.sqrt(hamiltonian_sum / point_count),
        "composite_momentum_l2": math.sqrt(momentum_sum / point_count),
        # Retain concise summary keys used by callers of the demonstration.
        "coarse_rms": math.sqrt(error_sum / point_count),
        "fine_rms": math.sqrt(error_sum / point_count),
    }

