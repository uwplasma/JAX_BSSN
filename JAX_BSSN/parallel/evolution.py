"""Local BSSN equations on a shared PyPIC3D device mesh."""

import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from JAX_BSSN.bssn.constraints import compute_all_constraints, compute_all_constraints_with_matter
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs, compute_bssn_rhs_with_matter, enforce_algebraic_constraints,
)
from .communication import refresh_tiled_field
from .layout import MESH_AXES, require_bssn_guards, tile_spec

CONSTRAINT_NAMES = ("hamiltonian", "momentum", "det_gamma", "trace_A", "gamma")


def local_parameters(params, layout):
    """Coordinates, owned bounds, and physical-face flags for the current tile."""
    indices = tuple(jax.lax.axis_index(name) for name in MESH_AXES)
    counts = layout.counts
    origins = (params.x_min, params.y_min, params.z_min)
    minima = tuple(origin + (index * width - layout.guard_cells) * params.dx
                   for origin, index, width in zip(origins, indices, layout.tile_shape))
    # Stay a Python int unless an endpoint really makes the count tile
    # dependent: a traced count turns every derivative closure index into a
    # dynamic gather/scatter.
    owned = tuple((width + (index == count - 1)) if endpoint else width
                  for width, index, count, endpoint
                  in zip(layout.tile_shape, indices, counts, layout.upper_endpoints))
    faces = tuple(flag for index, count in zip(indices, counts)
                  for flag in (index == 0, index == count - 1))
    reduced = (params.reduced_axes if params.reduced_axes is not None else
               tuple(n == 1 for n in layout.active_shape))
    return params._replace(x_min=minima[0], y_min=minima[1], z_min=minima[2],
                            guard_cells=layout.guard_cells, owned_shape=owned,
                            owned_faces=faces, reduced_axes=reduced)


def owned_mask(shape, params):
    """Select each physical sample once, including owned upper C endpoints."""
    mask = jnp.ones(shape, dtype=bool)
    for axis, width in enumerate(params.owned_shape):
        nodes = jnp.arange(shape[axis])
        dims = [1, 1, 1]
        dims[axis] = shape[axis]
        mask = mask & ((nodes >= params.guard_cells)
                       & (nodes < params.guard_cells + width)).reshape(dims)
    return mask


def map_tiled(function, values, params, layout, *, output_template=None):
    """Call ``function(local_values, local_params)`` once on every device.

    All array leaves have tensor indices followed by six tile/spatial axes.
    The function sees only the final three spatial axes. Supply a template
    when its output tree/component ranks differ from its inputs.
    """
    layout.validate()
    template = values if output_template is None else output_template
    in_specs = jax.tree.map(tile_spec, values)
    out_specs = jax.tree.map(tile_spec, template)

    def local_op(local_values):
        local_values = jax.tree.map(lambda a: a.reshape(a.shape[:-6] + a.shape[-3:]), local_values)
        result = function(local_values, local_parameters(params, layout))
        return jax.tree.map(lambda a: a.reshape(a.shape[:-3] + (1, 1, 1) + a.shape[-3:]), result)

    return jax.shard_map(local_op, mesh=layout.mesh, in_specs=(in_specs,),
                         out_specs=out_specs, check_vma=False)(values)


def refresh_bssn_halos(state, params, layout, *, ghost_fill=None):
    """Exchange neighbors, then call ``ghost_fill(tiled_state, params)``.

    The optional callback receives the global tiled tree after communication
    and returns the same shapes. FMR uses it to replace the patch exterior
    shell with interpolation from the matching parent RK stage.
    """
    require_bssn_guards(layout, params)
    spatial = state.conformal_factor.shape
    counts = [math.prod(value.shape[:-6]) for value in state]
    packed = jnp.concatenate(tuple(value.reshape((count,) + spatial)
                                   for value, count in zip(state, counts)), axis=0)
    packed = refresh_tiled_field(packed, params, layout)
    values, start = [], 0
    for value, count in zip(state, counts):
        values.append(packed[start:start + count].reshape(value.shape))
        start += count
    state = type(state)(*values)
    return state if ghost_fill is None else ghost_fill(state, params)


def project_owned(state, params, layout):
    """Re-impose the algebraic constraints on owned samples only."""
    def project(local, local_params):
        projected = enforce_algebraic_constraints(local)
        mask = owned_mask(local.conformal_factor.shape, local_params)
        return jax.tree.map(lambda new, old: jnp.where(mask, new, old), projected, local)

    return map_tiled(project, state, params, layout)


def prepare_tiled_state(state, params, layout, *, ghost_fill=None):
    """Project owned geometry, then refresh neighbors and physical exteriors."""
    return refresh_bssn_halos(project_owned(state, params, layout), params, layout,
                              ghost_fill=ghost_fill)


def compute_tiled_bssn_rhs(state, params, layout, *, ghost_fill=None):
    """Refresh state halos and evaluate the unchanged vacuum equations locally."""
    state = refresh_bssn_halos(state, params, layout, ghost_fill=ghost_fill)
    return map_tiled(compute_bssn_rhs, state, params, layout)


def compute_tiled_bssn_rhs_with_matter(state, params, layout, energy_density,
                                       momentum_density, spatial_stress, *, ghost_fill=None):
    """Evaluate matter-coupled equations with valid momentum derivative halos."""
    state = refresh_bssn_halos(state, params, layout, ghost_fill=ghost_fill)
    sources = tuple(refresh_tiled_field(value, params, layout)
                    for value in (energy_density, momentum_density, spatial_stress))
    return map_tiled(lambda local, p: compute_bssn_rhs_with_matter(local[0], p, *local[1:]),
                     (state, *sources), params, layout, output_template=state)


def tiled_rk4_step(state, params, layout, *, ghost_fill=None):
    """RK4 with algebraic projection and a complete halo refresh at each stage."""
    state = prepare_tiled_state(state, params, layout, ghost_fill=ghost_fill)
    k1 = map_tiled(compute_bssn_rhs, state, params, layout)
    stage = jax.tree.map(lambda q, k: q + params.dt * k / 2, state, k1)
    stage = prepare_tiled_state(stage, params, layout, ghost_fill=ghost_fill)
    k2 = map_tiled(compute_bssn_rhs, stage, params, layout)
    stage = jax.tree.map(lambda q, k: q + params.dt * k / 2, state, k2)
    stage = prepare_tiled_state(stage, params, layout, ghost_fill=ghost_fill)
    k3 = map_tiled(compute_bssn_rhs, stage, params, layout)
    stage = jax.tree.map(lambda q, k: q + params.dt * k, state, k3)
    stage = prepare_tiled_state(stage, params, layout, ghost_fill=ghost_fill)
    k4 = map_tiled(compute_bssn_rhs, stage, params, layout)
    result = jax.tree.map(lambda q, a, b, c, d: q + params.dt / 6 * (a + 2*b + 2*c + d),
                          state, k1, k2, k3, k4)
    return prepare_tiled_state(result, params, layout, ghost_fill=ghost_fill)


def constraint_statistics(violations, mask):
    """Mesh-wide (count, sum of squares, maximum) per constraint field.

    Only these scalar statistics cross the mesh; the violation fields stay
    local to their tile.
    """
    statistics = []
    for field in violations:
        count = jnp.sum(mask) * math.prod(field.shape[:-3])
        square = jnp.sum(jnp.where(mask, field**2, 0.0))
        maximum = jnp.max(jnp.where(mask, jnp.abs(field), 0.0))
        statistics.append((jax.lax.psum(count, MESH_AXES),
                           jax.lax.psum(square, MESH_AXES),
                           jax.lax.pmax(maximum, MESH_AXES)))
    return tuple(statistics)


def constraint_norms_from_totals(totals):
    """Combine per-patch statistics into composite point-count L2/Linf norms."""
    norms = {}
    for component, name in enumerate(CONSTRAINT_NAMES):
        count = sum(patch[component][0] for patch in totals)
        square = sum(patch[component][1] for patch in totals)
        norms[name + "_l2"] = jnp.sqrt(square / count)
        norms[name + "_linf"] = jnp.max(jnp.stack([patch[component][2] for patch in totals]))
    return norms


def tiled_constraint_norms(state, params, layout, rho=None, momentum=None, *,
                           ghost_fill=None):
    """Serial-compatible L2/Linf constraints, reducing only scalar statistics."""
    if (rho is None) != (momentum is None):
        raise ValueError("matter constraints need both an energy and a momentum density")
    state = refresh_bssn_halos(state, params, layout, ghost_fill=ghost_fill)
    values = (state,) if rho is None else (state, rho, momentum)
    specs = jax.tree.map(tile_spec, values)

    def local_norms(local):
        local = jax.tree.map(lambda a: a.reshape(a.shape[:-6] + a.shape[-3:]), local)
        p = local_parameters(params, layout)
        violations = (compute_all_constraints(local[0], p) if rho is None else
                      compute_all_constraints_with_matter(local[0], p, local[1], local[2]))
        return constraint_statistics(violations, owned_mask(local[0].conformal_factor.shape, p))

    totals = jax.shard_map(local_norms, mesh=layout.mesh, in_specs=(specs,),
                           out_specs=P(), check_vma=False)(values)
    return constraint_norms_from_totals((totals,))
