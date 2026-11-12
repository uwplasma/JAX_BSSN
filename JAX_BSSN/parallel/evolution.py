"""Local BSSN equations on a shared PyPIC3D device mesh."""

import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from JAX_BSSN.bssn.constraints import compute_all_constraints, compute_all_constraints_with_matter
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs, compute_bssn_rhs_with_matter, enforce_algebraic_constraints,
)
from .communication import MESH_AXES, refresh_tiled_field, tile_spec, validate_layout


def local_parameters(params, mesh, tile_shape, guard_cells, upper_endpoints):
    """Coordinates, owned bounds, and physical-face flags for the current tile."""
    indices = tuple(jax.lax.axis_index(name) for name in MESH_AXES)
    counts = mesh.devices.shape
    origins = (params.x_min, params.y_min, params.z_min)
    minima = tuple(origin + (index * width - guard_cells) * params.dx
                   for origin, index, width in zip(origins, indices, tile_shape))
    owned = tuple(width + (index == count - 1) * int(endpoint)
                  for width, index, count, endpoint in zip(tile_shape, indices, counts, upper_endpoints))
    faces = tuple(flag for index, count in zip(indices, counts)
                  for flag in (index == 0, index == count - 1))
    return params._replace(x_min=minima[0], y_min=minima[1], z_min=minima[2],
                            guard_cells=guard_cells, owned_shape=owned, owned_faces=faces)


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


def map_tiled(function, values, params, *, mesh, tile_shape, guard_cells=4,
               upper_endpoints=(False, False, False), output_template=None):
    """Call ``function(local_values, local_params)`` once on every device.

    All array leaves have tensor indices followed by six tile/spatial axes.
    The function sees only the final three spatial axes. Supply a template
    when its output tree/component ranks differ from its inputs.
    """
    validate_layout(mesh, tile_shape, guard_cells, upper_endpoints)
    template = values if output_template is None else output_template
    in_specs = jax.tree.map(tile_spec, values)
    out_specs = jax.tree.map(tile_spec, template)

    def local_op(local_values):
        local_values = jax.tree.map(lambda a: a.reshape(a.shape[:-6] + a.shape[-3:]), local_values)
        local_params = local_parameters(params, mesh, tile_shape, guard_cells, upper_endpoints)
        result = function(local_values, local_params)
        return jax.tree.map(lambda a: a.reshape(a.shape[:-3] + (1, 1, 1) + a.shape[-3:]), result)

    return jax.shard_map(local_op, mesh=mesh, in_specs=(in_specs,),
                         out_specs=out_specs, check_vma=False)(values)


def refresh_bssn_halos(state, params, **layout):
    """Batch all BSSN components into one owner-to-ghost communication sweep."""
    if layout.get("guard_cells", 4) < 4:
        raise ValueError("Tiled BSSN evolution requires four guards for the supported MAD stencils")
    spatial = state.conformal_factor.shape
    counts = [math.prod(value.shape[:-6]) for value in state]
    packed = jnp.concatenate(tuple(value.reshape((count,) + spatial)
                                   for value, count in zip(state, counts)), axis=0)
    packed = refresh_tiled_field(packed, params, **layout)
    values, start = [], 0
    for value, count in zip(state, counts):
        values.append(packed[start:start + count].reshape(value.shape))
        start += count
    return type(state)(*values)


def prepare_tiled_state(state, params, **layout):
    """Project owned geometry, then refresh neighbors and physical exteriors."""
    def project(local, local_params):
        projected = enforce_algebraic_constraints(local)
        mask = owned_mask(local.conformal_factor.shape, local_params)
        return jax.tree.map(lambda new, old: jnp.where(mask, new, old), projected, local)

    return refresh_bssn_halos(map_tiled(project, state, params, **layout), params, **layout)


def compute_tiled_bssn_rhs(state, params, **layout):
    """Refresh state halos and evaluate the unchanged vacuum equations locally."""
    state = refresh_bssn_halos(state, params, **layout)
    return map_tiled(compute_bssn_rhs, state, params, **layout)


def compute_tiled_bssn_rhs_with_matter(state, params, energy_density, momentum_density,
                                      spatial_stress, **layout):
    """Evaluate matter-coupled equations with valid momentum derivative halos."""
    state = refresh_bssn_halos(state, params, **layout)
    sources = tuple(refresh_tiled_field(value, params, **layout)
                    for value in (energy_density, momentum_density, spatial_stress))
    return map_tiled(lambda local, p: compute_bssn_rhs_with_matter(local[0], p, *local[1:]),
                       (state, *sources), params, output_template=state, **layout)


def tiled_rk4_step(state, params, **layout):
    """RK4 with algebraic projection and a complete halo refresh at each stage."""
    state = prepare_tiled_state(state, params, **layout)
    k1 = compute_tiled_bssn_rhs(state, params, **layout)
    stage = jax.tree.map(lambda q, k: q + params.dt * k / 2, state, k1)
    stage = prepare_tiled_state(stage, params, **layout)
    k2 = compute_tiled_bssn_rhs(stage, params, **layout)
    stage = jax.tree.map(lambda q, k: q + params.dt * k / 2, state, k2)
    stage = prepare_tiled_state(stage, params, **layout)
    k3 = compute_tiled_bssn_rhs(stage, params, **layout)
    stage = jax.tree.map(lambda q, k: q + params.dt * k, state, k3)
    stage = prepare_tiled_state(stage, params, **layout)
    k4 = compute_tiled_bssn_rhs(stage, params, **layout)
    result = jax.tree.map(lambda q, a, b, c, d: q + params.dt / 6 * (a + 2*b + 2*c + d),
                          state, k1, k2, k3, k4)
    return prepare_tiled_state(result, params, **layout)


def tiled_constraint_norms(state, params, rho=None, momentum=None, *, mesh,
                            tile_shape, guard_cells=4, upper_endpoints=(False, False, False)):
    """Serial-compatible L2/Linf constraints, reducing only scalar statistics."""
    layout = dict(mesh=mesh, tile_shape=tile_shape, guard_cells=guard_cells,
                  upper_endpoints=upper_endpoints)
    state = refresh_bssn_halos(state, params, **layout)
    values = (state,) if rho is None else (state, rho, momentum)
    specs = jax.tree.map(tile_spec, values)

    def local_norms(local):
        local = jax.tree.map(lambda a: a.reshape(a.shape[:-6] + a.shape[-3:]), local)
        p = local_parameters(params, mesh, tile_shape, guard_cells, upper_endpoints)
        violations = (compute_all_constraints(local[0], p) if rho is None else
                      compute_all_constraints_with_matter(local[0], p, local[1], local[2]))
        mask = owned_mask(local[0].conformal_factor.shape, p)
        norms = {}
        for name, field in zip(("hamiltonian", "momentum", "det_gamma", "trace_A", "gamma"), violations):
            count = jnp.sum(mask) * math.prod(field.shape[:-3])
            square = jnp.sum(jnp.where(mask, field**2, 0.0))
            maximum = jnp.max(jnp.where(mask, jnp.abs(field), 0.0))
            norms[name + "_l2"] = jnp.sqrt(jax.lax.psum(square, MESH_AXES)
                                            / jax.lax.psum(count, MESH_AXES))
            norms[name + "_linf"] = jax.lax.pmax(maximum, MESH_AXES)
        return norms

    return jax.shard_map(local_norms, mesh=mesh, in_specs=(specs,),
                         out_specs=P(), check_vma=False)(values)
