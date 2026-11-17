"""Stage-synchronous Cartesian FMR on a shared BSSN device mesh."""

import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from JAX_BSSN.bssn.variables import BSSNVariables
from JAX_BSSN.bssn.constraints import compute_all_constraints
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs
from JAX_BSSN.parallel import initialize_tiled_bssn, make_field_mesh
from JAX_BSSN.parallel.layout import FACE_NAMES, MESH_AXES, TileLayout, tile_spec
from JAX_BSSN.parallel.evolution import (
    constraint_norms_from_totals, constraint_statistics, local_parameters, map_tiled,
    owned_mask, prepare_tiled_state, project_owned, refresh_bssn_halos,
)
from .refinement import fine_active_shape, hierarchy_parents
from .transfer import fill_tiled_fine_ghosts, prolongation_routes, restrict_tiled_to_coarse


def patch_layout(shape, mesh):
    """Fit an exact active extent to uniform tiles, retaining an upper vertex."""
    widths, endpoints = [], []
    for n, count in zip(shape, mesh.devices.shape):
        if n % count == 0:
            widths.append(n // count)
            endpoints.append(False)
        elif (n - 1) % count == 0 and n > 1:
            widths.append((n - 1) // count)
            endpoints.append(True)
        else:
            raise ValueError("FMR active shape does not fit the shared device mesh")
    layout = TileLayout(mesh, tuple(widths), 4 + int(any(endpoints)), tuple(endpoints))
    layout.validate()
    return layout


def evolution_parameters(parameters, hierarchy):
    """Use the finest hierarchy spacing for MAD, and bulk equations on children."""
    finest_dx = min(p.dx for p in parameters)
    result = []
    for patch, params in enumerate(parameters):
        changes = dict(mad_q=(finest_dx / params.dx) ** 4 if hierarchy.use_mad else 1.0)
        if patch:
            changes.update({name: 0 for name in FACE_NAMES})
        result.append(params._replace(**changes))
    return tuple(result)


def validate_hierarchy(states, parameters, hierarchy, layouts):
    parents = hierarchy_parents(hierarchy)
    if not parameters or not len(parameters) == len(layouts) == len(parents):
        raise ValueError("FMR requires one parameter tuple and one layout per patch")
    if states is not None and len(states) != len(parents):
        raise ValueError("FMR requires one state per patch")
    mesh = layouts[0].mesh
    dt = float(parameters[0].dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("the shared FMR timestep must be finite and positive")
    root_codes = tuple(getattr(parameters[0], name) for name in FACE_NAMES)
    if any(code not in (0, 1) for code in root_codes):
        raise ValueError("FMR root boundaries support periodic and Sommerfeld codes")
    root_shape = layouts[0].active_shape
    for axis in range(3):
        if 1 in root_codes[2*axis:2*axis + 2]:
            accessible = min(root_shape[axis],
                             layouts[0].tile_shape[axis] + layouts[0].guard_cells)
            if accessible < 8:
                raise ValueError("Sommerfeld/KO closures require eight accessible inward samples")
    for child, (params, layout) in enumerate(zip(parameters, layouts)):
        if layout.mesh != mesh:
            raise ValueError("FMR patches must share the same device mesh and ordering")
        layout.validate()
        if layout.guard_cells < 4 + int(any(layout.upper_endpoints)):
            raise ValueError("FMR needs four exterior guards beyond every owned endpoint")
        expected = layout.counts + layout.allocated_shape
        if states is not None and any(field.shape[-6:] != expected for field in states[child]):
            raise ValueError("FMR state shape does not match its tiled layout")
        if not math.isclose(float(params.dt), dt, rel_tol=1.e-5, abs_tol=1.e-8):
            raise ValueError("stage-synchronous FMR requires identical timesteps")
        if child:
            parent = parents[child]
            spec = hierarchy.patches[child - 1]
            if layout.active_shape != fine_active_shape(spec):
                raise ValueError("FMR child active shape does not match its parent bounds")
            if not math.isclose(float(params.dx) * 2, float(parameters[parent].dx),
                                rel_tol=1.e-5, abs_tol=1.e-8):
                raise ValueError("successive FMR spacings must differ by exactly two")
            parent_shape = layouts[parent].active_shape
            if any(lo <= 0 or hi >= n - 1 for lo, hi, n in
                   zip(spec.coarse_lo, spec.coarse_hi, parent_shape)):
                raise ValueError("every child patch must be strictly inside its parent")
            for axis, name in enumerate(("x_min", "y_min", "z_min")):
                origin = getattr(parameters[parent], name) + spec.coarse_lo[axis] * parameters[parent].dx
                # Retain the serial geometry tolerance: legacy origins may
                # have been read from single-precision coordinate arrays.
                if not math.isclose(float(getattr(params, name)), float(origin),
                                    rel_tol=1.e-5, abs_tol=1.e-8):
                    raise ValueError("FMR child active origin is inconsistent with its parent")
            prolongation_routes(layouts[parent].geometry, layout.geometry, spec,
                                 parent == 0, tuple(getattr(parameters[parent], name) for name in FACE_NAMES))


def initialize_tiled_fmr(initial, root_params, root_shape, hierarchy, *, mesh=None):
    """Construct patches from a common ``initial(coordinates, params)`` callback.

    ``None`` initializes flat spacetime. A tuple of per-patch initial inputs
    also accepts the native arrays/mappings understood by initialize_tiled_bssn.
    Returned parameter origins refer to active samples, not padded allocations.
    The caller chooses the common timestep in ``root_params``.
    """
    mesh = make_field_mesh((1, 1, 1)) if mesh is None else mesh
    root_shape = tuple(root_shape)
    parents = hierarchy_parents(hierarchy)
    layouts = [patch_layout(root_shape, mesh)]
    reduced = (tuple(n == 1 for n in root_shape) if root_params.reduced_axes is None
               else root_params.reduced_axes)
    parameters = [root_params._replace(guard_cells=0, owned_shape=None, reduced_axes=reduced)]
    for child, spec in enumerate(hierarchy.patches, start=1):
        parent = parameters[parents[child]]
        origin = tuple(getattr(parent, name) + lo * parent.dx for name, lo in
                       zip(("x_min", "y_min", "z_min"), spec.coarse_lo))
        parameters.append(parent._replace(dx=parent.dx / 2, x_min=origin[0],
                                          y_min=origin[1], z_min=origin[2],
                                          **{name: 0 for name in FACE_NAMES}))
        layouts.append(patch_layout(fine_active_shape(spec), mesh))
    parameters = evolution_parameters(tuple(parameters), hierarchy)
    layouts = tuple(layouts)
    # Validate static nesting and all source stencils before allocating fields
    # or compiling communication kernels for an invalid hierarchy.
    validate_hierarchy(None, parameters, hierarchy, layouts)
    if isinstance(initial, (tuple, list)) and not isinstance(initial, BSSNVariables):
        inputs = tuple(initial)
        if len(inputs) != len(parameters):
            raise ValueError("one initial-data input is required per FMR patch")
    else:
        inputs = (initial,) * len(parameters)
    states = tuple(initialize_tiled_bssn(data, params, layout)
                   for data, params, layout in zip(inputs, parameters, layouts))
    validate_hierarchy(states, parameters, hierarchy, layouts)
    states = prepare_fmr_states(states, parameters, hierarchy, layouts=layouts, project=False)
    return states, parameters, layouts


def prepare_fmr_states(states, parameters, hierarchy, *, layouts, project=True):
    """Prepare owned geometry, then refill neighbors and parent-supplied ghosts.

    ``parameters`` must already carry the hierarchy-wide MAD weight and child
    face codes from :func:`evolution_parameters`.
    """
    parents = hierarchy_parents(hierarchy)
    prepared = []
    for child, (state, params, layout) in enumerate(zip(states, parameters, layouts)):
        ghost_fill = None
        if child:
            parent = parents[child]
            def ghost_fill(value, unused_params, parent=parent, child=child):
                return fill_tiled_fine_ghosts(
                    prepared[parent], value, hierarchy.patches[child - 1],
                    parent_layout=layouts[parent], child_layout=layouts[child],
                    parent_boundary_codes=tuple(getattr(parameters[parent], name) for name in FACE_NAMES),
                    parent_is_root=parent == 0)
        operation = prepare_tiled_state if project else refresh_bssn_halos
        prepared.append(operation(state, params, layout, ghost_fill=ghost_fill))
    return tuple(prepared)


def _stage(states, parameters, hierarchy, layouts):
    """One matching set of stage states and RHS values; ``parameters`` normalized."""
    states = prepare_fmr_states(states, parameters, hierarchy, layouts=layouts)
    rhs = tuple(map_tiled(compute_bssn_rhs, state, params, layout)
                for state, params, layout in zip(states, parameters, layouts))
    return states, rhs


def fmr_stage(states, parameters, hierarchy, *, layouts):
    """Matching stage states and RHS values, with no second halo exchange."""
    return _stage(states, evolution_parameters(parameters, hierarchy), hierarchy, layouts)


def fmr_rk4_step(states, parameters, hierarchy, *, layouts):
    """Advance all fixed patches using the same four RK stages and timestep."""
    states, parameters, layouts = tuple(states), tuple(parameters), tuple(layouts)
    validate_hierarchy(states, parameters, hierarchy, layouts)
    parameters = evolution_parameters(parameters, hierarchy)
    dt = parameters[0].dt
    s0, k1 = _stage(states, parameters, hierarchy, layouts)
    stage = jax.tree.map(lambda q, k: q + dt * k / 2, s0, k1)
    _, k2 = _stage(stage, parameters, hierarchy, layouts)
    stage = jax.tree.map(lambda q, k: q + dt * k / 2, s0, k2)
    _, k3 = _stage(stage, parameters, hierarchy, layouts)
    stage = jax.tree.map(lambda q, k: q + dt * k, s0, k3)
    _, k4 = _stage(stage, parameters, hierarchy, layouts)
    result = jax.tree.map(lambda q, a, b, c, d: q + dt * (a + 2*b + 2*c + d) / 6,
                          s0, k1, k2, k3, k4)
    result = [project_owned(state, params, layout)
              for state, params, layout in zip(result, parameters, layouts)]

    # Injection only reads owned vertices, so no intermediate halo exchange is
    # needed on the upward pass. Reproject each receiving parent's geometry.
    parents = hierarchy_parents(hierarchy)
    for child in range(len(result) - 1, 0, -1):
        parent = parents[child]
        result[parent] = restrict_tiled_to_coarse(
            result[parent], result[child], hierarchy.patches[child - 1],
            parent_layout=layouts[parent], child_layout=layouts[child])
        result[parent] = project_owned(result[parent], parameters[parent], layouts[parent])
    return prepare_fmr_states(tuple(result), parameters, hierarchy, layouts=layouts, project=False)


def _uncovered_mask(local_shape, p, layout, children):
    """Owned samples of this tile that no child patch of it also owns."""
    mask = owned_mask(local_shape, p)
    coordinates = tuple(jax.lax.axis_index(axis) * width + jnp.arange(n) - p.guard_cells
                        for axis, width, n in zip(MESH_AXES, layout.tile_shape, mask.shape))
    for spec in children:
        covered = jnp.ones(mask.shape, dtype=bool)
        for d, (indices, lo, hi) in enumerate(zip(coordinates, spec.coarse_lo, spec.coarse_hi)):
            shape = [1, 1, 1]
            shape[d] = indices.size
            covered &= ((indices >= lo) & (indices <= hi)).reshape(shape)
        mask &= ~covered
    return mask


def fmr_constraint_norms(states, parameters, hierarchy, *, layouts):
    """Composite point-count norms, excluding every child-covered parent point."""
    parameters = evolution_parameters(parameters, hierarchy)
    states = prepare_fmr_states(states, parameters, hierarchy, layouts=layouts, project=False)
    parents = hierarchy_parents(hierarchy)
    totals = []
    for patch, (state, params, layout) in enumerate(zip(states, parameters, layouts)):
        children = tuple(spec for child, spec in enumerate(hierarchy.patches, start=1)
                         if parents[child] == patch)

        def local_norms(local, params=params, layout=layout, children=children):
            local = jax.tree.map(lambda a: a.reshape(a.shape[:-6] + a.shape[-3:]), local)
            p = local_parameters(params, layout)
            mask = _uncovered_mask(local.conformal_factor.shape, p, layout, children)
            return constraint_statistics(compute_all_constraints(local, p), mask)

        totals.append(jax.shard_map(local_norms, mesh=layout.mesh,
                                    in_specs=(jax.tree.map(tile_spec, state),),
                                    out_specs=P(), check_vma=False)(state))
    return constraint_norms_from_totals(totals)
