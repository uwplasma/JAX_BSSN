"""PyPIC3D-compatible neighbor exchange for collocated Cartesian fields.

The mesh, slab exchange, and x -> y -> z sweep are adapted from
PyPIC3D/boundary_conditions/ghost_cells.py at
d91140bf489dda104e5b244f60c87db77edab4d7. Only owner-to-ghost transport is
copied: BSSN physical exteriors do not use electromagnetic wall parity.
"""

from functools import lru_cache
import math

import jax
import jax.numpy as jnp

from .layout import FACE_NAMES, MESH_AXES, TileLayout, VECTOR_TILE_SPEC


def make_field_mesh(tile_grid_shape):
    """Build a mesh with one tile per device, as in PyPIC3D."""
    tile_grid_shape = tuple(int(n) for n in tile_grid_shape)
    count = math.prod(tile_grid_shape)
    devices = jax.devices()
    if count > len(devices):
        raise ValueError(f"Tile topology {tile_grid_shape} requires {count} devices; "
                         f"JAX exposes {len(devices)}")
    return jax.make_mesh(tile_grid_shape, MESH_AXES, devices=devices[:count],
                         axis_types=(jax.sharding.AxisType.Auto,) * 3)


def axis_permutations(axis_size):
    """Return positive/negative neighbor sends, copied from PyPIC3D."""
    return (tuple((i, (i + 1) % axis_size) for i in range(axis_size)),
            tuple((i, (i - 1) % axis_size) for i in range(axis_size)))


def _plane(axis, index):
    result = [slice(None)] * 3
    result[axis] = index
    return tuple(result)


def _halo_slices(axis, g):
    return (_plane(axis, slice(0, g)), _plane(axis, slice(-g, None)),
            _plane(axis, slice(g, 2 * g)), _plane(axis, slice(-2 * g, -g)))


def _wrap_isolated_axis(tile, axis, width, g, endpoint):
    """Periodic self-exchange for an axis whose tiles all live on one device.

    A short patch can be narrower than its halo. Repeated wrapping reads only
    owned samples, even when a slab would span ghosts.
    """
    owned = width + int(endpoint)
    nodes = g + (jnp.arange(tile.shape[axis]) - g) % owned
    return jnp.take(tile, nodes, axis=axis)


def _exchange_axis_slabs(tile, axis, axis_name, count, width, g, endpoint):
    """One-hop owner-to-ghost slab transport along a distributed axis."""
    last = jax.lax.axis_index(axis_name) == count - 1
    lower, upper, low_owned, high_owned = _halo_slices(axis, g)
    positive, negative = axis_permutations(count)

    # Read BOTH sends before writing: source slabs can overlap for n < 2g.
    high_values = tile[high_owned]
    if endpoint:
        final_values = tile[_plane(axis, slice(width + 1, width + g + 1))]
        high_values = jnp.where(last, final_values, high_values)
    lower_values = jax.lax.ppermute(high_values, axis_name, positive)
    upper_values = jax.lax.ppermute(tile[low_owned], axis_name, negative)

    result = tile.at[lower].set(lower_values)
    if endpoint:
        # The final C endpoint occupies the first nominal upper-ghost slot.
        # ``tile`` is still the pre-exchange snapshot, and the lower write
        # above never reaches that slot, so this keeps the endpoint intact.
        shifted = jnp.concatenate((tile[_plane(axis, slice(g + width, g + width + 1))],
                                   upper_values[_plane(axis, slice(0, g - 1))]), axis=axis)
        upper_values = jnp.where(last, shifted, upper_values)
    return result.at[upper].set(upper_values)


def _fill_physical_exterior(tile, before, axis, side, count, width, g, endpoint,
                            exterior, code):
    """Close one physical face of this tile, discarding the wrapped send.

    The face test is per side, so mixed left/right boundary codes work too.
    A periodic face keeps its wrapped send and needs no conditional at all.
    """
    if not code:
        return tile
    index = jax.lax.axis_index(MESH_AXES[axis])
    on_face = index == (0 if side == 0 else count - 1)
    lower, upper, _, _ = _halo_slices(axis, g)
    target = lower if side == 0 else upper
    boundary = g if side == 0 else g + width - 1 + int(endpoint)
    nodes = jnp.arange(g) if side == 0 else jnp.arange(g + width, width + 2 * g)
    shape = [1, 1, 1]
    shape[axis] = g

    def fill(a):
        if exterior == "preserve":
            values = before[target]
        else:
            base = jnp.take(a, jnp.asarray([boundary]), axis=axis)
            if exterior == "constant":
                values = jnp.broadcast_to(base, a[target].shape)
            else:
                inward = boundary + (1 if side == 0 else -1)
                neighbor = jnp.take(a, jnp.asarray([inward]), axis=axis)
                slope = neighbor - base if side == 0 else base - neighbor
                values = base + (nodes - boundary).reshape(shape) * slope
        return a.at[target].set(values)

    return jax.lax.cond(on_face, fill, lambda a: a, tile)


def _sweep_tile(tile, boundary_codes, mesh_shape, tile_shape, g,
                upper_endpoints, exterior):
    """Exchange slabs in x/y/z order, carrying edges and corners with them."""
    for axis, axis_name in enumerate(MESH_AXES):
        width, count = tile_shape[axis], mesh_shape[axis]
        endpoint = upper_endpoints[axis]
        before = tile
        if count == 1:
            tile = _wrap_isolated_axis(tile, axis, width, g, endpoint)
        else:
            tile = _exchange_axis_slabs(tile, axis, axis_name, count, width, g, endpoint)
        for side in (0, 1):
            tile = _fill_physical_exterior(tile, before, axis, side, count, width, g,
                                           endpoint, exterior, boundary_codes[2 * axis + side])
    return tile


@lru_cache(maxsize=128)
def _halo_operator(mesh, geometry, boundary_conditions, exterior):
    """Compile one refresher per distinct mesh, geometry and face-code set."""
    mesh_shape, tile_shape, g, upper_endpoints = geometry
    allocated = tuple(n + 2 * g for n in tile_shape)

    def local_op(local):
        result = jax.vmap(lambda tile: _sweep_tile(
            tile, boundary_conditions, mesh_shape, tile_shape, g, upper_endpoints, exterior
        ))(local[:, 0, 0, 0])
        return result[:, None, None, None]

    sweep = jax.jit(jax.shard_map(local_op, mesh=mesh, in_specs=VECTOR_TILE_SPEC,
                                  out_specs=VECTOR_TILE_SPEC, check_vma=False))

    def run(values):
        if values.shape[-6:-3] != mesh_shape or values.shape[-3:] != allocated:
            raise ValueError("Field shape does not match the mesh, tile dimensions, and guard depth")
        leading = values.shape[:-6]
        stacked = values.reshape((math.prod(leading),) + values.shape[-6:])
        return sweep(stacked).reshape(leading + values.shape[-6:])

    return run


def make_tiled_halo_op(layout, boundary_conditions, *, exterior="linear"):
    """Make a scalar/vector/tensor refresher with PyPIC3D's tile layout.

    ``boundary_conditions`` contains six BSSN face codes (three paired codes
    are also accepted). ``exterior`` is linear, constant, or preserve. Leading
    component axes are flattened only for batched communication. Refreshers
    are cached, so a repeated layout reuses its compiled sweep. Face codes are
    static, like the mesh and the guard depth, so that cache key is stable.
    """
    layout.validate()
    if exterior not in ("linear", "constant", "preserve"):
        raise ValueError("Unknown physical exterior rule")
    boundary_conditions = tuple(int(code) for code in boundary_conditions)
    if len(boundary_conditions) == 3:
        boundary_conditions = tuple(bc for bc in boundary_conditions for _ in range(2))
    return _halo_operator(layout.mesh, layout.geometry, boundary_conditions, exterior)


def refresh_tiled_field(values, params, layout, *, exterior="linear"):
    """Refresh one collocated scalar or tensor using BSSN face codes."""
    bcs = tuple(getattr(params, name) for name in FACE_NAMES)
    return make_tiled_halo_op(layout, bcs, exterior=exterior)(values)
