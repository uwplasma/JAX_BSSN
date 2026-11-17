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

from .layout import FACE_NAMES, MESH_AXES, VECTOR_TILE_SPEC


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


def _plane(axis, index):
    result = [slice(None)] * 3
    result[axis] = index
    return tuple(result)


def _sweep_tile(tile, boundary_codes, mesh_shape, tile_shape, g,
                upper_endpoints, exterior):
    """Exchange slabs in x/y/z order, carrying edges and corners with them."""
    for axis, axis_name in enumerate(MESH_AXES):
        width, count = tile_shape[axis], mesh_shape[axis]
        endpoint = upper_endpoints[axis]
        lower, upper = _plane(axis, slice(0, g)), _plane(axis, slice(-g, None))
        before = tile

        if count == 1:
            # All of this axis' tiles live on one device, so periodicity is a
            # self-copy. A short patch can be narrower than its halo; repeated
            # wrapping reads only owned samples, even when a slab would span
            # ghosts.
            owned = width + int(endpoint)
            nodes = g + (jnp.arange(tile.shape[axis]) - g) % owned
            tile = jnp.take(tile, nodes, axis=axis)
        else:
            # One-hop owner-to-ghost slab transport. The neighbor permutations
            # are copied from PyPIC3D.
            last = jax.lax.axis_index(axis_name) == count - 1
            positive = tuple((i, (i + 1) % count) for i in range(count))
            negative = tuple((i, (i - 1) % count) for i in range(count))

            # Read BOTH sends before writing: source slabs can overlap for n < 2g.
            high_values = tile[_plane(axis, slice(-2 * g, -g))]
            if endpoint:
                final_values = tile[_plane(axis, slice(width + 1, width + g + 1))]
                high_values = jnp.where(last, final_values, high_values)
            lower_values = jax.lax.ppermute(high_values, axis_name, positive)
            upper_values = jax.lax.ppermute(tile[_plane(axis, slice(g, 2 * g))],
                                            axis_name, negative)
            if endpoint:
                # The final C endpoint occupies the first nominal upper-ghost
                # slot. ``tile`` is still the pre-exchange snapshot here, and
                # the lower write below never reaches that slot, so shifting
                # the received slab past it keeps the endpoint intact.
                shifted = jnp.concatenate((tile[_plane(axis, slice(g + width, g + width + 1))],
                                           upper_values[_plane(axis, slice(0, g - 1))]), axis=axis)
                upper_values = jnp.where(last, shifted, upper_values)
            tile = tile.at[lower].set(lower_values).at[upper].set(upper_values)

        # Close each physical face, discarding the wrapped send. The code is
        # static, so a periodic face emits no conditional at all, and the test
        # is per side, so mixed left/right boundary codes work too.
        for side in (0, 1):
            if not boundary_codes[2 * axis + side]:
                continue
            on_face = jax.lax.axis_index(axis_name) == (0 if side == 0 else count - 1)
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

            tile = jax.lax.cond(on_face, fill, lambda a: a, tile)
    return tile


@lru_cache(maxsize=128)
def make_tiled_halo_op(layout, boundary_conditions, *, exterior="linear"):
    """Make a scalar/vector/tensor refresher with PyPIC3D's tile layout.

    ``boundary_conditions`` is the six BSSN face codes as hashable ints, and
    ``exterior`` is linear, constant, or preserve. Every argument is static,
    so the compiled sweep is built once and a repeated layout reuses it rather
    than re-tracing. Leading component axes are flattened only for batched
    communication, which lets one refresher serve scalars, vectors and tensors.
    """
    layout.validate()
    if exterior not in ("linear", "constant", "preserve"):
        raise ValueError("Unknown physical exterior rule")
    counts, tile_shape, g, upper_endpoints = layout.geometry
    allocated = layout.allocated_shape

    def local_op(local):
        result = jax.vmap(lambda tile: _sweep_tile(
            tile, boundary_conditions, counts, tile_shape, g, upper_endpoints, exterior
        ))(local[:, 0, 0, 0])
        return result[:, None, None, None]

    sweep = jax.jit(jax.shard_map(local_op, mesh=layout.mesh, in_specs=VECTOR_TILE_SPEC,
                                  out_specs=VECTOR_TILE_SPEC, check_vma=False))

    def refresh(values):
        if values.shape[-6:-3] != counts or values.shape[-3:] != allocated:
            raise ValueError("Field shape does not match the mesh, tile dimensions, and guard depth")
        leading = values.shape[:-6]
        stacked = values.reshape((math.prod(leading),) + values.shape[-6:])
        return sweep(stacked).reshape(leading + values.shape[-6:])

    return refresh


def refresh_tiled_field(values, params, layout, *, exterior="linear"):
    """Refresh one collocated scalar or tensor using BSSN face codes."""
    bcs = tuple(int(getattr(params, name)) for name in FACE_NAMES)
    return make_tiled_halo_op(layout, bcs, exterior=exterior)(values)
