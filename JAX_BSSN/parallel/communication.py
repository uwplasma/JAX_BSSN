"""PyPIC3D-compatible neighbor exchange for collocated Cartesian fields.

The mesh, slab exchange, and x -> y -> z sweep are adapted from
PyPIC3D/boundary_conditions/ghost_cells.py at
d91140bf489dda104e5b244f60c87db77edab4d7. Only owner-to-ghost transport is
copied: BSSN physical exteriors do not use electromagnetic wall parity.
"""

import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

MESH_AXES = ("tile_x", "tile_y", "tile_z")
SCALAR_TILE_SPEC = P(*MESH_AXES, None, None, None)
VECTOR_TILE_SPEC = P(None, *MESH_AXES, None, None, None)


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


def tile_spec(value):
    """Replicate tensor indices and shard only the three tile indices."""
    return P(*(None,) * (value.ndim - 6), *MESH_AXES, None, None, None)


def axis_permutations(axis_size, boundary_condition=0):
    """Return positive/negative neighbor sends, copied from PyPIC3D."""
    if boundary_condition == 0:
        return (tuple((i, (i + 1) % axis_size) for i in range(axis_size)),
                tuple((i, (i - 1) % axis_size) for i in range(axis_size)))
    return (tuple((i, i + 1) for i in range(axis_size - 1)),
            tuple((i, i - 1) for i in range(1, axis_size)))


def _plane(axis, index):
    result = [slice(None)] * 3
    result[axis] = index
    return tuple(result)


def _halo_slices(axis, g):
    return (_plane(axis, slice(0, g)), _plane(axis, slice(-g, None)),
            _plane(axis, slice(g, 2 * g)), _plane(axis, slice(-2 * g, -g)))


def validate_layout(mesh, tile_shape, guard_cells, upper_endpoints):
    """Protect the single-hop slab stencil and the shared tile topology."""
    if tuple(mesh.axis_names) != MESH_AXES:
        raise ValueError(f"BSSN requires mesh axes {MESH_AXES}")
    if guard_cells < 1:
        raise ValueError("Tiled fields require at least one guard cell")
    for width, count, endpoint in zip(tile_shape, mesh.devices.shape, upper_endpoints):
        reduced = width == count == 1 and not endpoint
        if width < guard_cells and not reduced:
            raise ValueError("A resolved tile must be at least as wide as its halo")


def _sweep_tile(tile, boundary_codes, mesh_shape, tile_shape, g,
                upper_endpoints, exterior):
    """Exchange slabs in x/y/z order, carrying edges and corners with them."""
    for axis, axis_name in enumerate(MESH_AXES):
        width, count = tile_shape[axis], mesh_shape[axis]
        if width == count == 1 and not upper_endpoints[axis]:
            tile = jnp.broadcast_to(tile[_plane(axis, slice(g, g + 1))], tile.shape)
            continue

        index = jax.lax.axis_index(axis_name)
        last = index == count - 1
        endpoint = upper_endpoints[axis]
        lower, upper, low_owned, high_owned = _halo_slices(axis, g)
        positive, negative = axis_permutations(count)

        # Read BOTH sends before writing: source slabs can overlap for n<2g.
        high_values = tile[high_owned]
        if endpoint:
            final_values = tile[_plane(axis, slice(width + 1, width + g + 1))]
            high_values = jnp.where(last, final_values, high_values)
        lower_values = jax.lax.ppermute(high_values, axis_name, positive)
        upper_values = jax.lax.ppermute(tile[low_owned], axis_name, negative)
        before = tile
        tile = tile.at[lower].set(lower_values)
        if endpoint:
            # The final C endpoint occupies the first nominal upper-ghost slot.
            shifted = jnp.concatenate((before[_plane(axis, slice(g + width, g + width + 1))],
                                       upper_values[_plane(axis, slice(0, g - 1))]), axis=axis)
            upper_values = jnp.where(last, shifted, upper_values)
        tile = tile.at[upper].set(upper_values)

        # Face codes remain dynamic JAX values. Wrapped sends are discarded
        # only on physical faces, so mixed left/right boundary codes work too.
        for side in (0, 1):
            on_face = index == (0 if side == 0 else count - 1)
            physical = boundary_codes[2 * axis + side] != 0
            boundary = g if side == 0 else g + width - 1 + int(endpoint)
            target = lower if side == 0 else upper
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

            tile = jax.lax.cond(on_face & physical, fill, lambda a: a, tile)
    return tile


def make_tiled_halo_op(mesh, tile_shape, boundary_conditions, num_guard_cells=4,
                       *, upper_endpoints=(False, False, False), exterior="linear"):
    """Make a scalar/vector/tensor refresher with PyPIC3D's tile layout.

    ``boundary_conditions`` contains six BSSN face codes (three paired codes
    are also accepted). ``exterior`` is linear, constant, or preserve. Leading
    component axes are flattened only for batched communication.
    """
    tile_shape = tuple(tile_shape)
    upper_endpoints = tuple(upper_endpoints)
    g = int(num_guard_cells)
    validate_layout(mesh, tile_shape, g, upper_endpoints)
    if exterior not in ("linear", "constant", "preserve"):
        raise ValueError("Unknown physical exterior rule")
    if len(boundary_conditions) == 3:
        boundary_conditions = tuple(bc for bc in boundary_conditions for _ in range(2))
    mesh_shape = tuple(mesh.devices.shape)

    def run(values):
        if values.shape[-6:-3] != mesh_shape or values.shape[-3:] != tuple(n + 2*g for n in tile_shape):
            raise ValueError("Field shape does not match the mesh, tile dimensions, and guard depth")
        leading = values.shape[:-6]
        stacked = values.reshape((math.prod(leading),) + values.shape[-6:])

        def local_op(local):
            result = jax.vmap(lambda tile: _sweep_tile(
                tile, boundary_conditions, mesh_shape, tile_shape, g, upper_endpoints, exterior
            ))(local[:, 0, 0, 0])
            return result[:, None, None, None]

        result = jax.shard_map(local_op, mesh=mesh, in_specs=VECTOR_TILE_SPEC,
                               out_specs=VECTOR_TILE_SPEC, check_vma=False)(stacked)
        return result.reshape(leading + values.shape[-6:])

    return run


def refresh_tiled_field(values, params, *, mesh, tile_shape, guard_cells=4,
                        upper_endpoints=(False, False, False), exterior="linear"):
    """Refresh one collocated scalar or tensor using BSSN face codes."""
    bcs = tuple(getattr(params, name) for name in
                ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc"))
    return make_tiled_halo_op(mesh, tile_shape, bcs, guard_cells,
                              upper_endpoints=upper_endpoints, exterior=exterior)(values)
