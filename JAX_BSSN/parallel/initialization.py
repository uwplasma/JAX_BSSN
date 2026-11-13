"""Process-local construction of tiled initial data; no global device buffers."""

from collections.abc import Mapping
from os import PathLike

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from JAX_BSSN.bssn.variables import BSSNVariables
from JAX_BSSN.evolution.coordinates import grid_coordinates
from .communication import MESH_AXES, refresh_tiled_field, validate_layout


def initialize_tiled_field(initial, params, *, mesh, tile_shape, guard_cells=4,
                            upper_endpoints=(False, False, False), component_shape=(),
                            dtype=jnp.float64):
    """Load only addressable owned tiles, then fill their communication halos.

    ``initial`` is a native guard-free global array, a .npy filename, False
    (literal zeros), or ``initial(coordinates, local_params)``. Callback
    coordinates are broadcast Cartesian arrays on the OWNED samples, with
    local_params describing their origin/counts. NumPy is used only for host
    memory-mapped I/O and transfer buffers.
    """
    tile_shape = tuple(tile_shape)
    component_shape = tuple(component_shape)
    upper_endpoints = tuple(upper_endpoints)
    g = int(guard_cells)
    validate_layout(mesh, tile_shape, g, upper_endpoints)
    counts = tuple(mesh.devices.shape)
    physical_shape = tuple(n * count + int(end) for n, count, end in
                            zip(tile_shape, counts, upper_endpoints))
    face_names = (("xl_bc", "xr_bc"), ("yl_bc", "yr_bc"), ("zl_bc", "zr_bc"))
    for axis, names in enumerate(face_names):
        if any(int(getattr(params, name)) == 1 for name in names):
            if physical_shape[axis] == 1:
                raise ValueError("Reduced Cartesian axes must be periodic")
            accessible = min(physical_shape[axis], tile_shape[axis] + g)
            if accessible < 8:
                raise ValueError("Sommerfeld/KO closures require eight accessible inward samples")

    source = np.load(initial, mmap_mode="r", allow_pickle=False) if isinstance(initial, (str, PathLike)) else initial
    if source is not False and source is not None and not callable(source):
        if source.shape != component_shape + physical_shape:
            raise ValueError(f"Initial array has shape {source.shape}; expected "
                             f"{component_shape + physical_shape} on global C nodes")
    shape = component_shape + counts + tuple(n + 2*g for n in tile_shape)
    spec = P(*(None,) * len(component_shape), *MESH_AXES, None, None, None)
    sharding = NamedSharding(mesh, spec)
    origins = (params.x_min, params.y_min, params.z_min)

    def read_tile(index):
        tile = tuple(part.start or 0 for part in index[-6:-3])
        owned = tuple(n + int(end and i == count - 1)
                      for n, i, count, end in zip(tile_shape, tile, counts, upper_endpoints))
        offsets = tuple(i*n for i, n in zip(tile, tile_shape))
        if callable(source):
            minima = tuple(origin + offset * params.dx for origin, offset in zip(origins, offsets))
            faces = tuple(flag for i, count in zip(tile, counts) for flag in (i == 0, i == count - 1))
            p = params._replace(x_min=minima[0], y_min=minima[1], z_min=minima[2],
                                guard_cells=0, owned_shape=owned, owned_faces=faces)
            value = source(grid_coordinates(owned, p, dtype), p)
            if value.shape != component_shape + owned:
                raise ValueError("Initial-data callback must return the owned tile shape")
            value = np.asarray(jax.device_get(value), dtype=dtype)
        elif source is False or source is None:
            value = np.zeros(component_shape + owned, dtype=dtype)
        else:
            slices = tuple(slice(start, start+n) for start, n in zip(offsets, owned))
            value = np.asarray(source[(Ellipsis,) + slices], dtype=dtype)
        buffer = np.zeros(component_shape + (1, 1, 1) + tuple(n+2*g for n in tile_shape), dtype=dtype)
        interior = tuple(slice(g, g+n) for n in owned)
        buffer[(Ellipsis, 0, 0, 0) + interior] = value
        return buffer

    values = jax.make_array_from_callback(shape, sharding, read_tile, dtype=dtype)
    return refresh_tiled_field(values, params, mesh=mesh, tile_shape=tile_shape,
                                guard_cells=g, upper_endpoints=upper_endpoints)


def initialize_tiled_bssn(initial, params, *, mesh, tile_shape, guard_cells=4,
                           upper_endpoints=(False, False, False)):
    """Create a BSSNVariables tree from local callbacks or native initial data.

    None creates flat spacetime. A mapping uses BSSNVariables field names and
    native .npy filenames, with omitted/False entries remaining literal zero.
    Existing global BSSNVariables are supported for small-data compatibility.
    """
    # The additional owned C endpoint occupies the first nominal upper halo
    # slot. General MAD needs four samples beyond that endpoint on a periodic
    # axis; Sommerfeld axes use the existing shorter D4 stencil everywhere.
    face_names = (("xl_bc", "xr_bc"), ("yl_bc", "yr_bc"), ("zl_bc", "zr_bc"))
    if guard_cells < 5 and float(params.mad_q) != 1.0:
        for endpoint, names in zip(upper_endpoints, face_names):
            if endpoint and all(int(getattr(params, name)) != 1 for name in names):
                raise ValueError("General MAD with an owned periodic upper endpoint requires five guard cells")

    if initial is None:
        def initial(coordinates, p):
            one = jnp.ones(p.owned_shape, dtype=jnp.float64)
            zero = jnp.zeros_like(one)
            identity = jnp.eye(3)[:, :, None, None, None] * one
            vector = jnp.zeros((3,) + p.owned_shape, dtype=one.dtype)
            return BSSNVariables(identity, one, jnp.zeros_like(identity), zero, vector, one, vector)

    # Cache only owned LOCAL tiles, so a seven-leaf state callback runs once
    # per addressable tile rather than once per state component.
    cache = {}
    def component_callback(component):
        def evaluate(coordinates, p):
            key = (float(p.x_min), float(p.y_min), float(p.z_min))
            if key not in cache:
                # Do not accumulate all addressable tiles on the default GPU
                # while the seven component callbacks consume this cache.
                cache[key] = jax.tree.map(lambda value: np.asarray(jax.device_get(value)),
                                          initial(coordinates, p))
            return cache[key][component]
        return evaluate

    fields = []
    components = ((3, 3), (), (3, 3), (), (3,), (), (3,))
    for i, (name, leading) in enumerate(zip(BSSNVariables._fields, components)):
        source = (component_callback(i) if callable(initial) else
                  initial.get(name, False) if isinstance(initial, Mapping) else initial[i])
        fields.append(initialize_tiled_field(source, params, mesh=mesh, tile_shape=tile_shape,
                       guard_cells=guard_cells, upper_endpoints=upper_endpoints, component_shape=leading))
    return BSSNVariables(*fields)
