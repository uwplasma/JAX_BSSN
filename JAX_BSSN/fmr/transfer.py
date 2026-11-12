"""Static patch routes and bounded, device-to-device FMR transfers.

NumPy constructs integer geometry and interpolation coefficients on the host.
Evolved data stay tiled: each collective transports one stencil window, never
a whole patch. The six-node interpolation is separable in Cartesian axes.
"""

from functools import lru_cache
from itertools import product
import math

import jax
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.parallel.communication import MESH_AXES, VECTOR_TILE_SPEC


def layout_geometry(layout):
    return (tuple(layout["mesh"].devices.shape), tuple(layout["tile_shape"]),
            int(layout.get("guard_cells", 4)),
            tuple(layout.get("upper_endpoints", (False,) * 3)))


def active_shape(layout):
    counts, widths, _, endpoints = layout_geometry(layout)
    return tuple(n * c + int(e) for n, c, e in zip(widths, counts, endpoints))


def pack_state(state):
    spatial = state.conformal_factor.shape
    return jnp.concatenate(tuple(field.reshape((-1,) + spatial) for field in state))


def unpack_state(packed, template):
    fields, offset = [], 0
    for field in template:
        count = math.prod(field.shape[:-6])
        fields.append(packed[offset:offset + count].reshape(field.shape))
        offset += count
    return type(template)(*fields)


def tile_number(index, counts):
    return (index[0] * counts[1] + index[1]) * counts[2] + index[2]


def local_tile_number(counts):
    return tile_number(tuple(jax.lax.axis_index(axis) for axis in MESH_AXES), counts)


def transfer_rounds(edges):
    """Batch equally shaped windows with one send/receive per device."""
    groups = {}
    for edge in edges:
        key = (edge["window_shape"], edge["target_shape"])
        rounds = groups.setdefault(key, [])
        for batch in rounds:
            if all(edge["source"] != old["source"] and
                   edge["destination"] != old["destination"] for old in batch):
                batch.append(edge)
                break
        else:
            rounds.append([edge])
    return tuple(tuple(batch) for rounds in groups.values() for batch in rounds)


def interpolation_segments(tile, axis, parent_geometry, child_geometry,
                           spec, root, boundary_codes):
    """Split an axis where source ownership or exterior-shell status changes."""
    pc, pw, pg, pe = parent_geometry
    cc, cw, cg, ce = child_geometry
    parent_size = pw[axis] * pc[axis] + int(pe[axis])
    child_size = cw[axis] * cc[axis] + int(ce[axis])
    segments, current = [], []
    previous = None
    for local in range(cw[axis] + 2 * cg):
        fine = tile * cw[axis] + local - cg
        target = spec.coarse_lo[axis] + fine / spec.refinement_ratio
        base = math.floor(target)
        if root:
            # Each physical face is independent, including mixed face codes.
            if base - 2 < 0 and boundary_codes[2 * axis] != 0:
                raise ValueError("FMR interpolation stencil crosses a physical root boundary")
            if base + 3 >= parent_size and boundary_codes[2 * axis + 1] != 0:
                raise ValueError("FMR interpolation stencil crosses a physical root boundary")
        # Read an exterior stencil from that side's refreshed halo. Moving a
        # periodic target to the opposite owner is wrong for mixed root faces:
        # the opposite exterior may obey a physical boundary condition.
        owner = min(max(base // pw[axis], 0), pc[axis] - 1)
        local_target = target - owner * pw[axis] + pg
        low, high = math.floor(local_target) - 2, math.floor(local_target) + 3
        if low < 0 or high >= pw[axis] + 2 * pg:
            raise ValueError("child prolongation stencil exceeds parent allocation")
        exterior = -1 if fine < 0 else (1 if fine >= child_size else 0)
        key = (owner, exterior)
        if current and (key != previous[0] or not math.isclose(target - previous[1], .5)):
            segments.append(current)
            current = []
        current.append((local, local_target, owner, exterior))
        previous = (key, target)
    if current:
        segments.append(current)
    return segments


@lru_cache(maxsize=128)
def prolongation_routes(parent_geometry, child_geometry, spec, root, boundary_codes):
    counts = parent_geometry[0]
    if counts != child_geometry[0]:
        raise ValueError("FMR patches must use the same tile topology")
    edges = []
    for tile in product(*(range(n) for n in counts)):
        axes = [interpolation_segments(tile[d], d, parent_geometry, child_geometry,
                                       spec, root, boundary_codes) for d in range(3)]
        for segments in product(*axes):
            if not any(segment[0][3] for segment in segments):
                continue
            source = tuple(segment[0][2] for segment in segments)
            starts, shapes, indices, weights = [], [], [], []
            for segment in segments:
                targets = np.asarray([point[1] for point in segment])
                nodes = np.floor(targets).astype(np.int32)[:, None] + np.arange(-2, 4)
                start = int(nodes.min())
                starts.append(start)
                shapes.append(int(nodes.max()) - start + 1)
                indices.append(nodes - start)
                coefficient = np.ones(nodes.shape)
                for k in range(6):
                    for m in range(6):
                        if k != m:
                            coefficient[:, k] *= (targets - nodes[:, m]) / (nodes[:, k] - nodes[:, m])
                weights.append(coefficient)
            edges.append(dict(source=tile_number(source, counts),
                              destination=tile_number(tile, counts),
                              source_start=tuple(starts), window_shape=tuple(shapes),
                              target_start=tuple(segment[0][0] for segment in segments),
                              target_shape=tuple(len(segment) for segment in segments),
                              indices=tuple(indices), weights=tuple(weights)))
    return transfer_rounds(edges)


@lru_cache(maxsize=128)
def restriction_routes(parent_geometry, child_geometry, spec):
    counts, pw, pg, _ = parent_geometry
    _, cw, cg, ce = child_geometry
    edges = []
    for tile in product(*(range(n) for n in counts)):
        axes = []
        for d in range(3):
            owned = cw[d] + int(ce[d] and tile[d] == counts[d] - 1)
            groups = {}
            for local in range(owned):
                fine = tile[d] * cw[d] + local
                if fine % spec.refinement_ratio:
                    continue
                parent = spec.coarse_lo[d] + fine // spec.refinement_ratio
                owner = min(parent // pw[d], counts[d] - 1)
                groups.setdefault(owner, []).append((local + cg, parent - owner * pw[d] + pg))
            axes.append(tuple(groups.items()))
        for segments in product(*axes):
            shape = tuple(len(points) for _, points in segments)
            edges.append(dict(source=tile_number(tile, counts),
                              destination=tile_number(tuple(owner for owner, _ in segments), counts),
                              window_shape=shape, target_shape=shape,
                              target_start=tuple(points[0][1] for _, points in segments),
                              indices=tuple(np.asarray([p[0] for p in points], dtype=np.int32)
                                            for _, points in segments)))
    return transfer_rounds(edges)


def route_metadata(batch, counts, interpolate):
    """Small replicated geometry tables; no field values are replicated."""
    size = math.prod(counts)
    source_starts = np.zeros((size, 3), dtype=np.int32)
    target_starts = np.zeros((size, 3), dtype=np.int32)
    destination = np.zeros(size, dtype=bool)
    shape = batch[0]["target_shape"]
    indices = [np.zeros((size, n, 6) if interpolate else (size, n), dtype=np.int32)
               for n in shape]
    weights = [np.zeros((size, n, 6)) for n in shape] if interpolate else []
    for edge in batch:
        source, dest = edge["source"], edge["destination"]
        if interpolate:
            source_starts[source] = edge["source_start"]
        destination[dest] = True
        target_starts[dest] = edge["target_start"]
        for d in range(3):
            indices[d][dest if interpolate else source] = edge["indices"][d]
            if interpolate:
                weights[d][dest] = edge["weights"][d]
    return source_starts, target_starts, destination, indices, weights


@lru_cache(maxsize=128)
def transfer_operator(parent_geometry, child_geometry, spec, mesh, interpolate,
                      root=True, boundary_codes=(0,) * 6):
    routes = (prolongation_routes(parent_geometry, child_geometry, spec, root, boundary_codes)
              if interpolate else restriction_routes(parent_geometry, child_geometry, spec))
    counts = tuple(mesh.devices.shape)
    metadata = tuple(route_metadata(batch, counts, interpolate) for batch in routes)

    def local_transfer(src, dst):
        src, dst = src[:, 0, 0, 0], dst[:, 0, 0, 0]
        rank = local_tile_number(counts)
        for batch, tables in zip(routes, metadata):
            starts, targets, receives, indices, weights = tables
            if interpolate:
                start = jnp.asarray(starts)[rank]
                window = jax.lax.dynamic_slice(src, (jnp.int32(0), start[0], start[1], start[2]),
                                              (src.shape[0],) + batch[0]["window_shape"])
            else:
                window = src
                for d in range(3):
                    window = jnp.take(window, jnp.asarray(indices[d])[rank], axis=d + 1)
            permutation = tuple((edge["source"], edge["destination"]) for edge in batch)
            window = jax.lax.ppermute(window, MESH_AXES, permutation)
            if interpolate:
                for d in range(3):
                    window = jnp.take(window, jnp.asarray(indices[d])[rank], axis=d + 1)
                    weight = jnp.asarray(weights[d], dtype=src.dtype)[rank]
                    shape = [1] * window.ndim
                    shape[d + 1:d + 3] = weight.shape
                    window = jnp.sum(window * weight.reshape(shape), axis=d + 2)
            start = jnp.asarray(targets)[rank]
            updated = jax.lax.dynamic_update_slice(dst, window, (jnp.int32(0), start[0], start[1], start[2]))
            dst = jnp.where(jnp.asarray(receives)[rank], updated, dst)
        return dst[:, None, None, None]

    return jax.jit(jax.shard_map(local_transfer, mesh=mesh,
                                in_specs=(VECTOR_TILE_SPEC, VECTOR_TILE_SPEC),
                                out_specs=VECTOR_TILE_SPEC, check_vma=False))


def fill_tiled_fine_ghosts(parent, child, spec, *, parent_layout, child_layout,
                           parent_boundary_codes=(0,) * 6, parent_is_root=True):
    """Replace only the child's exterior shell from prepared parent data."""
    transfer = transfer_operator(layout_geometry(parent_layout), layout_geometry(child_layout),
                                 spec, child_layout["mesh"], True,
                                 parent_is_root, tuple(parent_boundary_codes))
    supplied = transfer(pack_state(parent), pack_state(child))
    return unpack_state(supplied, child)


def restrict_tiled_to_coarse(parent, child, spec, *, parent_layout, child_layout):
    """Inject owned coincident child vertices into their unique parent owner."""
    transfer = transfer_operator(layout_geometry(parent_layout), layout_geometry(child_layout),
                                 spec, parent_layout["mesh"], False)
    injected = transfer(pack_state(child), pack_state(parent))
    return unpack_state(injected, parent)
