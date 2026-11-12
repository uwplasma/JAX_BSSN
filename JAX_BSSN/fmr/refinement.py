"""Vertex-centred fixed mesh refinement and serial compatibility helpers.

The hierarchy is vertex centred with a fixed 2:1 refinement ratio. Legacy
child arrays retain four symmetric guards; tiled arrays reserve an additional
slot when an upper endpoint is owned. All levels advance with the common
finest-grid timestep. Spatial prolongation is degree five and restriction is
point injection at coincident vertices.
"""

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn.variables import BSSNVariables


class FMRPatchSpec(NamedTuple):
    """Inclusive parent-active-index bounds for one child patch."""

    coarse_lo: tuple[int, int, int]
    coarse_hi: tuple[int, int, int]
    refinement_ratio: int = 2
    ghost_width: int = 4
    parent: int | None = None


class FMRHierarchySpec(NamedTuple):
    """Parent-before-child patches and hierarchy-wide MAD choice."""

    patches: tuple[FMRPatchSpec, ...]
    use_mad: bool = True


def hierarchy_parents(hierarchy):
    """Flat parent indices, with omitted parents retaining the old chain."""
    parents = [-1]
    for child, spec in enumerate(hierarchy.patches, start=1):
        parent = child - 1 if spec.parent is None else spec.parent
        if not isinstance(parent, (int, np.integer)) or not 0 <= parent < child:
            raise ValueError("FMR patches must follow their parent in the hierarchy")
        parents.append(int(parent))
    for child, spec in enumerate(hierarchy.patches, start=1):
        for other in range(1, child):
            sibling = hierarchy.patches[other - 1]
            if parents[child] == parents[other] and all(
                max(a, b) <= min(c, d) for a, b, c, d in zip(
                    spec.coarse_lo, sibling.coarse_lo,
                    spec.coarse_hi, sibling.coarse_hi)
            ):
                raise ValueError("sibling FMR patches must not share owned vertices")
    return tuple(parents)


def hierarchy_depths(hierarchy):
    """Refinement depth of each flat patch, starting with the root."""
    depths = [0]
    for parent in hierarchy_parents(hierarchy)[1:]:
        depths.append(depths[parent] + 1)
    return tuple(depths)


def _validate_spec(spec: FMRPatchSpec) -> None:
    if len(spec.coarse_lo) != 3 or len(spec.coarse_hi) != 3:
        raise ValueError("FMR patch bounds must be three-dimensional")
    if spec.refinement_ratio != 2:
        raise ValueError("FMR refinement ratio must be 2:1")
    if spec.ghost_width != 4:
        raise ValueError("the BSSN FMR configuration requires four guard cells")
    if any(h < l for l, h in zip(spec.coarse_lo, spec.coarse_hi)):
        raise ValueError("coarse_hi must not precede coarse_lo")


def fine_active_shape(spec: FMRPatchSpec) -> tuple[int, int, int]:
    _validate_spec(spec)
    ratio = spec.refinement_ratio
    return tuple(ratio * (h - l) + 1
                 for l, h in zip(spec.coarse_lo, spec.coarse_hi))


def fine_active_slice(spec: FMRPatchSpec) -> tuple[slice, slice, slice]:
    g = spec.ghost_width
    return tuple(slice(g, g + n) for n in fine_active_shape(spec))


def fine_active_view(array: jnp.ndarray, spec: FMRPatchSpec) -> jnp.ndarray:
    return array[(...,) + fine_active_slice(spec)]


def fine_coordinates(spec: FMRPatchSpec, dx_coarse: float,
                     origin=(0.0, 0.0, 0.0)):
    """Coordinates of a padded child allocation from its parent active origin."""
    _validate_spec(spec)
    g = spec.ghost_width
    dtype = jnp.result_type(dx_coarse, 1.0)
    spacing = jnp.asarray(dx_coarse, dtype=dtype)
    ratio = jnp.asarray(spec.refinement_ratio, dtype=dtype)
    axes = [
        jnp.asarray(origin[d], dtype=dtype)
        + (jnp.asarray(spec.coarse_lo[d], dtype=dtype)
           + (jnp.arange(fine_active_shape(spec)[d] + 2*g, dtype=dtype)
              - jnp.asarray(g, dtype=dtype)) / ratio) * spacing
        for d in range(3)
    ]
    return jnp.meshgrid(*axes, indexing="ij")


@lru_cache(maxsize=None)
def _cached_interpolation_matrices(parent_shape, spec, parent_offset,
                                   periodic, dtype_name):
    dtype = np.dtype(dtype_name)
    offsets = np.arange(-2, 4, dtype=np.int32)
    matrices = []
    for direction in range(3):
        target_count = fine_active_shape(spec)[direction] + 2 * spec.ghost_width
        targets = (
            dtype.type(parent_offset[direction] + spec.coarse_lo[direction])
            + (np.arange(target_count, dtype=dtype)
               - dtype.type(spec.ghost_width))
            / dtype.type(spec.refinement_ratio)
        )
        base = np.floor(targets).astype(np.int32)
        nodes = base[:, None] + offsets[None, :]
        if not periodic and (
            np.min(nodes) < 0 or np.max(nodes) >= parent_shape[direction]
        ):
            raise ValueError("child prolongation stencil exceeds parent allocation")
        weights = np.ones(nodes.shape, dtype=dtype)
        for k in range(6):
            for m in range(6):
                if k != m:
                    weights[:, k] *= ((targets - nodes[:, m])
                                      / (nodes[:, k] - nodes[:, m]))
        matrix = np.zeros((target_count, parent_shape[direction]), dtype=dtype)
        source_nodes = nodes % parent_shape[direction] if periodic else nodes
        for target_index in range(target_count):
            for stencil_index in range(6):
                matrix[target_index, source_nodes[target_index, stencil_index]] += (
                    weights[target_index, stencil_index]
                )
        matrices.append(matrix)
    return tuple(matrices)


def _interpolation_matrices(parent, spec, parent_offset=(0, 0, 0),
                            periodic=True):
    host = _cached_interpolation_matrices(
        tuple(parent.shape[-3:]), spec, tuple(parent_offset), bool(periodic),
        str(parent.dtype),
    )
    return tuple(jnp.asarray(matrix, dtype=parent.dtype) for matrix in host)


def _prolongate_with_matrices(parent, matrices):
    matrix_x, matrix_y, matrix_z = matrices
    child = jnp.einsum("ia,...abc->...ibc", matrix_x, parent)
    child = jnp.einsum("jb,...ibc->...ijc", matrix_y, child)
    return jnp.einsum("kc,...ijc->...ijk", matrix_z, child)


def prolongate_to_fine(parent: jnp.ndarray, spec: FMRPatchSpec, *,
                       parent_ghost_width: int = 0,
                       periodic_parent: bool | None = None) -> jnp.ndarray:
    """Prolongate a parent allocation while preserving arbitrary leading axes."""
    if periodic_parent is None:
        periodic_parent = parent_ghost_width == 0
    offset = (int(parent_ghost_width),) * 3
    matrices = _interpolation_matrices(parent, spec, offset, periodic_parent)
    return _prolongate_with_matrices(parent, matrices)


def _fill_fine_ghosts_with_matrices(parent_vars, child_vars, spec, matrices):
    active = fine_active_slice(spec)
    fields = []
    for parent, child in zip(parent_vars, child_vars):
        supplied = _prolongate_with_matrices(parent, matrices)
        supplied = supplied.at[(...,) + active].set(child[(...,) + active])
        fields.append(supplied)
    return BSSNVariables(*fields)


def fill_fine_ghosts(parent_vars, child_vars, spec, *,
                     parent_ghost_width: int = 0,
                     periodic_parent: bool | None = None):
    """Refresh one child's guards without changing its active data."""
    if periodic_parent is None:
        periodic_parent = parent_ghost_width == 0
    matrices = _interpolation_matrices(
        parent_vars[0], spec, (int(parent_ghost_width),) * 3, periodic_parent
    )
    return _fill_fine_ghosts_with_matrices(parent_vars, child_vars, spec, matrices)


def restrict_to_coarse(parent_vars, child_vars, spec, *,
                       parent_ghost_width: int = 0):
    """Inject coincident active child vertices into its parent allocation."""
    offset = int(parent_ghost_width)
    covered = tuple(slice(offset + l, offset + h + 1)
                    for l, h in zip(spec.coarse_lo, spec.coarse_hi))
    stride = spec.refinement_ratio
    fields = []
    for parent, child in zip(parent_vars, child_vars):
        injected = fine_active_view(child, spec)[
            (..., slice(None, None, stride), slice(None, None, stride),
             slice(None, None, stride))
        ]
        fields.append(parent.at[(...,) + covered].set(injected))
    return BSSNVariables(*fields)


def _spatial_shape(variables):
    return tuple(int(value) for value in variables[0].shape[-3:])


def validate_hierarchy(states, parameters, hierarchy: FMRHierarchySpec) -> None:
    """Validate the legacy bare-root/symmetric-four-guard array convention."""
    parents = hierarchy_parents(hierarchy)
    if not states or not len(states) == len(parameters) == len(parents):
        raise ValueError("FMR states, parameters, and hierarchy must have equal lengths")
    for child, (state, params) in enumerate(zip(states, parameters)):
        if len({tuple(field.shape[-3:]) for field in state}) != 1:
            raise ValueError("FMR fields have inconsistent spatial shapes")
        if child:
            spec = hierarchy.patches[child - 1]
            expected = tuple(n + 2 * spec.ghost_width for n in fine_active_shape(spec))
            if _spatial_shape(state) != expected:
                raise ValueError("FMR child allocation does not match its patch specification")


def fmr_rk4_step(states, parameters, hierarchy: FMRHierarchySpec):
    """Advance legacy serial arrays through the common tiled FMR implementation.

    The root has no ghosts; children have four symmetric ghosts and parameter
    origins at the start of their padded allocations. Adapters use only JAX
    array operations, so existing jitted callers retain their calling convention.
    """
    import jax
    from JAX_BSSN.parallel import make_field_mesh
    from .tiled import tiled_fmr_rk4_step

    states, parameters = tuple(states), tuple(parameters)
    validate_hierarchy(states, parameters, hierarchy)
    mesh = make_field_mesh((1, 1, 1))
    layouts, tiled, active_parameters = [], [], []
    reduced = (tuple(n == 1 for n in _spatial_shape(states[0]))
               if parameters[0].reduced_axes is None else parameters[0].reduced_axes)
    for child, (state, params) in enumerate(zip(states, parameters)):
        shape = _spatial_shape(state) if child == 0 else fine_active_shape(hierarchy.patches[child - 1])
        layouts.append(dict(mesh=mesh, tile_shape=shape, guard_cells=4,
                            upper_endpoints=(False,) * 3))
        offset = 0 if child == 0 else 4 * params.dx
        active_parameters.append(params._replace(
            x_min=params.x_min + offset, y_min=params.y_min + offset,
            z_min=params.z_min + offset, guard_cells=0, owned_shape=None,
            reduced_axes=reduced,
            **({name: 0 for name in ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc")}
               if child else {})))
        if child == 0:
            state = jax.tree.map(lambda a: jnp.pad(a, ((0, 0),) * (a.ndim - 3) + ((4, 4),) * 3), state)
        tiled.append(jax.tree.map(lambda a: a.reshape(a.shape[:-3] + (1, 1, 1) + a.shape[-3:]), state))
    result = tiled_fmr_rk4_step(tuple(tiled), tuple(active_parameters), hierarchy, layouts=tuple(layouts))
    serial = []
    for child, state in enumerate(result):
        state = jax.tree.map(lambda a: a.reshape(a.shape[:-6] + a.shape[-3:]), state)
        if child == 0:
            state = jax.tree.map(lambda a: a[(..., slice(4, -4), slice(4, -4), slice(4, -4))], state)
        serial.append(state)
    return tuple(serial)


# Keep the historical module import usable while exposing the tiled path too.
from .transfer import fill_tiled_fine_ghosts, restrict_tiled_to_coarse
from .tiled import (initialize_tiled_fmr, prepare_fmr_states, tiled_fmr_stage,
                    tiled_fmr_rk4_step, fmr_constraint_norms)
