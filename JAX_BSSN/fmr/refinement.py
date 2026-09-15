"""Stage-synchronous fixed mesh refinement for one nested patch per level.

The hierarchy is vertex centred, has a fixed 2:1 refinement ratio and four
guard cells on every non-root level.  All levels advance with the finest-grid
timestep.  Spatial prolongation is degree five and restriction is point
injection at coincident vertices.
"""

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs,
    eliminate_trace_A,
    enforce_unit_determinant_conformal_metric,
)


class FMRPatchSpec(NamedTuple):
    """Inclusive parent-active-index bounds for one child patch."""

    coarse_lo: tuple[int, int, int]
    coarse_hi: tuple[int, int, int]
    refinement_ratio: int = 2
    ghost_width: int = 4


class FMRHierarchySpec(NamedTuple):
    """Ordered parent-to-child patch chain and hierarchy-wide MAD choice."""

    patches: tuple[FMRPatchSpec, ...]
    use_mad: bool = True


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
    """Validate static hierarchy geometry before compiling an evolution step."""
    states, parameters = tuple(states), tuple(parameters)
    patches = tuple(hierarchy.patches)
    if not states:
        raise ValueError("an FMR hierarchy requires at least one level")
    if len(parameters) != len(states):
        raise ValueError("FMR states and parameters must have equal lengths")
    if len(patches) != len(states) - 1:
        raise ValueError("an FMR hierarchy requires one patch per child level")
    for spec in patches:
        _validate_spec(spec)

    dt = float(parameters[0].dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("the shared FMR timestep must be finite and positive")
    for level, (state, params) in enumerate(zip(states, parameters)):
        if not np.isclose(float(params.dt), dt):
            raise ValueError("stage-synchronous FMR requires identical timesteps")
        shapes = {tuple(field.shape[-3:]) for field in state}
        if len(shapes) != 1:
            raise ValueError(f"level {level} fields have inconsistent spatial shapes")
        if level:
            parent_params = parameters[level - 1]
            if not np.isclose(float(params.dx) * 2.0, float(parent_params.dx)):
                raise ValueError("successive FMR spacings must differ by exactly two")
            spec = patches[level - 1]
            expected = tuple(n + 2 * spec.ghost_width
                             for n in fine_active_shape(spec))
            if _spatial_shape(state) != expected:
                raise ValueError(
                    f"level {level} allocation shape {_spatial_shape(state)} "
                    f"does not match expected {expected}"
                )

    for child_level, spec in enumerate(patches, start=1):
        parent_active = (_spatial_shape(states[0]) if child_level == 1
                         else fine_active_shape(patches[child_level - 2]))
        if any(lo <= 0 or hi >= size - 1 for lo, hi, size in zip(
            spec.coarse_lo, spec.coarse_hi, parent_active
        )):
            raise ValueError("every child patch must be strictly inside its parent")
        parent_ghost = 0 if child_level == 1 else patches[child_level - 2].ghost_width
        _interpolation_matrices(
            states[child_level - 1][0], spec, (parent_ghost,) * 3,
            child_level == 1,
        )
        parent_params, child_params = parameters[child_level - 1:child_level + 1]
        parent_active_origin = np.asarray(
            (parent_params.x_min, parent_params.y_min, parent_params.z_min),
            dtype=float,
        ) + parent_ghost * float(parent_params.dx)
        expected_child_origin = (
            parent_active_origin
            + np.asarray(spec.coarse_lo) * float(parent_params.dx)
            - spec.ghost_width * float(child_params.dx)
        )
        actual_child_origin = np.asarray(
            (child_params.x_min, child_params.y_min, child_params.z_min), dtype=float
        )
        if not np.allclose(actual_child_origin, expected_child_origin):
            raise ValueError(f"level {child_level} parameter origin is inconsistent")


def _project(variables):
    return eliminate_trace_A(enforce_unit_determinant_conformal_metric(variables))


def _add(variables, rhs, scale):
    return BSSNVariables(*(x + scale*y for x, y in zip(variables, rhs)))


def _fill_hierarchy(states, hierarchy, matrices):
    states = list(states)
    for child_level, (spec, interpolation) in enumerate(
        zip(hierarchy.patches, matrices), start=1
    ):
        states[child_level] = _fill_fine_ghosts_with_matrices(
            states[child_level - 1], states[child_level], spec, interpolation
        )
    return tuple(states)


def _stage(states, parameters, hierarchy, matrices):
    states = tuple(_project(state) for state in states)
    states = _fill_hierarchy(states, hierarchy, matrices)
    return states, tuple(compute_bssn_rhs(state, params)
                         for state, params in zip(states, parameters))


def fmr_rk4_step(states, parameters, hierarchy: FMRHierarchySpec):
    """Advance an ordered nested hierarchy through one synchronized RK4 step."""
    states, parameters = tuple(states), tuple(parameters)
    validate_hierarchy(states, parameters, hierarchy)
    finest_dx = parameters[-1].dx
    evolved_parameters = tuple(
        params._replace(mad_q=(finest_dx / params.dx) ** 4
                        if hierarchy.use_mad and level < len(parameters) - 1
                        else 1.0)
        for level, params in enumerate(parameters)
    )
    matrices = tuple(
        _interpolation_matrices(
            states[level][0], spec,
            ((0,) * 3 if level == 0
             else (hierarchy.patches[level - 1].ghost_width,) * 3),
            level == 0,
        )
        for level, spec in enumerate(hierarchy.patches)
    )
    dt = parameters[0].dt
    s0, k1 = _stage(states, evolved_parameters, hierarchy, matrices)
    _, k2 = _stage(tuple(_add(v, k, dt/2) for v, k in zip(s0, k1)),
                   evolved_parameters, hierarchy, matrices)
    _, k3 = _stage(tuple(_add(v, k, dt/2) for v, k in zip(s0, k2)),
                   evolved_parameters, hierarchy, matrices)
    _, k4 = _stage(tuple(_add(v, k, dt) for v, k in zip(s0, k3)),
                   evolved_parameters, hierarchy, matrices)
    result = [
        BSSNVariables(*(x + dt*(a + 2*b + 2*c + d)/6
                        for x, a, b, c, d in zip(v, r1, r2, r3, r4)))
        for v, r1, r2, r3, r4 in zip(s0, k1, k2, k3, k4)
    ]
    result = [_project(state) for state in result]
    for child_level in range(len(result) - 1, 0, -1):
        parent_ghost = (0 if child_level == 1
                        else hierarchy.patches[child_level - 2].ghost_width)
        result[child_level - 1] = _project(restrict_to_coarse(
            result[child_level - 1], result[child_level],
            hierarchy.patches[child_level - 1],
            parent_ghost_width=parent_ghost,
        ))
    return _fill_hierarchy(tuple(result), hierarchy, matrices)
