"""Locate puncture candidates from the lapse and coordinate shift divergence."""

from functools import partial

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import get_boundary_codes
from JAX_BSSN.evolution.spatial_derivatives import diff1_physical


@partial(jax.jit, static_argnames="max_punctures")
def identify_punctures(state, lapse_threshold, evolution_parameters, *, max_punctures):
    """Return minimum-lapse positions in positive-divergence low-lapse regions.

    The state must use global unpadded Cartesian arrays. Regions connect through
    shared faces, including opposite faces only when both are periodic. The
    returned positions have shape (max_punctures, 3), with NaN padding. Count
    includes all regions, so count > max_punctures reports truncation. Region
    order and equal-lapse ties use the smallest flattened grid index.

    These candidates do not establish the presence of an apparent horizon.
    """
    if not isinstance(max_punctures, int) or max_punctures < 1:
        raise ValueError("max_punctures must be a positive static integer")
    lapse, shift = state.lapse, state.shift
    if lapse.ndim != 3 or shift.shape != (3,) + lapse.shape:
        raise ValueError("Puncture detection requires global unpadded BSSN arrays, not tiled states")

    # Qualifying nodes: a collapsing lapse together with outward coordinate flow.
    div_shift = jnp.zeros_like(lapse)
    for axis in range(3):
        # A singleton global axis has no variation, including at open faces.
        if lapse.shape[axis] > 1:
            div_shift = div_shift + diff1_physical(shift[axis], axis, evolution_parameters)
    qualifying = (lapse < lapse_threshold) & (div_shift > 0.)

    # Wrapping across an axis requires both of its faces to be periodic.
    periodic = []
    for axis in range(3):
        left_bc, right_bc = get_boundary_codes(evolution_parameters, axis)
        periodic.append((left_bc == 0) & (right_bc == 0))

    # Label face-connected candidates. Background uses a sentinel larger than
    # every grid index. A component's smallest index propagates only through
    # qualifying face neighbors.
    size = lapse.size
    grid_index = jnp.arange(size, dtype=jnp.int32).reshape(lapse.shape)
    labels = jnp.where(qualifying, grid_index, size)

    def propagate(loop_state):
        labels, _ = loop_state
        updated = labels
        for axis in range(3):
            lower = jnp.roll(labels, 1, axis=axis)
            upper = jnp.roll(labels, -1, axis=axis)
            lower_face = (slice(None),) * axis + (0,) + (slice(None),) * (2 - axis)
            upper_face = (slice(None),) * axis + (-1,) + (slice(None),) * (2 - axis)
            lower = lower.at[lower_face].set(jnp.where(periodic[axis], lower[lower_face], size))
            upper = upper.at[upper_face].set(jnp.where(periodic[axis], upper[upper_face], size))
            updated = jnp.minimum(updated, jnp.minimum(lower, upper))
        updated = jnp.where(qualifying, updated, size)
        return updated, jnp.any(updated != labels)

    labels, _ = jax.lax.while_loop(lambda loop_state: loop_state[1], propagate,
                                  (labels, jnp.asarray(True)))

    # Select each region's minimum lapse, then the first index attaining it. The
    # extra reduction slot belongs to background and to padded output rows.
    flat_labels = labels.ravel()
    minimum_lapse = jnp.full((size + 1,), jnp.inf, dtype=lapse.dtype)
    minimum_lapse = minimum_lapse.at[flat_labels].min(
        jnp.where(qualifying, lapse, jnp.inf).ravel())
    at_minimum = qualifying.ravel() & (lapse.ravel() == minimum_lapse[flat_labels])
    representative = jnp.full((size + 1,), size, dtype=jnp.int32)
    representative = representative.at[flat_labels].min(
        jnp.where(at_minimum, grid_index.ravel(), size))

    # One root per region, carrying that region's label, ordered by grid index.
    roots = qualifying & (labels == grid_index)
    count = jnp.sum(roots, dtype=jnp.int32)
    root_indices = jnp.nonzero(roots.ravel(), size=max_punctures, fill_value=size)[0]
    selected = representative[root_indices]
    valid = selected < size

    # Convert the retained representatives to physical coordinates, padding the
    # unused output rows with NaN.
    indices = jnp.stack(jnp.unravel_index(jnp.where(valid, selected, 0), lapse.shape), axis=-1)
    origin = jnp.asarray((evolution_parameters.x_min, evolution_parameters.y_min,
                          evolution_parameters.z_min))
    positions = origin + evolution_parameters.dx * indices
    return jnp.where(valid[:, None], positions, jnp.nan), count
