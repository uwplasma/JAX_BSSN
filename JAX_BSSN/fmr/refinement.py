"""Vertex-centred fixed mesh refinement geometry and patch-tree ancestry.

The hierarchy is vertex centred with a fixed 2:1 refinement ratio, so a patch
owns both of its inclusive parent bounds. Tiled allocations reserve four
exterior guards plus one more slot when an upper endpoint is owned. Evolution,
prolongation and injection live in ``tiled`` and ``transfer``.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


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
