"""The tiled-patch layout value and the invariants every consumer shares.

A layout names the device mesh, the owned extent of one tile, and the guard
depth padding it.  Keeping it a single typed value removes the per-consumer
``layout.get("guard_cells", 4)`` defaults, which disagreed with
``fmr.tiled.patch_layout``'s ``4 + int(any(endpoints))`` for endpoint layouts.
"""

from typing import NamedTuple

from jax.sharding import PartitionSpec as P

MESH_AXES = ("tile_x", "tile_y", "tile_z")
SCALAR_TILE_SPEC = P(*MESH_AXES, None, None, None)
VECTOR_TILE_SPEC = P(None, *MESH_AXES, None, None, None)

FACE_NAMES = ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc")
FACE_NAME_PAIRS = (("xl_bc", "xr_bc"), ("yl_bc", "yr_bc"), ("zl_bc", "zr_bc"))

# Keep this module independent of the BSSN boundary modules, as
# ``evolution.derivatives`` does.  Boundary code 1 selects Sommerfeld.
SOMMERFELD_BC = 1


def tile_spec(value):
    """Replicate tensor indices and shard only the three tile indices."""
    return P(*(None,) * (value.ndim - 6), *MESH_AXES, None, None, None)


class TileLayout(NamedTuple):
    """One patch's device mesh, owned tile extent and guard depth.

    ``upper_endpoints`` marks an axis whose active extent is one sample longer
    than ``tile_shape * tiles``; the last tile owns that extra C endpoint, and
    it occupies the first nominal upper-guard slot rather than a slot of its
    own.  Everything downstream derives its geometry from these four fields.
    """

    mesh: object
    tile_shape: tuple
    guard_cells: int = 4
    upper_endpoints: tuple = (False, False, False)

    @property
    def counts(self):
        """Tiles per axis, in mesh order."""
        return tuple(int(n) for n in self.mesh.devices.shape)

    @property
    def geometry(self):
        """Hashable mesh-independent key for the cached transfer routes.

        Two layouts with equal geometry share route tables even when their
        mesh objects differ, which is what the previous ``layout_geometry``
        helper provided.  Coercion here keeps the key canonical.
        """
        return (self.counts, tuple(int(n) for n in self.tile_shape),
                int(self.guard_cells),
                tuple(bool(end) for end in self.upper_endpoints))

    @property
    def active_shape(self):
        """Physical samples per axis, including an owned upper endpoint."""
        return tuple(width * count + int(end) for width, count, end
                     in zip(self.tile_shape, self.counts, self.upper_endpoints))

    @property
    def allocated_shape(self):
        """Padded per-tile extent, counting both guard slabs."""
        return tuple(width + 2 * self.guard_cells for width in self.tile_shape)

    def validate(self):
        """Protect the single-hop slab stencil and the shared tile topology."""
        if tuple(self.mesh.axis_names) != MESH_AXES:
            raise ValueError(f"BSSN requires mesh axes {MESH_AXES}")
        if self.guard_cells < 1:
            raise ValueError("Tiled fields require at least one guard cell")
        for width, count, endpoint in zip(self.tile_shape, self.counts,
                                          self.upper_endpoints):
            if width < 1:
                raise ValueError("Tiled fields require at least one owned sample per axis")
            if count > 1 and width < self.guard_cells:
                raise ValueError("A tile with neighbors must be at least as wide as its halo")
            if endpoint and self.guard_cells < 2:
                raise ValueError("An owned upper endpoint consumes a guard slot, "
                                 "so its axis needs at least two")


def require_bssn_guards(layout, params):
    """Guard depth the MAD/KO stencils need, including owned upper endpoints.

    Shared by field construction and by every halo refresh, so a state built
    with one ``mad_q`` and evolved with another cannot slip past the rule.
    """
    if layout.guard_cells < 4:
        raise ValueError("Tiled BSSN evolution requires four guards for the supported MAD stencils")
    if layout.guard_cells >= 5 or float(params.mad_q) == 1.0:
        return
    # The additional owned C endpoint occupies the first nominal upper halo
    # slot.  General MAD needs four samples beyond that endpoint on a periodic
    # axis; Sommerfeld axes use the existing shorter D4 stencil everywhere.
    for endpoint, names in zip(layout.upper_endpoints, FACE_NAME_PAIRS):
        if endpoint and all(int(getattr(params, name)) != SOMMERFELD_BC for name in names):
            raise ValueError("General MAD with an owned periodic upper endpoint "
                             "requires five guard cells")
