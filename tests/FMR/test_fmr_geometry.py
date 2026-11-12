"""Hierarchy geometry and composite, uncovered-point constraint reductions."""

import unittest

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.fmr import (
    FMRPatchSpec, FMRHierarchySpec, hierarchy_parents, hierarchy_depths,
    fine_active_shape, fine_active_slice, fine_active_view,
    initialize_tiled_fmr, fmr_constraint_norms,
)
from JAX_BSSN.fmr.tiled import patch_layout, validate_hierarchy
from JAX_BSSN.parallel import make_field_mesh

jax.config.update("jax_enable_x64", True)


class TestFMRGeometry(unittest.TestCase):
    def test_patches_are_vertex_centered_with_inclusive_bounds(self):
        # Both parent bounds are owned, so a 2:1 patch has an odd active extent.
        spec = FMRPatchSpec((4, 4, 4), (9, 9, 9))
        self.assertEqual(fine_active_shape(spec), (11, 11, 11))
        self.assertEqual(fine_active_slice(spec),
                         (slice(4, 15), slice(4, 15), slice(4, 15)))
        padded = jnp.arange(19.**3).reshape((19,) * 3)
        view = fine_active_view(padded, spec)
        self.assertEqual(view.shape, (11, 11, 11))
        self.assertTrue(bool(jnp.array_equal(view, padded[4:15, 4:15, 4:15])))
        # Leading tensor axes pass through untouched.
        tensor = jnp.broadcast_to(padded, (3, 3) + padded.shape)
        self.assertEqual(fine_active_view(tensor, spec).shape, (3, 3, 11, 11, 11))

    def test_explicit_ancestry_and_inclusive_sibling_ownership(self):
        left = FMRPatchSpec((2, 2, 2), (6, 6, 6), parent=0)
        right = FMRPatchSpec((7, 2, 2), (11, 6, 6), parent=0)
        grandchild = FMRPatchSpec((2, 2, 2), (4, 4, 4), parent=1)
        hierarchy = FMRHierarchySpec((left, grandchild, right))
        self.assertEqual(hierarchy_parents(hierarchy), (-1, 0, 1, 0))
        self.assertEqual(hierarchy_depths(hierarchy), (0, 1, 2, 1))
        with self.assertRaisesRegex(ValueError, "share owned vertices"):
            hierarchy_parents(FMRHierarchySpec((left, right._replace(coarse_lo=(6, 2, 2)))))
        with self.assertRaisesRegex(ValueError, "follow their parent"):
            hierarchy_parents(FMRHierarchySpec((left._replace(parent=1),)))

    @unittest.skipUnless(jax.device_count() >= 4, "requires four logical devices")
    def test_patch_extent_must_fit_without_silent_padding(self):
        mesh = make_field_mesh((4, 1, 1))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            patch_layout((23, 9, 9), mesh)
        layout = patch_layout((25, 9, 9), mesh)
        self.assertEqual(layout["tile_shape"], (6, 9, 9))
        self.assertEqual(layout["upper_endpoints"], (True, False, False))
        self.assertEqual(layout["guard_cells"], 5)

    def test_sommerfeld_root_requires_valid_interpolation_stencils(self):
        params = BSSNParameters(xl_bc=1, xr_bc=1)
        hierarchy = FMRHierarchySpec((FMRPatchSpec((1, 3, 3), (5, 7, 7)),))
        with self.assertRaisesRegex(ValueError, "physical root boundary"):
            initialize_tiled_fmr(None, params, (16, 12, 12), hierarchy)

    def test_legacy_coordinate_roundoff_does_not_change_geometry_acceptance(self):
        mesh = make_field_mesh((1, 1, 1))
        root = BSSNParameters(dx=.1, dt=.001, x_min=-.45, y_min=-.45, z_min=-.45)
        spec = FMRPatchSpec((3, 3, 3), (8, 8, 8))
        hierarchy = FMRHierarchySpec((spec,))
        # Serial callers commonly recover their padded origin from a saved
        # float32 coordinate, then add the four guards when adapting it.
        origin = float(jnp.float32(-.35)) + 4 * .05
        child = root._replace(dx=.05, dt=float(jnp.float32(.001)),
                              x_min=origin, y_min=origin, z_min=origin)
        layouts = (patch_layout((12, 12, 12), mesh), patch_layout((11, 11, 11), mesh))
        validate_hierarchy(None, (root, child), hierarchy, layouts)
        with self.assertRaisesRegex(ValueError, "origin is inconsistent"):
            validate_hierarchy(None, (root, child._replace(x_min=origin + .001)),
                                     hierarchy, layouts)

    def test_composite_constraints_exclude_all_covered_parent_vertices(self):
        hierarchy = FMRHierarchySpec((
            FMRPatchSpec((3, 3, 3), (7, 7, 7), parent=0),
            FMRPatchSpec((2, 2, 2), (4, 4, 4), parent=1),
            FMRPatchSpec((10, 3, 3), (14, 7, 7), parent=0),
        ))
        states, params, layouts = initialize_tiled_fmr(
            None, BSSNParameters(dx=.1), (18, 12, 12), hierarchy)
        curvatures = (1., 2., 4., 3.)
        states = tuple(state._replace(trace_K=jnp.full_like(state.trace_K, K))
                       for state, K in zip(states, curvatures))
        norms = jax.jit(lambda values: fmr_constraint_norms(
            values, params, hierarchy, layouts=layouts))(states)
        # Flat metric and A_ij=0 give H=2 K^2/3 at every owned point.
        counts = jnp.asarray((18*12*12 - 2*5**3, 9**3 - 3**3, 5**3, 9**3))
        hamiltonian = 2*jnp.asarray(curvatures)**2/3
        expected = jnp.sqrt(jnp.sum(counts*hamiltonian**2)/jnp.sum(counts))
        self.assertTrue(bool(jnp.isclose(norms["hamiltonian_l2"], expected, rtol=2e-13, atol=2e-13)))
        self.assertTrue(bool(jnp.isclose(norms["hamiltonian_linf"], jnp.max(hamiltonian), rtol=2e-13)))
        self.assertLess(float(norms["det_gamma_linf"]), 1e-13)
        self.assertLess(float(norms["trace_A_linf"]), 1e-13)


if __name__ == "__main__":
    unittest.main()
