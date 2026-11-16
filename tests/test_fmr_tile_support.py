"""Tiled ghost callbacks and short patches used by fixed mesh refinement."""

import unittest
from unittest.mock import Mock

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.evolution.spatial_derivatives import (
    diff1_physical, diff2_physical, diff1_upwind_physical, ko_dissipation,
)
from JAX_BSSN.fmr import FMRPatchSpec
from JAX_BSSN.fmr.transfer import fill_tiled_fine_ghosts
from JAX_BSSN.parallel import (
    make_field_mesh, make_tiled_halo_op, initialize_tiled_field,
    initialize_tiled_bssn, map_tiled, owned_mask, refresh_bssn_halos, prepare_tiled_state,
    compute_tiled_bssn_rhs, compute_tiled_bssn_rhs_with_matter,
    tiled_rk4_step, tiled_constraint_norms,
)


jax.config.update("jax_enable_x64", True)


class TestFMRTileSupport(unittest.TestCase):
    def setUp(self):
        self.mesh = make_field_mesh((1, 1, 1))

    def assert_close(self, actual, expected, atol=2.e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=2.e-12, atol=atol)),
                        f"maximum error {float(jnp.max(jnp.abs(actual-expected)))}")

    def test_short_single_tiles_repeat_owned_samples_through_every_guard(self):
        for width in (1, 3, 5):
            for endpoint in (False, True):
                with self.subTest(width=width, endpoint=endpoint):
                    owned = width + int(endpoint)
                    field = jnp.arange(owned, dtype=jnp.float64)[:, None, None]
                    actual = initialize_tiled_field(
                        field, BSSNParameters(), mesh=self.mesh,
                        tile_shape=(width, 1, 1), guard_cells=4,
                        upper_endpoints=(endpoint, False, False),
                    )
                    expected = ((jnp.arange(width + 8) - 4) % owned)
                    expected = jnp.broadcast_to(expected[:, None, None], (width + 8, 9, 9))
                    self.assert_close(actual[0, 0, 0], expected)

    def test_preserved_physical_exteriors_survive_short_axis_self_exchange(self):
        shape = (1, 1, 1, 11, 9, 13)
        field = jnp.arange(11 * 9 * 13, dtype=jnp.float64).reshape(shape)
        refresh = make_tiled_halo_op(
            self.mesh, (3, 1, 5), (1,) * 6, exterior="preserve",
        )
        self.assert_close(refresh(field), field)

    @unittest.skipUnless(jax.device_count() >= 2, "requires two logical devices")
    def test_short_axes_with_neighbor_tiles_still_require_a_full_send_slab(self):
        mesh = make_field_mesh((2, 1, 1))
        with self.assertRaisesRegex(ValueError, "at least as wide as its halo"):
            make_tiled_halo_op(mesh, (3, 1, 1), (0,) * 6)

    def test_resolved_singleton_derivatives_use_parent_supplied_samples(self):
        dx, g = .25, 4
        x = (jnp.arange(9, dtype=jnp.float64) - g) * dx
        field = jnp.broadcast_to((x**2 + 2*x)[:, None, None], (9, 9, 9))
        params = BSSNParameters(dx=dx, nu=1., guard_cells=g, owned_shape=(1, 1, 1),
                                reduced_axes=(False, True, True))
        self.assert_close(diff1_physical(field, 0, params)[g, g, g], 2.)
        self.assert_close(diff2_physical(field, 0, params)[g, g, g], 2.)
        self.assert_close(diff1_upwind_physical(field, jnp.ones_like(field), 0, params)[g, g, g], 2.)
        sixth_degree = jnp.broadcast_to(x[:, None, None]**6, field.shape)
        self.assert_close(ko_dissipation(sixth_degree, params)[g, g, g], 720 * dx**5 / 64)

        legacy = params._replace(reduced_axes=None)
        self.assert_close(diff1_physical(field, 0, legacy), jnp.zeros_like(field))
        self.assert_close(diff2_physical(field, 0, legacy), jnp.zeros_like(field))
        self.assert_close(ko_dissipation(sixth_degree, legacy), jnp.zeros_like(field))

    def test_local_parameters_distinguish_domain_and_patch_singletons(self):
        layout = dict(mesh=self.mesh, tile_shape=(1, 1, 1))
        field = initialize_tiled_field(False, BSSNParameters(), **layout)

        def reduced_axis(local, params):
            return jnp.full_like(local, params.reduced_axes[0])

        inferred = map_tiled(reduced_axis, field, BSSNParameters(), **layout)
        resolved = map_tiled(reduced_axis, field,
                             BSSNParameters(reduced_axes=(False, True, True)), **layout)
        self.assert_close(inferred, jnp.ones_like(field))
        self.assert_close(resolved, jnp.zeros_like(field))

    def test_mixed_root_faces_supply_periodic_stencils_from_the_requested_side(self):
        root_layout = dict(mesh=self.mesh, tile_shape=(16, 16, 16), guard_cells=4)
        child_layout = dict(mesh=self.mesh, tile_shape=(7, 7, 7), guard_cells=5)
        root_field = jnp.broadcast_to(jnp.arange(16, dtype=jnp.float64)[:, None, None]**3,
                                      (16, 16, 16))
        cases = ((0, 1, 1, 4, -.5), (1, 0, 11, 14, 16.5))
        for left, right, low, high, target in cases:
            with self.subTest(left=left, right=right):
                params = BSSNParameters(dx=1., xl_bc=left, xr_bc=right)
                parent = initialize_tiled_bssn(None, params, **root_layout)
                parent = parent._replace(lapse=initialize_tiled_field(root_field, params, **root_layout))
                child = initialize_tiled_bssn(None, BSSNParameters(dx=.5), **child_layout)
                spec = FMRPatchSpec((low, 5, 5), (high, 8, 8))
                filled = fill_tiled_fine_ghosts(
                    parent, child, spec, parent_layout=root_layout, child_layout=child_layout,
                    parent_boundary_codes=(left, right, 0, 0, 0, 0),
                )

                # Independent six-node Lagrange reference with only this
                # periodic face wrapping. The opposite physical halo differs.
                nodes = jnp.floor(target) + jnp.arange(-2, 4)
                weights = jnp.ones(6, dtype=jnp.float64)
                for k in range(6):
                    for m in range(6):
                        if k != m:
                            weights = weights.at[k].multiply((target - nodes[m]) / (nodes[k] - nodes[m]))
                expected = jnp.sum(weights * (nodes % 16)**3)
                index = 5 + int(2 * (target - low))
                self.assert_close(filled.lapse[0, 0, 0, index, 5, 5], expected)
                self.assert_close(filled.lapse[0, 0, 0, 5:12, 5:12, 5:12], jnp.ones((7, 7, 7)))

    def test_ghost_callback_supplies_rhs_stages_and_constraint_diagnostics(self):
        layout = dict(mesh=self.mesh, tile_shape=(1, 1, 1))
        params = BSSNParameters(dx=.1, dt=1.e-5, nu=0., kappa=0., g=0., eta=0.,
                                zero_shift=1, reduced_axes=(False, True, True))
        state = initialize_tiled_bssn(None, params, **layout)
        g = 4

        def quadratic_lapse(tiled, parameters):
            def fill(local, p):
                x = p.x_min + jnp.arange(local.lapse.shape[0]) * p.dx
                lapse = jnp.broadcast_to((1 + x*x)[:, None, None], local.lapse.shape)
                return local._replace(lapse=jnp.where(owned_mask(lapse.shape, p), local.lapse, lapse))
            return map_tiled(fill, tiled, parameters, **layout)

        for prepare in (refresh_bssn_halos, prepare_tiled_state):
            prepared = prepare(state, params, ghost_fill=quadratic_lapse, **layout)
            self.assert_close(prepared.lapse[0, 0, 0, :, g, g],
                              1 + ((jnp.arange(9) - g) * params.dx)**2)

        # On flat geometry with alpha=1+x^2, K_t=-Laplacian(alpha)=-2.
        vacuum = jax.jit(lambda q: compute_tiled_bssn_rhs(
            q, params, ghost_fill=quadratic_lapse, **layout))(state)
        matter = jax.jit(lambda q: compute_tiled_bssn_rhs_with_matter(
            q, params, jnp.zeros_like(q.lapse), jnp.zeros_like(q.shift),
            jnp.zeros_like(q.conformal_metric), ghost_fill=quadratic_lapse, **layout))(state)
        owned = (0, 0, 0, g, g, g)
        self.assert_close(vacuum.trace_K[owned], -2.)
        self.assert_close(matter.trace_K[owned], -2.)

        # Count callback applications during tracing, not device execution:
        # once per prepared stage and once for the final valid ghost shell.
        ghost_fill = Mock(side_effect=quadratic_lapse)
        evolved = jax.jit(lambda q: tiled_rk4_step(
            q, params, ghost_fill=ghost_fill, **layout))(state)
        self.assertEqual(ghost_fill.call_count, 5)
        self.assert_close(evolved.trace_K[owned] / params.dt, -2., atol=2.e-7)
        exterior = jnp.arange(9) != g
        self.assert_close(evolved.lapse[0, 0, 0, :, g, g][exterior],
                          (1 + ((jnp.arange(9) - g) * params.dx)**2)[exterior])

        def quadratic_conformal_factor(tiled, parameters):
            def fill(local, p):
                x = p.x_min + jnp.arange(local.lapse.shape[0]) * p.dx
                factor = jnp.broadcast_to((1 + x*x/8)[:, None, None], local.lapse.shape)
                return local._replace(conformal_factor=jnp.where(
                    owned_mask(factor.shape, p), local.conformal_factor, factor))
            return map_tiled(fill, tiled, parameters, **layout)

        # gamma_ij=W^-2 delta_ij gives R=4 W Laplacian(W)-6 |grad W|^2=1 at x=0.
        norms = tiled_constraint_norms(state, params,
                                        ghost_fill=quadratic_conformal_factor, **layout)
        self.assert_close(norms["hamiltonian_linf"], 1.)


if __name__ == "__main__":
    unittest.main()
