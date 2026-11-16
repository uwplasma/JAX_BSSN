"""Padded tile stencils retain the serial Cartesian boundary equations."""

import unittest

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.boundaries import sommerfeld, sommerfeld_boundary_mask
from JAX_BSSN.evolution.derivatives import (
    diff1_field, diff1_upwind_field, diff2_field, diff6_field,
)
from JAX_BSSN.evolution.spatial_derivatives import (
    diff1_physical, diff1_upwind_physical, diff2_physical, ko_dissipation,
)


jax.config.update("jax_enable_x64", True)


class TestTiledBoundaries(unittest.TestCase):
    def setUp(self):
        self.guard = 4
        self.shape = (25, 9, 8)
        self.params = BSSNParameters(
            dx=0.2, nu=0.03, mad_q=0.35,
            xl_bc=1, xr_bc=1, zl_bc=1, zr_bc=1,
            x_min=-2.4, y_min=-0.8, z_min=-0.7,
        )
        x = self.params.x_min + self.params.dx * jnp.arange(self.shape[0])
        y = 2 * jnp.pi * jnp.arange(self.shape[1]) / self.shape[1]
        z = self.params.z_min + self.params.dx * jnp.arange(self.shape[2])
        self.field = (
            jnp.exp(0.2 * x[:, None, None])
            * (1 + 0.1 * jnp.cos(y[None, :, None]))
            + 0.07 * z[None, None, :]**4
            + 0.05 * x[:, None, None] * jnp.sin(y[None, :, None])
        )

    def assert_fields_close(self, actual, expected, atol=2.0e-11):
        error = float(jnp.max(jnp.abs(actual - expected)))
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=2.0e-12, atol=atol)), error)

    def padded_tile(self, field, tile_x, params=None):
        """Sample full corner halos, putting sentinels outside physical walls."""

        params = self.params if params is None else params
        start = (8 * tile_x, 0, 0)
        count = (9 if tile_x == 2 else 8, self.shape[1], self.shape[2])
        result = field
        codes = ((params.xl_bc, params.xr_bc), (params.yl_bc, params.yr_bc),
                 (params.zl_bc, params.zr_bc))
        for d in range(3):
            # The final conducting C endpoint occupies the first upper guard
            # slot in PyPIC3D's fixed-width tile allocation.
            width = 8 if d == 0 and params.xr_bc == 1 else count[d]
            indices = start[d] + jnp.arange(-self.guard, width + self.guard)
            axis = result.ndim - 3 + d
            result = jnp.take(result, indices % self.shape[d], axis=axis)
            outside = ((indices < 0) & (codes[d][0] == 1)) | (
                (indices >= self.shape[d]) & (codes[d][1] == 1)
            )
            mask_shape = [1] * result.ndim
            mask_shape[axis] = len(indices)
            result = jnp.where(outside.reshape(mask_shape), -12345.0, result)

        local_params = params._replace(
            guard_cells=self.guard, owned_shape=count,
            owned_faces=(tile_x == 0, tile_x == 2, True, True, True, True),
            x_min=params.x_min + (start[0] - self.guard) * params.dx,
            y_min=params.y_min - self.guard * params.dx,
            z_min=params.z_min - self.guard * params.dx,
        )
        owned = (...,) + tuple(slice(self.guard, self.guard + n) for n in count)
        global_owned = (...,) + tuple(slice(s, s + n) for s, n in zip(start, count))
        return result, local_params, owned, global_owned

    def check_stencils_match_serial(self, field, tiles):
        coefficient = jnp.where(self.field > 1, 0.3, -0.2)
        for tile_x in tiles:
            local, params, owned, global_owned = self.padded_tile(field, tile_x)
            local_coefficient, _, _, _ = self.padded_tile(coefficient, tile_x)
            for d in range(3):
                axis = field.ndim - 3 + d
                for derivative in (diff1_physical, diff2_physical):
                    self.assert_fields_close(
                        derivative(local, axis, params)[owned],
                        derivative(field, axis, self.params)[global_owned],
                    )
                self.assert_fields_close(
                    diff1_upwind_physical(local, local_coefficient, axis, params)[owned],
                    diff1_upwind_physical(field, coefficient, axis, self.params)[global_owned],
                )
            self.assert_fields_close(
                ko_dissipation(local, params)[owned],
                ko_dissipation(field, self.params)[global_owned],
            )

    def test_scalar_stencils_match_serial_on_every_tile(self):
        self.check_stencils_match_serial(self.field, range(3))

    def test_leading_tensor_axes_are_differenced_componentwise(self):
        """Axis plumbing only; the tile sweep above carries the numerical load.

        Each component gets a distinct scale, so a component mix-up in the local
        path shows up as a mismatch; broadcasting one field would hide it.
        """
        for leading_shape in ((3,), (3, 3)):
            scales = 1.0 + jnp.arange(3**len(leading_shape), dtype=jnp.float64)
            field = scales.reshape(leading_shape + (1, 1, 1)) * self.field
            self.check_stencils_match_serial(field, (1,))

    def test_mixed_derivatives_use_corner_halos_without_an_intermediate_exchange(self):
        for tile_x in range(3):
            local, params, owned, global_owned = self.padded_tile(self.field, tile_x)
            for first, second in ((0, 1), (1, 0), (0, 2), (2, 0)):
                expected = diff1_physical(
                    diff1_physical(self.field, first, self.params), second, self.params
                )[global_owned]
                actual = diff1_physical(
                    diff1_physical(local, first, params), second, params
                )[owned]
                self.assert_fields_close(actual, expected)

    def test_middle_tile_retains_global_sommerfeld_d4_policy(self):
        local, params, owned, global_owned = self.padded_tile(self.field, 1)
        actual = diff1_physical(local, 0, params)[owned]
        expected = diff1_field(self.field, 0, self.params.dx, 1, 1)[global_owned]
        self.assert_fields_close(actual, expected)
        local_mad = diff1_field(local, 0, params.dx, mad_q=params.mad_q)[owned]
        self.assertGreater(float(jnp.max(jnp.abs(actual - local_mad))), 1.0e-9)

    def test_periodic_mad_stencils_keep_the_requested_weight(self):
        params = self.params._replace(xl_bc=0, xr_bc=0, zl_bc=0, zr_bc=0)
        for tile_x in range(3):
            local, local_params, owned, global_owned = self.padded_tile(
                self.field, tile_x, params
            )
            self.assert_fields_close(
                diff1_upwind_physical(local, 0.4, 0, local_params)[owned],
                diff1_upwind_physical(self.field, 0.4, 0, params)[global_owned],
            )

    def test_sommerfeld_uses_global_coordinates_and_owned_endpoint(self):
        rhs = jnp.full_like(self.field, 7.0)
        expected = sommerfeld(self.field, rhs, jnp.asarray(0.5), self.params)
        for tile_x in range(3):
            local, params, owned, global_owned = self.padded_tile(self.field, tile_x)
            local_rhs = jnp.full_like(local, 7.0)
            actual = sommerfeld(local, local_rhs, jnp.asarray(0.5), params)
            self.assert_fields_close(actual[owned], expected[global_owned])
            mask = sommerfeld_boundary_mask(local.shape, params)
            owned_mask = jnp.zeros(local.shape, dtype=bool).at[owned].set(True)
            self.assertFalse(bool(jnp.any(mask & ~owned_mask)))
            self.assertTrue(bool(jnp.all(actual[~owned_mask] == 7.0)))
            if tile_x == 2:
                self.assertTrue(bool(mask[self.guard + 8, self.guard + 4, self.guard + 4]))

    def test_reduced_axes_have_zero_derivatives_and_no_spurious_faces(self):
        field = jnp.broadcast_to(self.field[:, :1, :], (25, 1, 8))
        padded = jnp.pad(field, ((4, 4), (4, 4), (4, 4)), mode="edge")
        params = self.params._replace(
            guard_cells=4, owned_shape=(25, 1, 8), yl_bc=1, yr_bc=1,
        )
        for derivative in (diff1_physical, diff2_physical):
            self.assertTrue(bool(jnp.all(derivative(padded, 1, params) == 0.0)))
        self.assertTrue(bool(jnp.all(diff1_upwind_physical(padded, 0.3, 1, params) == 0.0)))
        mask = sommerfeld_boundary_mask(padded.shape, params)
        self.assertFalse(bool(mask[10, 4, 7]))

    def test_default_derivative_bounds_are_exactly_unchanged(self):
        for derivative in (diff1_field, diff2_field, diff6_field):
            original = derivative(self.field, 0, self.params.dx, 1, 1)
            explicit = derivative(
                self.field, 0, self.params.dx, 1, 1,
                interior_start=0, interior_end=self.shape[0],
            )
            self.assertTrue(bool(jnp.array_equal(original, explicit)))
        original = diff1_upwind_field(self.field, 0.3, 0, self.params.dx, 1, 1)
        explicit = diff1_upwind_field(
            self.field, 0.3, 0, self.params.dx, 1, 1,
            interior_start=0, interior_end=self.shape[0],
        )
        self.assertTrue(bool(jnp.array_equal(original, explicit)))

    def test_initializer_rejects_sommerfeld_on_a_reduced_axis(self):
        from JAX_BSSN.parallel import initialize_tiled_bssn, make_field_mesh

        params = BSSNParameters(yl_bc=1, yr_bc=1)
        with self.assertRaisesRegex(ValueError, "Reduced Cartesian axes must be periodic"):
            initialize_tiled_bssn(
                None, params, mesh=make_field_mesh((1, 1, 1)), tile_shape=(8, 1, 1)
            )

    def test_periodic_mad_upper_endpoint_requires_five_guards(self):
        from JAX_BSSN.parallel import initialize_tiled_bssn, make_field_mesh, map_tiled

        params = BSSNParameters(dx=1.0, mad_q=0.35)
        layout = dict(
            mesh=make_field_mesh((1, 1, 1)), tile_shape=(8, 1, 1),
            upper_endpoints=(True, False, False),
        )
        with self.assertRaisesRegex(ValueError, "five guard cells"):
            initialize_tiled_bssn(None, params, guard_cells=4, **layout)

        def wave_lapse(coordinates, local_params):
            x, y, z = coordinates
            one = jnp.ones(local_params.owned_shape)
            zero = jnp.zeros_like(one)
            metric = jnp.eye(3)[:, :, None, None, None] * one
            vector = jnp.zeros((3,) + local_params.owned_shape)
            lapse = one + 0.1 * jnp.sin(2 * jnp.pi * x / 9)
            return BSSNVariables(metric, one, jnp.zeros_like(metric), zero,
                                 vector, lapse, vector)

        state = initialize_tiled_bssn(wave_lapse, params, guard_cells=5, **layout)
        derivative = map_tiled(
            lambda field, p: diff1_upwind_physical(field, 1.0, 0, p),
            state.lapse, params, guard_cells=5, **layout,
        )
        field = (1 + 0.1 * jnp.sin(2 * jnp.pi * jnp.arange(9) / 9))[:, None, None]
        expected = diff1_upwind_physical(field, 1.0, 0, params)[-1, 0, 0]
        actual = derivative[0, 0, 0, 5 + 8, 5, 5]
        self.assert_fields_close(actual, expected, atol=1.0e-14)


if __name__ == "__main__":
    unittest.main()
