"""Independent numerical checks for patch transfers and synchronized tiled RK4."""

import itertools
import math
import unittest

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.coordinates import grid_coordinates
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs
from JAX_BSSN.fmr import FMRHierarchySpec, FMRPatchSpec
from JAX_BSSN.parallel import make_field_mesh
from tests.initial_data import gauge_wave_analytic_state
from tests.support import fmr_reference as reference

jax.config.update("jax_enable_x64", True)


def untile_field(field, layout, ghosts=0):
    """Read each global vertex from its owner, including requested exterior guards."""
    counts = layout["mesh"].devices.shape
    width = layout["tile_shape"]
    endpoints = layout.get("upper_endpoints", (False,) * 3)
    guard = layout.get("guard_cells", 4)
    shape = tuple(n * count + int(endpoint)
                  for n, count, endpoint in zip(width, counts, endpoints))
    indices = []
    local_indices = []
    for axis in range(3):
        node = jnp.arange(-ghosts, shape[axis] + ghosts)
        owner = jnp.clip(node // width[axis], 0, counts[axis] - 1)
        dims = [1, 1, 1]
        dims[axis] = node.size
        indices.append(owner.reshape(dims))
        local_indices.append((node - owner * width[axis] + guard).reshape(dims))
    return field[(Ellipsis,) + tuple(indices + local_indices)]


def untile_state(state, layout, ghosts=0):
    return BSSNVariables(*(untile_field(field, layout, ghosts) for field in state))


def smooth_state(coordinates, params):
    """Smooth nonseparable geometry exercises mixed derivatives and both shift signs."""
    x, y, z = jnp.broadcast_arrays(*coordinates)
    phase = x + 0.03 * jnp.sin(2 * jnp.pi * y) * jnp.cos(2 * jnp.pi * z)
    state = gauge_wave_analytic_state(phase, y, z, amplitude=0.01)
    shift = jnp.stack((0.1 + 0.01*jnp.sin(2*jnp.pi*y),
                       -0.05 + 0.005*jnp.cos(2*jnp.pi*z),
                       0.025 + 0.003*jnp.sin(2*jnp.pi*x)))
    return state._replace(shift=shift)


def legacy_chain(root_shape, params, specs, initial=smooth_state):
    states = [initial(grid_coordinates(root_shape, params, jnp.float64), params)]
    parameters = [params]
    origin = (params.x_min, params.y_min, params.z_min)
    for spec in specs:
        coordinates = reference.fine_coordinates(spec, parameters[-1].dx, origin)
        dx = parameters[-1].dx / 2
        origin = tuple(start + lo * parameters[-1].dx
                       for start, lo in zip(origin, spec.coarse_lo))
        child_params = params._replace(
            dx=dx, x_min=origin[0]-4*dx, y_min=origin[1]-4*dx,
            z_min=origin[2]-4*dx,
        )
        states.append(initial(coordinates, child_params))
        parameters.append(child_params)
    return tuple(states), tuple(parameters)


def reference_stages(states, parameters, specs, use_mad=True, stage_count=4):
    hierarchy = reference.FMRHierarchySpec(tuple(specs), use_mad=use_mad)
    finest_dx = min(params.dx for params in parameters)
    parameters = tuple(params._replace(mad_q=(finest_dx/params.dx)**4 if use_mad else 1.0)
                       for params in parameters)
    matrices = tuple(reference._interpolation_matrices(
        states[level][0], spec, (0 if level == 0 else 4,) * 3, level == 0,
    ) for level, spec in enumerate(specs))
    base, rhs = reference._stage(states, parameters, hierarchy, matrices)
    stages = [(base, rhs)]
    for multiplier in (0.5, 0.5, 1.0)[:stage_count-1]:
        stage = tuple(reference._add(value, rate, parameters[0].dt*multiplier)
                      for value, rate in zip(base, rhs))
        prepared, rhs = reference._stage(stage, parameters, hierarchy, matrices)
        stages.append((prepared, rhs))
    return stages


class TestTiledFMR(unittest.TestCase):
    def assert_field_close(self, actual, expected, atol=3e-11, rtol=2e-11):
        difference = float(jnp.max(jnp.abs(actual-expected)))
        self.assertTrue(bool(jnp.allclose(actual, expected, atol=atol, rtol=rtol)),
                        f"maximum difference {difference}")

    def assert_state_close(self, actual, expected, **tolerances):
        for name, value, truth in zip(BSSNVariables._fields, actual, expected):
            with self.subTest(field=name):
                self.assert_field_close(value, truth, **tolerances)

    def check_rk_stages_against_serial(self, topology):
        from JAX_BSSN.fmr import initialize_tiled_fmr, fmr_stage, fmr_rk4_step

        n = 12 if topology == (1, 1, 1) else 24
        bounds = (((3,)*3, (8,)*3), ((2,)*3, (4,)*3)) if n == 12 else (
            ((4,)*3, (14,)*3), ((4,)*3, (12,)*3))
        specs = tuple(FMRPatchSpec(lo, hi) for lo, hi in bounds)
        hierarchy = FMRHierarchySpec(specs)
        params = BSSNParameters(dx=1/n, dt=0.0005, eta=0., g=0., nu=0.005,
                                 x_min=-0.5, y_min=-0.5, z_min=-0.5)
        initial, serial_params = legacy_chain((n,)*3, params, specs)
        expected_stages = reference_stages(initial, serial_params, specs)
        states, parameters, layouts = initialize_tiled_fmr(
            smooth_state, params, (n,)*3, hierarchy,
            mesh=make_field_mesh(topology),
        )
        evaluate_stage = jax.jit(lambda values: fmr_stage(
            values, parameters, hierarchy, layouts=layouts))
        stage = states
        base = None
        for number, (expected_state, expected_rhs) in enumerate(expected_stages):
            prepared, rhs = evaluate_stage(stage)
            if base is None:
                base = prepared
            for level, (value, rate, truth, true_rate) in enumerate(zip(
                    prepared, rhs, expected_state, expected_rhs)):
                self.assert_state_close(untile_state(value, layouts[level], 0 if level == 0 else 4), truth)
                active_rhs = true_rate if level == 0 else BSSNVariables(*(
                    reference.fine_active_view(field, specs[level-1]) for field in true_rate))
                self.assert_state_close(untile_state(rate, layouts[level]), active_rhs, atol=1e-9)
            if number < 3:
                scale = params.dt * (0.5 if number < 2 else 1.0)
                stage = jax.tree.map(lambda value, rate: value + scale*rate, base, rhs)
        result = jax.jit(lambda value: fmr_rk4_step(
            value, parameters, hierarchy, layouts=layouts))(states)
        expected = reference.fmr_rk4_step(initial, serial_params,
                                          reference.FMRHierarchySpec(specs))
        for level, (value, truth) in enumerate(zip(result, expected)):
            self.assert_state_close(untile_state(value, layouts[level], 0 if level == 0 else 4), truth)

    def test_every_rk_stage_and_final_injection_match_frozen_serial_method(self):
        self.check_rk_stages_against_serial((1, 1, 1))

    @unittest.skipUnless(jax.device_count() >= 8, "requires eight CPU devices")
    def test_eight_device_rk_stages_and_final_injection_match_frozen_serial_method(self):
        self.check_rk_stages_against_serial((2, 2, 2))

    def test_disabling_mad_matches_the_frozen_fourth_order_derivative_stage(self):
        from JAX_BSSN.fmr import initialize_tiled_fmr, fmr_stage

        specs = (FMRPatchSpec((3,)*3, (8,)*3), FMRPatchSpec((2,)*3, (4,)*3))
        hierarchy = FMRHierarchySpec(specs, use_mad=False)
        params = BSSNParameters(dx=1/12, dt=0.0005, eta=0., g=0., nu=0.005,
                               x_min=-0.5, y_min=-0.5, z_min=-0.5)
        initial, serial_params = legacy_chain((12,)*3, params, specs)
        expected_state, expected_rhs = reference_stages(
            initial, serial_params, specs, use_mad=False, stage_count=1)[0]
        states, parameters, layouts = initialize_tiled_fmr(
            smooth_state, params, (12,)*3, hierarchy, mesh=make_field_mesh((1,1,1)))
        self.assertTrue(all(param.mad_q == 1.0 for param in parameters))
        prepared, rhs = jax.jit(lambda values: fmr_stage(
            values, parameters, hierarchy, layouts=layouts))(states)
        for level, (value, rate, truth, true_rate) in enumerate(zip(
                prepared, rhs, expected_state, expected_rhs)):
            self.assert_state_close(untile_state(value, layouts[level], 0 if level == 0 else 4), truth)
            active_rhs = true_rate if level == 0 else BSSNVariables(*(
                reference.fine_active_view(field, specs[level-1]) for field in true_rate))
            self.assert_state_close(untile_state(rate, layouts[level]), active_rhs, atol=1e-9)

    def test_single_vertex_patch_keeps_resolved_derivatives_from_parent_guards(self):
        from JAX_BSSN.fmr import initialize_tiled_fmr, fmr_stage

        spec = FMRPatchSpec((3,)*3, (3,)*3)
        hierarchy = FMRHierarchySpec((spec,))
        params = BSSNParameters(dx=1/8, dt=0.0001, eta=0., g=0., nu=0.)
        initial, serial_params = legacy_chain((8,)*3, params, (spec,))
        _, expected = reference_stages(initial, serial_params, (spec,), stage_count=1)[0]
        states, parameters, layouts = initialize_tiled_fmr(
            smooth_state, params, (8,)*3, hierarchy, mesh=make_field_mesh((1,1,1)))
        _, rhs = jax.jit(lambda values: fmr_stage(
            values, parameters, hierarchy, layouts=layouts))(states)
        actual = untile_state(rhs[1], layouts[1])
        expected = BSSNVariables(*(reference.fine_active_view(field, spec)
                                   for field in expected[1]))
        self.assertGreater(float(jnp.max(jnp.abs(expected.conformal_connection))), 1e-8)
        self.assert_state_close(actual, expected, atol=1e-9)

    def test_sibling_order_and_shallower_last_patch_do_not_change_mad(self):
        from JAX_BSSN.fmr import initialize_tiled_fmr, fmr_stage

        # The last patch is deliberately shallower than its predecessor.
        specs = (FMRPatchSpec((2,)*3, (10,)*3, parent=0),
                 FMRPatchSpec((3,)*3, (9,)*3, parent=1),
                 FMRPatchSpec((14,)*3, (20,)*3, parent=0))
        swapped = (specs[2], specs[0], specs[1]._replace(parent=2))
        params = BSSNParameters(dx=1/24, dt=0.0001, eta=0., g=0., nu=0.)
        results = []
        for patches in (specs, swapped):
            hierarchy = FMRHierarchySpec(patches)
            states, parameters, layouts = initialize_tiled_fmr(
                smooth_state, params, (24,)*3, hierarchy, mesh=make_field_mesh((1,1,1)))
            prepared, rhs = jax.jit(lambda values: fmr_stage(
                values, parameters, hierarchy, layouts=layouts))(states)
            results.append(tuple(untile_state(value, layout) for value, layout in zip(rhs, layouts)))
            root = untile_state(prepared[0], layouts[0])
            expected = compute_bssn_rhs(root, params._replace(mad_q=(1/4)**4))
            self.assert_state_close(results[-1][0], expected, atol=1e-9)
        for before, after in zip(results[0], (results[1][0], results[1][2], results[1][3], results[1][1])):
            self.assert_state_close(before, after, atol=1e-9)

    def test_quintic_ghost_shell_and_endpoint_injection_keep_active_values(self):
        from JAX_BSSN.fmr import (
            initialize_tiled_fmr, fill_tiled_fine_ghosts, restrict_tiled_to_coarse,
        )

        def polynomial(coordinates, p):
            x, y, z = jnp.broadcast_arrays(*coordinates)
            scalar = x**5 + 2*y**4 - z**3 + x*y*z
            fields = []
            for number, components in enumerate(((3, 3), (), (3, 3), (), (3,), (), (3,))):
                coefficient = 1 + number + jnp.arange(math.prod(components)).reshape(components)
                coefficient = coefficient.reshape(components + (1, 1, 1))
                fields.append(coefficient * scalar + number)
            return BSSNVariables(*fields)

        spec = FMRPatchSpec((6,)*3, (14,)*3)
        params = BSSNParameters(dx=0.1)
        hierarchy = FMRHierarchySpec((spec,))
        topology = (2, 2, 2) if jax.device_count() >= 8 else (1, 1, 1)
        states, parameters, layouts = initialize_tiled_fmr(
            polynomial, params, (24,)*3, hierarchy, mesh=make_field_mesh(topology))
        child = BSSNVariables(*(jnp.full_like(field, -999.-number)
                                for number, field in enumerate(states[1])))
        filled = fill_tiled_fine_ghosts(states[0], child, spec,
                                      parent_layout=layouts[0], child_layout=layouts[1])

        # Check every allocated face, edge, and corner, including the extra
        # lower guard in endpoint layouts and the halos between child tiles.
        layout = layouts[1]
        widths, guard = layout["tile_shape"], layout["guard_cells"]
        active_shape = reference.fine_active_shape(spec)
        for tile in itertools.product(*(range(count) for count in topology)):
            indices = [jnp.arange(width + 2*guard) + owner*width - guard
                       for owner, width in zip(tile, widths)]
            global_nodes = jnp.meshgrid(*indices, indexing="ij")
            shell = jnp.zeros(global_nodes[0].shape, dtype=bool)
            for node, size in zip(global_nodes, active_shape):
                shell |= (node < 0) | (node >= size)
            coordinates = tuple((lo + node/2) * params.dx
                                for lo, node in zip(spec.coarse_lo, global_nodes))
            expected = polynomial(coordinates, params)
            for number, (actual, truth) in enumerate(zip(filled, expected)):
                actual = actual[(Ellipsis,) + tile + (slice(None),)*3]
                self.assert_field_close(actual[..., shell], truth[..., shell], atol=2e-9)
                self.assertTrue(bool(jnp.all(actual[..., ~shell] == -999.-number)))

        # Every scalar/vector/tensor component must inject both inclusive
        # endpoints exactly, without modifying any uncovered parent vertex.
        restricted = restrict_tiled_to_coarse(states[0], filled, spec,
                                             parent_layout=layouts[0], child_layout=layouts[1])
        covered = tuple(slice(lo, hi+1) for lo, hi in zip(spec.coarse_lo, spec.coarse_hi))
        outside = jnp.ones((24,)*3, dtype=bool).at[covered].set(False)
        for number, (before, after) in enumerate(zip(states[0], restricted)):
            root = untile_field(after, layouts[0])
            self.assertTrue(bool(jnp.all(root[(Ellipsis,) + covered] == -999.-number)))
            expected = untile_field(before, layouts[0])
            self.assert_field_close(root[..., outside], expected[..., outside], atol=0, rtol=0)

    def test_sommerfeld_root_does_not_apply_physical_faces_to_children(self):
        from JAX_BSSN.fmr import initialize_tiled_fmr, fmr_rk4_step

        spec = FMRPatchSpec((5,)*3, (11,)*3)
        params = BSSNParameters(dx=0.1, dt=0.0001, xl_bc=1, xr_bc=1,
                               yl_bc=1, yr_bc=1, zl_bc=1, zr_bc=1,
                               x_min=-0.8, y_min=-0.8, z_min=-0.8)
        hierarchy = FMRHierarchySpec((spec,))
        states, parameters, layouts = initialize_tiled_fmr(
            None, params, (17,)*3, hierarchy, mesh=make_field_mesh((1,1,1)))
        result = jax.jit(lambda values: fmr_rk4_step(
            values, parameters, hierarchy, layouts=layouts))(states)
        for old, new, layout in zip(states, result, layouts):
            self.assert_state_close(untile_state(new, layout), untile_state(old, layout), atol=2e-12)
        self.assertTrue(all(getattr(parameters[1], name) == 0 for name in
                            ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc")))

    @unittest.skipUnless(jax.device_count() >= 8, "requires eight CPU devices")
    def test_compiled_refinement_transfers_do_not_gather_whole_patches(self):
        from JAX_BSSN.fmr import initialize_tiled_fmr, fmr_stage

        hierarchy = FMRHierarchySpec((FMRPatchSpec((4,)*3, (14,)*3),))
        states, parameters, layouts = initialize_tiled_fmr(
            None, BSSNParameters(dx=0.1), (24,)*3, hierarchy, mesh=make_field_mesh((2,2,2)))
        text = jax.jit(lambda values: fmr_stage(
            values, parameters, hierarchy, layouts=layouts)).lower(states).compile().as_text().lower()
        text = text.replace("_", "-")
        self.assertNotIn("all-gather", text)
        self.assertIn("collective-permute", text)


if __name__ == "__main__":
    unittest.main()
