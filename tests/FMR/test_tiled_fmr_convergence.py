"""Spatial and temporal accuracy after a wave has crossed a refinement interface."""

import math
import unittest

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.fmr import FMRHierarchySpec, FMRPatchSpec
from JAX_BSSN.parallel import make_field_mesh
from tests.initial_data import gauge_wave_analytic_state
from tests.FMR.test_tiled_fmr import legacy_chain, untile_field
from tests.support import fmr_reference as reference

jax.config.update("jax_enable_x64", True)

AMPLITUDE = 0.001
FINAL_TIME = 0.25


def gauge_wave(coordinates, params):
    x, y, z = jnp.broadcast_arrays(*coordinates)
    return gauge_wave_analytic_state(x, y, z, amplitude=AMPLITUDE)


def evolve_wave(grid_size, steps, fixed_geometry=False):
    from JAX_BSSN.fmr import initialize_tiled_fmr, fmr_rk4_step

    dx = 1/grid_size
    if fixed_geometry:
        shape = (grid_size, grid_size//2, grid_size//2)
        spec = FMRPatchSpec((grid_size//4, grid_size//8, grid_size//8),
                            (3*grid_size//4, 3*grid_size//8, 3*grid_size//8))
    else:
        shape = (grid_size, 8, 8)
        spec = FMRPatchSpec((grid_size//4, 2, 2), (3*grid_size//4, 5, 5))
    hierarchy = FMRHierarchySpec((spec,))
    params = BSSNParameters(eta=0., kappa=0., g=0., nu=0., dx=dx,
                           dt=FINAL_TIME/steps, zero_shift=1, gauge=0)
    states, parameters, layouts = initialize_tiled_fmr(
        gauge_wave, params, shape, hierarchy, mesh=make_field_mesh((1,1,1)))
    evolve = jax.jit(lambda value: jax.lax.fori_loop(
        0, steps,
        lambda _, current: fmr_rk4_step(current, parameters, hierarchy, layouts=layouts),
        value,
    ))
    result = evolve(states)
    lapse = tuple(untile_field(state.lapse, layout) for state, layout in zip(result, layouts))
    root_x = dx*jnp.arange(grid_size)[:, None, None]
    fine_x = reference.fine_coordinates(spec, dx)[0][4:-4, 4:-4, 4:-4]
    exact = tuple(jnp.sqrt(1-AMPLITUDE*jnp.sin(2*jnp.pi*(x-FINAL_TIME)))
                  for x in (root_x, fine_x))
    uncovered = jnp.ones(shape, dtype=bool)
    covered = tuple(slice(lo, hi+1) for lo, hi in zip(spec.coarse_lo, spec.coarse_hi))
    uncovered = uncovered.at[covered].set(False)
    return lapse, exact, uncovered, dx


def evolve_reference_wave(grid_size, steps):
    """Evolve the fixed temporal-test geometry with the frozen serial algorithm."""
    spec = reference.FMRPatchSpec((grid_size//4, 2, 2), (3*grid_size//4, 5, 5))
    hierarchy = reference.FMRHierarchySpec((spec,))
    params = BSSNParameters(eta=0., kappa=0., g=0., nu=0., dx=1/grid_size,
                           dt=FINAL_TIME/steps, zero_shift=1, gauge=0)
    states, parameters = legacy_chain((grid_size, 8, 8), params, (spec,), initial=gauge_wave)
    evolve = jax.jit(lambda value: jax.lax.fori_loop(
        0, steps,
        lambda _, current: reference.fmr_rk4_step(current, parameters, hierarchy),
        value,
    ))
    result = evolve(states)
    return (result[0].lapse, reference.fine_active_view(result[1].lapse, spec))


def composite_norm(fields, uncovered, dx):
    """Volume-weight uncovered coarse vertices and active fine vertices once."""
    root, fine = fields
    volume = jnp.sum(uncovered)*dx**3 + fine.size*(dx/2)**3
    square = jnp.sum(jnp.where(uncovered, root**2, 0))*dx**3
    square += jnp.sum(fine**2)*(dx/2)**3
    return float(jnp.sqrt(square/volume))


class TestTiledFMRConvergence(unittest.TestCase):
    def test_spatial_accuracy_on_three_grids_after_interface_crossing(self):
        sizes = (16, 24, 32)
        errors = []
        for size in sizes:
            actual, exact, uncovered, dx = evolve_wave(size, 256, fixed_geometry=True)
            error = composite_norm(tuple((a-b)/AMPLITUDE for a, b in zip(actual, exact)),
                                   uncovered, dx)
            self.assertTrue(math.isfinite(error))
            errors.append(error)
            print("FMR spatial gauge-wave error", size, error, flush=True)
        orders = tuple(math.log(a/b)/math.log(n2/n1)
                       for a, b, n1, n2 in zip(errors, errors[1:], sizes, sizes[1:]))
        print("FMR spatial gauge-wave errors/orders", sizes, errors, orders, flush=True)
        self.assertTrue(all(order > 3.5 for order in orders), (errors, orders))

    def test_temporal_refinement_matches_frozen_serial_scheme(self):
        """Both implementations measure order 2.80 for this timestep range.

        The initial fourth-order hypothesis failed for this hierarchy in both
        methods. Verify preservation against the frozen algorithm while keeping
        the finite-range order visible; it does not establish asymptotic order.
        """
        steps = (20, 40, 80)
        tiled, serial, discrepancies = [], [], []
        for count in steps:
            actual, _, uncovered, dx = evolve_wave(24, count)
            expected = evolve_reference_wave(24, count)
            discrepancy = max(float(jnp.max(jnp.abs(a-b))) for a, b in zip(actual, expected))
            print("FMR temporal tiled/serial maximum lapse difference", count, discrepancy, flush=True)
            # Resolve the ~4e-11 lapse change between the two smallest steps;
            # the general one-step field tolerance is too loose for this check.
            self.assertLess(discrepancy, 1.e-14)
            tiled.append(actual)
            serial.append(expected)
            discrepancies.append(discrepancy)

        differences = []
        for solutions in (tiled, serial):
            values = [composite_norm(tuple((a-b)/AMPLITUDE for a, b in zip(coarse, fine)),
                                     uncovered, dx)
                      for coarse, fine in zip(solutions, solutions[1:])]
            self.assertTrue(all(math.isfinite(value) and value > 0 for value in values))
            self.assertLess(values[1], values[0])
            differences.append(values)
        orders = tuple(math.log(values[0]/values[1])/math.log(2) for values in differences)
        print("FMR temporal tiled/serial differences and measured orders", steps,
              differences, orders, "maximum lapse discrepancies", discrepancies, flush=True)
        for actual, expected in zip(*differences):
            self.assertAlmostEqual(actual, expected, delta=1.e-12)


if __name__ == "__main__":
    unittest.main()
