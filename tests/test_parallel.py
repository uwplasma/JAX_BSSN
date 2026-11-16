"""Serial equivalence of PyPIC3D-layout halo exchange and BSSN evolution.

Run in a fresh process with JAX_NUM_CPU_DEVICES=8 and JAX_ENABLE_X64=1.
All numerical references use JAX; gathering is confined to these small tests.
"""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.constraints import compute_all_constraints_with_matter
from JAX_BSSN.diagnostics.constraints import compute_constraint_norms
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs_with_matter, enforce_algebraic_constraints, rk4_step,
)
from JAX_BSSN.parallel import (
    make_field_mesh, initialize_tiled_field, initialize_tiled_bssn,
    compute_tiled_bssn_rhs_with_matter, prepare_tiled_state,
    refresh_tiled_field, tiled_rk4_step, tiled_constraint_norms,
)


def gather_small_field(value, widths, g=4, endpoints=(False, False, False)):
    tiles, local = [], []
    for count, width, endpoint in zip(value.shape[-6:-3], widths, endpoints):
        i = jnp.arange(count*width + int(endpoint))
        t = jnp.minimum(i // width, count - 1)
        tiles.append(t)
        local.append(g + i - t*width)
    tx, ty, tz = tiles
    ix, iy, iz = local
    return value[(Ellipsis, tx[:, None, None], ty[None, :, None], tz[None, None, :],
                  ix[:, None, None], iy[None, :, None], iz[None, None, :])]


def smooth_state(shape):
    x, y, z = jnp.meshgrid(*(2*jnp.pi*jnp.arange(n)/n for n in shape), indexing="ij")
    one = jnp.ones(shape)
    metric = jnp.eye(3)[:, :, None, None, None] * one
    metric = metric.at[0, 0].add(.012*jnp.sin(x+y))
    metric = metric.at[1, 1].add(.008*jnp.cos(y+z))
    metric = metric.at[0, 1].set(.003*jnp.cos(x+z))
    metric = metric.at[1, 0].set(metric[0, 1])
    curvature = .002*jnp.sin(x-y+z)*jnp.diag(jnp.array([1., -1., 0.]))[:, :, None, None, None]
    vector = jnp.stack((jnp.sin(y+z), jnp.cos(x+z), jnp.sin(x-y)))
    return enforce_algebraic_constraints(BSSNVariables(
        metric, one+.008*jnp.sin(x+y+z), curvature, .003*jnp.cos(x-y),
        .001*vector, one+.01*jnp.cos(y-z), .02*vector))


@unittest.skipUnless(jax.device_count() >= 8, "requires eight logical devices")
class TestParallelBSSN(unittest.TestCase):
    def assert_close(self, actual, expected, atol=3.e-11):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=3.e-11, atol=atol)),
                        f"maximum error {float(jnp.max(jnp.abs(actual-expected)))}")

    def test_faces_edges_corners_and_tensor_components(self):
        mesh = make_field_mesh((2, 2, 2))
        widths = (4, 4, 4)
        params = BSSNParameters()
        x, y, z = jnp.meshgrid(*(jnp.arange(8) for _ in range(3)), indexing="ij")
        encoded = x*10000 + y*100 + z
        for leading in ((), (3,), (3, 3)):
            component = jnp.arange(1 if not leading else 3**len(leading)).reshape(leading+(1, 1, 1))
            values = (encoded + 1000000*component).astype(jnp.float64)
            actual = initialize_tiled_field(values, params, mesh=mesh, tile_shape=widths,
                                            component_shape=leading)
            tx, ty, tz, ix, iy, iz = jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2),
                                                jnp.arange(12), jnp.arange(12), jnp.arange(12), indexing="ij")
            expected = ((tx*4+ix-4)%8)*10000 + ((ty*4+iy-4)%8)*100 + (tz*4+iz-4)%8
            expected = expected + 1000000*component.reshape(leading+(1,)*6)
            self.assertTrue(bool(jnp.all(actual == expected)))
            self.assertEqual(actual.sharding.mesh, mesh)

    def test_linear_exteriors_and_extra_center_endpoints(self):
        mesh = make_field_mesh((2, 2, 2))
        params = BSSNParameters(xl_bc=1, xr_bc=1, yl_bc=1, yr_bc=1, zl_bc=1, zr_bc=1)
        endpoints = (True, False, True)
        x, y, z = jnp.meshgrid(jnp.arange(9), jnp.arange(8), jnp.arange(9), indexing="ij")
        values = 7.*x + 3.*y - z
        actual = initialize_tiled_field(values, params, mesh=mesh, tile_shape=(4, 4, 4),
                                         upper_endpoints=endpoints)
        tx, ty, tz, ix, iy, iz = jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2),
                                            jnp.arange(12), jnp.arange(12), jnp.arange(12), indexing="ij")
        expected = 7*(tx*4+ix-4) + 3*(ty*4+iy-4) - (tz*4+iz-4)
        self.assert_close(actual, expected)
        self.assert_close(gather_small_field(actual, (4, 4, 4), endpoints=endpoints), values)

    def test_periodic_wrap_includes_extra_owned_endpoint(self):
        mesh = make_field_mesh((2, 1, 1))
        values = jnp.arange(9.)[:, None, None]
        actual = initialize_tiled_field(values, BSSNParameters(), mesh=mesh,
                                         tile_shape=(4, 1, 1), upper_endpoints=(True, False, False))
        x = (jnp.arange(2)[:, None]*4 + jnp.arange(12)[None, :] - 4) % 9
        self.assert_close(actual[:, 0, 0, :, 4, 4], x)

    def test_memory_mapped_field_input_and_local_callback(self):
        mesh = make_field_mesh((2, 1, 1))
        params = BSSNParameters(dx=.25, x_min=-2., y_min=1., z_min=3.)
        values = jnp.arange(8.)[:, None, None]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"field.npy"
            jnp.save(path, values)
            loaded = initialize_tiled_field(path, params, mesh=mesh, tile_shape=(4, 1, 1))
        callback = initialize_tiled_field(lambda xyz, p: (xyz[0]+2.)/.25 + 0*xyz[1]+0*xyz[2],
                                           params, mesh=mesh, tile_shape=(4, 1, 1))
        self.assert_close(callback, loaded)

    def test_full_matter_rhs_and_constraint_norms(self):
        mesh = make_field_mesh((2, 2, 2))
        widths = (4, 4, 4)
        state = smooth_state((8, 8, 8))
        params = BSSNParameters(dx=.2, dt=.001, nu=.02, kappa=.1, mad_q=.35)
        layout = dict(mesh=mesh, tile_shape=widths)
        tiled = initialize_tiled_bssn(state, params, **layout)
        rho = .01*state.lapse
        momentum = .03*state.shift
        stress = .004*state.conformal_metric
        sources = tuple(initialize_tiled_field(v, params, component_shape=v.shape[:-3], **layout)
                        for v in (rho, momentum, stress))
        rhs = jax.jit(lambda q, a, b, c: compute_tiled_bssn_rhs_with_matter(q, params, a, b, c, **layout))
        executable = rhs.lower(tiled, *sources).compile()
        actual = executable(tiled, *sources)
        expected = compute_bssn_rhs_with_matter(state, params, rho, momentum, stress)
        for a, b in zip(actual, expected):
            self.assert_close(gather_small_field(a, widths), b)
        norms = tiled_constraint_norms(tiled, params, sources[0], sources[1], **layout)
        reference = compute_constraint_norms(compute_all_constraints_with_matter(state, params, rho, momentum))
        for key in reference:
            self.assert_close(norms[key], reference[key])
        # Inspect after SPMD partitioning, which can introduce implicit gathers.
        hlo = executable.as_text().lower()
        self.assertNotIn("all-gather", hlo)
        self.assertIn("collective-permute", hlo)

    def test_mixed_sommerfeld_rhs_interior_tiles_and_endpoint(self):
        mesh = make_field_mesh((4, 1, 1))
        widths = (4, 8, 8)
        endpoints = (True, False, False)
        params = BSSNParameters(dx=.2, dt=.001, x_min=-2., y_min=-1., z_min=-1.,
                                xl_bc=1, xr_bc=1, zl_bc=1, zr_bc=1, mad_q=.35, kappa=.1)
        layout = dict(mesh=mesh, tile_shape=widths, upper_endpoints=endpoints)
        state = smooth_state((17, 8, 8))
        tiled = initialize_tiled_bssn(state, params, **layout)
        sources = (.01*state.lapse, .03*state.shift, .004*state.conformal_metric)
        tiled_sources = tuple(initialize_tiled_field(v, params, component_shape=v.shape[:-3], **layout)
                              for v in sources)
        actual = jax.jit(lambda q, a, b, c: compute_tiled_bssn_rhs_with_matter(q, params, a, b, c, **layout))(
            tiled, *tiled_sources)
        expected = compute_bssn_rhs_with_matter(state, params, *sources)
        for a, b in zip(actual, expected):
            self.assert_close(gather_small_field(a, widths, endpoints=endpoints), b)

    def test_three_rk4_steps_preserve_serial_solution_and_projection(self):
        mesh = make_field_mesh((2, 1, 1))
        widths = (4, 4, 4)
        params = BSSNParameters(dx=.2, dt=.001, nu=.02, kappa=.1)
        state = smooth_state((8, 4, 4))
        layout = dict(mesh=mesh, tile_shape=widths)
        tiled = initialize_tiled_bssn(state, params, **layout)
        advance = jax.jit(lambda q: tiled_rk4_step(q, params, **layout))
        for _ in range(3):
            tiled = advance(tiled)
            state = rk4_step(state, params)
        for a, b in zip(tiled, state):
            self.assert_close(gather_small_field(a, widths), b)
        norms = tiled_constraint_norms(tiled, params, **layout)
        self.assertLess(float(norms["det_gamma_linf"]), 3.e-14)
        self.assertLess(float(norms["trace_A_linf"]), 3.e-14)


if __name__ == "__main__":
    unittest.main()
