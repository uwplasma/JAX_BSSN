"""Wave diagnostic conventions and native FMR movie geometry."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import openpmd_api as io

from JAX_BSSN.diagnostics.openpmd import FMRPatchSeriesWriter
from JAX_BSSN.diagnostics.waves import composite_wave_diagnostics, diagnostic_fields, h_plus
from JAX_BSSN.fmr import FMRHierarchySpec, FMRPatchSpec
from JAX_BSSN.parallel import make_field_mesh
from demos.gravitational_waves import linear_wave
from demos.gravitational_waves.fmr_wave import branching_fmr as run
from demos.gravitational_waves.fmr_wave import make_branching_fmr_movie as movie

jax.config.update("jax_enable_x64", True)


class TestWaveDiagnostics(unittest.TestCase):
    def test_legacy_names_reuse_the_core_diagnostics(self):
        # These identities are what let the composite-norm tests below cover the
        # demo too, instead of a second copy of them living beside the writer tests.
        self.assertIs(linear_wave.h_plus, h_plus)
        self.assertIs(linear_wave.diagnostic_fields, diagnostic_fields)
        self.assertIs(linear_wave._composite_wave_diagnostics, composite_wave_diagnostics)

    def test_composite_norms_count_vertices_and_sum_momentum_components(self):
        hierarchy = FMRHierarchySpec((FMRPatchSpec((1, 1, 1), (3, 3, 2)),))
        root = jnp.ones((6, 5, 4))
        child = jnp.ones((5, 5, 3))
        error = (2*root).at[1:4, 1:4, 1:3].set(999.)
        fields = (
            {"h_plus": error, "hamiltonian_constraint": 5*root,
             "momentum_constraint": (root, 2*root, 2*root)},
            {"h_plus": 3*child, "hamiltonian_constraint": 7*child,
             "momentum_constraint": (2*child, 3*child, 6*child)},
        )
        result = composite_wave_diagnostics(
            fields, (0*root, 0*child), hierarchy, 0., 1., 1., active_fields=True,
        )
        root_count, child_count = 120 - 18, 75
        count = root_count + child_count
        self.assertAlmostEqual(result["composite_rms"], float(jnp.sqrt((root_count*4 + child_count*9)/count)))
        self.assertEqual(result["composite_max"], 3.)
        self.assertAlmostEqual(result["composite_hamiltonian_l2"], float(jnp.sqrt((root_count*25 + child_count*49)/count)))
        self.assertAlmostEqual(result["composite_momentum_l2"], float(jnp.sqrt((root_count*9 + child_count*49)/count)))

    def test_branching_tree_excludes_each_parent_s_own_covered_boxes(self):
        """Two siblings under the root and a grandchild under the first sibling.

        Coverage is per-parent, so the grandchild must be subtracted from its own
        sibling rather than from the root, and the second sibling keeps every vertex.
        """
        hierarchy = FMRHierarchySpec((
            FMRPatchSpec((1, 1, 1), (3, 3, 3), parent=0),
            FMRPatchSpec((4, 1, 1), (6, 3, 3), parent=0),
            FMRPatchSpec((1, 1, 1), (3, 3, 3), parent=1),
        ))
        shapes = ((8,) * 3,) + ((5,) * 3,) * 3
        covered = ((slice(1, 4),) * 3, (slice(4, 7),) + (slice(1, 4),) * 2)
        values = (1., 2., 3., 4.)
        fields = []
        for patch, (shape, value) in enumerate(zip(shapes, values)):
            wave = jnp.full(shape, value)
            hamiltonian = jnp.full(shape, value + 4.)
            # 999 anywhere a child owns the vertices must never reach a norm.
            boxes = covered if patch == 0 else (covered[0],) if patch == 1 else ()
            for box in boxes:
                wave = wave.at[box].set(999.)
                hamiltonian = hamiltonian.at[box].set(999.)
            fields.append({"h_plus": wave, "hamiltonian_constraint": hamiltonian,
                           "momentum_constraint": (wave, wave, wave)})
        coordinates = tuple(jnp.zeros(shape) for shape in shapes)
        result = composite_wave_diagnostics(
            tuple(fields), coordinates, hierarchy, 0., 1., 1., active_fields=True,
        )

        counts = (8**3 - 2*3**3, 5**3 - 3**3, 5**3, 5**3)
        total = sum(counts)
        self.assertEqual(total, 806)
        self.assertAlmostEqual(result["composite_rms"], float(jnp.sqrt(
            sum(n*v**2 for n, v in zip(counts, values))/total)))
        self.assertEqual(result["composite_max"], 4.)
        self.assertAlmostEqual(result["composite_hamiltonian_l2"], float(jnp.sqrt(
            sum(n*(v + 4.)**2 for n, v in zip(counts, values))/total)))
        self.assertAlmostEqual(result["composite_momentum_l2"], float(jnp.sqrt(
            3*sum(n*v**2 for n, v in zip(counts, values))/total)))

    def test_initial_wave_retains_the_analytic_connection_and_propagation_sign(self):
        coordinates = jnp.meshgrid(jnp.linspace(-.3, .2, 7), jnp.arange(3.), jnp.arange(2.), indexing="ij")
        amplitude, wavelength = .02, 1.3
        params = run.ROOT_PARAMETERS._replace(owned_shape=(7, 3, 2))
        state = run.linear_wave_initial_data(coordinates, params, amplitude=amplitude, wavelength=wavelength)
        wave_number = 2*jnp.pi/wavelength
        h = amplitude*jnp.sin(wave_number*coordinates[0])
        dhdx = wave_number*amplitude*jnp.cos(wave_number*coordinates[0])
        expected_connection = 2*h*dhdx/3*(1 - h*h)**(-2./3.)
        self.assertTrue(bool(jnp.allclose(h_plus(state), h, atol=2.e-16, rtol=0.)))
        self.assertTrue(bool(jnp.allclose(state.conformal_connection[0], expected_connection, atol=1.e-16, rtol=0.)))
        self.assertTrue(bool(jnp.all(state.conformal_connection[1:] == 0.)))
        # gamma_dot = -2 K with lapse=1 and zero shift must give h_dot=-h_x.
        physical_metric = state.conformal_metric/state.conformal_factor**2
        physical_K = state.traceless_K/state.conformal_factor**2 + physical_metric*state.trace_K/3
        self.assertTrue(bool(jnp.allclose(physical_K[1, 1] - physical_K[2, 2], dhdx, atol=1.e-16, rtol=0.)))


class TestLinearWaveRun(unittest.TestCase):
    """End-to-end cover for the demo driver. File naming and manifest format are
    checked against the writer itself in tests/FMR/test_fmr_output.py."""

    RECORDS = {"h_plus", "lapse", "shift", "K", "W",
               "hamiltonian_constraint", "momentum_constraint"}

    def run_demo(self, directory, **overrides):
        arguments = dict(grid_size=8, level_count=3, final_time=.02,
                         output_iterations=2, show_progress=False)
        arguments.update(overrides)
        base = Path(directory) / "linear_wave_fmr"
        _, result = linear_wave.run_linear_wave(output_path=base, **arguments)
        self.assertEqual(Path(result["output_path"]), base.with_suffix(".visit"))
        return base, result

    def test_three_centered_levels_halve_their_parent_and_stay_accurate(self):
        with tempfile.TemporaryDirectory() as directory:
            base, result = self.run_demo(directory)
            lines = base.with_suffix(".visit").read_text().splitlines()
            self.assertEqual(lines[0], "!NBLOCKS 3")

            # Each centered patch covers half its parent, so 8 -> 7 -> 5 vertices.
            for level, expected in enumerate(((8,) * 3, (7,) * 3, (5,) * 3)):
                path = base.parent / (f"linear_wave_fmr_level_{level:02d}"
                                      f"_patch_000_00000001.h5")
                series = io.Series(str(path), io.Access.read_only)
                iteration = series.iterations[1]
                self.assertEqual(set(iteration.meshes), self.RECORDS)
                component = iteration.meshes["h_plus"][io.Mesh_Record_Component.SCALAR]
                self.assertEqual(tuple(component.shape), expected)
                self.assertAlmostEqual(
                    float(iteration.meshes["h_plus"].grid_spacing[0]), .125 / 2**level)
                iteration.close()
                series.close()

            # The driver's own five-percent gate would have raised before here.
            self.assertLess(result["diagnostics"]["composite_rms"] / 1.e-8, 1.e-3)
            self.assertFalse(base.with_suffix(".h5").exists())

    @unittest.skipUnless(jax.device_count() >= 2, "requires two logical devices")
    def test_the_driver_splits_every_patch_over_a_supplied_device_mesh(self):
        """Plumbing only: tests/FMR/test_tiled_fmr.py checks distributed FMR itself.

        A 12-vertex root splits evenly; its centered child keeps 11 active
        vertices, so each tile stays wider than its five-deep halo.
        """
        with tempfile.TemporaryDirectory() as directory:
            _, result = self.run_demo(directory, grid_size=12, level_count=2,
                                      mesh=make_field_mesh((2, 1, 1)))
        root, child = result["layouts"]
        self.assertEqual(root["tile_shape"], (6, 12, 12))
        self.assertEqual(root["upper_endpoints"], (False,) * 3)
        self.assertEqual(child["tile_shape"], (5, 11, 11))
        self.assertEqual(child["upper_endpoints"], (True, False, False))
        for state in result["parameters"]:
            self.assertEqual(state.dt, result["parameters"][0].dt)
        # The driver's five-percent gate already ran; confirm it stayed tiny.
        self.assertLess(result["diagnostics"]["composite_rms"] / 1.e-8, 1.e-3)


class TestMovieGeometry(unittest.TestCase):
    def setUp(self):
        self.patches = []
        for index, dx in enumerate((.25, .125)):
            patch = dict(id=index, parent=index - 1, depth=index, ordinal=0,
                         origin=[-2*dx]*3, shape=[5]*3, dx=dx,
                         tile_shape=[2, 5, 5], tile_counts=[2, 1, 1],
                         upper_endpoints=[True, False, False])
            self.patches.append(patch)

    def prepare(self):
        for patch in self.patches:
            mesh = SimpleNamespace(geometry_string="cartesian", attributes=(),
                                   grid_global_offset=patch["origin"], grid_spacing=[patch["dx"]]*3)
            component = SimpleNamespace(shape=patch["shape"], position=(0.,)*3)
            movie.prepare_geometry(patch, mesh, component)

    def test_native_edges_tile_splits_and_child_coverage(self):
        self.prepare()
        root = self.patches[0]
        self.assertTrue(bool(jnp.array_equal(jnp.asarray(root["edges"][0]),
                                             jnp.array([-.5, -.375, -.125, .125, .375, .5]))))
        self.assertTrue(bool(jnp.array_equal(jnp.asarray(root["tile_splits"][0]), jnp.array([-.125]))))
        for normal in (1, 2):
            faces, samples = movie.plane_geometry(self.patches, normal)
            axes = [axis for axis in range(3) if axis != normal]
            areas = [0., 0.]
            for face, sample in zip(faces, samples):
                face = jnp.asarray(face)
                widths = jnp.max(face, axis=0) - jnp.min(face, axis=0)
                areas[sample[0]] += float(widths[axes[0]]*widths[axes[1]])
                center = jnp.mean(face, axis=0)
                inside_child = bool(jnp.all(jnp.abs(center[jnp.array(axes)]) < .25))
                self.assertEqual(sample[0] == 1, inside_child)
            self.assertEqual(areas, [.75, .25])

    def test_slice_interpolation_only_changes_the_normal_coordinate(self):
        axis = jnp.linspace(-.5, .5, 5)
        x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
        field = x + 2*y + 3*z
        actual = movie.slice_field(jax.device_get(field), self.patches[0], 2, .0625)
        self.assertTrue(bool(jnp.array_equal(jnp.asarray(actual), x[:, :, 0] + 2*y[:, :, 0] + 3*.0625)))

    def test_history_excludes_stale_manifest_frames(self):
        hierarchy = FMRHierarchySpec((FMRPatchSpec((1,)*3, (3,)*3),))
        fields = tuple({"h_plus": jnp.full((5,)*3, .1)} for _ in self.patches)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "wave"
            writer = FMRPatchSeriesWriter(base, dt=.1)
            for index in range(2):
                writer.write(fields, output_index=index, simulation_step=index, time=.1*index,
                             spacings=(.25, .125), origins=tuple(p["origin"] for p in self.patches),
                             ghost_cells=(0, 0), hierarchy=hierarchy)
            metadata = dict(manifest=writer.visit_path.name, amplitude=.1, patches=self.patches,
                            history=[dict(step=0, time=0.)])
            metadata_path = base.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata))
            _, frames, limit = movie.load_frames(metadata_path, 1)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][0], 0.)
            self.assertAlmostEqual(limit, 1.02)
            metadata["history"][0]["step"] = 7
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "index/step"):
                movie.load_frames(metadata_path, 1)


if __name__ == "__main__":
    unittest.main()
