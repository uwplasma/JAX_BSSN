"""Patch-tree output and composite ownership, using JAX numerical fixtures."""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import openpmd_api as io
from jax.sharding import NamedSharding, PartitionSpec

from JAX_BSSN.diagnostics.openpmd import FMRPatchSeriesWriter, tiled_output_fields
from JAX_BSSN.fmr import FMRHierarchySpec, FMRPatchSpec
from JAX_BSSN.parallel import make_field_mesh


class TestFMRPatchOutput(unittest.TestCase):
    def setUp(self):
        self.hierarchy = FMRHierarchySpec((
            FMRPatchSpec((1, 1, 1), (3, 3, 3), parent=0),
            FMRPatchSpec((4, 1, 1), (6, 3, 3), parent=0),
            FMRPatchSpec((1, 1, 1), (3, 3, 3), parent=1),
        ))
        self.origins = ((0., 0., 0.), (1., 1., 1.), (4., 1., 1.), (1.5,) * 3)
        self.spacings = (1., .5, .5, .25)
        self.shapes = ((8,) * 3,) + ((5,) * 3,) * 3
        self.values = tuple(
            index*1000 + jnp.arange(shape[0]**3, dtype=jnp.float64).reshape(shape)
            for index, shape in enumerate(self.shapes)
        )

    def test_branching_files_record_depth_ordinal_and_parent_and_restart(self):
        fields = tuple({"scalar": value if index == 0 else
                        jnp.pad(value, 4, constant_values=-999.),
                        "vector": tuple((value + component) if index == 0 else
                                        jnp.pad(value + component, 4, constant_values=-999.)
                                        for component in range(3))}
                       for index, value in enumerate(self.values))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "tree"
            arguments = dict(output_index=0, simulation_step=3, time=.03,
                             spacings=self.spacings, origins=self.origins,
                             ghost_cells=(0, 4, 4, 4), hierarchy=self.hierarchy)
            writer = FMRPatchSeriesWriter(base, dt=.01)
            writer.write(fields, **arguments)
            manifest = base.with_suffix(".visit").read_text()
            FMRPatchSeriesWriter(base, dt=.01).write(fields, **arguments)
            self.assertEqual(base.with_suffix(".visit").read_text(), manifest)
            pairs = ((0, 0), (1, 0), (1, 1), (2, 0))
            self.assertEqual(manifest.splitlines(), ["!NBLOCKS 4", "!TIME 0.03"] + [
                f"tree_level_{level:02d}_patch_{patch:03d}_00000000.opmd"
                for level, patch in pairs
            ])
            for index, ((level, patch), parent) in enumerate(zip(pairs, (-1, 0, 0, 1))):
                path = base.parent / f"tree_level_{level:02d}_patch_{patch:03d}_00000000.h5"
                series = io.Series(str(path), io.Access.read_only)
                iteration = series.iterations[0]
                self.assertEqual(iteration.get_attribute("fmrLevel"), level)
                self.assertEqual(iteration.get_attribute("fmrPatch"), patch)
                self.assertEqual(iteration.get_attribute("fmrId"), index)
                self.assertEqual(iteration.get_attribute("fmrParent"), parent)
                pending = iteration.meshes["scalar"][io.Mesh_Record_Component.SCALAR].load_chunk()
                series.flush()
                self.assertTrue(bool(jnp.array_equal(jnp.asarray(pending), self.values[index])))
                iteration.close()
                series.close()

    @unittest.skipUnless(jax.device_count() >= 2, "requires two logical devices")
    def test_tiled_export_preserves_owned_upper_endpoint_and_discards_all_guards(self):
        mesh = make_field_mesh((2, 1, 1))
        layouts = (dict(mesh=mesh, tile_shape=(4, 8, 8), guard_cells=4,
                        upper_endpoints=(False,) * 3),) + tuple(
            dict(mesh=mesh, tile_shape=(2, 4, 4), guard_cells=5,
                 upper_endpoints=(True,) * 3) for _ in range(3)
        )
        fields = []
        for value, layout in zip(self.values, layouts):
            widths, g, endpoints = layout["tile_shape"], layout["guard_cells"], layout["upper_endpoints"]
            padded = jnp.full((2, 1, 1) + tuple(n + 2*g for n in widths), -999.)
            for tile_x in range(2):
                owned = tuple(n + int(end and i == count - 1)
                              for n, end, i, count in zip(widths, endpoints, (tile_x, 0, 0), (2, 1, 1)))
                target = tuple(slice(g, g + n) for n in owned)
                source = (slice(tile_x*widths[0], tile_x*widths[0] + owned[0]),
                          slice(None), slice(None))
                padded = padded.at[(tile_x, 0, 0) + target].set(value[source])
            unsharded = tiled_output_fields({"scalar": padded}, layout)
            self.assertTrue(bool(jnp.array_equal(jnp.asarray(unsharded["scalar"]), value)))
            tiled = jax.device_put(padded, NamedSharding(
                mesh, PartitionSpec("tile_x", "tile_y", "tile_z", None, None, None),
            ))
            fields.append({"scalar": tiled, "vector": (tiled, tiled + 1, tiled + 2)})
            owned = tiled_output_fields(fields[-1], layout)
            self.assertTrue(bool(jnp.array_equal(jnp.asarray(owned["scalar"]), value)))
            self.assertTrue(bool(jnp.array_equal(jnp.asarray(owned["vector"][2]), value + 2)))

        with tempfile.TemporaryDirectory() as directory:
            writer = FMRPatchSeriesWriter(Path(directory) / "tiled" / "wave", dt=.01)
            geometry = dict(output_index=0, simulation_step=0, time=0.,
                            spacings=self.spacings, origins=self.origins, hierarchy=self.hierarchy)
            result = writer.write(fields, ghost_cells=(4, 5, 5, 5), layouts=layouts, **geometry)
            self.assertEqual(result, writer.visit_path)
            self.assertEqual(writer.visit_path.read_text().splitlines()[0], "!NBLOCKS 4")

            active = tuple(tiled_output_fields(field, layout) for field, layout in zip(fields, layouts))
            owned_writer = FMRPatchSeriesWriter(Path(directory) / "owned" / "wave", dt=.01)
            owned_writer.write(active, ghost_cells=(0,) * 4, **geometry)
            self.assertEqual(writer.visit_path.read_text(), owned_writer.visit_path.read_text())
            for name in writer.visit_path.read_text().splitlines()[2:]:
                left = io.Series(str((writer.base_path.parent / name).resolve()), io.Access.read_only)
                right = io.Series(str((owned_writer.base_path.parent / name).resolve()), io.Access.read_only)
                a, b = left.iterations[0], right.iterations[0]
                for attribute in a.attributes:
                    self.assertTrue(bool(jnp.array_equal(
                        jnp.asarray(a.get_attribute(attribute)), jnp.asarray(b.get_attribute(attribute)),
                    )))
                for field in a.meshes:
                    for attribute in ("grid_spacing", "grid_global_offset"):
                        self.assertTrue(bool(jnp.array_equal(
                            jnp.asarray(getattr(a.meshes[field], attribute)),
                            jnp.asarray(getattr(b.meshes[field], attribute)),
                        )))
                    for component in a.meshes[field]:
                        x = a.meshes[field][component].load_chunk()
                        y = b.meshes[field][component].load_chunk()
                        left.flush()
                        right.flush()
                        self.assertTrue(bool(jnp.array_equal(jnp.asarray(x), jnp.asarray(y))))
                a.close()
                b.close()
                left.close()
                right.close()
            # Restart validation accepts the same physical fields through either layout.
            FMRPatchSeriesWriter(owned_writer.base_path, dt=.01).write(
                fields, ghost_cells=(4, 5, 5, 5), layouts=layouts, **geometry,
            )

    def test_zero_ghost_input_still_requires_the_exact_owned_shape(self):
        fields = tuple({"scalar": value if index == 0 else jnp.pad(value, 4)}
                       for index, value in enumerate(self.values))
        with tempfile.TemporaryDirectory() as directory:
            writer = FMRPatchSeriesWriter(Path(directory) / "wrong_shape", dt=.01)
            with self.assertRaisesRegex(ValueError, "active shape"):
                writer.write(fields, output_index=0, simulation_step=0, time=0.,
                             spacings=self.spacings, origins=self.origins,
                             ghost_cells=(0,) * 4, hierarchy=self.hierarchy)

    def test_restarted_append_preserves_branch_parents_and_accepts_later_times(self):
        fields = tuple({"scalar": value if index == 0 else
                        jnp.pad(value, 4, constant_values=-999.)}
                       for index, value in enumerate(self.values))
        geometry = dict(spacings=self.spacings, origins=self.origins,
                        ghost_cells=(0, 4, 4, 4), hierarchy=self.hierarchy)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "restart"
            FMRPatchSeriesWriter(base, dt=.01).write(
                fields, output_index=0, simulation_step=0, time=0., **geometry,
            )
            # Reparent the grandchild to the second sibling at a valid new
            # origin. Its depth and filename stay the same; metadata must catch it.
            changed = self.hierarchy._replace(patches=(
                *self.hierarchy.patches[:2], self.hierarchy.patches[2]._replace(parent=2),
            ))
            with self.assertRaisesRegex(ValueError, "topology changed"):
                FMRPatchSeriesWriter(base, dt=.01).write(
                    fields, output_index=1, simulation_step=7, time=.07,
                    spacings=self.spacings,
                    origins=(*self.origins[:3], (4.5, 1.5, 1.5)),
                    ghost_cells=(0, 4, 4, 4), hierarchy=changed,
                )
            self.assertFalse((base.parent / "restart_level_00_patch_000_00000001.h5").exists())
            FMRPatchSeriesWriter(base, dt=.01).write(
                fields, output_index=1, simulation_step=7, time=.07, **geometry,
            )
            lines = base.with_suffix(".visit").read_text().splitlines()
            self.assertEqual(lines[1], "!TIME 0.0")
            self.assertEqual(lines[6], "!TIME 0.07")


if __name__ == "__main__":
    unittest.main()
