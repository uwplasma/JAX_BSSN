import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import openpmd_api as io
import pytest

from JAX_BSSN.bssn.variables import BSSNVariables
from JAX_BSSN.cartoon.spherical_symmetry import (
    cartoon_axis_output_fields,
    compact_cartoon_state,
)
from JAX_BSSN.diagnostics.openpmd import FMRPatchSeriesWriter, OpenPMDWriter
from JAX_BSSN.fmr.refinement import FMRHierarchySpec, FMRPatchSpec


PATCH_SPEC = FMRPatchSpec((1, 1, 1), (3, 3, 3))
FIELD_NAMES = {
    "h_plus",
    "lapse",
    "shift",
    "K",
    "W",
    "hamiltonian_constraint",
    "momentum_constraint",
}


def _diagnostic_field_maps():
    root_base = np.arange(5**3, dtype=np.float64).reshape((5, 5, 5))
    fine_active = 1000.0 + np.arange(5**3, dtype=np.float64).reshape((5, 5, 5))
    fine_base = np.full((13, 13, 13), -999.0, dtype=np.float64)
    fine_base[4:-4, 4:-4, 4:-4] = fine_active

    def fields(base):
        return {
            "h_plus": base,
            "lapse": base + 1.0,
            "shift": (base + 2.0, base + 3.0, base + 4.0),
            "K": base + 5.0,
            "W": base + 6.0,
            "hamiltonian_constraint": base + 7.0,
            "momentum_constraint": (
                base + 8.0,
                base + 9.0,
                base + 10.0,
            ),
        }

    return fields(jnp.asarray(root_base)), fields(jnp.asarray(fine_base)), root_base, fine_active


def _write_output(writer, output_index=0, simulation_step=0, time=0.0, **kwargs):
    root_fields, fine_fields, root_base, fine_active = _diagnostic_field_maps()
    patch_spec = kwargs.pop("patch_spec", PATCH_SPEC)
    geometry = dict(
        spacings=((1.0,) * 3, (0.5,) * 3),
        origins=((-2.0,) * 3, (-1.0,) * 3),
        ghost_cells=(0, 4),
        hierarchy=FMRHierarchySpec((patch_spec,)),
    )
    geometry.update(kwargs)
    result = writer.write(
        (root_fields, fine_fields),
        output_index=output_index,
        simulation_step=simulation_step,
        time=time,
        **geometry,
    )
    return result, root_fields, fine_fields, root_base, fine_active


def _read_patch(path, output_index):
    series = io.Series(str(path), io.Access.read_only)
    iteration = series.iterations[output_index]
    pending = {}
    metadata = {}
    for mesh_name in iteration.meshes:
        mesh = iteration.meshes[mesh_name]
        components = {}
        for component_name in mesh:
            component = mesh[component_name]
            components[component_name] = component.load_chunk()
        pending[mesh_name] = components
        first_component = mesh[next(iter(mesh))]
        metadata[mesh_name] = {
            "components": set(mesh),
            "spacing": tuple(mesh.grid_spacing),
            "origin": tuple(mesh.grid_global_offset),
            "position": tuple(first_component.position),
            "shape": tuple(first_component.shape),
            "geometry": mesh.geometry,
            "axis_labels": list(mesh.axis_labels),
            "data_order": mesh.data_order,
            "grid_unit_SI": mesh.grid_unit_SI,
            "unit_SI": mesh.unit_SI,
            "component_unit_SI": first_component.unit_SI,
        }
    attributes = {
        name: iteration.get_attribute(name) for name in iteration.attributes
    }
    time = iteration.time
    dt = iteration.dt
    series.flush()
    arrays = {
        mesh_name: {
            component_name: np.array(array, copy=True)
            for component_name, array in components.items()
        }
        for mesh_name, components in pending.items()
    }
    iteration.close()
    series.close()
    return {
        "arrays": arrays,
        "metadata": metadata,
        "attributes": attributes,
        "time": time,
        "dt": dt,
    }


def _patch_path(base, level, output_index, suffix="h5"):
    return base.parent / (
        f"{base.name}_level_{level:02d}_patch_000_{output_index:08d}.{suffix}"
    )


def test_patch_series_writes_native_root_and_fine_arrays_and_metadata(tmp_path):
    base = tmp_path / "linear_wave_fmr"
    writer = FMRPatchSeriesWriter(base, dt=0.125)
    _, root_fields, fine_fields, root_base, fine_active = _write_output(
        writer, output_index=0, simulation_step=11, time=0.25
    )

    root = _read_patch(_patch_path(base, 0, 0), 0)
    fine = _read_patch(_patch_path(base, 1, 0), 0)
    scalar = io.Mesh_Record_Component.SCALAR

    assert set(root["arrays"]) == set(fine["arrays"]) == FIELD_NAMES
    assert root["metadata"]["h_plus"]["components"] == {scalar}
    assert root["metadata"]["shift"]["components"] == {"x", "y", "z"}
    assert fine["metadata"]["momentum_constraint"]["components"] == {
        "x", "y", "z"
    }
    np.testing.assert_array_equal(root["arrays"]["h_plus"][scalar], root_base)
    np.testing.assert_array_equal(fine["arrays"]["h_plus"][scalar], fine_active)
    np.testing.assert_array_equal(
        fine["arrays"]["shift"]["z"], fine_active + 4.0
    )
    assert not np.any(fine["arrays"]["h_plus"][scalar] == -999.0)

    for patch, shape, spacing, origin in (
        (root, (5, 5, 5), (1.0, 1.0, 1.0), (-2.0, -2.0, -2.0)),
        (fine, (5, 5, 5), (0.5, 0.5, 0.5), (-1.0, -1.0, -1.0)),
    ):
        for metadata in patch["metadata"].values():
            assert metadata["shape"] == shape
            assert metadata["spacing"] == spacing
            assert metadata["origin"] == origin
            assert metadata["position"] == (0.0, 0.0, 0.0)
            assert metadata["geometry"] == io.Geometry.cartesian
            assert metadata["axis_labels"] == ["x", "y", "z"]
            expected_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"
            assert metadata["data_order"] == expected_order
            assert metadata["grid_unit_SI"] == 1.0
            assert metadata["unit_SI"] == 1.0
            assert metadata["component_unit_SI"] == 1.0
        assert patch["time"] == 0.25
        assert patch["dt"] == 0.125

    assert root["attributes"]["fmrLevel"] == 0
    assert root["attributes"]["fmrPatch"] == 0
    assert root["attributes"]["fmrParent"] == -1
    assert root["attributes"]["refinementRatio"] == 1
    assert root["attributes"]["coarseStart"] == [0, 0, 0]
    assert root["attributes"]["coarseStop"] == [5, 5, 5]
    assert root["attributes"]["simulationStep"] == 11
    assert fine["attributes"]["fmrLevel"] == 1
    assert fine["attributes"]["fmrPatch"] == 0
    assert fine["attributes"]["fmrParent"] == 0
    assert fine["attributes"]["refinementRatio"] == 2
    assert fine["attributes"]["coarseStart"] == [1, 1, 1]
    assert fine["attributes"]["coarseStop"] == [4, 4, 4]
    assert fine["attributes"]["simulationStep"] == 11

    # Slicing and device-to-host conversion must not mutate either JAX source.
    np.testing.assert_array_equal(np.asarray(root_fields["h_plus"]), root_base)
    fine_source = np.asarray(fine_fields["h_plus"])
    assert fine_source.shape == (13, 13, 13)
    assert np.all(fine_source[:4] == -999.0)


def test_default_demo_patch_has_32_root_and_31_active_fine_vertices(tmp_path):
    base = tmp_path / "linear_wave_fmr"
    dx = 1.0 / 32.0
    root_origin = (-(32 - 1) * dx / 2.0,) * 3
    patch_spec = FMRPatchSpec((8, 8, 8), (23, 23, 23))
    fine_origin = tuple(
        root_origin[axis] + patch_spec.coarse_lo[axis] * dx
        for axis in range(3)
    )
    root = np.arange(32**3, dtype=np.float64).reshape((32,) * 3)
    fine = np.full((39,) * 3, -1.0, dtype=np.float64)
    native_fine = np.arange(31**3, dtype=np.float64).reshape((31,) * 3)
    fine[4:-4, 4:-4, 4:-4] = native_fine

    FMRPatchSeriesWriter(base, dt=0.01).write(
        ({"h_plus": root}, {"h_plus": fine}),
        output_index=0,
        simulation_step=0,
        time=0.0,
        spacings=((dx,) * 3, (dx / 2.0,) * 3),
        origins=(root_origin, fine_origin),
        ghost_cells=(0, 4),
        hierarchy=FMRHierarchySpec((patch_spec,)),
    )

    root_patch = _read_patch(_patch_path(base, 0, 0), 0)
    fine_patch = _read_patch(_patch_path(base, 1, 0), 0)
    assert root_patch["metadata"]["h_plus"]["shape"] == (32, 32, 32)
    assert fine_patch["metadata"]["h_plus"]["shape"] == (31, 31, 31)


def test_patch_series_writes_three_level_parent_chain_and_manifest(tmp_path):
    base = tmp_path / "three_level"
    second_spec = FMRPatchSpec((1, 1, 1), (3, 3, 3))
    hierarchy = FMRHierarchySpec((PATCH_SPEC, second_spec))
    root = np.zeros((5, 5, 5))
    level_one = np.zeros((13, 13, 13))
    level_two = np.zeros((13, 13, 13))
    FMRPatchSeriesWriter(base, dt=0.1).write(
        ({"h_plus": root}, {"h_plus": level_one}, {"h_plus": level_two}),
        output_index=0, simulation_step=0, time=0.0,
        spacings=(1.0, 0.5, 0.25),
        origins=((-2.0,) * 3, (-1.0,) * 3, (-0.5,) * 3),
        ghost_cells=(0, 4, 4), hierarchy=hierarchy,
    )
    finest = _read_patch(_patch_path(base, 2, 0), 0)
    assert finest["attributes"]["fmrLevel"] == 2
    assert finest["attributes"]["fmrParent"] == 1
    assert finest["attributes"]["coarseStart"] == [1, 1, 1]
    lines = base.with_suffix(".visit").read_text().splitlines()
    assert lines[0] == "!NBLOCKS 3"
    assert lines[3].startswith("three_level_level_02_patch_000_")
    assert fine_patch["metadata"]["h_plus"]["spacing"] == (dx / 2.0,) * 3
    assert fine_patch["metadata"]["h_plus"]["origin"] == fine_origin
    np.testing.assert_array_equal(
        fine_patch["arrays"]["h_plus"][io.Mesh_Record_Component.SCALAR],
        np.asarray(native_fine),
    )


def test_cartoon_output_writes_complete_reflected_axis_with_parity(tmp_path):
    num_radial_points = 6
    full_nx = 2 * num_radial_points
    shape = (full_nx, 1, 1)
    positive = jnp.arange(num_radial_points, dtype=jnp.float64) + 1.0

    scalar = jnp.zeros(shape).at[num_radial_points:, 0, 0].set(positive)
    vector = jnp.zeros((3,) + shape)
    tensor = jnp.zeros((3, 3) + shape)
    for component in range(3):
        vector = vector.at[component, num_radial_points:, 0, 0].set(
            (component + 1.0) * positive
        )
    for i in range(3):
        for j in range(3):
            tensor = tensor.at[i, j, num_radial_points:, 0, 0].set(
                (3.0 * i + j + 1.0) * positive
            )

    full_vars = BSSNVariables(
        conformal_metric=tensor,
        conformal_factor=scalar,
        traceless_K=2.0 * tensor,
        trace_K=2.0 * scalar,
        conformal_connection=vector,
        lapse=3.0 * scalar,
        shift=2.0 * vector,
    )
    compact_vars = compact_cartoon_state(full_vars)
    fields = cartoon_axis_output_fields(
        compact_vars,
        4.0 * compact_vars.conformal_factor,
        5.0 * compact_vars.conformal_connection,
    )

    filename = tmp_path / "cartoon_axis.h5"
    with OpenPMDWriter(
        filename,
        grid_spacing=(0.25, 0.25, 0.25),
        grid_global_offset=(-1.375, 0.0, 0.0),
        dt=0.01,
    ) as writer:
        writer.write(fields, step=0, time=0.0)

    series = io.Series(str(filename), io.Access.read_only)
    iteration = series.iterations[0]
    pending = {}
    metadata = {}
    for mesh_name in iteration.meshes:
        mesh = iteration.meshes[mesh_name]
        pending[mesh_name] = {
            component_name: mesh[component_name].load_chunk()
            for component_name in mesh
        }
        metadata[mesh_name] = (
            tuple(mesh.grid_spacing),
            tuple(mesh.grid_global_offset),
        )
    series.flush()
    meshes = {
        mesh_name: {
            component_name: np.array(array, copy=True)
            for component_name, array in components.items()
        }
        for mesh_name, components in pending.items()
    }
    iteration.close()
    series.close()

    expected_names = {
        "W",
        "K",
        "lapse",
        "shift",
        "conformal_connection",
        "conformal_metric_xx",
        "conformal_metric_xy",
        "conformal_metric_xz",
        "conformal_metric_yy",
        "conformal_metric_yz",
        "conformal_metric_zz",
        "traceless_K_xx",
        "traceless_K_xy",
        "traceless_K_xz",
        "traceless_K_yy",
        "traceless_K_yz",
        "traceless_K_zz",
        "hamiltonian_constraint",
        "momentum_constraint",
    }
    assert set(meshes) == expected_names

    for name, components in meshes.items():
        assert metadata[name] == ((0.25, 0.25, 0.25), (-1.375, 0.0, 0.0))
        for array in components.values():
            assert array.shape == (full_nx, 1, 1)

    scalar_component = io.Mesh_Record_Component.SCALAR
    W = meshes["W"][scalar_component][:, 0, 0]
    np.testing.assert_array_equal(W[:num_radial_points], positive[::-1])
    np.testing.assert_array_equal(W[num_radial_points:], positive)

    shift = meshes["shift"]
    np.testing.assert_array_equal(
        shift["x"][:num_radial_points, 0, 0],
        -2.0 * np.asarray(positive[::-1]),
    )
    np.testing.assert_array_equal(
        shift["y"][:num_radial_points, 0, 0],
        4.0 * np.asarray(positive[::-1]),
    )

    metric_xy = meshes["conformal_metric_xy"][scalar_component][:, 0, 0]
    metric_yz = meshes["conformal_metric_yz"][scalar_component][:, 0, 0]
    np.testing.assert_array_equal(
        metric_xy[:num_radial_points], -2.0 * np.asarray(positive[::-1])
    )
    np.testing.assert_array_equal(
        metric_yz[:num_radial_points], 6.0 * np.asarray(positive[::-1])
    )


def test_output_indices_simulation_steps_aliases_helpers_and_manifest_are_restart_safe(tmp_path):
    base = tmp_path / "linear_wave_fmr"
    writer = FMRPatchSeriesWriter(base, dt=0.1)
    _write_output(writer, output_index=0, simulation_step=0, time=0.0)
    _write_output(writer, output_index=1, simulation_step=7, time=0.7)

    for output_index in (0, 1):
        for level in (0, 1):
            h5_path = _patch_path(base, level, output_index)
            alias_path = _patch_path(base, level, output_index, "opmd")
            assert h5_path.is_file()
            assert alias_path.is_symlink()
            assert os.readlink(alias_path) == h5_path.name
        patch = _read_patch(_patch_path(base, 0, output_index), output_index)
        assert patch["attributes"]["simulationStep"] == (0, 7)[output_index]
        assert patch["time"] == (0.0, 0.7)[output_index]

    assert (tmp_path / "linear_wave_fmr_level_00_patch_000.pmd").read_text() == (
        "linear_wave_fmr_level_00_patch_000_%08T.h5\n"
    )
    assert (tmp_path / "linear_wave_fmr_level_01_patch_000.pmd").read_text() == (
        "linear_wave_fmr_level_01_patch_000_%08T.h5\n"
    )
    assert not (tmp_path / "linear_wave_fmr.pmd").exists()
    expected_manifest = (
        "!NBLOCKS 2\n"
        "!TIME 0.0\n"
        "linear_wave_fmr_level_00_patch_000_00000000.opmd\n"
        "linear_wave_fmr_level_01_patch_000_00000000.opmd\n"
        "!TIME 0.7\n"
        "linear_wave_fmr_level_00_patch_000_00000001.opmd\n"
        "linear_wave_fmr_level_01_patch_000_00000001.opmd\n"
    )
    assert base.with_suffix(".visit").read_text() == expected_manifest

    # A fresh writer represents a restarted process. Rewriting an identical
    # saved index retains correct aliases and does not duplicate the group.
    restarted = FMRPatchSeriesWriter(base, dt=0.1)
    _write_output(restarted, output_index=1, simulation_step=7, time=0.7)
    assert base.with_suffix(".visit").read_text() == expected_manifest


@pytest.mark.parametrize("conflict_kind", ["wrong_symlink", "file", "directory"])
def test_patch_series_refuses_conflicting_alias_paths(tmp_path, conflict_kind):
    base = tmp_path / "linear_wave_fmr"
    alias = _patch_path(base, 0, 0, "opmd")
    if conflict_kind == "wrong_symlink":
        alias.symlink_to("unrelated.h5")
    elif conflict_kind == "file":
        alias.write_text("not an alias")
    else:
        alias.mkdir()

    with pytest.raises(FileExistsError, match="alias"):
        _write_output(FMRPatchSeriesWriter(base, dt=0.1))
    assert not _patch_path(base, 0, 0).exists()


@pytest.mark.parametrize(
    "manifest",
    [
        "!NBLOCKS 1\n",
        "!NBLOCKS 2\n!TIME 0.0\nroot.opmd\n",
        (
            "!NBLOCKS 2\n!TIME 0.0\n"
            "linear_wave_fmr_level_01_patch_000_00000000.opmd\n"
            "linear_wave_fmr_level_00_patch_000_00000000.opmd\n"
        ),
        (
            "!NBLOCKS 2\n!TIME 0.0\n"
            "linear_wave_fmr_level_00_patch_000_00000000.opmd\n"
            "linear_wave_fmr_level_01_patch_000_00000001.opmd\n"
        ),
    ],
)
def test_patch_series_rejects_malformed_incomplete_or_changed_manifests(
    tmp_path, manifest
):
    base = tmp_path / "linear_wave_fmr"
    base.with_suffix(".visit").write_text(manifest)
    with pytest.raises(ValueError, match="manifest|block|NBLOCKS|indices"):
        _write_output(FMRPatchSeriesWriter(base, dt=0.1))


def test_patch_series_rejects_conflicting_manifest_times(tmp_path):
    base = tmp_path / "linear_wave_fmr"
    writer = FMRPatchSeriesWriter(base, dt=0.1)
    _write_output(writer, output_index=0, simulation_step=0, time=0.0)
    with pytest.raises(ValueError, match="conflicting time"):
        _write_output(writer, output_index=0, simulation_step=0, time=0.25)

    with pytest.raises(ValueError, match="conflicting simulationStep"):
        _write_output(writer, output_index=0, simulation_step=3, time=0.0)


def test_patch_series_rejects_sparse_output_indices_before_writing(tmp_path):
    base = tmp_path / "linear_wave_fmr"
    with pytest.raises(ValueError, match="contiguous"):
        _write_output(
            FMRPatchSeriesWriter(base, dt=0.1),
            output_index=2,
            simulation_step=9,
            time=0.9,
        )
    assert not _patch_path(base, 0, 2).exists()


def test_patch_series_rejects_topology_and_field_changes(tmp_path):
    base = tmp_path / "linear_wave_fmr"
    writer = FMRPatchSeriesWriter(base, dt=0.1)
    _write_output(writer)

    root_fields, fine_fields, _, _ = _diagnostic_field_maps()
    fine_fields = dict(fine_fields)
    fine_fields.pop("K")
    with pytest.raises(ValueError, match="incompatible field sets"):
        writer.write(
            (root_fields, fine_fields),
            output_index=1,
            simulation_step=1,
            time=0.1,
            spacings=(1.0, 0.5),
            origins=((-2.0,) * 3, (-1.0,) * 3),
            ghost_cells=(0, 4),
            hierarchy=FMRHierarchySpec((PATCH_SPEC,)),
        )

    with pytest.raises(ValueError, match="topology changed"):
        writer.write(
            (root_fields, _diagnostic_field_maps()[1]),
            output_index=1,
            simulation_step=1,
            time=0.1,
            spacings=(1.0, 0.5),
            origins=((-2.0,) * 3, (-1.0,) * 3),
            ghost_cells=(0, 4),
            hierarchy=FMRHierarchySpec((PATCH_SPEC,)),
            grid_positions=((0.5,) * 3, (0.5,) * 3),
        )

    non_ratio_two = PATCH_SPEC._replace(refinement_ratio=4)
    with pytest.raises(ValueError, match="2:1"):
        _write_output(
            FMRPatchSeriesWriter(tmp_path / "ratio", dt=0.1),
            patch_spec=non_ratio_two,
        )


def test_uniform_write_keeps_existing_single_grid_behavior(tmp_path):
    field = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    filename = tmp_path / "uniform.h5"

    with OpenPMDWriter(
        filename,
        grid_spacing=(0.5, 0.25, 0.125),
        grid_global_offset=(-1.0, -2.0, -3.0),
        dt=0.1,
    ) as writer:
        writer.write({"chi": field}, step=0, time=0.0)

    mesh = _read_patch(filename, 0)
    scalar = io.Mesh_Record_Component.SCALAR
    assert mesh["metadata"]["chi"]["spacing"] == (0.5, 0.25, 0.125)
    assert mesh["metadata"]["chi"]["origin"] == (-1.0, -2.0, -3.0)
    np.testing.assert_array_equal(mesh["arrays"]["chi"][scalar], field)
