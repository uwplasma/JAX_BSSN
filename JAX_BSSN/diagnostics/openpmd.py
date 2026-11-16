"""Synchronous openPMD output for Cartesian mesh fields."""

from collections.abc import Mapping
import os
from pathlib import Path
import re
import tempfile

import jax
import numpy as np
import openpmd_api as io


_ALIGNMENT_RTOL = 1.0e-9
_ALIGNMENT_ATOL = 1.0e-10
_VECTOR_COMPONENTS = ("x", "y", "z")


def _as_3tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value, value)


def _float_3tuple(value, name):
    values = tuple(float(item) for item in _as_3tuple(value))
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or a 3-tuple")
    return values


def _integer_3tuple(value, name):
    values = tuple(int(item) for item in _as_3tuple(value))
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or a 3-tuple")
    return values


def _ensure_openpmd_array(data):
    """Return host-owned floating-point data suitable for openPMD."""

    array = np.asarray(jax.device_get(data))
    if not np.issubdtype(array.dtype, np.floating):
        array = np.asarray(array, dtype=np.float64)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, copy=True, order="C")

    return array


def _strip_ghost_cells(field_map, ghost_cells):
    """Remove the configured outer ghost width from scalar and vector fields."""

    ghost_cells = _integer_3tuple(ghost_cells, "ghost_cells")
    if any(width < 0 for width in ghost_cells):
        raise ValueError("ghost_cells must be a non-negative scalar or 3-tuple")
    interior = tuple(
        slice(width, -width) if width > 0 else slice(None)
        for width in ghost_cells
    )

    interior_fields = {}
    for name, field in field_map.items():
        if _field_kind(field) == "vector":
            interior_fields[name] = tuple(
                component[interior] for component in field
            )
        else:
            interior_fields[name] = field[interior]

    return interior_fields


def _field_kind(field):
    is_vector = isinstance(field, (tuple, list)) and len(field) == 3
    return "vector" if is_vector else "scalar"


def _active_field_shape(field, field_name, level_name):
    components = field if _field_kind(field) == "vector" else (field,)
    shapes = [tuple(component.shape) for component in components]
    if any(len(shape) != 3 for shape in shapes):
        raise ValueError(
            f"field {field_name!r} on {level_name} must have "
            "three-dimensional components"
        )
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(
            f"vector field {field_name!r} on {level_name} has components "
            "with different active shapes"
        )
    if any(size < 1 for size in shapes[0]):
        raise ValueError(
            f"ghost stripping leaves field {field_name!r} on {level_name} "
            "with an empty axis"
        )
    return shapes[0]


def _validate_field_maps(root_fields, fine_fields):
    if not root_fields or not fine_fields:
        raise ValueError("root and fine patches must both contain fields")

    root_names = set(root_fields)
    fine_names = set(fine_fields)
    if root_names != fine_names:
        missing = sorted(root_names - fine_names)
        extra = sorted(fine_names - root_names)
        raise ValueError(
            "root and fine patches have incompatible field sets; "
            f"missing={missing}, extra={extra}"
        )

    patch_shapes = []
    for level_name, fields in (("root patch", root_fields),
                               ("fine patch", fine_fields)):
        active_shape = None
        for field_name, field in fields.items():
            shape = _active_field_shape(field, field_name, level_name)
            if active_shape is None:
                active_shape = shape
            elif shape != active_shape:
                raise ValueError(
                    f"fields on {level_name} do not share one active mesh shape"
                )
        patch_shapes.append(active_shape)

    for name in root_names:
        root_kind = _field_kind(root_fields[name])
        fine_kind = _field_kind(fine_fields[name])
        if root_kind != fine_kind:
            raise ValueError(
                f"field {name!r} changes from {root_kind} on the root patch "
                f"to {fine_kind} on the fine patch"
            )

    return tuple(patch_shapes)


def _configure_mesh(mesh, grid_spacing, grid_global_offset):
    mesh.geometry = io.Geometry.cartesian
    mesh.data_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"
    mesh.axis_labels = ["x", "y", "z"]
    mesh.grid_spacing = list(grid_spacing)
    mesh.grid_global_offset = list(grid_global_offset)
    mesh.grid_unit_SI = 1.0
    mesh.unit_SI = 1.0


def _write_scalar_mesh(
    iteration,
    name,
    data,
    grid_spacing,
    grid_global_offset,
    grid_position,
):
    mesh = iteration.meshes[name]
    _configure_mesh(mesh, grid_spacing, grid_global_offset)

    array = _ensure_openpmd_array(data)
    component = mesh[io.Mesh_Record_Component.SCALAR]
    component.position = list(grid_position)
    component.reset_dataset(io.Dataset(array.dtype, array.shape))
    component.store_chunk(array, [0, 0, 0], array.shape)
    component.unit_SI = 1.0
    return array


def _write_vector_mesh(
    iteration,
    name,
    components,
    grid_spacing,
    grid_global_offset,
    grid_position,
):
    mesh = iteration.meshes[name]
    _configure_mesh(mesh, grid_spacing, grid_global_offset)

    arrays = []
    for component_name, data in zip(_VECTOR_COMPONENTS, components):
        array = _ensure_openpmd_array(data)
        component = mesh[component_name]
        component.position = list(grid_position)
        component.reset_dataset(io.Dataset(array.dtype, array.shape))
        component.store_chunk(array, [0, 0, 0], array.shape)
        component.unit_SI = 1.0
        arrays.append(array)
    return arrays


def _write_field_map(
    series,
    iteration,
    field_map,
    grid_spacing,
    grid_global_offset,
    grid_position,
):
    # openPMD stores chunks asynchronously. Retain every host buffer until the
    # flush that makes this operation synchronous.
    pending_arrays = []
    for field_name, field in field_map.items():
        if _field_kind(field) == "vector":
            pending_arrays.extend(
                _write_vector_mesh(
                    iteration,
                    field_name,
                    field,
                    grid_spacing,
                    grid_global_offset,
                    grid_position,
                )
            )
        else:
            pending_arrays.append(
                _write_scalar_mesh(
                    iteration,
                    field_name,
                    field,
                    grid_spacing,
                    grid_global_offset,
                    grid_position,
                )
            )
    series.flush()


class OpenPMDWriter:
    """Own one synchronous openPMD series for Cartesian mesh diagnostics."""

    def __init__(
        self,
        filename,
        grid_spacing,
        grid_global_offset,
        dt,
        ghost_cells=0,
        grid_position=(0.0, 0.0, 0.0),
    ):
        output_path = Path(filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".h5")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.filename = output_path
        self.grid_spacing = _float_3tuple(grid_spacing, "grid_spacing")
        self.grid_global_offset = _float_3tuple(
            grid_global_offset,
            "grid_global_offset",
        )
        self.grid_position = _float_3tuple(grid_position, "grid_position")
        self.dt = float(dt)
        self.ghost_cells = ghost_cells

        self.series = io.Series(str(output_path), io.Access.create)
        self.series.set_attribute("software", "JAX_BSSN")

    def _iteration(self, step, time):
        iteration = self.series.iterations[int(step)]
        iteration.time = float(time)
        iteration.dt = self.dt
        iteration.time_unit_SI = 1.0
        return iteration

    def write(self, field_map, step, time):
        """Write one iteration after removing ghost cells from each field."""

        iteration = self._iteration(step, time)
        interior_fields = _strip_ghost_cells(field_map, self.ghost_cells)
        _write_field_map(
            self.series,
            iteration,
            interior_fields,
            self.grid_spacing,
            self.grid_global_offset,
            self.grid_position,
        )
        iteration.close()

    def close(self):
        if self.series is not None:
            self.series.close()
            self.series = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def tiled_output_fields(field_map, layout):
    """Copy owned tile samples into host arrays for single-process output.

    Scalars and vector component tuples follow the ordinary writer format.
    Only this diagnostic/output boundary assembles complete patch arrays.
    """
    if jax.process_count() != 1:
        raise ValueError("tiled FMR output currently supports one process")
    counts = tuple(layout["mesh"].devices.shape)
    widths = tuple(layout["tile_shape"])
    endpoints = tuple(layout.get("upper_endpoints", (False,) * 3))
    g = layout.get("guard_cells", 4)
    shape = tuple(count * width + int(endpoint)
                  for count, width, endpoint in zip(counts, widths, endpoints))

    def owned_array(value):
        if value.shape != counts + tuple(width + 2*g for width in widths):
            raise ValueError("output field does not match its tile layout")
        result = np.empty(shape, dtype=value.dtype)
        for shard in value.addressable_shards:
            starts = tuple(index.start or 0 for index in shard.index[:3])
            buffer = np.asarray(jax.device_get(shard.data))
            for local_tile in np.ndindex(buffer.shape[:3]):
                tile = tuple(start + index for start, index in zip(starts, local_tile))
                owned = tuple(width + int(endpoint and index == count - 1)
                              for width, endpoint, index, count in
                              zip(widths, endpoints, tile, counts))
                source = tuple(slice(g, g + n) for n in owned)
                target = tuple(slice(index * width, index * width + n)
                               for index, width, n in zip(tile, widths, owned))
                result[target] = buffer[local_tile + source]
        return result

    return {name: tuple(owned_array(component) for component in field)
            if _field_kind(field) == "vector" else owned_array(field)
            for name, field in field_map.items()}


class FMRPatchSeriesWriter:
    """Write a fixed patch tree as one native-resolution series per patch."""

    def __init__(self, base_name, dt):
        base_path = Path(base_name)
        if base_path.suffix in (".h5", ".visit", ".pmd", ".opmd"):
            base_path = base_path.with_suffix("")
        base_path.parent.mkdir(parents=True, exist_ok=True)

        self.base_path = base_path
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.visit_path = base_path.with_suffix(".visit")
        self._topology = None
        self._level_count = None

    def _pattern(self, level, patch, suffix="h5"):
        return self.base_path.parent / (
            f"{self.base_path.name}_level_{level:02d}_patch_{patch:03d}_"
            f"%08T.{suffix}"
        )

    def _output_path(self, level, patch, output_index, suffix="h5"):
        return Path(
            str(self._pattern(level, patch, suffix)).replace(
                "%08T", f"{output_index:08d}"
            )
        )

    def _helper_path(self, level, patch):
        return self.base_path.parent / (
            f"{self.base_path.name}_level_{level:02d}_patch_{patch:03d}.pmd"
        )

    @staticmethod
    def _iteration_attributes(index, active_shape, hierarchy, simulation_step):
        from JAX_BSSN.fmr import hierarchy_parents, hierarchy_depths

        parents = hierarchy_parents(hierarchy)
        depths = hierarchy_depths(hierarchy)
        depth = depths[index]
        patch = sum(value == depth for value in depths[:index])
        if index == 0:
            start = (0, 0, 0)
            stop = tuple(int(size) for size in active_shape)
            ratio = 1
        else:
            spec = hierarchy.patches[index - 1]
            start = tuple(int(value) for value in spec.coarse_lo)
            stop = tuple(int(value) + 1 for value in spec.coarse_hi)
            ratio = int(spec.refinement_ratio)

        attributes = {
            "fmrLevel": np.int64(depth),
            "fmrPatch": np.int64(patch),
            "fmrParent": np.int64(parents[index]),
            "refinementRatio": np.int64(ratio),
            "coarseStart": np.asarray(start, dtype=np.int64),
            "coarseStop": np.asarray(stop, dtype=np.int64),
            "simulationStep": np.int64(simulation_step),
        }
        # Existing chain files retain their exact metadata. Branching trees
        # need an explicit flat index so fmrParent identifies its parent.
        if parents != tuple(range(-1, len(parents) - 1)):
            attributes["fmrId"] = np.int64(index)
        return attributes

    def _write_patch(
        self,
        level,
        field_map,
        active_shape,
        spacing,
        origin,
        grid_position,
        hierarchy,
        output_index,
        simulation_step,
        time,
    ):
        pattern = self._pattern(*self._patches[level])
        series = io.Series(str(pattern), io.Access.create)
        try:
            series.set_attribute("software", "JAX_BSSN")
            iteration = series.iterations[output_index]
            iteration.time = float(time)
            iteration.dt = self.dt
            iteration.time_unit_SI = 1.0
            for name, value in self._iteration_attributes(
                level, active_shape, hierarchy, simulation_step
            ).items():
                iteration.set_attribute(name, value)

            _write_field_map(
                series,
                iteration,
                field_map,
                spacing,
                origin,
                grid_position,
            )
            iteration.close()
        finally:
            series.close()

    def _validate_topology(self, fields, spacings, origins, positions, hierarchy):
        from JAX_BSSN.fmr import fine_active_shape, hierarchy_parents, hierarchy_depths

        parents = hierarchy_parents(hierarchy)
        depths = hierarchy_depths(hierarchy)
        self._patches = tuple((depth, depths[:i].count(depth))
                              for i, depth in enumerate(depths))
        count = len(fields)
        if count == 0 or len(hierarchy.patches) != count - 1:
            raise ValueError("hierarchy output requires one field map per patch")
        if not (len(spacings) == len(origins) == len(positions) == count):
            raise ValueError("per-level output geometry lengths do not match")
        shapes = []
        root_names = set(fields[0])
        root_kinds = {name: _field_kind(field) for name, field in fields[0].items()}
        for level, field_map in enumerate(fields):
            if not isinstance(field_map, Mapping) or set(field_map) != root_names:
                raise ValueError("FMR levels have incompatible field sets")
            if ({name: _field_kind(field) for name, field in field_map.items()}
                    != root_kinds):
                raise ValueError("field scalar/vector kinds must match across levels")
            shape = _active_field_shape(next(iter(field_map.values())),
                                        next(iter(field_map)), f"level {level}")
            for name, field in field_map.items():
                if _active_field_shape(field, name, f"level {level}") != shape:
                    raise ValueError(f"level {level} fields have inconsistent shapes")
            shapes.append(shape)
            if any(value <= 0.0 for value in spacings[level]):
                raise ValueError("patch grid spacings must be positive")
            if level:
                parent = parents[level]
                spec = hierarchy.patches[level - 1]
                if int(spec.refinement_ratio) != 2:
                    raise ValueError("FMR output supports only 2:1 refinement")
                expected = fine_active_shape(spec)
                if shape != expected:
                    raise ValueError(
                        f"level {level} active shape {shape} does not match {expected}"
                    )
                if not np.allclose(np.asarray(spacings[level]) * 2.0,
                                   spacings[parent], rtol=_ALIGNMENT_RTOL,
                                   atol=_ALIGNMENT_ATOL):
                    raise ValueError("successive FMR spacings must differ by two")
                expected_origin = tuple(
                    origin + lo * spacing for origin, lo, spacing in zip(
                        origins[parent], spec.coarse_lo, spacings[parent]
                    )
                )
                if not np.allclose(origins[level], expected_origin,
                                   rtol=_ALIGNMENT_RTOL, atol=_ALIGNMENT_ATOL):
                    raise ValueError("child active origin is inconsistent with parent")
        signature = (
            tuple(shapes), tuple(spacings), tuple(origins), tuple(positions), parents,
            tuple((tuple(p.coarse_lo), tuple(p.coarse_hi), p.refinement_ratio)
                  for p in hierarchy.patches), tuple(sorted(root_kinds.items())),
        )
        if self._topology is not None and signature != self._topology:
            raise ValueError("FMR patch-series topology changed between outputs")
        self._topology, self._level_count = signature, count
        return tuple(shapes)

    def _expected_alias_names(self, output_index, level_count):
        return tuple(self._output_path(level, patch, output_index, "opmd").name
                     for level, patch in self._patches)

    def _parse_manifest(self, level_count):
        if not self.visit_path.exists():
            return {}
        if not self.visit_path.is_file() or self.visit_path.is_symlink():
            raise ValueError(f"VisIt manifest path is not a regular file: {self.visit_path}")

        lines = self.visit_path.read_text(encoding="utf-8").splitlines()
        header = f"!NBLOCKS {level_count}"
        if not lines or lines[0] != header:
            raise ValueError(f"VisIt manifest must begin with exactly {header!r}")
        group_size = level_count + 1
        if (len(lines) - 1) % group_size != 0:
            raise ValueError("VisIt manifest contains an incomplete block group")

        groups = {}
        name = re.escape(self.base_path.name)
        patterns = [re.compile(
            rf"^{name}_level_{level:02d}_patch_{patch:03d}_(\d{{8}})\.opmd$"
        ) for level, patch in self._patches]
        for offset in range(1, len(lines), group_size):
            time_line = lines[offset]
            aliases = tuple(lines[offset + 1:offset + group_size])
            if not time_line.startswith("!TIME "):
                raise ValueError("VisIt manifest group is missing its !TIME line")
            try:
                time = float(time_line[6:])
            except ValueError as exc:
                raise ValueError("VisIt manifest contains an invalid time") from exc
            if not np.isfinite(time):
                raise ValueError("VisIt manifest contains a non-finite time")
            matches = [pattern.fullmatch(alias)
                       for pattern, alias in zip(patterns, aliases)]
            if any(match is None for match in matches):
                raise ValueError(
                    "VisIt manifest block order or patch topology is invalid"
                )
            indices_for_levels = [int(match.group(1)) for match in matches]
            if len(set(indices_for_levels)) != 1:
                raise ValueError("VisIt manifest patch output indices do not match")
            output_index = indices_for_levels[0]
            if output_index in groups:
                raise ValueError(
                    f"VisIt manifest duplicates output index {output_index}"
                )
            groups[output_index] = (time, *aliases)
        indices = sorted(groups)
        if indices != list(range(len(indices))):
            raise ValueError(
                "VisIt manifest output indices must be contiguous and start at zero"
            )
        ordered_times = [groups[index][0] for index in indices]
        if any(later < earlier and not np.isclose(
            later, earlier, rtol=_ALIGNMENT_RTOL, atol=_ALIGNMENT_ATOL
        ) for earlier, later in zip(ordered_times, ordered_times[1:])):
            raise ValueError(
                "VisIt manifest times must be nondecreasing by output index"
            )
        return groups

    @staticmethod
    def _ensure_relative_alias(alias_path, target_path):
        target = target_path.name
        if alias_path.is_symlink():
            if os.readlink(alias_path) == target:
                return
            raise FileExistsError(
                f"alias {alias_path} points to {os.readlink(alias_path)!r}, "
                f"expected {target!r}"
            )
        if os.path.lexists(alias_path):
            raise FileExistsError(
                f"refusing to overwrite conflicting alias path {alias_path}"
            )
        alias_path.symlink_to(target)

    @staticmethod
    def _preflight_relative_alias(alias_path, target_path):
        if alias_path.is_symlink():
            if os.readlink(alias_path) != target_path.name:
                raise FileExistsError(
                    f"alias {alias_path} points to {os.readlink(alias_path)!r}, "
                    f"expected {target_path.name!r}"
                )
        elif os.path.lexists(alias_path):
            raise FileExistsError(
                f"refusing to overwrite conflicting alias path {alias_path}"
            )

    @staticmethod
    def _ensure_helper(helper_path, pattern_name):
        expected = pattern_name + "\n"
        if os.path.lexists(helper_path):
            if (helper_path.is_file() and not helper_path.is_symlink()
                    and helper_path.read_text(encoding="utf-8") == expected):
                return
            raise FileExistsError(
                f"refusing to overwrite conflicting ParaView helper {helper_path}"
            )
        helper_path.write_text(expected, encoding="utf-8")

    def _rewrite_hierarchy_manifest(self, output_index, time, level_count):
        groups = self._parse_manifest(level_count)
        aliases = self._expected_alias_names(output_index, level_count)
        if output_index in groups:
            old_time, *old_aliases = groups[output_index]
            if not np.isclose(old_time, time, rtol=_ALIGNMENT_RTOL,
                              atol=_ALIGNMENT_ATOL):
                raise ValueError(
                    f"output index {output_index} already has conflicting time"
                )
            if tuple(old_aliases) != aliases:
                raise ValueError("saved hierarchy aliases conflict")
        groups[output_index] = (float(time), *aliases)
        indices = sorted(groups)
        if indices != list(range(len(indices))):
            raise ValueError("saved output indices must be contiguous and start at zero")
        times = [groups[index][0] for index in indices]
        if any(later < earlier and not np.isclose(
            later, earlier, rtol=_ALIGNMENT_RTOL, atol=_ALIGNMENT_ATOL
        ) for earlier, later in zip(times, times[1:])):
            raise ValueError("VisIt manifest times must be nondecreasing")
        lines = [f"!NBLOCKS {level_count}"]
        for index in indices:
            group_time, *group_aliases = groups[index]
            lines.append(f"!TIME {group_time}")
            lines.extend(group_aliases)
        contents = "\n".join(lines) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.visit_path.parent, prefix=f".{self.visit_path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.visit_path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def write(
        self,
        level_fields,
        *,
        output_index,
        simulation_step,
        time,
        spacings,
        origins,
        ghost_cells,
        hierarchy,
        grid_positions=None,
        layouts=None,
    ):
        """Write every patch, optionally extracting owned tiled samples.

        Tiled export supports one process and copies addressable owned chunks
        to host memory; it never assembles a global field on an accelerator.
        Origins are active patch origins for both array layouts.
        Already-owned patch maps use zero ghost cells on every patch and no
        layouts; their shapes and hierarchy metadata are validated as usual.
        """
        level_fields = tuple(level_fields)
        level_count = len(level_fields)
        output_index, simulation_step = int(output_index), int(simulation_step)
        time = float(time)
        if output_index < 0 or simulation_step < 0:
            raise ValueError("output_index and simulation_step must be non-negative")
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        if len(ghost_cells) != level_count:
            raise ValueError("one ghost-cell specification is required per level")
        spacings = tuple(_float_3tuple(value, "spacing") for value in spacings)
        origins = tuple(_float_3tuple(value, "origin") for value in origins)
        if grid_positions is None:
            grid_positions = ((0.0, 0.0, 0.0),) * level_count
        positions = tuple(_float_3tuple(value, "grid_position")
                          for value in grid_positions)
        ghosts = tuple(_integer_3tuple(value, "ghost_cells")
                       for value in ghost_cells)
        if layouts is None:
            if ghosts[0] != (0, 0, 0):
                raise ValueError("the FMR root level must not have ghost cells")
            for level in range(1, level_count):
                expected = (int(hierarchy.patches[level - 1].ghost_width),) * 3
                if ghosts[level] not in ((0, 0, 0), expected):
                    raise ValueError("child ghost width does not match hierarchy")
            active_fields = tuple(_strip_ghost_cells(fields, ghost)
                                  for fields, ghost in zip(level_fields, ghosts))
        else:
            if jax.process_count() != 1:
                raise ValueError("tiled FMR output currently supports one process")
            if len(layouts) != level_count:
                raise ValueError("one tile layout is required per patch")
            if any(ghost != (layout.get("guard_cells", 4),) * 3
                   for ghost, layout in zip(ghosts, layouts)):
                raise ValueError("ghost width does not match the tile layout")
            active_fields = tuple(tiled_output_fields(fields, layout)
                                  for fields, layout in zip(level_fields, layouts))
        shapes = self._validate_topology(
            active_fields, spacings, origins, positions, hierarchy
        )

        existing = self._parse_manifest(level_count)
        candidates = sorted(set(existing) | {output_index})
        if candidates != list(range(len(candidates))):
            raise ValueError("saved output indices must be contiguous and start at zero")
        if output_index in existing and not np.isclose(
            existing[output_index][0], time, rtol=_ALIGNMENT_RTOL,
            atol=_ALIGNMENT_ATOL,
        ):
            raise ValueError("output index already has conflicting time")
        if existing:
            # Appending after a process restart must preserve the topology
            # recorded by the previous output, just as rewriting an index does.
            saved_index = output_index if output_index in existing else max(existing)
            for level in range(level_count):
                path = self._output_path(*self._patches[level], saved_index, "h5")
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"VisIt manifest references a missing patch: {path}")
                series = io.Series(str(path), io.Access.read_only)
                try:
                    iteration = series.iterations[saved_index]
                    saved_step = int(iteration.get_attribute("simulationStep"))
                    if saved_index == output_index and saved_step != simulation_step:
                        raise ValueError("output index has conflicting simulationStep")
                    if not np.isclose(iteration.dt, self.dt,
                                      rtol=_ALIGNMENT_RTOL, atol=_ALIGNMENT_ATOL):
                        raise ValueError("FMR patch-series solver timestep changed")
                    expected_attributes = self._iteration_attributes(
                        level, shapes[level], hierarchy, saved_step
                    )
                    for name, expected in expected_attributes.items():
                        if not np.array_equal(
                            np.asarray(iteration.get_attribute(name)),
                            np.asarray(expected),
                        ):
                            raise ValueError(
                                f"FMR topology changed for attribute {name}"
                            )
                    if set(iteration.meshes) != set(active_fields[level]):
                        raise ValueError("FMR field set changed on restart")
                    for name in active_fields[level]:
                        mesh = iteration.meshes[name]
                        if not np.allclose(mesh.grid_spacing, spacings[level],
                                           rtol=_ALIGNMENT_RTOL,
                                           atol=_ALIGNMENT_ATOL):
                            raise ValueError("FMR spacing changed on restart")
                        if not np.allclose(mesh.grid_global_offset, origins[level],
                                           rtol=_ALIGNMENT_RTOL,
                                           atol=_ALIGNMENT_ATOL):
                            raise ValueError("FMR origin changed on restart")
                        for component_name in mesh:
                            component = mesh[component_name]
                            if tuple(component.shape) != shapes[level]:
                                raise ValueError("FMR active shape changed on restart")
                            if not np.allclose(component.position, positions[level],
                                               rtol=_ALIGNMENT_RTOL,
                                               atol=_ALIGNMENT_ATOL):
                                raise ValueError("FMR grid position changed on restart")
                    iteration.close()
                finally:
                    series.close()
        for level in range(level_count):
            h5_path = self._output_path(*self._patches[level], output_index, "h5")
            alias_path = self._output_path(*self._patches[level], output_index, "opmd")
            self._preflight_relative_alias(alias_path, h5_path)

        for level in range(level_count):
            self._write_patch(
                level, active_fields[level], shapes[level], spacings[level],
                origins[level], positions[level], hierarchy, output_index,
                simulation_step, time,
            )
        if jax.process_index() == 0:
            for level in range(level_count):
                h5_path = self._output_path(*self._patches[level], output_index, "h5")
                alias_path = self._output_path(*self._patches[level], output_index, "opmd")
                self._ensure_relative_alias(alias_path, h5_path)
                self._ensure_helper(
                    self._helper_path(*self._patches[level]),
                    self._pattern(*self._patches[level], "h5").name
                )
            self._rewrite_hierarchy_manifest(output_index, time, level_count)
        return self.visit_path

    def close(self):
        """Retained for context-manager symmetry; writes close immediately."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
