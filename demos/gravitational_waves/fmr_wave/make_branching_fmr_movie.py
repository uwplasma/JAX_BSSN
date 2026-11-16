"""Render saved FMR wave samples and their actual patch/tile geometry.

Edit the settings below, then run ``python make_branching_fmr_movie.py``.
Only the slice-normal direction is interpolated; transverse samples stay at
their native resolution and movie frames use saved times without interpolation.
"""

from itertools import product
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib

matplotlib.use("Agg")

from matplotlib import colormaps
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import openpmd_api as io

from JAX_BSSN.diagnostics.mesh_coordinates import mesh_axis_coordinates


# Movie settings. Paths are relative to this script, independent of the cwd.
METADATA_PATH = SCRIPT_DIR / "output" / "branching_fmr.json"
OUTPUT_PATH = METADATA_PATH.with_suffix(".mp4")
FPS = 12
FRAME_STRIDE = 1
FIGURE_SIZE = (16, 9)
DPI = 120
BITRATE = 4500
COLOR_MAP = "RdBu_r"
VIEW_ELEVATION = 24
VIEW_AZIMUTH = -58


def prepare_geometry(patch, mesh, component):
    """Cache native display edges and tile splits shared by both views."""
    coordinates = tuple(mesh_axis_coordinates(mesh, component, axis) for axis in range(3))
    edges = tuple(mesh_axis_coordinates(mesh, component, axis, edges=True) for axis in range(3))
    # The plot ends at the owned vertices, not half a cell beyond each patch.
    for axis in range(3):
        edges[axis][0], edges[axis][-1] = coordinates[axis][0], coordinates[axis][-1]
    patch["edges"] = edges
    patch["lower"] = np.asarray([axis[0] for axis in coordinates])
    patch["upper"] = np.asarray([axis[-1] for axis in coordinates])
    # Adjacent owned tile vertex sets meet halfway between their samples.
    patch["tile_splits"] = tuple(
        patch["origin"][axis] + (np.arange(1, count)*width - .5)*patch["dx"]
        for axis, (count, width) in enumerate(zip(patch["tile_counts"], patch["tile_shape"]))
    )


def slice_field(field, patch, normal, coordinate=0.):
    """Interpolate saved samples only along the normal to a shared plane."""
    index = (coordinate - patch["origin"][normal])/patch["dx"]
    if index < -1.e-10 or index > field.shape[normal] - 1 + 1.e-10:
        return None
    index = np.clip(index, 0., field.shape[normal] - 1)
    lower = int(np.floor(index))
    upper = min(lower + 1, field.shape[normal] - 1)
    fraction = index - lower
    return ((1 - fraction)*np.take(field, lower, axis=normal)
            + fraction*np.take(field, upper, axis=normal))


def load_frames(metadata_path, stride):
    """Read chronological native output and retain just the two plotted slices."""
    metadata = json.loads(metadata_path.read_text())
    if not np.isfinite(metadata["amplitude"]) or metadata["amplitude"] == 0.:
        raise ValueError("movie normalization requires finite nonzero amplitude")
    manifest = metadata_path.parent / metadata["manifest"]
    patches = metadata["patches"]
    lines = manifest.read_text().splitlines()
    count = int(lines[0].split()[1])
    if count != len(patches):
        raise ValueError("metadata and manifest have different patch counts")
    snapshots = [(float(lines[start].split()[1]), lines[start + 1:start + 1 + count])
                 for start in range(1, len(lines), count + 1)]
    history = metadata.get("history")
    if history is not None:
        # A shorter rerun can leave later groups in the fixed-series manifest.
        # The sidecar records exactly the contiguous outputs from this run.
        if len(history) > len(snapshots):
            raise ValueError("metadata records more snapshots than the manifest")
        snapshots = snapshots[:len(history)]
        if any(not np.isclose(saved["time"], time)
               for saved, (time, _) in zip(history, snapshots)):
            raise ValueError("metadata history times disagree with the manifest")
    if not snapshots:
        raise ValueError("no saved snapshots are recorded for this run")
    selected = list(range(0, len(snapshots), stride))
    if selected[-1] != len(snapshots) - 1:
        selected.append(len(snapshots) - 1)
    frames = []
    limit = 1.
    for snapshot_index in selected:
        time, names = snapshots[snapshot_index]
        slices = []
        for patch, name in zip(patches, names):
            # VisIt uses .opmd aliases; openPMD selects its backend from .h5.
            series = io.Series(str((manifest.parent / name).resolve()), io.Access.read_only)
            try:
                saved_index = next(iter(series.iterations))
                iteration = series.iterations[saved_index]
                if (saved_index != snapshot_index or (history is not None
                        and int(iteration.get_attribute("simulationStep")) != history[snapshot_index]["step"])):
                    raise ValueError(f"saved index/step disagrees with metadata: {name}")
                mesh = iteration.meshes["h_plus"]
                component = mesh[io.Mesh_Record_Component.SCALAR]
                if (tuple(component.shape) != tuple(patch["shape"])
                        or not np.allclose(mesh.grid_global_offset, patch["origin"])
                        or not np.allclose(mesh.grid_spacing, patch["dx"])
                        or not np.allclose(component.position, 0.)
                        or not np.isclose(iteration.time, time)):
                    raise ValueError(f"saved geometry/time disagrees with metadata: {name}")
                if not frames:
                    prepare_geometry(patch, mesh, component)
                pending = component.load_chunk()
                series.flush()
                field = np.array(pending, copy=True)/metadata["amplitude"]
                if not np.isfinite(field).all():
                    raise ValueError(f"non-finite numerical wave in {name}")
                slices.append({axis: slice_field(field, patch, axis) for axis in (2, 1)})
                limit = max(limit, float(np.max(np.abs(field))))
                iteration.close()
            finally:
                series.close()
        frames.append((time, slices))
    return metadata, frames, 1.02*limit


def uncovered_rectangles(rectangle, covered):
    """Clip a native sample's display pixel against child patch rectangles."""
    pieces = [rectangle]
    for child in covered:
        remainder = []
        for x0, x1, y0, y1 in pieces:
            lo_x, hi_x = max(x0, child[0]), min(x1, child[1])
            lo_y, hi_y = max(y0, child[2]), min(y1, child[3])
            if lo_x >= hi_x or lo_y >= hi_y:
                remainder.append((x0, x1, y0, y1))
                continue
            if x0 < lo_x:
                remainder.append((x0, lo_x, y0, y1))
            if hi_x < x1:
                remainder.append((hi_x, x1, y0, y1))
            if y0 < lo_y:
                remainder.append((lo_x, hi_x, y0, lo_y))
            if hi_y < y1:
                remainder.append((lo_x, hi_x, hi_y, y1))
        pieces = remainder
    return pieces


def plane_geometry(patches, normal):
    """Build native pixel faces, removing parent regions covered by children."""
    axes = [axis for axis in range(3) if axis != normal]
    polygons, samples = [], []
    for index, patch in enumerate(patches):
        lo, hi = patch["lower"], patch["upper"]
        if not lo[normal] <= 0. <= hi[normal]:
            continue
        covered = []
        for child in patches:
            if child["parent"] != patch["id"]:
                continue
            child_lo, child_hi = child["lower"], child["upper"]
            if child_lo[normal] <= 0. <= child_hi[normal]:
                covered.append((child_lo[axes[0]], child_hi[axes[0]],
                                child_lo[axes[1]], child_hi[axes[1]]))
        a, b = (patch["edges"][axis] for axis in axes)
        for i, j in product(range(len(a) - 1), range(len(b) - 1)):
            for x0, x1, y0, y1 in uncovered_rectangles(
                    (a[i], a[i + 1], b[j], b[j + 1]), covered):
                corners = np.zeros((4, 3))
                corners[:, axes[0]] = (x0, x1, x1, x0)
                corners[:, axes[1]] = (y0, y0, y1, y1)
                polygons.append(corners)
                samples.append((index, normal, i, j))
    return polygons, samples


def box_lines(lo, hi):
    """The twelve edges of a patch's inclusive owned-vertex box."""
    lines = []
    for axis in range(3):
        transverse = [i for i in range(3) if i != axis]
        for choices in product((0, 1), repeat=2):
            start, stop = lo.copy(), lo.copy()
            stop[axis] = hi[axis]
            for other, upper in zip(transverse, choices):
                start[other] = stop[other] = hi[other] if upper else lo[other]
            lines.append((start, stop))
    return lines


def draw_geometry_3d(axis, patches):
    """Draw patch edges and actual tile ownership splits, without ghost cells."""
    patch_lines, tile_lines = [], []
    for patch in patches:
        lo, hi = patch["lower"], patch["upper"]
        patch_lines.extend(box_lines(lo, hi))
        for direction in range(3):
            for split in patch["tile_splits"][direction]:
                transverse = [axis for axis in range(3) if axis != direction]
                corners = np.tile(lo, (4, 1))
                corners[:, direction] = split
                corners[1:3, transverse[0]] = hi[transverse[0]]
                corners[2:4, transverse[1]] = hi[transverse[1]]
                tile_lines.extend((corners[i], corners[(i + 1) % 4]) for i in range(4))
    if tile_lines:
        axis.add_collection3d(Line3DCollection(tile_lines, colors="black", linewidths=.55, zorder=4))
    axis.add_collection3d(Line3DCollection(patch_lines, colors="black", linewidths=1.4, zorder=5))


def render_movie(metadata_path, output_path, *, fps=12, stride=1):
    if fps < 1 or stride < 1:
        raise ValueError("fps and stride must be positive")
    metadata, frames, limit = load_frames(metadata_path, stride)
    patches = metadata["patches"]
    root = patches[0]
    lo, hi = root["lower"], root["upper"]
    cmap, norm = colormaps[COLOR_MAP], Normalize(-limit, limit)

    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    slice_axis = figure.add_axes((.065, .23, .425, .60))
    domain_axis = figure.add_axes((.505, .16, .435, .72), projection="3d", computed_zorder=False)
    title = figure.suptitle("", fontsize=19, y=.95)
    slice_axis.set(title="Finest available samples at $z=0$", xlabel="$x$", ylabel="$y$",
                   xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
    slice_axis.set_title(slice_axis.get_title(), fontsize=14, pad=16)
    images = []
    for index in sorted(range(len(patches)), key=lambda i: patches[i]["depth"]):
        patch = patches[index]
        values = frames[0][1][index][2]
        if values is None:
            continue
        x, y = patch["edges"][:2]
        layer = 1 + 3*patch["depth"]
        image = slice_axis.pcolormesh(x, y, values.T, cmap=cmap, norm=norm,
                                     shading="flat", rasterized=True, zorder=layer)
        images.append((index, image))
        for direction in (0, 1):
            for split in patch["tile_splits"][direction]:
                if direction == 0:
                    slice_axis.plot((split, split), (y[0], y[-1]), color="black", lw=.6, zorder=layer + 1)
                else:
                    slice_axis.plot((x[0], x[-1]), (split, split), color="black", lw=.6, zorder=layer + 1)
        slice_axis.add_patch(Rectangle((x[0], y[0]), x[-1] - x[0], y[-1] - y[0],
                                      fill=False, edgecolor="black", linewidth=1.9, zorder=layer + 2))
        if index:
            slice_axis.text(x[0] + .005, y[-1] - .009, f"P{patch['id']} · L{patch['depth']}",
                            fontsize=9, ha="left", va="top", zorder=20,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))

    polygons, samples = [], []
    for normal in (2, 1):
        faces, locations = plane_geometry(patches, normal)
        polygons.extend(faces)
        samples.extend(locations)
    planes = Poly3DCollection(polygons, edgecolors="none", antialiased=False, zsort="average", zorder=1)
    domain_axis.add_collection3d(planes)
    draw_geometry_3d(domain_axis, patches)
    domain_axis.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
                    xlabel="$x$", ylabel="$y$", zlabel="$z$")
    domain_axis.set_box_aspect(hi - lo)
    domain_axis.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    domain_axis.set_proj_type("ortho")
    figure.text(.72, .85, "Whole domain · numerical planes at $y=0$, $z=0$",
                ha="center", fontsize=14)
    domain_axis.grid(False)
    color_axis = figure.add_axes((.945, .275, .014, .45))
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    color_axis.set_title("$h_+/A$", fontsize=13, pad=12)
    figure.legend((Line2D([], [], color="black", lw=1.9), Line2D([], [], color="black", lw=.6)),
                  ("Patch boundary", "Internal tile ownership boundary"),
                  loc="lower center", bbox_to_anchor=(.50, .085), ncol=2, frameon=False, fontsize=11)
    figure.text(.5, .055, "Native transverse samples; linear interpolation only normal to each slice. "
                "Tile splits lie halfway between adjacent owned vertices.",
                ha="center", fontsize=10, color=".3")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=BITRATE,
                         extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(figure, str(output_path), dpi=DPI):
        for frame_index, (time, slices) in enumerate(frames):
            title.set_text(f"Branching fixed mesh refinement · numerical gravitational wave     t = {time:.5f}")
            for index, image in images:
                image.set_array(slices[index][2].T)
            values = np.asarray([slices[index][normal][i, j] for index, normal, i, j in samples])
            planes.set_facecolor(cmap(norm(values)))
            writer.grab_frame()
            print(f"Rendered saved snapshot {frame_index + 1}/{len(frames)} at t={time:.8g}", flush=True)
    plt.close(figure)
    return output_path


def main():
    output = render_movie(METADATA_PATH.resolve(), OUTPUT_PATH.resolve(),
                          fps=FPS, stride=FRAME_STRIDE)
    print(output)
    return output


if __name__ == "__main__":
    main()
