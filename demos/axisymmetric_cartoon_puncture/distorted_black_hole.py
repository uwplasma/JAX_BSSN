"""Evolve a puncture with a superposed axisymmetric Brill-wave distortion."""

import argparse
from pathlib import Path

from brill_wave_common import (
    build_distorted_puncture_plane_data,
    evolve_axisymmetric_demo,
)


MASS = 1.0
AMPLITUDE = 0.25
RHO_MAX = 24.0
Z_HALF_WIDTH = 24.0
NUM_RADIAL_POINTS = 192
NUM_Z_POINTS = 384
CFL = 0.2
FINAL_TIME = 30.0
SNAPSHOT_COUNT = 120

RHO_WIDTH = 4.0
Z_WIDTH = 4.0
CENTER_Z = 0.0


def run_distorted_black_hole(
    mass: float = MASS,
    amplitude: float = AMPLITUDE,
    rho_max: float = RHO_MAX,
    z_half_width: float = Z_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    rho_width: float = RHO_WIDTH,
    z_width: float = Z_WIDTH,
    center_z: float = CENTER_Z,
    output_dir: str | Path = "output",
    show_progress: bool = True,
):
    """Build a distorted puncture slice and evolve its ringdown."""

    dx = rho_max / num_radial_points
    plane_vars, initial_data = build_distorted_puncture_plane_data(
        num_radial_points,
        num_z_points,
        dx,
        mass,
        amplitude,
        rho_width,
        z_width,
        center_z=center_z,
    )
    vars, evolution = evolve_axisymmetric_demo(
        plane_vars,
        rho_max=rho_max,
        z_half_width=z_half_width,
        num_radial_points=num_radial_points,
        num_z_points=num_z_points,
        cfl=cfl,
        final_time=final_time,
        snapshot_count=snapshot_count,
        output_dir=output_dir,
        output_stem="distorted_black_hole",
        progress_label="Evolving distorted puncture",
        show_progress=show_progress,
    )

    note_path = Path(output_dir) / "distorted_black_hole_notes.txt"
    with note_path.open("w", encoding="utf-8") as note_file:
        note_file.write(
            "# Superposed Brill-wave puncture initial data\n"
            "# This demo multiplies a Schwarzschild puncture by a unit-determinant\n"
            "# Brill deformation without solving the mixed elliptic constraint.\n"
            "# Monitor the written constraint norms when tuning amplitude.\n"
        )

    return vars, {
        **initial_data,
        **evolution,
        "mass": mass,
        "amplitude": amplitude,
        "rho_width": rho_width,
        "z_width": z_width,
        "note_path": note_path,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=MASS)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--rho-max", type=float, default=RHO_MAX)
    parser.add_argument("--z-half-width", type=float, default=Z_HALF_WIDTH)
    parser.add_argument("--num-rho", type=int, default=NUM_RADIAL_POINTS)
    parser.add_argument("--num-z", type=int, default=NUM_Z_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT)
    parser.add_argument("--rho-width", type=float, default=RHO_WIDTH)
    parser.add_argument("--z-width", type=float, default=Z_WIDTH)
    parser.add_argument("--center-z", type=float, default=CENTER_Z)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_distorted_black_hole(
        mass=args.mass,
        amplitude=args.amplitude,
        rho_max=args.rho_max,
        z_half_width=args.z_half_width,
        num_radial_points=args.num_rho,
        num_z_points=args.num_z,
        cfl=args.cfl,
        final_time=args.final_time,
        snapshot_count=args.snapshots,
        rho_width=args.rho_width,
        z_width=args.z_width,
        center_z=args.center_z,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()