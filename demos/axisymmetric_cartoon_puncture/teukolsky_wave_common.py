"""Evolve a robust axisymmetric Teukolsky-wave collapse case."""

import argparse
from pathlib import Path

from brill_wave_common import (
    BRILL_MAX_ITERATIONS,
    BRILL_RELAXATION,
    BRILL_TOLERANCE,
    build_teukolsky_wave_plane_data,
    evolve_axisymmetric_demo,
)


AMPLITUDE = 0.0018
RHO_MAX = 12.0
Z_HALF_WIDTH = 12.0
NUM_RADIAL_POINTS = 192
NUM_Z_POINTS = 384
CFL = 0.2
FINAL_TIME = 40.0
SNAPSHOT_COUNT = 160

WIDTH = 0.5
CENTER_RADIUS = 2.0


def run_teukolsky_wave_common(
    amplitude: float = AMPLITUDE,
    rho_max: float = RHO_MAX,
    z_half_width: float = Z_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    width: float = WIDTH,
    center_radius: float = CENTER_RADIUS,
    output_dir: str | Path = "output",
    tolerance: float = BRILL_TOLERANCE,
    max_iterations: int = BRILL_MAX_ITERATIONS,
    relaxation: float = BRILL_RELAXATION,
    show_progress: bool = True,
):
    """Build and evolve one axisymmetric Teukolsky wave with zero shift."""

    dx = rho_max / num_radial_points
    plane_vars, initial_data = build_teukolsky_wave_plane_data(
        num_radial_points,
        num_z_points,
        dx,
        amplitude,
        width,
        center_radius,
        tolerance=tolerance,
        max_iterations=max_iterations,
        relaxation=relaxation,
        verbose=show_progress,
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
        output_stem="teukolsky_wave_common",
        progress_label="Evolving Teukolsky wave",
        zero_shift=1,
        show_progress=show_progress,
    )

    residual_path = Path(output_dir) / "teukolsky_wave_initial_residual.txt"
    with residual_path.open("w", encoding="utf-8") as residual_file:
        residual_file.write("# iteration max_update\n")
        for iteration, residual in enumerate(initial_data["residual_history"]):
            residual_file.write(f"{iteration:d} {residual:.16e}\n")

    return vars, {
        **initial_data,
        **evolution,
        "amplitude": amplitude,
        "width": width,
        "center_radius": center_radius,
        "residual_path": residual_path,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--rho-max", type=float, default=RHO_MAX)
    parser.add_argument("--z-half-width", type=float, default=Z_HALF_WIDTH)
    parser.add_argument("--num-rho", type=int, default=NUM_RADIAL_POINTS)
    parser.add_argument("--num-z", type=int, default=NUM_Z_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT)
    parser.add_argument("--width", type=float, default=WIDTH)
    parser.add_argument("--center-radius", type=float, default=CENTER_RADIUS)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--tolerance", type=float, default=BRILL_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=BRILL_MAX_ITERATIONS)
    parser.add_argument("--relaxation", type=float, default=BRILL_RELAXATION)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_teukolsky_wave_common(
        amplitude=args.amplitude,
        rho_max=args.rho_max,
        z_half_width=args.z_half_width,
        num_radial_points=args.num_rho,
        num_z_points=args.num_z,
        cfl=args.cfl,
        final_time=args.final_time,
        snapshot_count=args.snapshots,
        width=args.width,
        center_radius=args.center_radius,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()