"""Evolve an axisymmetric Brill wave and scan amplitudes across collapse."""

import argparse
from pathlib import Path

from brill_wave_common import (
    BRILL_MAX_ITERATIONS,
    BRILL_RELAXATION,
    BRILL_TOLERANCE,
    build_brill_wave_plane_data,
    evolve_axisymmetric_demo,
)


AMPLITUDE = 3.0
RHO_MAX = 20.0
Z_HALF_WIDTH = 20.0
NUM_RADIAL_POINTS = 192
NUM_Z_POINTS = 384
CFL = 0.2
FINAL_TIME = 30.0
SNAPSHOT_COUNT = 120

RHO_WIDTH = 3.0
Z_WIDTH = 3.0
CENTER_Z = 0.0


def run_brill_wave_collapse(
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
    tolerance: float = BRILL_TOLERANCE,
    max_iterations: int = BRILL_MAX_ITERATIONS,
    relaxation: float = BRILL_RELAXATION,
    show_progress: bool = True,
):
    """Solve Brill-wave initial data and evolve the resulting wave packet."""

    dx = rho_max / num_radial_points
    plane_vars, initial_data = build_brill_wave_plane_data(
        num_radial_points,
        num_z_points,
        dx,
        amplitude,
        rho_width,
        z_width,
        center_z=center_z,
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
        output_stem="brill_wave_collapse",
        progress_label="Evolving Brill wave",
        show_progress=show_progress,
    )

    residual_path = Path(output_dir) / "brill_wave_initial_residual.txt"
    with residual_path.open("w", encoding="utf-8") as residual_file:
        residual_file.write("# iteration max_update\n")
        for iteration, residual in enumerate(initial_data["residual_history"]):
            residual_file.write(f"{iteration:d} {residual:.16e}\n")

    return vars, {
        **initial_data,
        **evolution,
        "amplitude": amplitude,
        "rho_width": rho_width,
        "z_width": z_width,
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
    parser.add_argument("--rho-width", type=float, default=RHO_WIDTH)
    parser.add_argument("--z-width", type=float, default=Z_WIDTH)
    parser.add_argument("--center-z", type=float, default=CENTER_Z)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--tolerance", type=float, default=BRILL_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=BRILL_MAX_ITERATIONS)
    parser.add_argument("--relaxation", type=float, default=BRILL_RELAXATION)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_brill_wave_collapse(
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
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()