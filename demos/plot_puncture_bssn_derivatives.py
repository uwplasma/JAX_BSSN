import argparse
import os
import sys

# Ensure Python can import JAX_BSSN from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import jax

from JAX_BSSN import setup_jax_config
from JAX_BSSN.bssn import (
    BSSNParameters,
    evolve_conformal_factor,
    evolve_conformal_connection,
    evolve_conformal_metric,
    evolve_lapse,
    evolve_shift,
    evolve_trace_extrinsic_curvature,
    evolve_traceless_extrinsic_curvature,
)
from JAX_BSSN.evolve import evolve_shift_or_freeze
from JAX_BSSN.initialization import create_coordinate_arrays, get_initial_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot initial time derivatives of BSSN variables."
    )
    parser.add_argument("--nx", type=int, default=400, help="Grid points per dimension")
    parser.add_argument("--domain-radius", type=float, default=30.0, help="Domain half-length L")
    parser.add_argument("--mass", type=float, default=1.0, help="Puncture mass M")
    parser.add_argument("--eta", type=float, default=2.0, help="Gamma-driver shift parameter")
    parser.add_argument("--gamma-driver", type=float, default=2.5, help="Gamma-driver parameter")
    parser.add_argument("--ko", type=float, default=0.02, help="Kreiss-Oliger dissipation coefficient")
    parser.add_argument("--use-supergaussian", action="store_true", help="Enable super-Gaussian boundary condition")
    parser.add_argument("--output-file", default="bssn_initial_derivatives.png", help="Filename to save output plot")
    return parser.parse_args()


def extract_xy_plane(field_3d):
    """Extract 2D xy-plane slice from JAX/GPU array to CPU numpy array."""
    if field_3d.ndim > 3:  # Handles tensors like gamma_ij or shift_i
        gpu_slice = field_3d[..., 0]
    else:
        gpu_slice = field_3d[:, :, 0]
    return np.array(jax.device_get(gpu_slice))


def main():
    setup_jax_config(enable_x64=False, verbose=True)
    args = parse_args()

    nx = args.nx
    ny = args.nx
    nz = 1
    dx = 2.0 * args.domain_radius / nx
    dt = 0.1 * dx

    # Set boundary flag (0 for Periodic, 1 for Super-Gaussian)
    bc_flag = 1 if args.use_supergaussian else 0

    params = BSSNParameters(
        eta=args.eta,
        kappa=0.02,
        nu=args.ko,
        g=args.gamma_driver,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=bc_flag, xr_bc=bc_flag,
        yl_bc=bc_flag, yr_bc=bc_flag,
        zl_bc=bc_flag, zr_bc=bc_flag,
    )

    # 1. Initialize variables
    vars = get_initial_data(
        "puncture_black_hole",
        nx, ny, nz, dx,
        mass=args.mass,
        lapse_puncture=True,
        center_between_points=True,
    )

    print("Computing initial time derivatives (RHS)...")
    # 2. Compute RHS / time derivatives at timestep 0
    dt_gamma = evolve_conformal_metric(vars, params)
    dt_W = evolve_conformal_factor(vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(vars, params)
    dt_K = evolve_trace_extrinsic_curvature(vars, params)
    dt_Gamma = evolve_conformal_connection(vars, params)
    dt_alpha = evolve_lapse(vars, params)
    dt_beta = evolve_shift_or_freeze(vars, params)

    # List of derivatives to display
    deriv_fields = [
        (r"$\partial_t \alpha$", extract_xy_plane(dt_alpha), "coolwarm"),
        (r"$\partial_t W$", extract_xy_plane(dt_W), "coolwarm"),
        (r"$\partial_t \beta^x$", extract_xy_plane(dt_beta[0]), "coolwarm"),
        (r"$\partial_t \tilde{\Gamma}^x$", extract_xy_plane(dt_Gamma[0]), "coolwarm"),
        (r"$\partial_t \tilde{\gamma}_{xx}$", extract_xy_plane(dt_gamma[0, 0]), "coolwarm"),
        (r"$\partial_t \tilde{\gamma}_{yy}$", extract_xy_plane(dt_gamma[1, 1]), "coolwarm"),
        (r"$\partial_t \tilde{A}_{xx}$", extract_xy_plane(dt_A[0, 0]), "coolwarm"),
        (r"$\partial_t \tilde{A}_{yy}$", extract_xy_plane(dt_A[1, 1]), "coolwarm"),
        (r"$\partial_t K$", extract_xy_plane(dt_K), "coolwarm"),
    ]

    # 3. Render 3x3 grid of initial rates of change
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    extent = [-args.domain_radius, args.domain_radius, -args.domain_radius, args.domain_radius]

    for ax, (title, data, cmap) in zip(axes.flat, deriv_fields):
        im = ax.imshow(data.T, extent=extent, origin="lower", cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)

    bc_type_str = "Super-Gaussian" if args.use_supergaussian else "Periodic"
    fig.suptitle(f"BSSN Time Derivatives at t = 0 ({bc_type_str} BCs)", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=200)
    print(f"Plot saved successfully to {args.output_file}")


if __name__ == "__main__":
    main()