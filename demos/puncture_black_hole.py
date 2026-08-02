import argparse
import os
import jax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from JAX_BSSN import setup_jax_config
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.initialization import create_coordinate_arrays, get_initial_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve moving puncture black hole and stream 2D heatmap movies."
    )
    parser.add_argument("--nx", type=int, default=800, help="Grid points per dimension")
    parser.add_argument("--domain-radius", type=float, default=30.0, help="Domain half-length L")
    parser.add_argument("--mass", type=float, default=1.0, help="Puncture mass M")
    parser.add_argument("--final-time", type=float, default=30.0, help="Final simulation time")
    parser.add_argument("--dt-factor", type=float, default=0.1, help="Time step ratio relative to dx")
    parser.add_argument("--eta", type=float, default=2.0, help="Gamma-driver shift parameter")
    parser.add_argument("--gamma-driver", type=float, default=2.5, help="Gamma-driver parameter")
    parser.add_argument("--ko", type=float, default=0.02, help="Kreiss-Oliger dissipation coefficient")
    parser.add_argument("--snapshot-every", type=int, default=50, help="Capture frame every N steps")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second in MP4 output")
    parser.add_argument("--output-dir", default="movies_2d", help="Directory to save output movies")
    return parser.parse_args()


def extract_xy_plane(field_3d):
    """Extract 2D xy-plane directly on device, then copy to CPU."""
    gpu_slice = field_3d[:, :, 0]
    return np.array(jax.device_get(gpu_slice))


def main():
    setup_jax_config(enable_x64=False, verbose=True)
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    nx = args.nx
    ny = args.nx
    nz = 1
    dx = 2.0 * args.domain_radius / nx
    dt = args.dt_factor * dx
    num_steps = int(round(args.final_time / dt))

    X, Y, Z = create_coordinate_arrays(nx, ny, nz, dx)

    params = BSSNParameters(
        eta=args.eta,
        kappa=0.02,
        nu=args.ko,
        g=args.gamma_driver,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=0, 
        xr_bc=0,
        yl_bc=0,
        yr_bc=0,
        zl_bc=0,
        zr_bc=0,
    )

    vars = get_initial_data(
        "puncture_black_hole",
        nx,
        ny,
        nz,
        dx,
        mass=args.mass,
        lapse_puncture=True,
        center_between_points=True,
    )

    # Configuration mapping key -> (Label, Title, Getter, Colormap)
    fields_config = [
        ("alpha", r"$\alpha$", "Moving-puncture alpha evolution", lambda v: v.lapse, "viridis"),
        ("chi", r"$\chi$", "Moving-puncture chi evolution", lambda v: v.conformal_factor, "viridis"),
        ("beta", r"$\beta^x$", "Moving-puncture beta evolution", lambda v: v.shift[0], "coolwarm"),
        ("Gamma", r"$\tilde{\Gamma}^x$", "Moving-puncture Gamma evolution", lambda v: v.conformal_connection[0], "coolwarm"),
        ("conformal_gxx", r"$\tilde{\gamma}_{xx}$", "Moving-puncture conformal_gxx evolution", lambda v: v.conformal_metric[0, 0], "viridis"),
        ("conformal_gt", r"$\tilde{\gamma}_T$", "Moving-puncture conformal_gt evolution", lambda v: v.conformal_metric[1, 1], "viridis"),
        ("Axx", r"$\tilde{A}_{xx}$", "Moving-puncture Axx evolution", lambda v: v.traceless_K[0, 0], "coolwarm"),
        ("At", r"$\tilde{A}_T$", "Moving-puncture At evolution", lambda v: v.traceless_K[1, 1], "coolwarm"),
        ("Kh", r"$K$", "Moving-puncture Kh evolution", lambda v: v.trace_K, "viridis"),
    ]

    extent = [-args.domain_radius, args.domain_radius, -args.domain_radius, args.domain_radius]
    
    # Store dynamic render state
    render_pipeline = {}

    for key, ylabel_str, title_str, getter, cmap_choice in fields_config:
        fig, ax = plt.subplots(figsize=(8, 7))
        
        initial_data = extract_xy_plane(getter(vars))
        im = ax.imshow(
            initial_data.T, 
            extent=extent, 
            origin='lower', 
            cmap=cmap_choice, 
            animated=True
        )
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(ylabel_str, fontsize=14)
        
        ax.set_xlabel("x (code units)", fontsize=11)
        ax.set_ylabel("y (code units)", fontsize=11)
        ax.set_title(title_str, fontsize=13)

        time_text = ax.text(
            0.03, 0.95, "", transform=ax.transAxes, fontsize=12, color='white', verticalalignment="top",
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

        movie_filename = os.path.join(args.output_dir, f"moving_puncture_2D_{key}.mp4")
        
        # Unique writer instance per file prevents MP4 stream corruption
        writer = FFMpegWriter(fps=args.fps, metadata=dict(artist="JAX_BSSN"), bitrate=3000)
        writer_ctx = writer.saving(fig, movie_filename, dpi=150)
        writer_ctx.__enter__()

        render_pipeline[key] = {
            "fig": fig,
            "im": im,
            "time_text": time_text,
            "writer": writer,
            "writer_ctx": writer_ctx,
            "getter": getter,
        }

    def record_and_stream_frame(t_val, step_num):
        for key in render_pipeline:
            pipe = render_pipeline[key]
            data_2d = extract_xy_plane(pipe["getter"](vars))
            
            pipe["im"].set_array(data_2d.T)
            
            # Dynamic min/max color limits so feature changes stay visible
            v_min, v_max = float(np.nanmin(data_2d)), float(np.nanmax(data_2d))
            if v_min == v_max:
                v_min -= 1e-5
                v_max += 1e-5
            pipe["im"].set_clim(vmin=v_min, vmax=v_max)

            pipe["time_text"].set_text(f"step {step_num}/{num_steps}, t = {t_val:.3f}")
            pipe["writer"].grab_frame()

    print(f"\n--- Starting 2D Evolution (Nt = {num_steps}, dx = {dx:.4e}, dt = {dt:.4e}) ---")
    
    # Record t=0 frame
    record_and_stream_frame(0.0, 0)

    # Evolution Loop
    for step in tqdm(range(1, num_steps + 1)):
        vars = rk4_step(vars, params)
        if step % args.snapshot_every == 0 or step == num_steps:
            record_and_stream_frame(step * dt, step)

    # Clean up and finalize MP4 containers cleanly
    for key in render_pipeline:
        pipe = render_pipeline[key]
        pipe["writer_ctx"].__exit__(None, None, None)
        plt.close(pipe["fig"])

    print(f"\nSuccessfully generated all {len(fields_config)} MP4 videos cleanly!")


if __name__ == "__main__":
    main()