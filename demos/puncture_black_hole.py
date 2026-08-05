import argparse
import gc
import glob
import os
import subprocess
import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from JAX_BSSN import setup_jax_config
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.initialization import create_coordinate_arrays, get_initial_data


def resolve_ffmpeg_binary():
    """Locate the static FFmpeg binary regardless of script execution path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath("demos/ffmpeg-7.0.2-amd64-static/ffmpeg"),
        os.path.abspath("ffmpeg-7.0.2-amd64-static/ffmpeg"),
        os.path.join(script_dir, "ffmpeg-7.0.2-amd64-static/ffmpeg"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return "ffmpeg"


STATIC_FFMPEG = resolve_ffmpeg_binary()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve moving puncture black hole and render 2D heatmap movies."
    )
    parser.add_argument("--nx", type=int, default=800, help="Grid points per dimension")
    parser.add_argument("--domain-radius", type=float, default=30.0, help="Domain half-length L")
    parser.add_argument("--mass", type=float, default=1.0, help="Puncture mass M")
    parser.add_argument("--final-time", type=float, default=30.0, help="Final simulation time")
    parser.add_argument("--dt-factor", type=float, default=0.025, help="Time step ratio relative to dx (CFL condition)")
    parser.add_argument("--eta", type=float, default=2.0, help="Gamma-driver shift parameter")
    parser.add_argument("--gamma-driver", type=float, default=2.5, help="Gamma-driver parameter")
    parser.add_argument("--ko", type=float, default=0.2, help="Kreiss-Oliger dissipation coefficient")
    parser.add_argument("--snapshot-every", type=int, default=50, help="Capture frame every N steps")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second in MP4 output")
    parser.add_argument("--output-dir", default="movies_2d", help="Directory to save output movies")
    return parser.parse_args()


def extract_xy_plane(field_3d):
    """Safely transfer the 3D field or slice to CPU numpy before indexing."""
    cpu_arr = np.asarray(jax.device_get(field_3d))
    if cpu_arr.ndim == 3:
        return cpu_arr[:, :, 0]
    elif cpu_arr.ndim == 2:
        return cpu_arr
    else:
        return cpu_arr[..., 0]


def main():
    setup_jax_config(enable_x64=False, verbose=True)
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "_temp_frames")
    os.makedirs(frames_dir, exist_ok=True)

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

    render_pipeline = {}
    for key, ylabel_str, title_str, getter, cmap_choice in fields_config:
        fig, ax = plt.subplots(figsize=(8, 7))
        initial_data = extract_xy_plane(getter(vars))
        initial_data = np.nan_to_num(initial_data, nan=0.0, posinf=1e5, neginf=-1e5)

        im = ax.imshow(initial_data.T, extent=extent, origin="lower", cmap=cmap_choice)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(ylabel_str, fontsize=14)
        ax.set_xlabel("x (code units)", fontsize=11)
        ax.set_ylabel("y (code units)", fontsize=11)
        ax.set_title(title_str, fontsize=13)
        time_text = ax.text(
            0.03, 0.95, "", transform=ax.transAxes, fontsize=12, color="white", verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
        )
        render_pipeline[key] = {
            "fig": fig,
            "im": im,
            "time_text": time_text,
            "getter": getter,
        }

    frame_counter = 0

    def record_frame_snapshots(t_val, step_num, frame_idx):
        for key, pipe in render_pipeline.items():
            data_2d = extract_xy_plane(pipe["getter"](vars))

            # Clean NaNs/Infs to prevent corrupt PNG headers
            data_2d = np.nan_to_num(data_2d, nan=0.0, posinf=1e5, neginf=-1e5)

            pipe["im"].set_array(data_2d.T)

            v_min, v_max = float(np.min(data_2d)), float(np.max(data_2d))
            if v_min == v_max:
                v_min -= 1e-5
                v_max += 1e-5
            pipe["im"].set_clim(vmin=v_min, vmax=v_max)
            pipe["time_text"].set_text(f"step {step_num}/{num_steps}, t = {t_val:.3f}")

            frame_path = os.path.join(frames_dir, f"{key}_{frame_idx:05d}.png")
            pipe["fig"].savefig(frame_path, dpi=120, bbox_inches="tight")

        jax.block_until_ready(vars)
        gc.collect()

    print(f"\n--- Starting 2D Evolution (Nt = {num_steps}, dx = {dx:.4e}, dt = {dt:.4e}) ---")

    record_frame_snapshots(0.0, 0, frame_counter)
    frame_counter += 1

    for step in tqdm(range(1, num_steps + 1)):
        vars = rk4_step(vars, params)
        if step % args.snapshot_every == 0 or step == num_steps:
            record_frame_snapshots(step * dt, step, frame_counter)
            frame_counter += 1

    for key, pipe in render_pipeline.items():
        plt.close(pipe["fig"])

    print(f"\n--- Evolution complete. Compiling MP4 videos with FFmpeg ({STATIC_FFMPEG}) ---")

    for key, _, _, _, _ in fields_config:
        pattern = os.path.join(frames_dir, f"{key}_%05d.png")
        movie_filename = os.path.join(args.output_dir, f"moving_puncture_2D_{key}.mp4")

        cmd = [
            STATIC_FFMPEG,
            "-y",
            "-framerate", str(args.fps),
            "-i", pattern,
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Pads odd image dimensions (e.g. 907x748 -> 908x748)
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            movie_filename,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Generated: {movie_filename}")
        else:
            print(f"Failed to generate {movie_filename}. Error output:\n{result.stderr}")

    # Clean up temporary PNG frames
    for f in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(f)
    if os.path.exists(frames_dir):
        os.rmdir(frames_dir)

    print(f"\nSuccessfully generated all MP4 videos!")


if __name__ == "__main__":
    main()