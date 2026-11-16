"""Recover the Newtonian potential of a static uniform-density planet with periodic BSSN.

A uniform sphere (Earth's radius and mean density by default) sits at the centre of a
periodic box, in geometric units (G = c = 1, lengths in metres). Conformally flat,
constant-K initial data are solved from the periodic Hamiltonian constraint, then the
full BSSN system is evolved with the planet as a static dust source through
compute_bssn_rhs_with_matter. With 1+log slicing the lapse relaxes until dK/dt ~ 0,
which in the weak field is Poisson's equation, so 1/2(alpha^2 - 1) and -tr(delta gamma)/6
both approach the Newtonian potential. They are compared, after removing the periodic
zero mode, with the FFT solution of lap(Phi) = 4 pi (rho - <rho>).

Writes output/static_planetary_mass.png. The functions are also imported by
static_planetary_mass_convergence.py.
"""

import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs_with_matter, enforce_algebraic_constraints


G_OVER_C2 = 6.67430e-11 / 2.99792458e8**2  # converts kg to m (and kg/m^3 to 1/m^2)
RADIUS = 6.3817e6                         # planet radius [m]
RHO_SI = 5513.0                           # planet density [kg/m^3]
BOX_WIDTH = 10.0 * RADIUS                 # periodic box width [m]
NUM_POINTS = 256                          # grid points per dimension
CFL = 0.4                                 # dt / dx
FINAL_TIME = 2.97e8                       # evolution time [m], several box light-crossing times
RADIAL_BINS = 60                          # radial shells used for the profiles
OUTPUT_DIR = Path("output")


def planet_grid(num_points):
    """
    Build the cell-centred periodic grid and the planet's density.

    Args:
        num_points: Grid points per dimension

    Returns:
        Tuple (dx, r, rho): spacing [m], radius from the centre (n, n, n) [m],
        and density in geometric units (n, n, n) [1/m^2]
    """
    dx = BOX_WIDTH / num_points
    x = (jnp.arange(num_points) + 0.5) * dx - BOX_WIDTH / 2
    r = jnp.sqrt(x[:, None, None]**2 + x[None, :, None]**2 + x[None, None, :]**2)
    return dx, r, jnp.where(r < RADIUS, G_OVER_C2 * RHO_SI, 0.0)


def periodic_poisson(source, dx):
    """
    Solve lap(u) = source - <source> on the periodic box by FFT.

    Args:
        source: Right-hand side (n, n, n)
        dx: Grid spacing [m]

    Returns:
        Zero-mean periodic solution u (n, n, n)
    """
    k = 2 * jnp.pi * jnp.fft.fftfreq(source.shape[0], d=dx)
    k2 = k[:, None, None]**2 + k[None, :, None]**2 + k[None, None, :]**2
    u_hat = -jnp.fft.fftn(source) / jnp.where(k2 > 0, k2, 1.0)
    return jnp.fft.ifftn(jnp.where(k2 > 0, u_hat, 0.0)).real


def initial_data(rho, dx, tol=1e-13, max_iter=100, relaxation=0.6):
    """
    Solve the periodic conformally flat constant-K Hamiltonian constraint.

    The constraint 8 lap(psi) = (2/3 K^2 - 16 pi rho) psi^5 has a periodic solution
    only if its right-hand side averages to zero, which fixes K at every iteration.

    Args:
        rho: Density in geometric units (n, n, n) [1/m^2]
        dx: Grid spacing [m]
        tol: Stop once the largest change in psi falls below this
        max_iter: Maximum number of fixed-point iterations
        relaxation: Under-relaxation weight for each psi update

    Returns:
        Tuple (BSSNVariables, K) of the initial state and the constant trace of K [1/m]
    """
    psi = jnp.ones_like(rho)
    for _ in range(max_iter):
        psi5 = psi**5
        k2 = 24 * jnp.pi * jnp.mean(rho * psi5) / jnp.mean(psi5)
        update = 1 + periodic_poisson((k2 / 12 - 2 * jnp.pi * rho) * psi5, dx)
        change = jnp.max(jnp.abs(update - psi))
        psi = (1 - relaxation) * psi + relaxation * update
        if float(change) < tol:
            break
    shape = rho.shape
    eye = jnp.broadcast_to(jnp.eye(3)[:, :, None, None, None], (3, 3) + shape)
    vector = jnp.zeros((3,) + shape)
    lapse = jnp.sqrt(jnp.maximum(2 - psi**4, 1e-14))  # alpha^2 = 1 + 2 Phi with Phi = (1 - psi^4)/2
    K = -jnp.sqrt(k2)  # negative K: the periodic matter-filled box expands
    return BSSNVariables(eye, psi**-2, jnp.zeros_like(eye), jnp.full(shape, K), vector, lapse, vector), float(K)


@jax.jit
def rk4_step_with_matter(state, params, rho):
    """
    Advance one RK4 step with a static dust source (rho only, zero S_i and S_ij).

    Args:
        state: Current BSSNVariables
        params: BSSNParameters
        rho: Density in geometric units (n, n, n) [1/m^2]

    Returns:
        BSSNVariables after one time step, with the algebraic constraints enforced
    """
    zero_i, zero_ij = jnp.zeros((3,) + rho.shape), jnp.zeros((3, 3) + rho.shape)
    rhs = lambda s: compute_bssn_rhs_with_matter(s, params, rho, zero_i, zero_ij)
    stage = lambda k, h: enforce_algebraic_constraints(jax.tree.map(lambda q, dq: q + h * dq, state, k))
    state = enforce_algebraic_constraints(state)
    k1 = rhs(state)
    k2 = rhs(stage(k1, params.dt / 2))
    k3 = rhs(stage(k2, params.dt / 2))
    k4 = rhs(stage(k3, params.dt))
    return enforce_algebraic_constraints(jax.tree.map(
        lambda q, a, b, c, d: q + params.dt / 6 * (a + 2 * b + 2 * c + d), state, k1, k2, k3, k4))


def run_static_mass(num_points):
    """
    Solve the initial data and evolve the planet, printing progress throughout.

    Args:
        num_points: Grid points per dimension

    Returns:
        Dict with dx, r, rho, the final BSSNVariables and the potentials from
        potentials()
    """
    dx, r, rho = planet_grid(num_points)
    num_steps = math.ceil(FINAL_TIME / (CFL * dx))
    print(f"[{num_points}^3] dx = {dx:.4e} m, {num_steps} RK4 steps to t = {FINAL_TIME:.3e} m")
    print(f"[{num_points}^3] solving the periodic Hamiltonian constraint ...")
    state, K = initial_data(rho, dx)
    print(f"[{num_points}^3] constant K = {K:.4e} 1/m, max |psi - 1| = "
          f"{float(jnp.max(jnp.abs(state.conformal_factor**-0.5 - 1))):.4e}")
    params = BSSNParameters(eta=0.0, kappa=1.0, nu=0.25, g=0.0, dx=dx, dt=FINAL_TIME / num_steps,
                            zero_shift=1, gauge=1)  # frozen zero shift, 1+log slicing, periodic faces
    for _ in tqdm(range(num_steps), desc=f"Evolving {num_points}^3", unit="step"):
        state = rk4_step_with_matter(state, params, rho)
    return {"dx": dx, "r": r, "rho": rho, "state": state, **potentials(state, rho, dx)}


def potentials(state, rho, dx):
    """
    Extract the weak-field potentials and the periodic Poisson reference.

    Args:
        state: Final BSSNVariables
        rho: Density in geometric units (n, n, n) [1/m^2]
        dx: Grid spacing [m]

    Returns:
        Dict of zero-mean fields lapse, metric and poisson (n, n, n), plus the
        removed means lapse_mean and metric_mean
    """
    lapse = 0.5 * (state.lapse**2 - 1)
    gamma_trace = jnp.einsum("ii...->...", state.conformal_metric) / state.conformal_factor**2
    metric = (3 - gamma_trace) / 6  # gamma_ij = (1 - 2 Phi) delta_ij, so Phi = -tr(gamma - delta) / 6
    return {"lapse": lapse - lapse.mean(), "metric": metric - metric.mean(),
            "poisson": periodic_poisson(4 * jnp.pi * rho, dx),
            "lapse_mean": float(lapse.mean()), "metric_mean": float(metric.mean())}


def radial_profile(field, r, bins=RADIAL_BINS):
    """
    Average a 3D field over spherical shells.

    Args:
        field: 3D field (n, n, n)
        r: Radius from the centre (n, n, n) [m]
        bins: Number of shells between 0 and max(r)

    Returns:
        Tuple (shell-centre radii, shell averages), skipping empty shells
    """
    r, field = np.ravel(r), np.ravel(field)
    edges = np.linspace(0, r.max(), bins + 1)
    index = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
    count = np.bincount(index, minlength=bins)
    keep = count > 0
    return 0.5 * (edges[:-1] + edges[1:])[keep], (np.bincount(index, field, bins)[keep] / count[keep])


if __name__ == "__main__":
    mass = G_OVER_C2 * RHO_SI * 4 / 3 * math.pi * RADIUS**3
    print(f"Planet: R = {RADIUS:.4e} m, rho = {RHO_SI} kg/m^3, M = {mass:.4e} m "
          f"({mass / G_OVER_C2:.4e} kg), box = {BOX_WIDTH:.4e} m")
    result = run_static_mass(NUM_POINTS)

    r = result["r"]
    profiles = {name: radial_profile(result[name], r) for name in ("lapse", "metric", "poisson")}
    shells, phi_poisson = profiles["poisson"]
    surface = np.argmin(np.abs(shells - RADIUS))
    rms = lambda name: float(jnp.sqrt(jnp.mean((result[name] - result["poisson"])**2)))
    print(f"Poisson Phi(R)           = {phi_poisson[surface]:.4e}")
    print(f"BSSN lapse Phi(R)        = {profiles['lapse'][1][surface]:.4e}")
    print(f"BSSN metric Phi(R)       = {profiles['metric'][1][surface]:.4e}")
    print(f"zero modes removed       = lapse {result['lapse_mean']:.4e}, metric {result['metric_mean']:.4e}")
    print(f"RMS(lapse  - Poisson)    = {rms('lapse'):.4e}")
    print(f"RMS(metric - Poisson)    = {rms('metric'):.4e}")

    r_iso = np.linspace(0.01 * RADIUS, shells[-1], 500)
    phi_iso = np.where(r_iso < RADIUS, -mass / (2 * RADIUS) * (3 - (r_iso / RADIUS)**2), -mass / r_iso)
    _, metric_raw = radial_profile(result["metric"] + result["metric_mean"], r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(shells, phi_poisson, "k--", lw=2, label=r"Periodic Poisson $\Phi_{\mathrm{per}}$")
    ax1.plot(shells, profiles["lapse"][1], lw=1.2, label=r"BSSN lapse $\frac{1}{2}(\alpha^2 - 1) - \langle\Phi\rangle$")
    ax1.plot(shells, profiles["metric"][1], ":", lw=1.2, label=r"BSSN metric $-\frac{1}{6}\mathrm{tr}\,\delta\gamma - \langle\Phi\rangle$")
    ax1.set_title("Periodic BSSN potentials vs periodic Poisson reference")
    ax2.plot(shells, phi_poisson, "k--", lw=2, label=r"Periodic Poisson $\Phi_{\mathrm{per}}$")
    ax2.plot(shells, metric_raw, ":", lw=1.2, label=r"Raw BSSN metric $\Phi_{\mathrm{metric}}$")
    ax2.plot(shells, profiles["metric"][1], "o-", ms=3, lw=1, label=r"Zero-mean BSSN metric $\Phi_{\mathrm{metric}}$")
    ax2.plot(r_iso, phi_iso, color="tab:green", lw=1, label=r"Isolated Newtonian $\Phi_{\mathrm{iso}}$")
    ax2.set_title("Periodic zero-mode diagnostic")
    for ax in (ax1, ax2):
        ax.axvline(RADIUS, ls=":", color="grey", label=f"R = {RADIUS:.2e} m")
        ax.set(xlabel="r [m]", ylabel=r"$\Phi$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "static_planetary_mass.png", dpi=300)
    print(f"Saved {OUTPUT_DIR / 'static_planetary_mass.png'}")
