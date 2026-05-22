import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.bssn import BSSNVariables, BSSNParameters, compute_momentum_constraint
from JAX_BSSN.errors import compute_hamiltonian_constraint
from JAX_BSSN.derivatives import diff1_field
from JAX_BSSN.tensor_algebra import (
    invert_3x3_metric,
    christoffel_symbols_second_kind,
    trace_tensor,
    traceless_part,
)

"""Evolve flat spacetime + spherical mass density in 2D BSSN.

Initialises Minkowski initial data (alpha=1, flat metric, K=0)
with a caller-supplied density field.

Params:
- nx: number of grid points along x and y axes
- nz: number of grid points along z axis (set to 1 for 2D)
- L: physical size of the domain along each axis (total domain is L^3)
- dt_factor: factor to multiply the CFL time step by (e.g. 0.5 for safety)
- Nt: total number of time steps to evolve
- snapshot_stride: how often to save snapshots of the lapse for plotting
- rho_field_fn: function that takes the 1-D coordinate array and returns the mass-density field in geometric units
- show_progress: whether to display a progress bar during evolution
"""
def run_static_mass(R_sphere, rho_geom, nx, nz, L, dt_factor, Nt, snapshot_stride, rho_field_fn,
                    show_progress=True):
    dx = L / nx
    dz = L / nz
    dt = dt_factor * dx
    coords_x = -L/2 + (jnp.arange(nx) + 0.5) * dx  # cell-centred x
    coords_z = -L/2 + (jnp.arange(nz) + 0.5) * dz  # cell-centred z

    # Build the density on this grid via the user-supplied function
    rho_field = rho_field_fn(coords_x, coords_z, R_sphere, rho_geom)

    x3d, y3d, z3d = jnp.meshgrid(coords_x, coords_x, coords_z, indexing='ij')
    r3d = jnp.sqrt(x3d**2 + y3d**2 + z3d**2)
    r3d = jnp.maximum(r3d, 1e-30)

    # Weak-field initial data from the Newtonian potential
    M_geom = (4.0/3.0) * jnp.pi * R_sphere**3 * rho_geom

    def newtonian_potential(r, M, R):
        inside  = -M / (2.0 * R) * (3.0 - r**2 / R**2)
        outside = -M / r
        return jnp.where(r <= R, inside, outside)

    Phi = newtonian_potential(r3d, M_geom, R_sphere)
    alpha = 1.0 + Phi
    beta = jnp.zeros((3, nx, nx, nz))
    conformal_factor = 1.0 + Phi

    gamma = jnp.zeros((3, 3, nx, nx, nz))
    gamma = gamma.at[0, 0].set(1.0)
    gamma = gamma.at[1, 1].set(1.0)
    gamma = gamma.at[2, 2].set(1.0)

    traceless_K = jnp.zeros((3, 3, nx, nx, nz))
    trace_K     = jnp.zeros((nx, nx, nz))
    conformal_connection = jnp.zeros((3, nx, nx, nz))

    vars = BSSNVariables(
        conformal_metric=gamma,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=alpha,
        shift=beta,
        rho=rho_field,
    )

    params = BSSNParameters(
        eta=0.0,
        kappa=0.0,
        nu=0.25,
        f=1.0,      # 1+log slicing -> lapse settles to ~ 1 + Phi/c^2
        g=0.0,
        dx=dx,
        dt=dt,
    )

    # Snapshot storage
    times_list = [0.0]
    lapse_snapshots = [vars.lapse]

    iterator = range(1, Nt + 1)
    if show_progress:
        iterator = tqdm(iterator, leave=False)

    for step in iterator:
        vars = rk4_step(vars, params)
        if (step % snapshot_stride == 0) or (step == Nt):
            times_list.append(step * dt)
            lapse_snapshots.append(vars.lapse)
    
    return (
        jnp.asarray(times_list),
        lapse_snapshots,
        coords_x,
        coords_z,
        r3d,
        rho_field,
        dx,
    )

"""Create spherical density field centered at origin."""
"""Params:
- coords_x: 1D array of x coordinate values
- coords_z: 1D array of z coordinate values  
- R_sphere: radius of the spherical mass distribution
- rho_geom: uniform density within the sphere (geometric units)"""
def make_rho_field(coords_x, coords_z, R_sphere, rho_geom):
    # Create the 3D grid from the 1D coordinate arrays provided by the solver
    X, Y, Z = jnp.meshgrid(coords_x, coords_x, coords_z, indexing='ij')
    
    # Calculate radial distance from the center (0,0,0)
    r_squared = X**2 + Y**2 + Z**2
    
    # Apply the density condition
    rho_field = jnp.where(r_squared < R_sphere**2, rho_geom, 0.0)
    return rho_field

if __name__ == "__main__":
    print("Running Earth simulation")
    # Initialize parameters for the static mass evolution
    # Earth density (SI): 5513 kg/m^3
    #
    # To convert mass density to geometric units (1/m^2):
    #   rho_geom = (G / c^2) * rho_SI
    #
    # G = 6.67430e-11 m^3 kg^-1 s^-2
    # c = 2.99792e8   m s^-1
    # G/c^2 = 7.426e-28 m/kg

    G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
    c_SI = 2.99792458e8         # m s^-1

    # G/c^2 in m/kg
    G_over_c2_m = G_SI / (c_SI**2) 

    R_sphere = 6.3781e6         # meters (Earth radius)
    rho_SI = 5513.0             # kg/m^3

    # rho_geom has units of 1/m^2
    rho_geom = G_over_c2_m * rho_SI   

    M_sphere = (4.0/3.0) * jnp.pi * R_sphere**3 * rho_SI   # kg
    M_geom   = G_over_c2_m * M_sphere # Dimensionless mass in units of meters

    print(f'Earth density (SI):        {rho_SI:.4f} kg/m^3')
    print(f'G/c^2 (m/kg):              {G_over_c2_m:.4e}')
    print(f'rho_geom (1/m^2):          {rho_geom:.4e}')
    print(f'Sphere mass:               {float(M_sphere):.4e} kg')
    print(f'Phi/c^2 at surface ~ GM/(c^2 R) = {float(G_over_c2_m * M_sphere / R_sphere):.4e}')

    # Run the static-mass evolution
    L_domain = 20.0 * R_sphere    # larger domain for weaker periodic-image influence
    nx_mass  = 128                # finer 2D grid for the sphere
    nz_mass  = 1                  # Single z-slice for 2D
    dt_fac   = 0.1                # CFL factor
    Nt_mass  = 2000               # 2*10^3 timesteps
    snap_stride = 100

    print(f'Domain: [{-L_domain/2:.2f}, {L_domain/2:.2f}] m   ({nx_mass}x{nx_mass}x{nz_mass} grid)')
    print(f'dx = {L_domain/nx_mass:.4f} m,  dt = {dt_fac * L_domain/nx_mass:.4e} m')
    print(f'Nt = {Nt_mass},  rho_geom = {rho_geom:.4e} m^-2')

    times_mass, lapse_snaps, coords_x_mass, coords_z_mass, r3d_mass, rho_field_mass, dx_mass = run_static_mass(
        R_sphere=R_sphere,
        rho_geom=rho_geom,
        nx=nx_mass,
        nz=nz_mass,
        L=L_domain,
        dt_factor=dt_fac,
        Nt=Nt_mass,
        snapshot_stride=snap_stride,
        rho_field_fn=make_rho_field,
        show_progress=True,
    )

    print(f'\nEvolution finished: {len(times_mass)} snapshots over t=[0, {float(times_mass[-1]):.4f}] m')

    # Save results to a file
    output_filename = "simulation_results_2d.npz"
    jnp.savez(
        output_filename,
        times_mass=times_mass,
        lapse_snaps=jnp.array(lapse_snaps), # Convert list to array for saving
        coords_x_mass=coords_x_mass,
        coords_z_mass=coords_z_mass,
        r3d_mass=r3d_mass,
        R_sphere=R_sphere,
        rho_geom=rho_geom
    )

    print(f"Results saved to {output_filename}")
