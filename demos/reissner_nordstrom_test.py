import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from JAX_BSSN.bssn import (
    BSSNVariables,
    BSSNParameters,
    compute_em_sources,
    compute_em_momentum_density,
)
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.errors import compute_hamiltonian_constraint

jax.config.update("jax_enable_x64", True)


def main():
    nx = 32
    L = 5.0
    x = jnp.linspace(-L, L, nx)
    y = jnp.linspace(-L, L, nx)
    z = jnp.linspace(-L, L, nx)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

    conformal_metric = jnp.eye(3)[:, :, None, None, None] * jnp.ones((3, 3, nx, nx, nx))
    conformal_factor = jnp.ones((nx, nx, nx))
    params = BSSNParameters(dx=2 * L / (nx - 1), dt=0.15 * 2 * L / (nx - 1), nu=0.0)

    Qs = [1.0, 0.1, 0.01]
    steps = 100

    for Q in Qs:
        r = jnp.sqrt(X**2 + Y**2 + Z**2) + 1e-6
        E_flat = jnp.stack([Q * X / r**3, Q * Y / r**3, Q * Z / r**3], axis=0)
        B_flat = jnp.zeros_like(E_flat)

        rho_em, S_ij_em = compute_em_sources(
            conformal_metric=conformal_metric,
            conformal_factor=conformal_factor,
            E_flat=E_flat,
            B_flat=B_flat,
        )
        J_em = compute_em_momentum_density(
            conformal_metric=conformal_metric,
            conformal_factor=conformal_factor,
            E_flat=E_flat,
            B_flat=B_flat,
        )

        vars_rn = BSSNVariables(
            conformal_metric=conformal_metric,
            conformal_factor=conformal_factor,
            traceless_K=jnp.zeros((3, 3, nx, nx, nx)),
            trace_K=jnp.zeros((nx, nx, nx)),
            conformal_connection=jnp.zeros((3, nx, nx, nx)),
            lapse=jnp.ones((nx, nx, nx)),
            shift=jnp.zeros((3, nx, nx, nx)),
            rho=rho_em,
            S_ij=S_ij_em,
            momentum_density=J_em,
        )

        trace_K_history = []
        lapse_history = []
        ham_history = []
        
        # Store final lapse for radial profile comparison
        final_lapse = None

        for step in range(steps):
            H = compute_hamiltonian_constraint(vars_rn, params)
            max_H = jnp.max(jnp.abs(H))
            print(f"Step {step}: Max H = {max_H}") # Check if this turns into NaN or Inf
            ham_history.append(float(jnp.max(jnp.abs(H))))
            trace_K_history.append(float(jnp.max(jnp.abs(vars_rn.trace_K))))
            lapse_history.append(float(jnp.max(jnp.abs(vars_rn.lapse - 1.0))))
            vars_rn = rk4_step(vars_rn, params)
        
        final_lapse = vars_rn.lapse

        radial_profile = jnp.asarray(rho_em[:, nx // 2, nx // 2])
        radial_x = jnp.asarray(x)
        
        # Compute analytical Reissner-Nordström lapse: α(r) = sqrt(1 - 2M/r + Q²/r²)
        # Assume mass M ~ Q for dimensional analysis (can be adjusted)
        M = Q
        r_radial = jnp.abs(radial_x) + 1e-6
        alpha_analytical = jnp.sqrt(jnp.maximum(1.0 - 2*M/r_radial + Q**2/r_radial**2, 1e-6))
        lapse_radial_sim = jnp.asarray(final_lapse[:, nx // 2, nx // 2])

        fig, axs = plt.subplots(1, 4, figsize=(24, 5))
        axs[0].plot(radial_x, radial_profile, label='EM energy density')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Radius')
        axs[0].set_ylabel('$\\rho_{EM}$')
        axs[0].set_title(f'Q={Q}: EM energy density profile')
        axs[0].legend()

        axs[1].plot(range(steps), trace_K_history, marker='o', label='max |K|')
        axs[1].plot(range(steps), lapse_history, marker='x', label='max |α - 1|')
        axs[1].set_xlabel('RK4 step')
        axs[1].set_ylabel('Max value')
        axs[1].set_title(f'Q={Q}: Short evolution')
        axs[1].legend()

        axs[2].plot(range(steps), ham_history, marker='s', color='r', label='max |H|')
        axs[2].set_xlabel('RK4 step')
        axs[2].set_ylabel('Max |H|')
        axs[2].set_title(f'Q={Q}: Hamiltonian')
        axs[2].legend()
        
        axs[3].plot(radial_x, lapse_radial_sim, 'b-', linewidth=2, label='Simulated α(r)')
        axs[3].plot(radial_x, alpha_analytical, 'r--', linewidth=2, label=f'Analytical RN α(r), M={M}')
        axs[3].set_xlabel('Radius')
        axs[3].set_ylabel('Lapse α(r)')
        axs[3].set_title(f'Q={Q}: Lapse Comparison')
        axs[3].legend()

        fig.tight_layout()
        out_path = f"demos/reissner_nordstrom_Q{Q}_results.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
