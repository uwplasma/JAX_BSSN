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
    steps = 6

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

        for _ in range(steps):
            H = compute_hamiltonian_constraint(vars_rn, params)
            ham_history.append(float(jnp.max(jnp.abs(H))))
            trace_K_history.append(float(jnp.max(jnp.abs(vars_rn.trace_K))))
            lapse_history.append(float(jnp.max(jnp.abs(vars_rn.lapse - 1.0))))
            vars_rn = rk4_step(vars_rn, params)

        radial_profile = jnp.asarray(rho_em[:, nx // 2, nx // 2])
        radial_x = jnp.asarray(x)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].plot(radial_x, radial_profile, label='EM energy density')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Radius')
        axs[0].set_ylabel('$\\rho_{EM}$')
        axs[0].set_title(f'Q={Q}: EM energy density profile')
        axs[0].legend()

        axs[1].plot(range(steps), trace_K_history, marker='o', label='max |K|')
        axs[1].plot(range(steps), lapse_history, marker='x', label='max |\\alpha - 1|')
        axs[1].set_xlabel('RK4 step')
        axs[1].set_ylabel('Max value')
        axs[1].set_title(f'Q={Q}: Short evolution')
        axs[1].legend()

        axs[2].plot(range(steps), ham_history, marker='s', color='r', label='max |H|')
        axs[2].set_xlabel('RK4 step')
        axs[2].set_ylabel('Max |H|')
        axs[2].set_title(f'Q={Q}: Hamiltonian')
        axs[2].legend()

        fig.tight_layout()
        out_path = f"demos/reissner_nordstrom_Q{Q}_results.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
