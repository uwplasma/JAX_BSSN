import jax.numpy as jnp


def vacuum_matter_fields(shape, dtype=jnp.float64):
    """Return explicit zero matter sources for vacuum test states."""

    rho = jnp.zeros(shape, dtype=dtype)
    stress_tensor = jnp.zeros((3, 3) + shape, dtype=dtype)
    momentum_density = jnp.zeros((3,) + shape, dtype=dtype)

    return rho, stress_tensor, momentum_density