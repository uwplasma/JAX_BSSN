import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.cartoon.interpolation import (
    lagrange6_nonperiodic,
    lagrange6_stencil,
)


jax.config.update("jax_enable_x64", True)


def test_nonperiodic_lagrange6_reproduces_degree_five_polynomials():
    source_x = jnp.arange(12, dtype=jnp.float64)
    targets = jnp.asarray([0.1, 1.75, 5.4, 10.8, 11.0])

    for degree in range(6):
        interpolated = lagrange6_nonperiodic(source_x**degree, targets)
        np.testing.assert_allclose(
            interpolated,
            targets**degree,
            rtol=2.0e-12,
            atol=2.0e-10,
        )


def test_nonperiodic_lagrange6_preserves_leading_and_target_axes():
    source_x = jnp.arange(12, dtype=jnp.float64)
    targets = jnp.asarray([[0.25, 4.5], [9.75, 11.0]])
    profiles = jnp.stack([source_x**5, 3.0 * source_x**5])

    interpolated = lagrange6_nonperiodic(profiles, targets)

    assert interpolated.shape == (2, 2, 2)
    np.testing.assert_allclose(
        interpolated[0], targets**5, rtol=2.0e-12, atol=2.0e-10
    )
    np.testing.assert_allclose(
        interpolated[1], 3.0 * targets**5, rtol=2.0e-12, atol=5.0e-10
    )


def test_nonperiodic_lagrange6_exposes_out_of_domain_targets():
    source = jnp.arange(12, dtype=jnp.float64)
    result = lagrange6_nonperiodic(source, jnp.asarray([-0.01, 11.01]))
    assert np.all(np.isnan(result))


def test_signed_axis_origin_stencil_uses_negative_parity_ghosts():
    # q = r/dx + 3.5 = 4 at the first half-cell radius.  Indices 2 and 3
    # are the -1.5dx and -0.5dx parity ghosts; 4--7 are positive samples.
    nodes = lagrange6_stencil(jnp.asarray(4.0), source_size=16)
    np.testing.assert_array_equal(nodes, np.arange(2, 8))


def test_lagrange6_requires_six_source_samples():
    with pytest.raises(ValueError, match="needs six samples"):
        lagrange6_nonperiodic(jnp.arange(5.0), jnp.asarray(2.0))
