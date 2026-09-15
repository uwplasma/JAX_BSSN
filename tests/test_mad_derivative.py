"""Focused accuracy tests for fourth-order mesh-adapted differencing."""

import numpy as np
import jax
import jax.numpy as jnp

from JAX_BSSN.evolution.derivatives import diff1_field, diff2_field

jax.config.update("jax_enable_x64", True)


def _error(n, q):
    dx = 2.0 * np.pi / n
    x = jnp.arange(n, dtype=jnp.float64) * dx
    field = jnp.sin(x)[:, None, None]
    exact = jnp.cos(x)[:, None, None]
    return float(jnp.sqrt(jnp.mean((diff1_field(field, 0, dx, mad_q=q) - exact) ** 2)))


def _second_error(n, q):
    dx = 2.0 * np.pi / n
    x = jnp.arange(n, dtype=jnp.float64) * dx
    field = jnp.sin(x)[:, None, None]
    exact = -field
    return float(jnp.sqrt(jnp.mean((diff2_field(field, 0, dx, mad_q=q) - exact) ** 2)))


def _orders(errors):
    return [np.log2(a / b) for a, b in zip(errors[:-1], errors[1:])]


def test_q_one_is_historical_d4_to_roundoff():
    n = 73
    dx = 2.0 * np.pi / n
    x = jnp.arange(n, dtype=jnp.float64) * dx
    f = jnp.sin(3.0 * x)[:, None, None]
    old = (2.0 / 3.0 * (jnp.roll(f, -1, 0) - jnp.roll(f, 1, 0))
           - 1.0 / 12.0 * (jnp.roll(f, -2, 0) - jnp.roll(f, 2, 0))) / dx
    np.testing.assert_allclose(diff1_field(f, 0, dx, mad_q=1.0), old,
                               rtol=0.0, atol=np.finfo(float).eps * 2)


def test_d4_mad_and_d6_formal_orders():
    ns = (32, 64, 128, 256)
    d4 = [_error(n, 1.0) for n in ns]
    mad = [_error(n, 1.0 / 16.0) for n in ns]
    d6 = [_error(n, 0.0) for n in ns]
    print("D4 errors/orders", d4, _orders(d4))
    print("MAD errors/orders", mad, _orders(mad))
    print("D6 errors/orders", d6, _orders(d6))
    assert min(_orders(d4)[-2:]) > 3.95
    assert min(_orders(mad)[-2:]) > 3.9
    assert min(_orders(d6)[:2]) > 5.9


def test_leading_error_matches_fine_d4_at_coincident_nodes():
    # A coarse grid with H=2h and q=(h/H)^4 must have the same leading error.
    ratios = []
    for coarse_n in (16, 32, 64, 128, 256):
        fine_error = _error(2 * coarse_n, 1.0)
        coarse_error = _error(coarse_n, 1.0 / 16.0)
        ratios.append(coarse_error / fine_error)
    print("MAD(H) / D4(h) leading-error ratios", ratios)
    assert abs(ratios[-1] - 1.0) < 2.0e-3
    assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)


def test_second_derivative_d4_mad_and_d6_formal_orders():
    ns = (32, 64, 128, 256)
    d4 = [_second_error(n, 1.0) for n in ns]
    mad = [_second_error(n, 1.0 / 16.0) for n in ns]
    d6 = [_second_error(n, 0.0) for n in ns]
    assert min(_orders(d4)[-2:]) > 3.9
    assert min(_orders(mad)[-2:]) > 3.8
    assert min(_orders(d6)[:2]) > 5.8


def test_second_derivative_leading_error_matches_fine_d4():
    ratios = []
    for coarse_n in (16, 32, 64, 128, 256):
        fine_error = _second_error(2 * coarse_n, 1.0)
        coarse_error = _second_error(coarse_n, 1.0 / 16.0)
        ratios.append(coarse_error / fine_error)
    assert abs(ratios[-1] - 1.0) < 3.0e-3
    assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)
