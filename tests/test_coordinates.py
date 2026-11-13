"""Uniform NR coordinates and the shared upper-half-cell Yee convention.

C coordinates are independent of PyPIC3D. V denotes the upper half-cell
neighbour of C: V[i] = C[i] + dx/2. Integration tests compare this geometry
convention with the native PyPIC3D staggered field locations.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.evolution.coordinates import axis_coordinates, cylindrical_volume_weights, grid_coordinates
from tests.initial_data import create_coordinate_arrays, linear_wave_data

jax.config.update('jax_enable_x64', True)

YEE_LOCATIONS = (
    ("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"),
    ("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"),
)
SIZE = 12


def params(dx=.25):
    return BSSNParameters(dx=dx, dt=.1 * dx, x_min=-3.5 * dx, y_min=-4 * dx, z_min=1.)


@pytest.mark.parametrize('direction', range(3))
def test_vertex_sites_sit_half_a_cell_above_centers(direction):
    p = params()
    centers = axis_coordinates(SIZE, direction, p, jnp.float64, "C")
    vertices = axis_coordinates(SIZE, direction, p, jnp.float64, "V")
    assert bool(jnp.allclose(vertices - centers, .5 * p.dx, atol=1.e-15))


@pytest.mark.parametrize('direction', range(3))
def test_center_sites_are_the_unshifted_physical_samples(direction):
    p = params()
    minimum = (p.x_min, p.y_min, p.z_min)[direction]
    expected = minimum + p.dx * jnp.arange(SIZE)
    explicit = axis_coordinates(SIZE, direction, p, jnp.float64, "C")
    assert bool(jnp.allclose(explicit, expected, atol=1.e-15))
    assert bool(jnp.array_equal(axis_coordinates(SIZE, direction, p, jnp.float64), expected))


@pytest.mark.parametrize('location', YEE_LOCATIONS)
def test_grid_coordinates_staggers_each_axis_independently(location):
    p, shape = params(), (SIZE, SIZE + 1, SIZE + 2)
    minima = (p.x_min, p.y_min, p.z_min)
    for direction, coordinate in enumerate(grid_coordinates(shape, p, jnp.float64, location)):
        expected = minima[direction] + p.dx * (jnp.arange(shape[direction]) + (.5 if location[direction] == "V" else 0))
        assert bool(jnp.allclose(jnp.ravel(coordinate), expected, atol=1.e-15))


@pytest.mark.parametrize('location', YEE_LOCATIONS)
def test_cylindrical_weights_use_the_radial_site_of_the_field(location):
    p, shape = params(), (SIZE, 1, SIZE + 2)
    rho = p.x_min + p.dx * (jnp.arange(shape[0]) + (.5 if location[0] == "V" else 0))
    expected = jnp.broadcast_to(jnp.abs(rho)[:, None], (shape[0], shape[-1]))
    assert bool(jnp.allclose(cylindrical_volume_weights(shape, p, jnp.float64, location), expected, atol=1.e-15))


def test_default_grid_coordinates_are_unstaggered():
    p, shape = params(), (SIZE, SIZE + 1, SIZE + 2)
    for explicit, default in zip(grid_coordinates(shape, p, jnp.float64, ("C", "C", "C")), grid_coordinates(shape, p, jnp.float64)):
        assert bool(jnp.array_equal(explicit, default))


def test_shared_fixture_arrays_are_strongly_typed():
    """Weak-typed fixtures would silently promote in convergence measurements.

    ``tests/initial_data`` feeds the gauge-wave and tiled convergence tests, whose
    observed orders depend on the sampled dtype rather than on x64 promotion rules.
    """
    coordinates = create_coordinate_arrays(8, 8, 8, 0.125)
    variables = linear_wave_data(8, 8, 8, 0.125)
    single = create_coordinate_arrays(8, 8, 8, np.float32(0.125))

    assert all(not coordinate.weak_type for coordinate in coordinates)
    assert all(not field.weak_type for field in variables)
    assert all(coordinate.dtype == jnp.float32 for coordinate in single)
