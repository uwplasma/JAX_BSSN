"""
Unit tests for finite difference derivatives.

This module tests all derivative operations against known analytical solutions
using various test functions with known derivatives.
"""

import unittest
import jax.numpy as jnp
import numpy as np
from jax import jit
import jax
from scipy import stats

from JAX_BSSN.derivatives import (
    diff1_field,
    diff6_field,
    compute_all_derivatives,
    gradient_3d,
    divergence_3d,
    laplacian_3d,
    periodic_indexing,
    get_stencil_indices
)


class TestDerivatives(unittest.TestCase):
    """Test suite for finite difference derivatives."""

    def test_first_derivative_1d_trigonometric(self):
        """
        Test that diff1_field correctly computes the first derivative of a 1D trigonometric field
        embedded in a 3D periodic domain.
        Specifically:
        - Construct a periodic 3D grid with coordinates x, y, z in [-pi, pi) and resolution 100 per axis.
        - Define the scalar field f(X,Y,Z) = sin(X), which varies only along the x-axis.
        - Compute the analytical derivative d/dx f = cos(X).
        - Use diff1_field to compute numerical derivatives along axes 0 (x), 1 (y) and 2 (z).
        Assertions:
        - The numerical d/dx matches the analytical result with mean absolute error < 1e-6 and
            maximum absolute error < 5e-6.
        - The numerical d/dy and d/dz are effectively zero with the same error tolerances.
        Purpose:
        - Verify that the finite-difference implementation in diff1_field accurately differentiates
            smooth periodic trigonometric functions along the direction of variation and does not
            introduce spurious cross-axis derivatives.
        Notes:
        - The test relies on periodic sampling (endpoint=False) and a uniform spacing dx = x[1] - x[0].
        - Failures indicate either insufficient stencil accuracy or incorrect axis handling in diff1_field.
        """


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X)
        # create a sin vector field along the x direction
        dfdx_analytical = jnp.cos(X)
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically

        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt to x

        dfdy_error = diff1_field(f, 1, dx)
        # derivative should be zero

        dfdz_error = diff1_field(f, 2, dx)
        # derivative should be zero

        mean_error_x = jnp.mean(jnp.abs(dfdx_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 1e-6, msg=f"Trigonometric derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 1e-6, msg=f"Trigonometric derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 1e-6, msg=f"Trigonometric derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error))
        max_error_y = jnp.max(jnp.abs(dfdy_error))
        max_error_z = jnp.max(jnp.abs(dfdz_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-6, msg=f'Trigonometric derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-6, msg=f'Trigonometric derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-6, msg=f'Trigonometric derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold

    def test_first_derivative_2d_trigonometric(self):
        """
        Test that the first-order finite-difference derivative routine correctly
        computes partial derivatives for a separable 2D trigonometric field
        embedded in a 3D periodic grid.
        Setup:
        - Construct a uniform, periodic grid in each coordinate on [-pi, pi)
            with 100 points per axis and spacing dx determined from the x-grid.
        - Build the scalar field f(X, Y, Z) = sin(X) * cos(Y) on the meshgrid.
        Analytical derivatives:
        - ∂f/∂x =  cos(X) * cos(Y)
        - ∂f/∂y = -sin(X) * sin(Y)
        - ∂f/∂z =  0
        What is tested:
        - Compute numerical derivatives using diff1_field(f, axis, dx) for axis
            = 0 (x), 1 (y) and 2 (z).
        - Compute absolute errors between numerical and analytical results.
        - Assert that the mean absolute error for x, y and z is below 1e-6.
        - Assert that the maximum absolute error for x, y and z is below 5e-6.
        Assumptions and notes:
        - The derivative routine handles periodic boundaries consistently with the
            constructed grid.
        - dx is uniform and equal to the spacing along the x-axis; the test
            assumes the same spacing is applicable for y and z directions.
        - Thresholds are chosen to reflect expected accuracy of the finite-difference
            implementation on the given resolution.
        """


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X) * jnp.cos(Y)
        # create a sin vector field along the x direction
        dfdx_analytical = jnp.cos(X) * jnp.cos(Y)
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically
        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt x

        dfdy_analytical = -1 * jnp.sin(X) * jnp.sin(Y)
        # define the derivative of the vector field analytically
        dfdy_numerical = diff1_field(f, 1, dx)
        # calculate the derivative of the vector field numerically
        dfdy_error = dfdy_numerical - dfdy_analytical
        # calculate the error in the derivative wrt y

        dfdz_error = diff1_field(f, 2, dx)
        # derivative should be zero

        mean_error_x = jnp.mean(jnp.abs(dfdx_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 1e-6, msg=f"Trigonometric derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 1e-6, msg=f"Trigonometric derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 1e-6, msg=f"Trigonometric derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error))
        max_error_y = jnp.max(jnp.abs(dfdy_error))
        max_error_z = jnp.max(jnp.abs(dfdz_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-6, msg=f'Trigonometric derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-6, msg=f'Trigonometric derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-6, msg=f'Trigonometric derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold

    def test_first_derivative_3d_trigonometric(self):
        """
        Test that numerical first derivatives computed by diff1_field match analytical
        derivatives for a smooth 3D trigonometric scalar field.
        This test builds a periodic 3D grid on [-pi, pi) with 100 points per axis and
        spacing dx, defines the scalar field
            f(x,y,z) = sin(x) * cos(y) * sin(z)
        and its analytical first derivatives
            df/dx =  cos(x) * cos(y) * sin(z)
            df/dy = -sin(x) * sin(y) * sin(z)
            df/dz =  sin(x) * cos(y) * cos(z)
        It computes numerical derivatives along each axis with diff1_field and compares
        them to the analytical expressions. The test asserts that, for each axis,
        the mean absolute error is below 1e-6 and the maximum absolute error is below
        5e-6. These tolerances reflect the expected accuracy of the finite-difference
        (or spectral) derivative implementation on a smooth, periodic trigonometric
        field using the chosen resolution.
        Notes:
        - The test uses jax.numpy (jnp) arrays and assumes periodicity consistent with
          the grid construction (endpoint=False).
        - Failure indicates a regression or bug in diff1_field or an unexpected change
          in boundary handling / grid spacing.
        """



        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)
        # create a sin vector field along the x direction
        dfdx_analytical = jnp.cos(X) * jnp.cos(Y) * jnp.sin(Z)
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically
        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt x

        dfdy_analytical = -1 * jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
        # define the derivative of the vector field analytically
        dfdy_numerical = diff1_field(f, 1, dx)
        # calculate the derivative of the vector field numerically
        dfdy_error = dfdy_numerical - dfdy_analytical
        # calculate the error in the derivative wrt y


        dfdz_analytical = jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
        # define the derivative of the vector field analytically
        dfdz_numerical  = diff1_field(f, 2, dx)
        # calculate the derivative of the vector field numerically
        dfdz_error = dfdz_numerical - dfdz_analytical
        # derivative should be zero

        mean_error_x = jnp.mean(jnp.abs(dfdx_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 1e-6, msg=f"Trigonometric derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 1e-6, msg=f"Trigonometric derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 1e-6, msg=f"Trigonometric derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error))
        max_error_y = jnp.max(jnp.abs(dfdy_error))
        max_error_z = jnp.max(jnp.abs(dfdz_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-6, msg=f'Trigonometric derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-6, msg=f'Trigonometric derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-6, msg=f'Trigonometric derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_first_derivative_1d_quadratic(self):
        """
        Test the first-order finite-difference derivative operator on a simple 1D quadratic field
        embedded in a 3D grid.
        This test constructs a regular 3D mesh (X, Y, Z) from 1D coordinates spaced by dx and
        defines a scalar field f(X,Y,Z) = X**2 (i.e. a quadratic function varying only along
        the x-axis). The analytical x-derivative is 2*X, while derivatives along y and z
        should be zero.
        The numerical derivatives are computed with diff1_field for axes 0, 1 and 2 using the
        grid spacing dx. To avoid boundary artifacts introduced by the non-periodic quadratic
        function, the first and last three grid points are excluded from the error metrics.
        Assertions:
        - Mean absolute error for each axis (over the interior slice) is below 5e-6.
        - Maximum absolute error for each axis (over the interior slice) is below 5e-5.
        Raises:
        - AssertionError if any of the error thresholds are exceeded.
        """


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = X**2
        # create a sin vector field along the x direction
        dfdx_analytical = 2 * X
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically

        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt to x

        dfdy_error = diff1_field(f, 1, dx)
        # derivative should be zero

        dfdz_error = diff1_field(f, 2, dx)
        # derivative should be zero


        _slice = slice(3, -3) # ignore the first 3 points of the grid because quadratic is not periodic

        mean_error_x = jnp.mean(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the mean error


        self.assertLess(mean_error_x, 5e-6, msg=f"Quadratic polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 5e-6, msg=f"Quadratic polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 5e-6, msg=f"Quadratic polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        max_error_y = jnp.max(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        max_error_z = jnp.max(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the max error

        self.assertLess(max_error_x, 5e-5, msg=f'Quadratic polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-5, msg=f'Quadratic polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-5, msg=f'Quadratic polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_first_derivative_2d_quadratic(self):
        """
        Test that the first-order finite-difference operator `diff1_field` computes
        accurate partial derivatives for a simple 2D quadratic scalar field embedded
        in a 3D periodic grid.
        Setup:
        - Construct a uniform grid in x, y, z with 100 points each over [-pi, pi)
            (endpoint=False) and compute the grid spacing dx from the x array.
        - Create meshgrid with indexing='ij' and define the scalar field
            f(x,y,z) = x^2 + y^2 (no z-dependence).
        - Analytical derivatives:
                d/dx f = 2*x
                d/dy f = 2*y
                d/dz f = 0
        What is tested:
        - Compute numerical partial derivatives using diff1_field along axes 0, 1, 2.
        - Evaluate absolute errors between numerical and analytical derivatives.
        - Ignore boundary points (slice(3, -3)) because the polynomial is not periodic
            and finite-difference stencils touch boundaries.
        - Check that the mean absolute error over the interior is below 5e-6 for x and y
            (and near zero for z), and that the maximum absolute error over the interior
            is below 5e-5 for all three directions.
        Purpose:
        - Verify correctness and expected accuracy of diff1_field on smooth fields,
            ensure correct axis handling, spacing usage, and interior-boundary treatment.
        Notes:
        - Uses JAX numpy arrays (jnp) and assumes uniform spacing dx.
        - Thresholds are chosen to reflect expected truncation error for the stencil
            and resolution used.
        """


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = X**2 + Y**2
        # create a polynomial

        dfdx_analytical = 2 * X
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically
        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt to x

        dfdy_analytical = 2 * Y
        # define the derivative of the vector field analytically
        dfdy_numerical  = diff1_field(f, 1, dx)
        # calculate the derivative of the vector field numerically
        dfdy_error = dfdy_numerical - dfdy_analytical
        # calculate the error in the derivative wrt to y

        dfdz_error = diff1_field(f, 2, dx)
        # derivative should be zero


        _slice = slice(3, -3) # ignore the first 3 points of the grid because quadratic is not periodic

        mean_error_x = jnp.mean(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the mean error


        self.assertLess(mean_error_x, 5e-6, msg=f"Quadratic polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 5e-6, msg=f"Quadratic polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 5e-6, msg=f"Quadratic polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        max_error_y = jnp.max(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        max_error_z = jnp.max(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the max error

        self.assertLess(max_error_x, 5e-5, msg=f'Quadratic polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-5, msg=f'Quadratic polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-5, msg=f'Quadratic polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_first_derivative_3d_quadratic(self):
        """
        Test the numerical first-derivative operator on a 3D quadratic field.
        This test constructs a uniform 3D Cartesian grid (100 points per axis) over
        [-π, π) in each coordinate, builds the scalar field
            f(x, y, z) = x**2 + y**2 + z**2,
        and compares the numerically computed first partial derivatives returned by
        diff1_field to the analytical derivatives
            ∂f/∂x = 2*x,  ∂f/∂y = 2*y,  ∂f/∂z = 2*z.
        Key details:
        - The grid spacing dx is taken from the 1D linspace.
        - diff1_field is invoked with axis indices 0, 1, 2 to compute derivatives
          along x, y, z respectively.
        - Boundary points are excluded from the error statistics via _slice = slice(3, -3)
          because the quadratic field is not periodic and derivative stencils can be
          inaccurate near the domain boundaries.
        - The test asserts both mean and maximum absolute errors on the interior:
            mean error < 5e-6
            max  error < 5e-5
          These tolerances encode the expected numerical accuracy of diff1_field for this
          smooth polynomial input.
        A failure indicates an incorrect implementation or incorrect handling of grid
        spacing, axis ordering, boundary conditions, or stencil accuracy in diff1_field.
        """


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = X**2 + Y**2 + Z**2
        # create a polynomial

        dfdx_analytical = 2 * X
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically
        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt to x

        dfdy_analytical = 2 * Y
        # define the derivative of the vector field analytically
        dfdy_numerical = diff1_field(f, 1, dx)
        # calculate the derivative of the vector field numerically
        dfdy_error = dfdy_numerical - dfdy_analytical
        # calculate the error in the derivative wrt to y

        dfdz_analytical = 2 * Z
        # define the derivative of the vector field analytically
        dfdz_numerical = diff1_field(f, 2, dx)
        # calculate the derivative of the vector field numerically
        dfdz_error = dfdz_numerical - dfdz_analytical
        # calculate the error in the derivative wrt to z


        _slice = slice(3, -3) # ignore the first 3 points of the grid because quadratic is not periodic

        mean_error_x = jnp.mean(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the mean error


        self.assertLess(mean_error_x, 5e-6, msg=f"Quadratic polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 5e-6, msg=f"Quadratic polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 5e-6, msg=f"Quadratic polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        max_error_y = jnp.max(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        max_error_z = jnp.max(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the max error

        self.assertLess(max_error_x, 5e-5, msg=f'Quadratic polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-5, msg=f'Quadratic polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-5, msg=f'Quadratic polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_compute_all_derivatives(self):
        """
        Test that compute_all_derivatives correctly computes first-order spatial derivatives
        on a 3D periodic grid for a simple analytic field.
        This test constructs a uniform, periodic 3D meshgrid using jnp.linspace over
        [-pi, pi) with 100 points in each dimension and spacing dx. It defines a scalar
        field f(x,y,z) = sin(x) (i.e., variation only along the x-axis) and the known
        analytical derivative dfdx = cos(x). The test invokes compute_all_derivatives(f, dx)
        to obtain numerical derivatives along the x, y and z axes.
        Assertions:
        - The mean absolute error between the numerical dfdx and analytical cos(x) is below 1e-6.
        - The mean absolute errors for the derivatives in y and z are below 1e-6 (they should be zero).
        - The maximum absolute error for dfdx is below 5e-6.
        - The maximum absolute errors for dy and dz are below 5e-6.
        This verifies both accuracy (against the analytical derivative) and that the
        implementation does not introduce spurious derivatives in directions where the
        field is constant. Uses JAX (jnp) arrays and assumes periodic boundary behavior
        consistent with the linspace/endpoint=False setup.
        """

    
        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X)
        # create a sin vector field along the x direction
        dfdx_analytical = jnp.cos(X)
        # define the derivative of the vector field analytically
                
        all_derivs = compute_all_derivatives(f, dx)
        # calculate all derivatives at once

        dfdx_error = all_derivs[0] - dfdx_analytical
        dfdy_error = all_derivs[1]
        dfdz_error = all_derivs[2]
        # calculate the errors in the derivatives

        mean_error_x = jnp.mean(jnp.abs(dfdx_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 1e-6, msg=f"Trigonometric derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 1e-6, msg=f"Trigonometric derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 1e-6, msg=f"Trigonometric derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error))
        max_error_y = jnp.max(jnp.abs(dfdy_error))
        max_error_z = jnp.max(jnp.abs(dfdz_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-6, msg=f'Trigonometric derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-6, msg=f'Trigonometric derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-6, msg=f'Trigonometric derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_gradient_3d(self):
        """
        Test the 3D finite-difference gradient implementation on a separable trigonometric field.
        Sets up a uniform periodic grid in x, y, z over [-pi, pi) with 100 points per axis and spacing dx = x[1] - x[0].
        Defines a scalar field f(X,Y,Z) = sin(X) that depends only on x, computes its numerical gradient via
        gradient_3d(f, dx), and compares the result to the analytic derivatives:
            - ∂f/∂x = cos(X)
            - ∂f/∂y = 0
            - ∂f/∂z = 0
        Checks both mean and maximum absolute errors:
            - mean absolute errors for each component must be < 1e-6
            - maximum absolute errors for each component must be < 5e-6
        Failure messages include the measured error to aid debugging.
        """

  
        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X)
        # create a sin vector field along the x direction

        grad = gradient_3d(f, dx)
        # compute the gradient numerically

        dfdx = jnp.cos(X)
        # define the derivative of the vector field analytically

        dfdx_error = grad[0] - dfdx
        # calculate the error in the derivative wrt to x
        dfdy_error = grad[1]
        # derivative should be zero
        dfdz_error = grad[2]
        # derivative should be zero

        mean_error_x = jnp.mean(jnp.abs(dfdx_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 1e-6, msg=f"Trigonometric derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 1e-6, msg=f"Trigonometric derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 1e-6, msg=f"Trigonometric derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error))
        max_error_y = jnp.max(jnp.abs(dfdy_error))
        max_error_z = jnp.max(jnp.abs(dfdz_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-6, msg=f'Trigonometric derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-6, msg=f'Trigonometric derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-6, msg=f'Trigonometric derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold

    def test_divergence_3d_constant_field(self):
        """
        Test the numerical divergence implementation on a simple, analytic 3D vector field.
        This test constructs a periodic 3D grid on the domain [-pi, pi) with 100 points per axis
        and uniform spacing dx. It defines a vector field F = (sin(x), sin(y), 0) on the mesh
        (using meshgrid with indexing='ij') and computes the divergence using divergence_3d.
        The analytic divergence of this field is div(F) = cos(x) + cos(y). The test compares the
        numerical divergence to the analytic result and verifies accuracy by asserting that:
        - the mean absolute error is below 5e-6, and
        - the maximum absolute error is below 1e-5.
        Notes:
        - dx is computed as the uniform spacing between consecutive grid points.
        - The test assumes divergence_3d implements a finite-difference scheme compatible with
            periodic boundaries for this grid setup.
        - Tolerances are chosen for the given resolution (100 points per axis) and expected
            convergence behavior of the derivative scheme.
        Returns:
                None. The function raises assertion errors if the numerical divergence does not meet
                the prescribed accuracy thresholds.
        """


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        fx = jnp.sin(X)
        fy = jnp.sin(Y)
        # create a sin vector field along the x and y direction

        vector_field = jnp.stack([fx, fy, jnp.zeros_like(fx)], axis=0)
        # create the vector field

        div = divergence_3d(vector_field, dx)
        # compute the divergence numerically

        analytical_div = jnp.cos(X) + jnp.cos(Y)
        # define the divergence of the vector field analytically

        error = div - analytical_div

        mean_error = jnp.mean(jnp.abs(error))
        self.assertLess(mean_error, 5e-6, msg=f"Divergence of vector field failed with mean error {mean_error}" )
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error = jnp.max(jnp.abs(error))
        self.assertLess(max_error, 1e-5, msg=f"Divergence of vector field failed with max error {max_error}" )
        # ensure the max errors in the derivatives are below a certain threshold

    def test_laplacian_3d(self):
        """
        Unit test for laplacian_3d accuracy on a smooth, periodic 3D field.
        This test verifies that the numerical Laplacian implementation `laplacian_3d`
        produces results consistent with the analytical Laplacian for the
        trigonometric test field f(X,Y,Z) = sin(X) + sin(Y) + sin(Z).
        Procedure:
        - Construct a uniform, periodic 3D grid on the domain [-pi, pi) in each
            coordinate with 100 points per axis and compute the grid spacing dx.
        - Build the scalar field f = sin(X) + sin(Y) + sin(Z).
        - The analytical Laplacian of this field is lapl_analytical = -f.
        - Compute the numerical Laplacian using laplacian_3d(f, dx).
        - Compute absolute errors, then check that the mean absolute error is
            below 5e-5 and the maximum absolute error is below 5e-4.
        Purpose:
        - Serves as a regression test to ensure the discrete Laplacian operator
            is implemented correctly and is accurate for smooth, periodic fields.
        - The tolerances are chosen to reflect expected numerical error for the
            chosen resolution and finite-difference / spectral stencil used.
        """


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X) + jnp.sin(Y) + jnp.sin(Z)
        # create a trignometric field

        lapl_analytical = -f
        lapl_numerical = laplacian_3d(f, dx)
        # compute the laplacian numerically
        lapl_error = lapl_numerical - lapl_analytical
        # calculate the error in the laplacian

        mean_error = jnp.mean(jnp.abs(lapl_error))

        self.assertLess(mean_error, 5e-5, msg=f"Trigonometric derivative failed in x direction with mean error {mean_error}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error = jnp.max(jnp.abs(lapl_error))
        # calculate the max error

        self.assertLess(max_error, 5e-4, msg=f'Trigonometric derivative failed with max error {max_error}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_first_derivative_convergence(self):
        """Convergence test for the first derivative on a trigonometric function.

        Verifies that the numerical derivative error decreases with grid
        refinement and that the observed convergence order is consistent with
        a high-order finite-difference stencil (expecting order > 3).
        """
        Ns = [32, 64, 128, 256]
        errors = []
        dxs = []

        for N in Ns:
            x = jnp.linspace(-jnp.pi, jnp.pi, N, endpoint=False)
            y = x
            z = x
            dx = x[1] - x[0]
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            f = jnp.sin(X)
            # create a sin vector field along the x direction
            dfdx_analytical = jnp.cos(X)
            # define the derivative of the vector field analytically

            dfdx_numerical = diff1_field(f, 0, dx)
            # calculate the derivative of the vector field numerically

            mse_error = jnp.mean((dfdx_numerical - dfdx_analytical)**2)
            # compute the mean squared error

            errors.append(mse_error)
            dxs.append(dx)

        errors = jnp.array(errors)
        dxs = jnp.array(dxs)
        # convert to jnp arrays for processing

        res = stats.linregress( jnp.log(dxs), jnp.log(errors) + 3*jnp.log(dxs) )
        slope = jnp.abs( res.slope )
        # compute the order of the convergence using a line fit of the log(y)/log(x)


        self.assertGreater(slope, 4.0, msg=f"First derivative convergence test failed with observed order {slope}")
        # ensure the observed order is at least 4


    def test_first_derivative_mixed_derivatives(self):
        """
        Test that mixed first partial derivatives commute up to numerical tolerance.
        This test constructs a 3D periodic grid in x, y, z and evaluates the scalar field
        f(x,y,z) = x^2 * y^2 on that grid. It computes the mixed derivatives
        d/dx(d/dy f) and d/dy(d/dx f) using the numerical finite-difference helper
        diff1_field (first derivative along a given axis with grid spacing dx) and
        compares the two results.
        Because the polynomial is not periodic, a small number of boundary points
        are excluded from the error measurement (slice(3, -3)). The test asserts that
        the mean absolute difference between the two mixed derivatives is below
        5e-5 and that the maximum absolute difference is below 5e-4. If either
        threshold is exceeded, the test fails with a message including the observed
        mean or maximum error.
        """


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = X**2 * Y**2
        # create a polynomial

        dfdx = diff1_field(f, 0, dx)
        dfdxdy = diff1_field(dfdx, 1, dx)
        # compute dfdxdy numerically

        dfdy = diff1_field(f, 1, dx)
        dfdydx = diff1_field(dfdy, 0, dx)
        # compute dfdydx numerically

        error = dfdxdy - dfdydx
        # compute the error in the mixed derivatives

        _slice = slice(3, -3) # ignore the first 3 points of the grid because quadratic is not periodic

        mean_error = jnp.mean(jnp.abs(error[_slice, _slice, _slice]))
        # calculate the mean error

        self.assertLess(mean_error, 5e-5, msg=f"Mixed derivatives failed in x direction with mean error {mean_error}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error = jnp.max(jnp.abs(error[_slice, _slice, _slice]))
        # calculate the max error

        self.assertLess(max_error, 5e-4, msg=f'Mixed derivatives failed with max error {max_error}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_sixth_derivative_1d_trigonometric(self):
        """
        Test that the sixth-order finite-difference derivative routine correctly computes
        the derivative of a trigonometric field and returns near-zero derivatives in
        directions with no variation.
        This unit test constructs a 3D periodic grid (linspace with endpoint=False) and
        defines a scalar field f(x,y,z) = sin(x) that only varies along the x-axis.
        A relatively high wave number is chosen to exercise the stencil and reduce
        noise sensitivity. The analytical sixth derivative of sin(x) is -sin(x).
        What is verified:
        - The numerical sixth-order derivative along the x-axis (computed by diff6_field)
            matches the analytical result up to tolerances for both mean and maximum
            absolute errors.
        - The numerical sixth-order derivatives along the y- and z-axes are essentially
            zero (since the field has no y or z dependence), again checked for mean and
            maximum absolute errors.
        Key numerical settings used in the test (for context; not mutated by the test):
        - Wave number k = 4 * (2*pi) and grid resolution n = 150 (uniform dx).
        - The derivative operator under test is invoked as diff6_field(f, axis, dx).
        Failure mode:
        - The test fails if the mean or max absolute errors exceed the specified
            thresholds (mean_x < 5e-2, mean_y/z < 5e-4; max_x < 5e-2, max_y/z < 6e-4),
            indicating an incorrect implementation or insufficient resolution for the
            chosen wave number.
        """

        #### TOO MANY POINTS PER WAVELENGTH CAN CAUSE THE TEST TO FAIL DUE TO FLOATING POINT PRECISION ISSUES ####
        k = 4 * (2*jnp.pi)
        n = 150
        # wave number and numerical resolution

        x = jnp.linspace(-k, k, n, endpoint=False)
        y = jnp.linspace(-k, k, n, endpoint=False)
        z = jnp.linspace(-k, k, n, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X)
        # create a sin vector field along the x direction
        dfdx6_analytical = -1 * jnp.sin(X)
        # define the derivative of the vector field analytically
        dfdx6_numerical = diff6_field(f, 0, dx)
        # calculate the derivative of the vector field numerically

        dfdx6_error = dfdx6_numerical - dfdx6_analytical
        # calculate the error in the derivative wrt to x

        dfdy6_error = diff6_field(f, 1, dx)
        # derivative should be zero

        dfdz6_error = diff6_field(f, 2, dx)
        # derivative should be zero

        mean_error_x = jnp.mean(jnp.abs(dfdx6_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy6_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz6_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 5e-2, msg=f"Trigonometric derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 5e-4, msg=f"Trigonometric derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 5e-4, msg=f"Trigonometric derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx6_error))
        max_error_y = jnp.max(jnp.abs(dfdy6_error))
        max_error_z = jnp.max(jnp.abs(dfdz6_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-2, msg=f'Trigonometric derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 6e-4, msg=f'Trigonometric derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 6e-4, msg=f'Trigonometric derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold




if __name__ == '__main__':
    # Configure JAX for testing
    from JAX_BSSN import setup_jax_config
    setup_jax_config(enable_x64=True, verbose=True)
    
    unittest.main(verbosity=2)