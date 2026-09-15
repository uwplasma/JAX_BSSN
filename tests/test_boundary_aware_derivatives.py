"""Tests for periodic and Sommerfeld finite-difference closures."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.evolution.boundaries import (
    PERIODIC_BC,
    SOMMERFELD_BC,
)
from JAX_BSSN.evolution.derivatives import (
    diff1_field,
    diff1_upwind_field,
    diff2_field,
    diff6_field,
)


class TestBoundaryAwareDerivatives(unittest.TestCase):
    def setUp(self):
        self.n = 12
        self.dx = 0.2
        self.x = -0.7 + self.dx * jnp.arange(self.n, dtype=jnp.float64)

    def test_periodic_first_and_sixth_derivatives_are_unchanged(self):
        field = jnp.sin(self.x) + 0.2 * jnp.cos(2.0 * self.x)

        expected_first = (
            2.0 / 3.0 * (jnp.roll(field, -1) - jnp.roll(field, 1)) / self.dx
            - 1.0 / 12.0 * (jnp.roll(field, -2) - jnp.roll(field, 2)) / self.dx
        )
        expected_sixth = (
            jnp.roll(field, -3)
            - 6.0 * jnp.roll(field, -2)
            + 15.0 * jnp.roll(field, -1)
            - 20.0 * field
            + 15.0 * jnp.roll(field, 1)
            - 6.0 * jnp.roll(field, 2)
            + jnp.roll(field, 3)
        ) / self.dx**6

        np.testing.assert_allclose(
            diff1_field(field, 0, self.dx, PERIODIC_BC, PERIODIC_BC),
            expected_first,
            rtol=0.0,
            atol=np.finfo(float).eps * 4,
        )
        np.testing.assert_allclose(
            diff6_field(field, 0, self.dx, PERIODIC_BC, PERIODIC_BC),
            expected_sixth,
            rtol=1.0e-13,
            atol=1.0e-10,
        )

    def test_upwind_sommerfeld_faces_use_centered_d4_width_three(self):
        field = jnp.exp(self.x) + 0.1 * self.x**4
        coefficient = jnp.resize(
            jnp.asarray([1.0, -1.0, 0.0], dtype=field.dtype), (self.n,)
        )
        centered_d4 = diff1_field(
            field, 0, self.dx, SOMMERFELD_BC, SOMMERFELD_BC, mad_q=1.0
        )
        upwind_d4 = diff1_upwind_field(
            field,
            coefficient,
            0,
            self.dx,
            SOMMERFELD_BC,
            SOMMERFELD_BC,
            mad_q=1.0,
        )
        upwind_requested_d6 = diff1_upwind_field(
            field,
            coefficient,
            0,
            self.dx,
            SOMMERFELD_BC,
            SOMMERFELD_BC,
            mad_q=0.0,
        )

        np.testing.assert_allclose(
            upwind_d4[:3], centered_d4[:3], rtol=0.0, atol=1.0e-13
        )
        np.testing.assert_allclose(
            upwind_d4[-3:], centered_d4[-3:], rtol=0.0, atol=1.0e-13
        )
        np.testing.assert_allclose(
            upwind_requested_d6, upwind_d4, rtol=0.0, atol=1.0e-13
        )

    def test_upwind_physical_face_closures_do_not_use_wrapped_values(self):
        field = jnp.sin(self.x) + 0.2 * self.x**2

        changed_right = field.at[-4:].set(
            jnp.asarray([1.0e6, -2.0e6, 3.0e6, -4.0e6])
        )
        left_reference = diff1_upwind_field(
            field,
            -1.0,
            0,
            self.dx,
            SOMMERFELD_BC,
            PERIODIC_BC,
        )
        left_with_opposite_sentinel = diff1_upwind_field(
            changed_right,
            -1.0,
            0,
            self.dx,
            SOMMERFELD_BC,
            PERIODIC_BC,
        )
        np.testing.assert_array_equal(
            left_with_opposite_sentinel[:3], left_reference[:3]
        )

        changed_left = field.at[:4].set(
            jnp.asarray([-5.0e6, 6.0e6, -7.0e6, 8.0e6])
        )
        right_reference = diff1_upwind_field(
            field,
            1.0,
            0,
            self.dx,
            PERIODIC_BC,
            SOMMERFELD_BC,
        )
        right_with_opposite_sentinel = diff1_upwind_field(
            changed_left,
            1.0,
            0,
            self.dx,
            PERIODIC_BC,
            SOMMERFELD_BC,
        )
        np.testing.assert_array_equal(
            right_with_opposite_sentinel[-3:], right_reference[-3:]
        )

    def test_second_derivative_lopsided_coefficients_and_array_axes(self):
        field_1d = self.x**5 - 0.4 * self.x**4 + 0.2 * self.x**2
        exact_1d = 20.0 * self.x**3 - 4.8 * self.x**2 + 0.4

        derivative = diff2_field(
            field_1d, 0, self.dx, SOMMERFELD_BC, SOMMERFELD_BC
        )
        boundary_indices = jnp.asarray([0, 1, self.n - 2, self.n - 1])
        np.testing.assert_allclose(
            derivative[boundary_indices],
            exact_1d[boundary_indices],
            rtol=0.0,
            atol=2.0e-11,
        )

        for direction in (0, 1, 2, 3, 4):
            reshape = (1,) * direction + (self.n,) + (1,) * (4 - direction)
            shape = (2,) * direction + (self.n,) + (2,) * (4 - direction)
            field = jnp.broadcast_to(field_1d.reshape(reshape), shape)
            exact = jnp.broadcast_to(exact_1d.reshape(reshape), shape)
            numerical = diff2_field(
                field, direction, self.dx, SOMMERFELD_BC, SOMMERFELD_BC
            )
            selected = jnp.take(numerical, boundary_indices, axis=direction)
            expected = jnp.take(exact, boundary_indices, axis=direction)
            np.testing.assert_allclose(selected, expected, rtol=0.0, atol=2.0e-11)

    def test_second_derivative_boundary_closure_is_fourth_order(self):
        errors = []
        for n in (17, 33, 65):
            dx = 1.0 / (n - 1)
            x = dx * jnp.arange(n, dtype=jnp.float64)
            field = jnp.exp(x)
            derivative = diff2_field(
                field, 0, dx, SOMMERFELD_BC, SOMMERFELD_BC
            )
            indices = jnp.asarray([0, 1, n - 2, n - 1])
            errors.append(float(jnp.max(jnp.abs(derivative[indices] - field[indices]))))

        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        self.assertTrue(np.all(orders > 3.7), orders)

    def test_first_derivative_lopsided_coefficients_and_array_axes(self):
        field_1d = self.x**4
        exact_1d = 4.0 * self.x**3

        left_0 = (
            -25.0 / 12.0 * field_1d[0]
            + 4.0 * field_1d[1]
            - 3.0 * field_1d[2]
            + 4.0 / 3.0 * field_1d[3]
            - 1.0 / 4.0 * field_1d[4]
        ) / self.dx
        left_1 = (
            -1.0 / 4.0 * field_1d[0]
            - 5.0 / 6.0 * field_1d[1]
            + 3.0 / 2.0 * field_1d[2]
            - 1.0 / 2.0 * field_1d[3]
            + 1.0 / 12.0 * field_1d[4]
        ) / self.dx
        right_1 = (
            -1.0 / 12.0 * field_1d[-5]
            + 1.0 / 2.0 * field_1d[-4]
            - 3.0 / 2.0 * field_1d[-3]
            + 5.0 / 6.0 * field_1d[-2]
            + 1.0 / 4.0 * field_1d[-1]
        ) / self.dx
        right_0 = (
            1.0 / 4.0 * field_1d[-5]
            - 4.0 / 3.0 * field_1d[-4]
            + 3.0 * field_1d[-3]
            - 4.0 * field_1d[-2]
            + 25.0 / 12.0 * field_1d[-1]
        ) / self.dx

        derivative = diff1_field(
            field_1d, 0, self.dx, SOMMERFELD_BC, SOMMERFELD_BC
        )
        np.testing.assert_allclose(
            derivative[jnp.asarray([0, 1, self.n - 2, self.n - 1])],
            jnp.asarray([left_0, left_1, right_1, right_0]),
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(derivative, exact_1d, rtol=0.0, atol=2.0e-13)

        for direction in (0, 1, 2, 3, 4):
            reshape = (1,) * direction + (self.n,) + (1,) * (4 - direction)
            field = jnp.broadcast_to(field_1d.reshape(reshape), (2,) * direction + (self.n,) + (2,) * (4 - direction))
            exact = jnp.broadcast_to(exact_1d.reshape(reshape), field.shape)
            numerical = diff1_field(
                field, direction, self.dx, SOMMERFELD_BC, SOMMERFELD_BC
            )
            np.testing.assert_allclose(numerical, exact, rtol=0.0, atol=2.0e-13)

    def test_mixed_first_derivative_boundaries_only_replace_selected_side(self):
        field = jnp.exp(self.x)
        periodic = diff1_field(field, 0, self.dx)

        left_only = diff1_field(
            field, 0, self.dx, SOMMERFELD_BC, PERIODIC_BC
        )
        right_only = diff1_field(
            field, 0, self.dx, PERIODIC_BC, SOMMERFELD_BC
        )

        np.testing.assert_array_equal(left_only[-2:], periodic[-2:])
        np.testing.assert_array_equal(right_only[:2], periodic[:2])
        self.assertFalse(np.array_equal(np.asarray(left_only[:2]), np.asarray(periodic[:2])))
        self.assertFalse(np.array_equal(np.asarray(right_only[-2:]), np.asarray(periodic[-2:])))

    def test_first_derivative_boundary_closure_is_fourth_order(self):
        errors = []
        for n in (17, 33, 65):
            dx = 1.0 / (n - 1)
            x = dx * jnp.arange(n, dtype=jnp.float64)
            field = jnp.exp(x)
            derivative = diff1_field(
                field, 0, dx, SOMMERFELD_BC, SOMMERFELD_BC
            )
            indices = jnp.asarray([0, 1, n - 2, n - 1])
            errors.append(float(jnp.max(jnp.abs(derivative[indices] - field[indices]))))

        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        self.assertTrue(np.all(orders > 3.8), orders)

    def test_sixth_derivative_lopsided_closures(self):
        field = self.x**7
        exact = 5040.0 * self.x
        derivative = diff6_field(
            field, 0, self.dx, SOMMERFELD_BC, SOMMERFELD_BC
        )

        boundary_indices = jnp.asarray(
            [0, 1, 2, self.n - 3, self.n - 2, self.n - 1]
        )
        np.testing.assert_allclose(
            derivative[boundary_indices],
            exact[boundary_indices],
            rtol=0.0,
            atol=2.0e-9,
        )

    def test_sixth_derivative_boundary_closure_is_second_order(self):
        errors = []
        for n in (17, 33, 65):
            dx = 1.0 / (n - 1)
            x = dx * jnp.arange(n, dtype=jnp.float64)
            field = jnp.exp(x)
            derivative = diff6_field(
                field, 0, dx, SOMMERFELD_BC, SOMMERFELD_BC
            )
            indices = jnp.asarray([0, 1, 2, n - 3, n - 2, n - 1])
            errors.append(float(jnp.max(jnp.abs(derivative[indices] - field[indices]))))

        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        self.assertTrue(np.all(orders > 1.8), orders)


if __name__ == "__main__":
    unittest.main()
