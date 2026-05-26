"""
Unit tests for tensor algebra operations in numerical relativity.

This module tests all tensor algebra functions against known analytical solutions
for flat space (Minkowski) and Schwarzschild metrics.
"""

import unittest
import jax.numpy as jnp
import numpy as np
from jax import jit
import jax

from JAX_BSSN.tensor_algebra import (
    invert_3x3_metric,
    determinant_3x3_metric,
    raise_index,
    lower_index,
    christoffel_symbols_first_kind,
    christoffel_symbols_second_kind,
    riemann_tensor,
    ricci_tensor,
    ricci_scalar,
    trace_tensor,
    traceless_part,
    lie_derivative_metric,
    lie_derivative_conformal_metric
)

from JAX_BSSN.derivatives import diff1_field, compute_all_derivatives


class TestTensorAlgebra(unittest.TestCase):
    """Test suite for tensor algebra operations."""
    
    def setUp(self):
        """Set up test parameters and grid."""
        self.n = 50  # Grid size
        self.dx = 0.01  # Grid spacing
        self.tol = 1e-5  # Numerical tolerance

        # Create coordinate grids
        x = jnp.linspace(-1.0, 1.0, self.n)
        y = jnp.linspace(-1.0, 1.0, self.n)
        z = jnp.linspace(-1.0, 1.0, self.n)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # Radial coordinate for Schwarzschild
        self.r = jnp.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        # Avoid singularity at origin
        self.r = jnp.where(self.r < 0.1, 0.1, self.r)
    
    def create_flat_metric(self):
        """Create flat space metric in Cartesian coordinates."""
        metric = jnp.zeros((3, 3, self.n, self.n, self.n))
        metric = metric.at[0, 0].set(1.0)  # g_xx = 1
        metric = metric.at[1, 1].set(1.0)  # g_yy = 1
        metric = metric.at[2, 2].set(1.0)  # g_zz = 1
        return metric
    
    def create_schwarzschild_metric(self, mass=1.0):
        """
        Create Schwarzschild metric in isotropic coordinates.
        
        ds² = ψ⁴(dr² + r²dθ² + r²sin²θdφ²)
        where ψ = (1 + M/(2r))
        
        In Cartesian coordinates: g_ij = ψ⁴ δ_ij
        """
        psi = 1.0 + mass / (2.0 * self.r)
        psi4 = psi**4
        
        metric = jnp.zeros((3, 3, self.n, self.n, self.n))
        metric = metric.at[0, 0].set(psi4)  # g_xx = ψ⁴
        metric = metric.at[1, 1].set(psi4)  # g_yy = ψ⁴
        metric = metric.at[2, 2].set(psi4)  # g_zz = ψ⁴
        return metric
    
    def compute_metric_derivatives(self, metric):
        """Compute derivatives of metric tensor."""

        derivs = jnp.stack( [diff1_field(metric, d+2, self.dx) for d in range(3)], axis=0)

        return derivs
    
    def test_flat_metric_inversion(self):
        """Test metric inversion for flat space."""
        metric = self.create_flat_metric()
        inv_metric = invert_3x3_metric(metric)
        
        # For flat space, inverse should be identity
        expected = jnp.zeros((3, 3, self.n, self.n, self.n))
        expected = expected.at[0, 0].set(1.0)
        expected = expected.at[1, 1].set(1.0)
        expected = expected.at[2, 2].set(1.0)
        
        np.testing.assert_allclose(inv_metric, expected, atol=self.tol)
    
    def test_schwarzschild_metric_inversion(self):
        """Test metric inversion for Schwarzschild metric."""
        mass = 0.5
        metric = self.create_schwarzschild_metric(mass)
        inv_metric = invert_3x3_metric(metric)
        
        # For Schwarzschild in isotropic coords: g^ij = ψ⁻⁴ δ_ij
        psi = 1.0 + mass / (2.0 * self.r)
        psi_inv4 = psi**(-4)
        
        expected = jnp.zeros((3, 3, self.n, self.n, self.n))
        expected = expected.at[0, 0].set(psi_inv4)
        expected = expected.at[1, 1].set(psi_inv4)
        expected = expected.at[2, 2].set(psi_inv4)
        
        np.testing.assert_allclose(inv_metric, expected, atol=self.tol)
    
    def test_flat_metric_determinant(self):
        """Test determinant computation for flat space."""
        metric = self.create_flat_metric()
        det = determinant_3x3_metric(metric)
        
        # Flat space determinant should be 1
        expected = jnp.ones((self.n, self.n, self.n))
        np.testing.assert_allclose(det, expected, atol=self.tol)
    
    def test_schwarzschild_metric_determinant(self):
        """Test determinant computation for Schwarzschild metric."""
        mass = 0.5
        metric = self.create_schwarzschild_metric(mass)
        det = determinant_3x3_metric(metric)
        
        # For Schwarzschild: det(g) = ψ¹²
        psi = 1.0 + mass / (2.0 * self.r)
        expected = psi**12
        
        np.testing.assert_allclose(det, expected, atol=self.tol)
    
    def test_flat_christoffel_symbols(self):
        """Test Christoffel symbols for flat space."""
        metric = self.create_flat_metric()
        metric_derivs = self.compute_metric_derivatives(metric)
        
        # First kind
        christoffel_1 = christoffel_symbols_first_kind(metric_derivs)
        # For flat space, all Christoffel symbols should be zero
        expected = jnp.zeros((3, 3, 3, self.n, self.n, self.n))
        error = christoffel_1 - expected
        max_error = jnp.max(jnp.abs(error))
        self.assertLess(max_error, self.tol, msg=f"Christoffel symbols (1st kind) failed with max error {max_error}")

        # Second kind
        inv_metric = invert_3x3_metric(metric)
        christoffel_2 = christoffel_symbols_second_kind(inv_metric, metric_derivs)
        error = christoffel_2 - expected
        max_error = jnp.max(jnp.abs(error))
        self.assertLess(max_error, self.tol, msg=f"Christoffel symbols (2nd kind) failed with max error {max_error}")

    def test_schwarzschild_christoffel_symbols(self):
        """Test Christoffel symbols for Schwarzschild metric."""
        mass = 0.5
        metric = self.create_schwarzschild_metric(mass)
        metric_derivs = self.compute_metric_derivatives(metric)
        inv_metric = invert_3x3_metric(metric)
        
        christoffel_2 = christoffel_symbols_second_kind(inv_metric, metric_derivs)
        
        # For Schwarzschild in isotropic coordinates, we expect non-zero symbols
        # but they should be symmetric in lower indices
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    np.testing.assert_allclose(
                        christoffel_2[i, j, k], 
                        christoffel_2[i, k, j], 
                        atol=1e-12,
                        err_msg=f"Christoffel symbol not symmetric: Γ^{i}_{j}{k} ≠ Γ^{i}_{k}{j}"
                    )
    
    def test_flat_riemann_tensor(self):
        """Test Riemann tensor for flat space."""
        metric = self.create_flat_metric()
        metric_derivs = self.compute_metric_derivatives(metric)
        inv_metric = invert_3x3_metric(metric)
        
        christoffel = christoffel_symbols_second_kind(inv_metric, metric_derivs)
        
        # Compute derivatives of Christoffel symbols
        christoffel_derivs = jnp.zeros((3, 3, 3, 3, self.n, self.n, self.n))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        christoffel_derivs = christoffel_derivs.at[l, i, j, k].set(
                            diff1_field(christoffel[i, j, k], l, self.dx)
                        )
        
        riemann = riemann_tensor(christoffel, christoffel_derivs)
        
        # For flat space, Riemann tensor should be zero
        expected = jnp.zeros((3, 3, 3, 3, self.n, self.n, self.n))
        np.testing.assert_allclose(riemann, expected, atol=1e-8)
    
    def test_flat_ricci_tensor(self):
        """Test Ricci tensor for flat space."""
        # Create zero Riemann tensor for flat space
        riemann = jnp.zeros((3, 3, 3, 3, self.n, self.n, self.n))
        ricci = ricci_tensor(riemann)
        
        # For flat space, Ricci tensor should be zero
        expected = jnp.zeros((3, 3, self.n, self.n, self.n))
        np.testing.assert_allclose(ricci, expected, atol=self.tol)
    
    def test_flat_ricci_scalar(self):
        """Test Ricci scalar for flat space."""
        # Create zero Ricci tensor for flat space
        ricci = jnp.zeros((3, 3, self.n, self.n, self.n))
        metric = self.create_flat_metric()
        inv_metric = invert_3x3_metric(metric)
        
        scalar = ricci_scalar(ricci, inv_metric)
        
        # For flat space, Ricci scalar should be zero
        expected = jnp.zeros((self.n, self.n, self.n))
        np.testing.assert_allclose(scalar, expected, atol=self.tol)
    
    def test_index_raising_lowering(self):
        """Test index raising and lowering operations."""
        metric = self.create_flat_metric()
        inv_metric = invert_3x3_metric(metric)
        
        # Create a test tensor
        tensor = jnp.zeros((3, 3, self.n, self.n, self.n))
        tensor = tensor.at[0, 1].set(self.X)
        tensor = tensor.at[1, 2].set(self.Y)
        tensor = tensor.at[2, 0].set(self.Z)
        
        # For flat space, raising and lowering should be identity operations
        raised_0 = raise_index(tensor, inv_metric, 0)
        raised_1 = raise_index(tensor, inv_metric, 1)
        
        lowered_0 = lower_index(raised_0, metric, 0)
        lowered_1 = lower_index(raised_1, metric, 1)
        
        np.testing.assert_allclose(tensor, lowered_0, atol=self.tol)
        np.testing.assert_allclose(tensor, lowered_1, atol=self.tol)
    
    def test_trace_operations(self):
        """Test trace and traceless operations."""
        metric = self.create_flat_metric()
        inv_metric = invert_3x3_metric(metric)
        
        # Create a test tensor with known trace
        tensor = jnp.zeros((3, 3, self.n, self.n, self.n))
        tensor = tensor.at[0, 0].set(2.0)
        tensor = tensor.at[1, 1].set(3.0)
        tensor = tensor.at[2, 2].set(4.0)
        tensor = tensor.at[0, 1].set(self.X)
        tensor = tensor.at[1, 0].set(self.X)
        
        # Test trace computation
        trace = trace_tensor(tensor, inv_metric)
        expected_trace = 2.0 + 3.0 + 4.0  # Sum of diagonal elements for flat metric
        np.testing.assert_allclose(trace, expected_trace * jnp.ones_like(trace), atol=self.tol)
        
        # Test traceless part
        traceless = traceless_part(tensor, metric, inv_metric)
        traceless_trace = trace_tensor(traceless, inv_metric)
        
        # Traceless part should have zero trace
        np.testing.assert_allclose(traceless_trace, jnp.zeros_like(traceless_trace), atol=self.tol)
    
    def test_lie_derivative_flat_space(self):
        """Test Lie derivative for flat space with constant vector field."""
        metric = self.create_flat_metric()
        
        # Constant vector field
        vector = jnp.zeros((3, self.n, self.n, self.n))
        vector = vector.at[0].set(1.0)  # v^x = 1
        vector = vector.at[1].set(2.0)  # v^y = 2
        vector = vector.at[2].set(3.0)  # v^z = 3
        
        lie_deriv = lie_derivative_metric(vector, metric, self.dx)
        
        # For constant vector and flat metric, Lie derivative should be zero
        expected = jnp.zeros((3, 3, self.n, self.n, self.n))
        np.testing.assert_allclose(lie_deriv, expected, atol=1e-8)
    
    def test_conformal_lie_derivative(self):
        """Test conformal Lie derivative."""
        metric = self.create_flat_metric()
        
        # Linear vector field to test divergence term
        vector = jnp.zeros((3, self.n, self.n, self.n))
        vector = vector.at[0,...].set(self.X)  # v^x = x
        vector = vector.at[1,...].set(self.Y)  # v^y = y
        vector = vector.at[2,...].set(self.Z)  # v^z = z
        
        regular_lie = lie_derivative_metric(vector, metric, self.dx)
        conformal_lie = lie_derivative_conformal_metric(vector, metric, self.dx)
        
        # Compute the actual divergence of the vector field
        div_v = jnp.zeros(vector.shape[1:])
        for k in range(3):
            div_v += diff1_field(vector[k,...], k, self.dx)

        # The conformal correction term is -(2/3) * metric * div(v)
        metric_div_v = jnp.einsum('ij..., ...->ij...', metric, div_v)
        expected_correction = -(2.0/3.0) * metric_div_v

        error = conformal_lie - (regular_lie + expected_correction)
        mean_error = jnp.mean(jnp.abs(error))
        self.assertLess(mean_error, 1e-6, msg=f"Conformal Lie derivative failed with mean error {mean_error}")


    def test_metric_identity_property(self):
        """Test g_ij * g^jk = δ_i^k."""
        metric = self.create_schwarzschild_metric(0.3)
        inv_metric = invert_3x3_metric(metric)
        
        # Compute g_ij * g^jk
        identity = jnp.einsum('ij...,jk...->ik...', metric, inv_metric)
        
        # Should give Kronecker delta
        expected = jnp.zeros((3, 3, self.n, self.n, self.n))
        expected = expected.at[0, 0].set(1.0)
        expected = expected.at[1, 1].set(1.0)
        expected = expected.at[2, 2].set(1.0)
        
        np.testing.assert_allclose(identity, expected, atol=self.tol)
    
    def test_christoffel_symmetry(self):
        """Test that Christoffel symbols are symmetric in lower indices."""
        mass = 0.3
        metric = self.create_schwarzschild_metric(mass)
        metric_derivs = self.compute_metric_derivatives(metric)
        
        christoffel_1 = christoffel_symbols_first_kind(metric_derivs)
        
        # Test symmetry: Γ_ijk = Γ_ikj
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    np.testing.assert_allclose(
                        christoffel_1[i, j, k],
                        christoffel_1[i, k, j],
                        atol=1e-12,
                        err_msg=f"First kind Christoffel not symmetric: Γ_{i}{j}{k} ≠ Γ_{i}{k}{j}"
                    )
    
    def test_riemann_symmetries(self):
        """Test Riemann tensor symmetries."""
        # Create a simple non-flat case
        metric = self.create_schwarzschild_metric(0.1)  # Small mass to avoid numerical issues
        metric_derivs = self.compute_metric_derivatives(metric)
        inv_metric = invert_3x3_metric(metric)
        
        christoffel = christoffel_symbols_second_kind(inv_metric, metric_derivs)
        
        # Compute Christoffel derivatives (simplified for test)
        christoffel_derivs = jnp.zeros((3, 3, 3, 3, self.n, self.n, self.n))
        
        riemann = riemann_tensor(christoffel, christoffel_derivs)
        
        # Test antisymmetry: R^i_jkl = -R^i_jlk
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        np.testing.assert_allclose(
                            riemann[i, j, k, l],
                            -riemann[i, j, l, k],
                            atol=1e-10,
                            err_msg=f"Riemann tensor antisymmetry failed: R^{i}_{j}{k}{l} ≠ -R^{i}_{j}{l}{k}"
                        )


if __name__ == '__main__':
    # Configure JAX for testing
    from JAX_BSSN import setup_jax_config
    setup_jax_config(enable_x64=True, verbose=True)
    
    unittest.main(verbosity=2)
