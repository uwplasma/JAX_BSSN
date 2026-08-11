import jax.numpy as jnp
import unittest
from scipy import stats

from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.bssn import (BSSNVariables, BSSNParameters)
from JAX_BSSN.tensor_algebra import (invert_3x3_metric, christoffel_symbols_second_kind)
from JAX_BSSN.derivatives import (diff1_field)
from JAX_BSSN.errors import (compute_hamiltonian_constraint)
from tests.helpers import vacuum_matter_fields

class TestGaugeWave(unittest.TestCase):

    def test_gauge_wave_evolution(self):

        ##################### SIMULATION PARAMETERS #####################
        x_wind = 1.0
        nx = 60

        x = jnp.linspace(-x_wind/2, x_wind/2, nx, endpoint=False)
        y = jnp.linspace(-x_wind/2, x_wind/2, nx, endpoint=False)
        z = jnp.linspace(-x_wind/2, x_wind/2, nx, endpoint=False)

        dx = x[1] - x[0]
        dt = dx
        ################################################################

        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        H = 0.1 * jnp.sin(2 * jnp.pi * X )
        dHdt = -2 * jnp.pi * 0.1 * jnp.sin(2 * jnp.pi * X / x_wind)
        g_00 = -1 * (1 - H)
        g_11 = 1 - H
        g_22 = 1.0 * jnp.ones_like(H)
        g_33 = g_22
        # metric tensor being evolved

        def conformal_metric_analytic_00(t):
            H = 0.1 * jnp.sin(2 * jnp.pi * (X - t) / x_wind)
            return jnp.power(1 - H, 2/3)

        def conformal_metric_analytic_11(t):
            H = 0.1 * jnp.sin(2 * jnp.pi * (X - t) / x_wind)
            return jnp.power(1 - H, -1/3)

        def conformal_metric_analytic_22(t):
            H = 0.1 * jnp.sin(2 * jnp.pi * (X - t) / x_wind)
            return jnp.power(1 - H, -1/3)
        # analytic solution for conformal metric components

        def conformal_factor_analytic(t):
            H = 0.1 * jnp.sin(2 * jnp.pi * (X - t) / x_wind)
            return jnp.power(1 - H, -1/6)
        # analytic solution for conformal factor

        induced_metric = jnp.zeros(shape = (3, 3, nx, nx, nx) )

        induced_metric = induced_metric.at[0,0].set(g_11)
        induced_metric = induced_metric.at[1,1].set(g_22)
        induced_metric = induced_metric.at[2,2].set(g_33)
        # add the space components of the metric to the induced metric variable

        conformal_factor = jnp.power(1 - H, -1/6)
        initial_conformal_metric = jnp.zeros(shape = (3, 3, nx, nx, nx) )
        initial_conformal_metric = initial_conformal_metric.at[0,0].set( conformal_metric_analytic_00(0.0) )
        initial_conformal_metric = initial_conformal_metric.at[1,1].set( conformal_metric_analytic_11(0.0) )
        initial_conformal_metric = initial_conformal_metric.at[2,2].set( conformal_metric_analytic_22(0.0) )
        # compute initial conformal metric

        initial_lapse = jnp.sqrt( -1 * g_00 )
        initial_shift = jnp.zeros(shape=(3,nx,nx,nx) )

        derivs = jnp.stack( [diff1_field(initial_conformal_metric, d+2, dx) for d in range(3)], axis=0)
        inv_metric = invert_3x3_metric(initial_conformal_metric)
        christoffel_2 = christoffel_symbols_second_kind(inv_metric, derivs)
        # compute Christoffel symbols for initial conformal metric

        inv_conformal_metric = invert_3x3_metric(initial_conformal_metric)
        initial_conformal_connection = jnp.einsum('mn..., imn... -> i...', inv_conformal_metric, christoffel_2)
        # compute initial conformal connection functions

        extrinsic_curvature = jnp.zeros_like(induced_metric)
        extrinsic_curvature = extrinsic_curvature.at[0,0].set( -0.1*jnp.pi*jnp.cos(2*jnp.pi*X) / initial_lapse )
        trace_extrinsic_curvature = jnp.einsum('mn..., mn... -> ...', inv_metric, extrinsic_curvature)
        traceless_extrinsic_curvature = conformal_factor**2 * ( extrinsic_curvature - initial_conformal_metric * trace_extrinsic_curvature / 3 )
        # intitial extrinsic curvature
        rho, stress_tensor, momentum_density = vacuum_matter_fields(
            (nx, nx, nx), initial_conformal_metric.dtype
        )

        vars = BSSNVariables(
            conformal_metric=initial_conformal_metric,
            conformal_factor=conformal_factor,
            traceless_K=traceless_extrinsic_curvature,
            trace_K=trace_extrinsic_curvature,
            conformal_connection=initial_conformal_connection,
            lapse=initial_lapse,
            shift=initial_shift,
            rho=rho,
            S_ij=stress_tensor,
            momentum_density=momentum_density,
        )

        params = BSSNParameters(
            eta=0.0,          # Damping parameter for Γ^i evolution
            kappa = 0.0,      # no momentum damping in the analytic gauge-wave check
            g=0.0,           # Gamma driver shift parameter
            dx=dx,           # Grid spacing
            dt=dt         # Time step
        )

        initial_hamiltonian_constraint = compute_hamiltonian_constraint(vars, params)
        # compute initial Hamiltonian constraint violation

        for t in range(50):
            vars = rk4_step(vars, params)

        final_hamiltonian_constraint = compute_hamiltonian_constraint(vars, params)
        # compute final Hamiltonian constraint violation

        numerical_conformal_metric_00 = vars.conformal_metric[0,0]
        numerical_conformal_metric_11 = vars.conformal_metric[1,1]
        numerical_conformal_metric_22 = vars.conformal_metric[2,2]
        numerical_conformal_factor = vars.conformal_factor
        # extract numerical solutions

        analytical_conformal_metric_00 = conformal_metric_analytic_00(50*dt)
        analytical_conformal_metric_11 = conformal_metric_analytic_11(50*dt)
        analytical_conformal_metric_22 = conformal_metric_analytic_22(50*dt)
        analytical_conformal_factor = conformal_factor_analytic(50*dt)
        # compute analytical solutions at final time

        relative_g00_error = jnp.sqrt( jnp.mean( ((numerical_conformal_metric_00 - analytical_conformal_metric_00)/analytical_conformal_metric_00)**2 ) )
        relative_g11_error = jnp.sqrt( jnp.mean( ((numerical_conformal_metric_11 - analytical_conformal_metric_11)/analytical_conformal_metric_11)**2 ) )
        relative_g22_error = jnp.sqrt( jnp.mean( ((numerical_conformal_metric_22 - analytical_conformal_metric_22)/analytical_conformal_metric_22)**2 ) )
        relative_cf_error  = jnp.sqrt( jnp.mean( ((numerical_conformal_factor - analytical_conformal_factor)/analytical_conformal_factor)**2 ) )
        # compute relative L2 errors

        self.assertLess(relative_g00_error, 5e-5)
        self.assertLess(relative_g11_error, 5e-5)
        self.assertLess(relative_g22_error, 5e-5)
        self.assertLess(relative_cf_error, 1e-4)
        # check that errors are below threshold


    def test_convergence(self):

        def run_simulation(nx):
            ##################### SIMULATION PARAMETERS #####################
            x_wind = 1.0
            x = jnp.linspace(-x_wind/2, x_wind/2, nx, endpoint=False)
            y = jnp.linspace(-x_wind/2, x_wind/2, nx, endpoint=False)
            z = jnp.linspace(-x_wind/2, x_wind/2, nx, endpoint=False)

            dx = x[1] - x[0]
            dt = dx
            ################################################################

            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            H = 0.1 * jnp.sin(2 * jnp.pi * X )
            dHdt = -2 * jnp.pi * 0.1 * jnp.sin(2 * jnp.pi * X / x_wind)
            g_00 = -1 * (1 - H)
            g_11 = 1 - H
            g_22 = 1.0 * jnp.ones_like(H)
            g_33 = g_22
            # metric tensor being evolved

            def conformal_metric_analytic_00(t):
                H = 0.1 * jnp.sin(2 * jnp.pi * (X - t) / x_wind)
                return jnp.power(1 - H, 2/3)

            def conformal_metric_analytic_11(t):
                H = 0.1 * jnp.sin(2 * jnp.pi * (X - t) / x_wind)
                return jnp.power(1 - H, -1/3)

            def conformal_metric_analytic_22(t):
                H = 0.1 * jnp.sin(2 * jnp.pi * (X - t) / x_wind)
                return jnp.power(1 - H, -1/3)
            # analytic solution for conformal metric components

            def conformal_factor_analytic(t):
                H = 0.1 * jnp.sin(2 * jnp.pi * (X - t) / x_wind)
                return jnp.power(1 - H, -1/6)
            # analytic solution for conformal factor

            induced_metric = jnp.zeros(shape = (3, 3, nx, nx, nx) )

            induced_metric = induced_metric.at[0,0].set(g_11)
            induced_metric = induced_metric.at[1,1].set(g_22)
            induced_metric = induced_metric.at[2,2].set(g_33)
            # add the space components of the metric to the induced metric variable

            conformal_factor = jnp.power(1 - H, -1/6)
            initial_conformal_metric = jnp.zeros(shape = (3, 3, nx, nx, nx) )
            initial_conformal_metric = initial_conformal_metric.at[0,0].set( conformal_metric_analytic_00(0.0) )
            initial_conformal_metric = initial_conformal_metric.at[1,1].set( conformal_metric_analytic_11(0.0) )
            initial_conformal_metric = initial_conformal_metric.at[2,2].set( conformal_metric_analytic_22(0.0) )
            # compute initial conformal metric

            initial_lapse = jnp.sqrt( -1 * g_00 )
            initial_shift = jnp.zeros(shape=(3,nx,nx,nx) )

            derivs = jnp.stack( [diff1_field(initial_conformal_metric, d+2, dx) for d in range(3)], axis=0)
            inv_metric = invert_3x3_metric(initial_conformal_metric)
            christoffel_2 = christoffel_symbols_second_kind(inv_metric, derivs)
            # compute Christoffel symbols for initial conformal metric

            inv_conformal_metric = invert_3x3_metric(initial_conformal_metric)
            initial_conformal_connection = jnp.einsum('mn..., imn... -> i...', inv_conformal_metric, christoffel_2)
            # compute initial conformal connection functions

            extrinsic_curvature = jnp.zeros_like(induced_metric)
            extrinsic_curvature = extrinsic_curvature.at[0,0].set( -0.1*jnp.pi*jnp.cos(2*jnp.pi*X) / initial_lapse )
            trace_extrinsic_curvature = jnp.einsum('mn..., mn... -> ...', inv_metric, extrinsic_curvature)
            traceless_extrinsic_curvature = conformal_factor**2 * ( extrinsic_curvature - initial_conformal_metric * trace_extrinsic_curvature / 3 )
            # intitial extrinsic curvature
            rho, stress_tensor, momentum_density = vacuum_matter_fields(
                (nx, nx, nx), initial_conformal_metric.dtype
            )

            vars = BSSNVariables(
                conformal_metric=initial_conformal_metric,
                conformal_factor=conformal_factor,
                traceless_K=traceless_extrinsic_curvature,
                trace_K=trace_extrinsic_curvature,
                conformal_connection=initial_conformal_connection,
                lapse=initial_lapse,
                shift=initial_shift,
                rho=rho,
                S_ij=stress_tensor,
                momentum_density=momentum_density,
            )

            params = BSSNParameters(
                eta=0.0,          # Damping parameter for Γ^i evolution
                kappa = 0.0,      # no momentum damping in the analytic gauge-wave check
                g=0.0,           # Gamma driver shift parameter
                dx=dx,           # Grid spacing
                dt=dt         # Time step
            )

            initial_hamiltonian_constraint = compute_hamiltonian_constraint(vars, params)
            # compute initial Hamiltonian constraint violation

            for t in range(50):
                vars = rk4_step(vars, params)

            final_hamiltonian_constraint = compute_hamiltonian_constraint(vars, params)
            # compute final Hamiltonian constraint violation

            numerical_conformal_metric_00 = vars.conformal_metric[0,0]
            numerical_conformal_metric_11 = vars.conformal_metric[1,1]
            numerical_conformal_metric_22 = vars.conformal_metric[2,2]
            numerical_conformal_factor = vars.conformal_factor
            # extract numerical solutions

            analytical_conformal_metric_00 = conformal_metric_analytic_00(50*dt)
            analytical_conformal_metric_11 = conformal_metric_analytic_11(50*dt)
            analytical_conformal_metric_22 = conformal_metric_analytic_22(50*dt)
            analytical_conformal_factor = conformal_factor_analytic(50*dt)
            # compute analytical solutions at final time

            g00_error = jnp.sqrt( jnp.mean( (numerical_conformal_metric_00 - analytical_conformal_metric_00)**2 ) )
            g11_error = jnp.sqrt( jnp.mean( (numerical_conformal_metric_11 - analytical_conformal_metric_11)**2 ) )
            g22_error = jnp.sqrt( jnp.mean( (numerical_conformal_metric_22 - analytical_conformal_metric_22)**2 ) )
            cf_error  = jnp.sqrt( jnp.mean( (numerical_conformal_factor - analytical_conformal_factor)**2 ) )
            # compute L2 errors
            
            return g00_error, g11_error, g22_error, cf_error
            
        nx_values = [30, 50, 70]
        dxs       = jnp.asarray([1.0/nx for nx in nx_values])
        # define resolutions to test

        g00_errors = []
        g11_errors = []
        g22_errors = []
        cf_errors  = []

        for nx in nx_values:
            g00_, g11_, g22_, cf_ = run_simulation(nx)
            g00_errors.append( g00_ )
            g11_errors.append( g11_ )
            g22_errors.append( g22_ )
            cf_errors.append(  cf_  )
        g00_errors = jnp.asarray(g00_errors)
        g11_errors = jnp.asarray(g11_errors)
        g22_errors = jnp.asarray(g22_errors)
        cf_errors  = jnp.asarray(cf_errors)
        # run simulations and collect errors

        res_00 = stats.linregress( jnp.log(dxs), jnp.log(g00_errors) + 3*jnp.log(dxs) )
        slope_00 = jnp.abs( res_00.slope )

        res_11 = stats.linregress( jnp.log(dxs), jnp.log(g11_errors) + 3*jnp.log(dxs) )
        slope_11 = jnp.abs( res_11.slope )

        res_22 = stats.linregress( jnp.log(dxs), jnp.log(g22_errors) + 3*jnp.log(dxs) )
        slope_22 = jnp.abs( res_22.slope )

        res_cf = stats.linregress( jnp.log(dxs), jnp.log(cf_errors) + 3*jnp.log(dxs) )
        slope_cf = jnp.abs( res_cf.slope )

        self.assertGreater(slope_00, 4.0)
        self.assertGreater(slope_11, 4.0)
        self.assertGreater(slope_22, 4.0)
        self.assertGreater(slope_cf, 4.0)
        # check for at least 4th order convergence
