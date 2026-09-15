Architecture
============

The overall architecture of JAX BSSN keeps time derivative equations 
as separate pure functions, and the evolution package decides when to call them. 
This allows future experimentation with different timestepping algorithms and 
boundary treatments without changing the equations themselves.


Package map
-----------

``JAX_BSSN.bssn``
   the individual BSSN time-derivative equations, and raw constraint fields.

``JAX_BSSN.evolution``
   Finite-difference operators, physical boundary treatments, complete RHS
   assembly, algebraic projections, and unigrid RK4.

``JAX_BSSN.cartoon``
   Shared interpolation plus separate ``spherical_symmetry`` and
   ``axisymmetry`` cartoon algorithms, RK4, and constraint paths.

``JAX_BSSN.fmr``
   Geometry and transfers for a nested chain of vertex-centered refinement
   patches.

``JAX_BSSN.diagnostics``
   Constraint reductions and reporting, plotting, NumPy snapshots, and
   synchronous openPMD output. Evolution equations never import this package.

``demos``
   Physics demonstrations and their initial data including waves and single 
   puncture black holes.

``tests``
   Unittest suite that contains API tests and regression tests for the BSSN equations, cartoon symmetry, and FMR.


Cartesian RK4 algorithm
-----------------------

``JAX_BSSN.evolution.time_evolve.rk4_step`` performs one update in this exact
order:

1. Rescale the conformal metric to determinant one.
2. Project conformal extrinsic curvature to its trace-free part.
3. Assemble ``k1`` from all seven equation functions and replace the RHS on
   active Sommerfeld faces.
4. Form the ``k2`` midpoint state, then repeat steps 1--3.
5. Form the ``k3`` midpoint state, then repeat steps 1--3.
6. Form the ``k4`` endpoint state, then repeat steps 1--3.
7. Combine the classical RK4 stages and apply both algebraic projections once
   more.

Cartoon RK4 algorithms
----------------------

The spherical ``cartoon_rk4_step`` and axisymmetric
``axisymmetric_rk4_step`` own separate compact evolution paths. At every RK
stage they apply the algebraic projections, refresh four radial parity ghosts,
reconstruct temporary Cartesian support, evaluate the ordinary Cartesian RHS,
and project the independent reference data back to compact storage. Spherical
support has ``9 x 9`` transverse points; z-axis axisymmetric support has nine
y planes and retains the complete physical z domain.
