Demos
=====

The supported physical examples are gauge wave, linear wave, spherical Cartoon
puncture, and axisymmetric Cartoon collision. Each demo owns its coordinates,
initial data, parameters, and run loop.

Install demo support first:

.. code-block:: bash

   python -m pip install -e ".[demos]"

Gauge wave
----------

The gauge-wave demo evolves the analytic positive-x harmonic gauge wave on a
periodic uniform grid with harmonic slicing and fixed zero shift. It reports
RMS errors against the analytic solution at the final time.

.. code-block:: bash

   python demos/gravitational_waves/gauge_wave.py

The default configuration uses ``grid_size=60``, amplitude ``0.1``, wavelength
``1``, CFL ``1``, and one wavelength of evolution.

Linear wave with FMR
--------------------

The linear-wave demo evolves a plus-polarized wave through a configurable
centered chain of 2:1 refinement patches (three total levels by default). It
uses synchronized RK4 and reports composite diagnostics using the finest
available value at each physical location.

.. code-block:: bash

   python demos/gravitational_waves/linear_wave.py

The default run writes the VisIt multiblock collection
``demos/gravitational_waves/output/linear_wave_fmr.visit``. Open that file in VisIt to
load the root and fine patches together. The collection spatially aligns the
overlapping blocks; it does not remove root cells covered by the fine patch or
encode native AMR nesting.

Each patch is also an independent file-based openPMD series, with per-patch
``.pmd`` helpers for ParaView. ``grid_size`` must be divisible by four so 
the centered inclusive patch bounds land on coarse vertices. The demo rejects 
a final normalized coarse or fine RMS wave error at or above five percent.



Spherical Cartoon puncture
--------------------------

The Cartoon puncture demo evolves time-symmetric Schwarzschild data
on ``Nr`` positive half-cell radii with four parity ghosts. Each RK stage
reconstructs a temporary Cartesian support grid, so the installed Cartesian
BSSN equations remain the single equation implementation.

.. code-block:: bash

   python demos/spherically_symmetric_cartoon_puncture/cartoon_puncture.py

The production defaults reproduce the comparison configuration:
``Nr=2000``, ``rmax=100M``, CFL ``0.2``, final time ``100M``, and 500 output
intervals. ``run_cartoon_puncture`` accepts smaller values for smoke runs.

Output is written under ``output/`` as ``cartoon_puncture.h5`` and
``constraint_l2.txt``. The openPMD series contains the complete reflected
``2*Nr x 1 x 1`` axis, including all BSSN tensor components and Hamiltonian and
momentum constraints. Constraint norms use only independent positive radii and
exclude the four outer stencil-affected samples.

Axisymmetric Cartoon Bowen–York collision
----------------------------------------

The collision demo initializes two nonspinning punctures on the z axis with
opposite inward momenta. It solves the Hamiltonian constraint directly on the
axisymmetric meridional grid and evolves the resulting x–z half-plane with
``JAX_BSSN.cartoon.axisymmetry.axisymmetric_rk4_step``.

.. code-block:: bash

   python demos/axisymmetric_blackhole_collision/axisymmetric_blackhole_collision.py
   python demos/axisymmetric_blackhole_collision/make_movies.py


The initial solve runs directly on the ``Nrho x 2*Nrho`` cell-centered
meridional grid. Newton–Raphson uses the centered finite-difference
axisymmetric Laplacian, with even reflection at the axis and zero regular
correction on the outer rho and z cells. The singular Brill–Lindquist factor
and Bowen–York curvature are evaluated analytically on the same plane.
Initialization sets ``W=psi**-2``, unit conformal metric, BSSN traceless
curvature ``psi**-6 * A_BY``, lapse ``W``, and zero trace curvature,
connection functions, and shift.
