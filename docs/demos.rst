Demos
=====

The supported physical examples are gauge wave, linear wave, Cartesian single
puncture, axisymmetric Cartoon puncture, and spherical Cartoon puncture. Each demo contains its own
coordinates, initial data, parameters, and run loop.

Install demo support first:

.. code-block:: bash

   python -m pip install -e ".[demos]"

Gauge wave
----------

The gauge-wave demo evolves the analytic positive-x harmonic gauge wave on a
periodic uniform grid with harmonic slicing and fixed zero shift. It reports
RMS errors against the analytic solution at the final time.

.. code-block:: bash

   python demos/gauge_wave/gauge_wave.py

The default configuration uses ``grid_size=60``, amplitude ``0.1``, wavelength
``1``, CFL ``1``, and one wavelength of evolution.

Linear wave with FMR
--------------------

The linear-wave demo evolves a plus-polarized wave through a configurable
centered chain of 2:1 refinement patches (three total levels by default). It
uses synchronized RK4 and reports composite diagnostics using the finest
available value at each physical location.

.. code-block:: bash

   python demos/linear_wave/linear_wave.py

The default run writes the VisIt multiblock collection
``demos/linear_wave/output/linear_wave_fmr.visit``. Open that file in VisIt to
load the root and fine patches together. The collection spatially aligns the
overlapping blocks; it does not remove root cells covered by the fine patch or
encode native AMR nesting.

Each patch is also an independent file-based openPMD series, with per-patch
``.pmd`` helpers for ParaView. ``grid_size`` must be divisible by four so 
the centered inclusive patch bounds land on coarse vertices. The demo rejects 
a final normalized coarse or fine RMS wave error at or above five percent.

Single puncture
---------------

The single-puncture demo constructs time-symmetric Schwarzschild puncture data
with :math:`W=\psi^{-2}` and evolves it with 1+log lapse and an evolved
Gamma-driver shift. Its grid avoids sampling :math:`R=0` directly.

Run from the demo directory so its snapshots and the movie helper agree on the
output location:

.. code-block:: bash

   cd demos/single_puncture
   python single_puncture.py

The default run writes per-field ``.npy`` snapshots and
``constraint_l2.txt`` under ``output/``, plus a final field plot. Grid size,
domain width, CFL, final time, and snapshot cadence are explicit near the top
of ``main``.

After a run, optional MP4 rendering requires an ``ffmpeg`` executable visible
to Matplotlib:

.. code-block:: bash

   python make_movies.py

The renderer produces lapse, shift, and conformal-factor movies from the
synchronized snapshots.

Spherical Cartoon puncture
--------------------------

The Cartoon puncture demo evolves the same time-symmetric Schwarzschild data
on ``Nr`` positive half-cell radii with four parity ghosts. Each RK stage
reconstructs a temporary Cartesian support grid, so the installed Cartesian
BSSN equations remain the single equation implementation.

.. code-block:: bash

   python demos/cartoon_puncture/cartoon_puncture.py

The production defaults reproduce the comparison configuration:
``Nr=2000``, ``rmax=100M``, CFL ``0.2``, final time ``100M``, and 500 output
intervals. ``run_cartoon_puncture`` accepts smaller values for smoke runs.

Output is written under ``output/`` as ``cartoon_puncture.h5`` and
``constraint_l2.txt``. The openPMD series contains the complete reflected
``2*Nr x 1 x 1`` axis, including all BSSN tensor components and Hamiltonian and
momentum constraints. Constraint norms use only independent positive radii and
exclude the four outer stencil-affected samples.

Axisymmetric Cartoon puncture
-----------------------------

The parallel z-axis axisymmetric executable stores a positive-rho by full-z
plane, reconstructs nine Cartesian y planes at every RK stage, and writes an
expanded signed x-z plane to a distinct openPMD file:

.. code-block:: bash

   python demos/axisymmetric_cartoon_puncture/axisymmetric_cartoon_puncture.py

Its defaults are ``rho_max=32M``, ``z in [-32M,32M]``, ``Nrho=256``,
``Nz=512``, CFL ``0.2``, and final time ``10M``. Constraints are evaluated only at
the initial, output, and final iterations.

Axisymmetric boosted Bowen--York puncture
------------------------------------------

The boosted Bowen--York demo first solves the three-dimensional Cartesian
Hamiltonian constraint for a single puncture with linear momentum along the
z axis. Its exact ``y=0`` plane is converted to the repository's W-form BSSN
variables and evolved with the existing axisymmetric Cartoon RK4 path.

.. code-block:: bash

   cd demos/axisymmetric_bowen_york
   python axisymmetric_bowen_york.py
   python make_movies.py

The defaults use ``M=1``, ``Pz=0.5M``, ``rho_max=12M``,
``z in [-12M,12M]``, ``Nrho=48``, ``Nz=96``, CFL ``0.2``, and final time
``10M``. The elliptic grid is ``96 x 97 x 96``: its central y sample is
exactly zero, while the even x and z dimensions keep the puncture between
grid points. The run writes complete BSSN fields and constraints to
``output/axisymmetric_bowen_york.h5``. The renderer creates grouped H.264
movies and ``movies/puncture_trajectory.txt`` using the minimum of W near the
symmetry axis.

The elliptic solve is second-order and fixes the regular correction ``u`` to
zero on the finite outer grid layers. This approximates asymptotic flatness
and should be checked by enlarging the domain. Because the initial solve is
fully three-dimensional, its memory use is much larger than the compact
Cartoon evolution that follows.
