Constraints and diagnostics
===========================

Diagnostics include hamiltonian and momentum constraint monitoring,
as well as plotting and openPMD output.

Raw constraint fields
---------------------

``JAX_BSSN.bssn.constraints`` produces arrays:

.. list-table:: Constraint fields
   :header-rows: 1
   :widths: 24 22 54

   * - Field
     - Shape
     - Meaning
   * - ``hamiltonian``
     - ``(Nx, Ny, Nz)``
     - Vacuum Hamiltonian residual, using the denominator-free trace of
       :math:`W^2R_{ij}`.
   * - ``momentum``
     - ``(3, Nx, Ny, Nz)``
     - Covariant momentum residual used by both diagnostics and evolution.
   * - ``det_gamma``
     - ``(Nx, Ny, Nz)``
     - :math:`\det\tilde{\gamma}-1`.
   * - ``trace_A``
     - ``(Nx, Ny, Nz)``
     - :math:`\tilde{\gamma}^{ij}\tilde{A}_{ij}`.



Plotting and NumPy snapshots
----------------------------

``diagnostics.plotting`` provides central-slice plots, constraint-history
plots, ``.npz`` state snapshots, and a callback-based text writer. Plotting and
filesystem output are side effects and should be called outside compiled
timesteps.

openPMD output
--------------

``OpenPMDWriter`` writes synchronous Cartesian mesh iterations. Scalar fields
are three-dimensional arrays. Vector fields are three-component tuples or
lists whose components share one three-dimensional shape.