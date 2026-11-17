Constraints and diagnostics
===========================

Diagnostics include hamiltonian and momentum constraint monitoring,
as well as openPMD output.

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



openPMD output
--------------

``OpenPMDWriter`` writes synchronous Cartesian mesh iterations. Scalar fields
are three-dimensional arrays. Vector fields are three-component tuples or
lists whose components share one three-dimensional shape.
Apparent horizons
-----------------

``JAX_BSSN.utilities.find_apparent_horizon`` solves
:math:`D_i s^i + K_{ij}s^i s^j - K = 0` on a closed star-shaped surface
:math:`r=h(\theta,\phi)`. Supply physical covariant ``gamma`` and ``Kij``:

* ``(r,)``: positive radial samples on the +x line, tensors ``(3,3,nr)``.
* ``(x,z)``: signed Cartesian y=0 plane with symmetry about z, tensors ``(3,3,nx,nz)``.
* ``(x,y,z)``: full Cartesian volume, tensors ``(3,3,nx,ny,nz)``.

Metric interpolation lives in ``utilities/hermite_interpolation.py``. All
horizon geometry reconstruction, spatial differentiation, and Newton logic
live in ``utilities/apparent_horizon.py``.

Every layout uses the same full two-dimensional angular grid. Only the tensor
reconstruction depends on symmetry. Reduced data remain symmetric about their
coordinate origin/axis; changing the horizon center does not translate them.
Axes must be uniform. The default interpolation margin is four cells.

Both the center and initial guess are required::

    jax.config.update("jax_enable_x64", True)
    horizon = find_apparent_horizon(
        (x, z), gamma, Kij, center=(0., 0., 0.), initial_radius=1.0)

``initial_radius`` may instead be a positive ``(n_theta,n_phi)`` array, whose
shape overrides the starting angular resolution. The center stays fixed.

The finder uses cubic Hermite interpolation. JAX differentiates this interpolant, 
including the rotation of reduced tensors. Angular derivatives use fourth-order differences,
periodic azimuths, and polar reflection with a half-turn azimuthal shift; no
node lies at a pole. The complete dense Newton Jacobian is assembled by AD in
batches of 16 columns and solved directly.