The BSSN system
===============

JAX BSSN evolves the vacuum Cartesian BSSN system in the ``W`` conformal-factor
formulation. Every array ends with the three spatial axes ``(Nx, Ny, Nz)``.
Tensor and vector component axes precede those spatial axes.

Evolved variables
-----------------

The field order is a numerical contract because RK4 and FMR combine states by
tuple position.

.. list-table:: ``BSSNVariables`` layout
   :header-rows: 1
   :widths: 22 20 26 32

   * - Field
     - Symbol
     - Shape
     - Meaning
   * - ``conformal_metric``
     - :math:`\tilde{\gamma}_{ij}`
     - ``(3, 3, Nx, Ny, Nz)``
     - Symmetric conformal spatial metric, projected to determinant one.
   * - ``conformal_factor``
     - :math:`W`
     - ``(Nx, Ny, Nz)``
     - Conformal factor with :math:`\gamma_{ij}=W^{-2}\tilde{\gamma}_{ij}`.
   * - ``traceless_K``
     - :math:`\tilde{A}_{ij}`
     - ``(3, 3, Nx, Ny, Nz)``
     - Conformal trace-free extrinsic curvature.
   * - ``trace_K``
     - :math:`K`
     - ``(Nx, Ny, Nz)``
     - Trace of the physical extrinsic curvature.
   * - ``conformal_connection``
     - :math:`\tilde{\Gamma}^{i}`
     - ``(3, Nx, Ny, Nz)``
     - Evolved conformal connection functions.
   * - ``lapse``
     - :math:`\alpha`
     - ``(Nx, Ny, Nz)``
     - Lapse function.
   * - ``shift``
     - :math:`\beta^{i}`
     - ``(3, Nx, Ny, Nz)``
     - Contravariant shift vector.


Shared geometry
---------------

``bssn.geometry`` reconstructs the physical metric, packs and unpacks
symmetric tensors, and computes the curvature and lapse-Hessian sources shared
by the extrinsic curvature equations.

The physical metric uses

.. math::

   \gamma_{ij} = W^{-2}\tilde{\gamma}_{ij}.

Inverse powers used during physical-metric reconstruction and conformal-
connection evolution apply ``W_FLOOR_VALUE = 1e-12``. This protects only the
inverse operation: the evolved ``W`` array is not clamped or overwritten.

The traceless-curvature RHS consumes scaled quantities :math:`W^2 D_iD_j\alpha` 
and :math:`W^2 R_{ij}` to reduce unnecessary divisons by the conformal factor. 
These objects are computed in ``bssn.geometry`` and passed to the extrinsic 
curvature equations.

Gauge choices
-------------

``BSSNParameters.gauge`` selects the lapse source inside if logic:

* ``0``: harmonic slicing,
  :math:`\partial_t\alpha-\beta^i\partial_i\alpha=-\alpha^2K`.
* ``1``: 1+log slicing,
  :math:`\partial_t\alpha-\beta^i\partial_i\alpha=-2\alpha K`.

The shift uses a single-variable Gamma driver with advection and linear
damping. ``zero_shift = 1`` suppresses the shift RHS at every RK stage;
otherwise the configured driver is evolved.

Shift advection
---------------

All seven evolved fields use the shared ``compute_shift_advection`` path for
terms of the form

.. math::

   +\beta^i\partial_i u

on the right-hand side: :math:`\tilde{\gamma}_{ij}`, :math:`W`,
:math:`\tilde{A}_{ij}`, :math:`K`, :math:`\tilde{\Gamma}^i`, :math:`\alpha`,
and :math:`\beta^i` itself. The positive sign here is the RHS-coefficient
convention used by ``diff1_upwind_field``.

Algebraic constraints
---------------------

The timestep enforces

.. math::

   \det\tilde{\gamma}=1,
   \qquad
   \tilde{\gamma}^{ij}\tilde{A}_{ij}=0

before each RHS evaluation and after the final RK4 combination.