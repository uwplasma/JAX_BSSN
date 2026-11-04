Cartoon symmetry evolution
==========================

The Cartoon package is divided into two explicit implementations:
``JAX_BSSN.cartoon.spherical_symmetry`` and
``JAX_BSSN.cartoon.axisymmetry``.

Spherical symmetry
------------------

``JAX_BSSN.cartoon`` evolves a spherically symmetric state while reusing the
Cartesian BSSN equations and finite-difference operators. The compact state
stores only positive half-cell radii plus the reflected samples required by
the Cartesian stencils.

Compact radial storage
^^^^^^^^^^^^^^^^^^^^^^

For :math:`N_r` independent points, the positive coordinates are

.. math::

   r_i = (i + 1/2)\Delta x,
   \qquad i=0,\ldots,N_r-1.

The allocated field shape is ``(Nr + 4, 1, 1)``. The first four entries are
negative-:math:`x` parity ghosts and the remaining entries are the
authoritative positive-radius data. The half-cell placement avoids sampling
the coordinate origin or a puncture at :math:`r=0`.

Scalars are even under reflection. For Cartesian vectors the :math:`x`
component is odd and the transverse components are even. Rank-two tensor
parity is the outer product of those vector parities.

Spherical reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^

``reconstruct_cartoon_support`` builds the Cartesian ``(Nr + 4) x 9 x 9``
support grid from the positive radius in two explicit steps. It reads only
the degrees of freedom the symmetry leaves independent -- the :math:`rr` and
tangential tensor entries and the radial vector component -- so it needs no
symmetry projection of its own.

``radial_interpolate`` evaluates one radial profile at
:math:`r=\sqrt{x^2+y^2+z^2}` with six-point, degree-five nonperiodic Lagrange
interpolation. The positive half-axis is the only authoritative data: samples
through the origin come from its parity reflection, and samples past the outer
face come from three reconstruction-only Sommerfeld :math:`1/r` continuation
cells, so no target ever extrapolates.

``transformation_matrix(x, y, z)`` returns the rotation
:math:`\Lambda^{i}{}_{\bar{\jmath}} =
\partial x^{i}/\partial\bar{x}^{\bar{\jmath}}`
from the on-axis frame :math:`\bar{x}=(r,0,0)` to the support point, whose
columns are the radial, azimuthal and polar unit vectors. Scalars are the
interpolated profile itself, vectors are
:math:`\beta^{i}=\Lambda^{i}{}_{\bar{\jmath}}\bar{\beta}^{\bar{\jmath}}`,
and tensors are
:math:`\tilde{\gamma}_{ij}=\Lambda_{i\bar{a}}\Lambda_{j\bar{b}}
\bar{\gamma}_{\bar{a}\bar{b}}`. Because :math:`\Lambda` is a rotation it
serves both index positions. Only the positive centerline RHS is retained
after the Cartesian equations are evaluated; four reflected ghosts are then
regenerated from that result.

RK4 algorithm
^^^^^^^^^^^^^^^^^^

``cartoon_rk4_step`` is a separate entry point from Cartesian ``rk4_step``.
For each classical RK4 stage it performs this sequence:

1. Enforce spherical symmetry: discard the components the symmetry forbids --
   on the reference axis a vector is :math:`(v_r,0,0)` and a tensor is
   :math:`\operatorname{diag}(T_{rr},T_{tt},T_{tt})`, with the ``yy`` entry
   authoritative for the tangential pair -- and refresh the compact origin
   ghosts from reflection parity.
2. Enforce determinant-one and trace-free BSSN algebraic constraints. This
   order matters: discarding the forbidden components moves
   :math:`\det\tilde\gamma` and :math:`\operatorname{tr}\tilde{A}`, so step 1
   run second would undo step 2. The reverse is safe, because the algebraic
   projection maps :math:`\operatorname{diag}(r,t,t)` to
   :math:`\operatorname{diag}(r,t',t')` and scales by parity-even factors,
   preserving both the symmetric subspace and the ghost parity.
3. Reconstruct Cartesian support from the parity-filled signed axis.
4. Evaluate the ordinary Cartesian BSSN RHS and its outer Sommerfeld face.
5. Project the support RHS back onto compact radial storage.

The compact grid uses a non-Sommerfeld left ``x`` code because those entries
are internal parity ghosts, not a physical face. The outer ``x`` face is
Sommerfeld and all temporary ``y``/``z`` faces are periodic. Coordinate minima
must be ``(-3.5dx, -4dx, -4dx)``.

The current ``9 x 9`` support conservatively accommodates the fourth-order
first-, second-, mixed-, and upwind shift-advection stencils. In particular,
the D4 upwind operator reaches three points toward the transport direction;
the centerline has four reconstructed samples available on either side.
Cartoon requires ``mad_q = 1``, which takes the D4 fast path and does not
evaluate the wider D6 upwind stencil.

Z-axis axisymmetry
------------------

``JAX_BSSN.cartoon.axisymmetry`` evolves a complete :math:`x`--:math:`z`
half-plane while retaining the Cartesian BSSN RHS. The symmetry axis is
:math:`z`, the reference plane is :math:`y=0`, and :math:`x=\rho` contains
positive half-cell centres. A scalar has shape ``(Nrho + 4, 1, Nz)``; vectors
and tensors prepend their component dimensions. The full physical
:math:`z` interval is stored, so no equatorial symmetry is assumed.

The first four radial entries are parity ghosts generated by a rotation of
:math:`\pi` around :math:`z`. Vector parity is ``(-1, -1, +1)`` and tensor
parity is its outer product. Consequently azimuthal vector components and
the :math:`\rho\phi` and :math:`\phi z` tensor twists are retained.

For each of the nine temporary :math:`y/\Delta x=-4,\ldots,4` planes,
``reconstruct_axisymmetric_support`` computes

.. math::

   \rho=\sqrt{x^2+y^2}, \qquad \phi=\operatorname{atan2}(y,x),

interpolates each complete field at ``(rho, 0, z)`` with
``radial_interpolate``, and rotates full vectors and both tensor indices with
:math:`R_z(\phi)`, returned by ``transformation_matrix(x, y)``. Unlike the
spherical case no component is discarded: axisymmetry forbids nothing
pointwise on the reference plane, so ``fill_axisymmetric_ghosts`` is parity
reflection alone. Three reconstruction-only
outer radial samples continue deviations from flat-space values with
:math:`1/\sqrt{\rho^2+z^2}` falloff at fixed :math:`z`.

``axisymmetric_rk4_step`` applies algebraic projection, parity refresh,
support reconstruction, the unchanged ``compute_bssn_rhs``, and compact
projection at every stage.

The compact contract requires at least six positive radial cells, at least
nine z cells, ``mad_q=1``, ``x_min=-3.5*dx``, and ``y_min=-4*dx``. The radial
ghost face and reconstructed y faces are periodic stencil faces; outer rho is
Sommerfeld. Either periodic or Sommerfeld behavior may be selected at each
physical z face.

Uniform physical coordinates
----------------------------

All Cartoon solvers use uniform physical Cartesian spacing ``params.dx`` in
x, y, and z. Coordinate minima refer to physical grid samples. Positive radial
centers are ``(i+1/2)*dx``; native Yee vertices lie half a cell above centers.
``axis_coordinates`` and ``grid_coordinates`` return these sample positions.
Cartoon reconstruction rotates physical Cartesian components and interpolates
at the corresponding uniform radial index. Yee transfers retain equal local
averages on this grid.

The BSSN operators retain their fourth-order stencils, and the EM operators
retain their second-order stencils. Boundary closures, MAD weights, and KO
dissipation use the same physical spacing. The demos choose ``dt=CFL*dx``;
library steppers use the caller's ``params.dt``.

Cylindrical norms retain the radial volume weight ``abs(rho)`` at each field's
native site. Physical metric factors remain part of the Maxwell equations,
stress tensor, and metric contractions. These factors are independent of grid
spacing.

Output uses Cartesian openPMD geometry. Readers reject legacy mapped meshes
and checkpoints; old uniform checkpoints with absent or null mapping fields
remain readable. Historical output files are not converted.
