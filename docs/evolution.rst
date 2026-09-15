Evolution and boundaries
========================

The evolution package owns the discretization and the placement of boundary
and algebraic constraint operations for the BSSN equations.


Derivatives with mesh adapted differencing (MAD)
------------------------------------------------

With ``mad_q = 1`` the result is exactly the historical centered fourth-order
operator

.. math::
   
   D_4 f_i = \frac{2}{3h}(f_{i+1}-f_{i-1})
             -\frac{1}{12h}(f_{i+2}-f_{i-2}).

For other values, mesh-adapted differencing blends that operator with the
centered sixth-order-accurate first derivative:

.. math::

   D_{\mathrm{MAD}} = qD_4 + (1-q)D_6.

This is done to eliminate abrupt jumps in the leading order truncation error 
at mesh-refinement interfaces. The ``diff2_field`` applies the same MAD weight 
to dedicated centered second derivative stencils.  With ``mad_q = 1`` it uses 
the five-point fourth-order operator; ``mad_q = 0`` selects the seven-point 
sixth-order operator.


.. note::
   Baker, J. G., & Van Meter, J. R. (2005). 
   Reducing reflections from mesh refinement 
   interfaces in numerical relativity. 
   Physical Review D—Particles, Fields, 
   Gravitation, and Cosmology, 72(10), 104010.



Upwind shift advection
----------------------

``diff1_upwind_field`` is the first-derivative operator used only for shift
advection. Its ``rhs_coefficient`` argument is the coefficient :math:`c` in

.. math::

   \partial_t u = \cdots + c\,\partial_i u.

This convention is important: the corresponding transport velocity is
:math:`-c`. A positive RHS coefficient therefore selects the forward-biased
fourth-order stencil

.. math::

   D^{+}_4 f_i = \frac{-3f_{i-1}-10f_i+18f_{i+1}
                     -6f_{i+2}+f_{i+3}}{12h},

while a negative coefficient selects

.. math::

   D^{-}_4 f_i = \frac{-f_{i-3}+6f_{i-2}-18f_{i-1}
                     +10f_i+3f_{i+1}}{12h}.

As with centered first derivatives, MAD blends fourth- and sixth-order
operators:

.. math::

   D_{\mathrm{upwind,MAD}} = qD_{4,\mathrm{upwind}}
                              +(1-q)D_{6,\mathrm{upwind}}.


Kreiss--Oliger dissipation
--------------------------

``diff6_field`` computes the centered seven-point sixth derivative. Each
evolution equation adds the existing dissipation term

.. math::

   \frac{\nu}{64}h^5
   \left(\partial_x^6+
         \partial_y^6+
         \partial_z^6\right)u.

Periodic axes use wrapped stencils. Sommerfeld faces replace the outer closure
points with the implemented lopsided coefficients.

Boundary codes
--------------

Each low/high face has an integer code in ``BSSNParameters``:

.. list-table:: Boundary families
   :header-rows: 1
   :widths: 18 25 57

   * - Code
     - Name
     - Numerical role
   * - ``0``
     - Periodic
     - Centered derivative stencils wrap with ``jnp.roll``.
   * - ``1``
     - Sommerfeld
     - Use nonperiodic derivative closures and replace the assembled RHS on
       the selected outer plane.

Sommerfeld uses the coordinate origin supplied through ``x_min``, ``y_min``,
and ``z_min``. It evaluates

.. math::

   \partial_t u = -\left(\partial_r u + \frac{u-u_0}{r}\right)
