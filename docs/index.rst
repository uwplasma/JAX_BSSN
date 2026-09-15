JAX BSSN
========

.. container:: hero

   JAX BSSN is a autodifferentiable and jit-compiled implementation of the
   BSSN numerical relativity evolution system in JAX. JAX BSSN is a research 
   level codebase designed for rapid prototyping and experimentation with gradient-
   based methods in numerical relativity. JAX BSSN features a vacuum BSSN with fourth 
   order finite differencing, fourth order Runge-Kutta time evolution, Kreiss-Oliger 
   dissipation, periodic and Sommerfeld boundary conditions, and cartoon based methods 
   for spherical and axisymmetric spacetimes.

   .. container:: hero-actions

      :doc:`Check out some cool demos <demos>`


Quick navigation
----------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Install JAX BSSN
      :link: installation
      :link-type: doc

      Set up the codebase.

   .. grid-item-card:: Follow the architecture
      :link: architecture
      :link-type: doc

      View the overall structure and design.

   .. grid-item-card:: Inspect the BSSN system
      :link: bssn
      :link-type: doc

      Review the BSSN equations and their implementation.

   .. grid-item-card:: Trace an RK4 step
      :link: evolution
      :link-type: doc

      Locate derivatives, dissipation, boundaries, and projections.

   .. grid-item-card:: Understand FMR
      :link: fmr
      :link-type: doc

      Examine our implementation of fixed mesh refinement.

   .. grid-item-card:: Use Cartoon symmetry
      :link: cartoon
      :link-type: doc

      Review the implementation of cartoon-based symmetry.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   architecture
   bssn
   evolution
   cartoon
   fmr
   diagnostics
   demos

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
