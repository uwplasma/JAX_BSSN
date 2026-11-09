Generic matter sources
======================

JAX BSSN evolves spacetime from externally supplied Eulerian matter projections::

    compute_bssn_rhs_with_matter(state, evolution_parameters, rho, S_i, S_ij)
    enforce_algebraic_constraints(state)

``rho`` is physical energy density, ``S_i`` covariant momentum density, and
``S_ij`` covariant spatial stress. Their array layouts are ``(nx,ny,nz)``,
``(3,nx,ny,nz)``, and ``(3,3,nx,ny,nz)``. Matter sources are applied before
Sommerfeld face equations. Matter-aware constraint diagnostics use the same
projections.

The caller owns matter evolution, units and source construction. JAX BSSN has
no PyPIC3D dependency. Maxwell, PIC, coupled time stepping and their integration
tests belong to PyPIC3D; see that project's ``docs/einstein_maxwell.rst``.
