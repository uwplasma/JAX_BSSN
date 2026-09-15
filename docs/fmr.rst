Fixed mesh refinement
=====================

JAX BSSN supports a single nested refinement level with a 2:1 refinement ratio 
and four ghost cells on the fine level. Fifth degree Lagrange interpolation is 
used to interpolate coarse data to fine ghost cells, and vise versa for restriction. 
Right now, the interpolation indicies use periodic wrapping, so the current FMR 
implementation is only valid with periodic boundary conditions.

FMR RK4
---------------------

Every level uses the finest-grid timestep. There is no subcycling. At each RK
stage, ghost filling proceeds from root to finest.

1. Project both level states to determinant-one and trace-free form.
2. Fill fine guard cells from the matching coarse stage.
3. Evaluate the coarse and fine BSSN right-hand sides.
4. Form matching coarse/fine ``k2``, ``k3``, and ``k4`` states and repeat
   steps 1--3 for each stage.
5. Combine both RK4 solutions.
6. Project the fine result.
7. Inject coincident active fine points into the coarse result and project the
   coarse state.
8. Refill final fine guard cells from the updated coarse state.
