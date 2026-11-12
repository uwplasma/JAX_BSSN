"""Fixed Cartesian refinement using the shared BSSN tile implementation."""

from .refinement import (
    FMRPatchSpec, FMRHierarchySpec, hierarchy_parents, hierarchy_depths,
    fine_active_shape, fine_active_slice, fine_active_view, fine_coordinates,
    prolongate_to_fine, fill_fine_ghosts, restrict_to_coarse, fmr_rk4_step,
    initialize_tiled_fmr, prepare_fmr_states, tiled_fmr_stage,
    tiled_fmr_rk4_step, fmr_constraint_norms,
    fill_tiled_fine_ghosts, restrict_tiled_to_coarse,
)
