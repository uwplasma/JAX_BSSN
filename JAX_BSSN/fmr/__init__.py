"""Fixed Cartesian refinement using the shared BSSN tile implementation."""

from .refinement import (
    FMRPatchSpec, FMRHierarchySpec, hierarchy_parents, hierarchy_depths,
    fine_active_shape, fine_active_slice, fine_active_view,
)
from .tiled import (
    initialize_tiled_fmr, patch_layout, validate_hierarchy, prepare_fmr_states,
    fmr_stage, fmr_rk4_step, fmr_constraint_norms,
)
from .transfer import fill_tiled_fine_ghosts, restrict_tiled_to_coarse
