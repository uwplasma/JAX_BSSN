Fixed mesh refinement
=====================

FMR uses the Cartesian tile evolution machinery for a fixed tree of nested
patches. Each patch has its own resolution and tile layout. Same-resolution
tile neighbors exchange their owned samples; a child's outer ghost cells are
interpolated from its parent at the same RK stage. Refinement interfaces use
interior derivative stencils, not physical Sommerfeld closures.

The numerical method remains vertex centred with a 2:1 refinement ratio,
degree-five Lagrange prolongation, and injection of coincident fine vertices
into the parent. All patches use the finest-grid timestep, with no subcycling.
The root may have periodic or Sommerfeld faces. Children must lie strictly
inside their parent, with enough parent samples for the complete interpolation
stencil; stencils may not cross a Sommerfeld root face. Siblings must not
overlap. Moving patches and AMR are not supported.

Patch geometry and initialization
---------------------------------

``FMRPatchSpec(coarse_lo, coarse_hi, parent=...)`` uses inclusive indices into
the parent's active data. A child has ``2*(hi-lo)+1`` active vertices along
each axis. States are kept in a flat tuple: the root is index zero and the
child described by ``hierarchy.patches[i]`` is index ``i+1``. Its parent must
precede it. Omitting ``parent`` retains the historical parent-to-child chain.

.. code-block:: python

    import jax
    from JAX_BSSN.bssn import BSSNParameters
    from JAX_BSSN.fmr import (
        FMRHierarchySpec, FMRPatchSpec, initialize_tiled_fmr,
        tiled_fmr_rk4_step,
    )

    hierarchy = FMRHierarchySpec((
        FMRPatchSpec((3, 4, 4), (9, 11, 11), parent=0),
        FMRPatchSpec((14, 4, 4), (20, 11, 11), parent=0),
        FMRPatchSpec((3, 3, 3), (9, 11, 11), parent=1),
    ))
    params = BSSNParameters(dx=1/24, dt=1/768)
    states, parameters, layouts = initialize_tiled_fmr(
        None, params, (24, 16, 16), hierarchy,
    )
    advance = jax.jit(lambda values: tiled_fmr_rk4_step(
        values, parameters, hierarchy, layouts=layouts,
    ))
    states = advance(states)

``None`` initializes flat spacetime. An initial-data callback receives
``(coordinates, local_params)`` on the owned samples, following
``initialize_tiled_bssn``. A tuple of initial inputs supplies separate native
arrays, field mappings, or callbacks for the patches. Patch spacings and active
origins are derived from the root geometry. Pass ``mesh=...`` to use an
existing field mesh; the default uses one device.
The returned layouts describe each patch's mesh, owned tile widths, allocated
guard depth, and upper endpoint ownership; retain them with the tiled states.
The caller selects the common ``dt`` to resolve the finest spacing; initialization
does not silently change it.

All patches reuse the same device mesh and ordering. Along each axis an active
extent must equal ``tile_count * tile_width`` or that quantity plus one upper
vertex. The initializer rejects incompatible extents rather than padding the
physical domain. Layouts with an extra endpoint allocate five guards, since
the endpoint occupies one upper slot and MAD needs four exterior samples.
Single-device patches may be smaller than their guard depth. A singleton
refined extent retains its parent's resolved directions through
``BSSNParameters.reduced_axes``.

RK4 and numerical contracts
----------------------------

At each initial/intermediate stage:

1. Project owned geometry to determinant-one and trace-free form.
2. Refresh tile neighbors and root physical exteriors.
3. Fill child ghosts from the matching parent stage, proceeding outward from
   the root so deeper patches see valid parent ghosts.
4. Evaluate the unchanged local BSSN equations and construct the next stage.

After the RK4 combination, project the result, inject each child before its
parent, project updated parents, and refresh final ghosts.
Restriction remains point injection; it is not a finite-volume conservative
average. ``use_mad=True`` retains the mesh-adapted derivative coefficient
``(dx_finest/dx_patch)**4``. Sommerfeld axes retain the existing D4 physical
boundary policy.

Legacy serial calls
--------------------

``fmr_rk4_step(states, parameters, hierarchy)`` continues to accept a bare
root array and child arrays with symmetric guards. Existing
``fine_coordinates``, ``fine_active_view``, ``prolongate_to_fine``, and
``restrict_to_coarse`` callers retain their geometry conventions. The adapter
returns the same serial array shapes. Serial child parameter origins still
refer to the padded allocation; tiled parameter origins refer to active data.
The small
``demos/gravitational_waves/linear_wave.py`` API and output naming are retained.

Output and diagnostics
-----------------------

``FMRPatchSeriesWriter.write`` accepts one field map per patch in hierarchy
order. Existing serial calls retain ``ghost_cells=(0, 4, ...)``. For tiled
fields pass ``layouts=layouts`` and each layout's allocated guard depth;
``origins`` are active patch origins. Owned chunks are copied directly to
host patch arrays for output, including each final upper vertex once.
Already-owned field maps can instead be passed with zero ghost cells on
every patch and no layouts; they retain the same shape and topology checks.
This writer currently supports tiled output from one process only.

Filenames retain ``level_DD_patch_PPP_%08T.h5`` with relative ``.opmd`` aliases,
per-patch ``.pmd`` helpers, and a combined ``.visit`` manifest. The level is
the refinement depth and patch numbers count patches at that depth in
hierarchy order. Chain output therefore retains all existing filenames and
metadata. Branching output additionally records ``fmrId``; ``fmrParent`` is
the parent's flat hierarchy index, with ``-1`` for the root.

Composite diagnostics exclude every direct child's covered parent region,
so each physical degree of freedom contributes once. Use
``fmr_constraint_norms(states, parameters, hierarchy, layouts=layouts)`` for
point-count L2 and Linf norms; it refreshes refinement ghosts before taking
derivatives, then reduces scalar statistics across devices. Run
``python -m demos.gravitational_waves.fmr_wave.branching_fmr`` for a periodic
linear wave with two siblings and a nested child, written at their native
resolutions. From ``demos/gravitational_waves/fmr_wave``, run
``python branching_fmr.py`` directly. Edit the settings at the top of the
script to choose the geometry, hierarchy, wave, timestep, and output cadence.
Shared wave diagnostics live in ``JAX_BSSN.diagnostics.waves``; the older
linear-wave demo retains its existing diagnostic imports and norm definitions.
The demonstration reports wave error and constraint norms;
a short smoke run is not a long-time stability qualification.

Whole-domain wave movie
-----------------------

From ``demos/gravitational_waves/fmr_wave``, save a trajectory on two CPU tiles per
patch, then render its native patch output:

.. code-block:: bash

    python branching_fmr.py
    python make_branching_fmr_movie.py

There are no command-line arguments. Defaults are 480 steps, output every
four steps, timestep ``1/768``, and wave amplitude ``1e-8``. The run script
configures two logical CPU devices before constructing the shared mesh. The
121 snapshots cover ``0 <= t <= 0.625``: the initial trough crosses the
nested child, leaves the left sibling, crosses the coarse gap, and leaves
the right sibling. Set ``PLATFORM`` and ``TILE_COUNTS`` at the top of the run
script to select a different device setup. The output backend remains
single-process. New data go into this folder's ``output/branching_fmr``
series; existing archives in the parent demo's output folder are unchanged.

The JSON sidecar records actual patch origins, shapes, ancestry, and tile
ownership from the initializer, together with diagnostics at each saved
time. The read-only renderer writes an MP4 beside it. Edit ``METADATA_PATH``,
``OUTPUT_PATH``, ``FPS``, and ``FRAME_STRIDE`` at its top to select an older
archive or change playback; figure size, resolution, color map, and camera
settings are there too. A complete central slice shows the finest available samples,
and a 3D view shows the whole domain with two numerical slices. Bold black
lines mark patch boundaries; thin black lines mark internal tile ownership
boundaries, halfway between adjacent owned vertices. Ghost cells are omitted.
The color scale is fixed across all frames and displays ``h_plus/amplitude``.
Only the slice-normal coordinate is linearly interpolated to place each
patch on a common physical plane; transverse samples retain their native
resolution, and every movie frame comes from an actual saved timestep.

The 480-step CPU run on 2026-11-12 produced 121 finite wave snapshots. A
sine/cosine fit to the saved root fields measured propagation speed
``1.00000208`` (expected ``1``), with a final composite wave RMS error divided
by amplitude of ``9.21623e-6``. Final point-count Hamiltonian and momentum L2
norms were ``1.85093e-10`` and ``6.59251e-11``. These measurements describe
this short propagation example; they do not establish long-time stability.

Migration validation
---------------------

CPU float64 validation on 2026-11-12 used JAX 0.9.2. The existing FMR,
MAD, boundary, evolution, and output regressions passed, including the
``8^3 -> 7^3 -> 5^3`` production-output case. The existing linear-wave
spatial test retained its acceptance thresholds and measured order 4.4874.
New checks cover branching ancestry, composite norms, quintic transfers for
scalar/vector/tensor components, inclusive endpoint injection, short patches,
resolved singleton derivatives, mixed root faces, and restart topology.
Two- and three-level comparisons against a frozen serial implementation
passed, including every RK stage, active RHS, and final restricted result
on an eight-device mesh with nonzero shift and MAD. Independent branch
ordering and the MAD-disabled derivative path also passed their checks.
Two independent CPU processes passed interpolation, injection, and full RK4
evolution while retaining distributed arrays. The partitioned eight-device
executable contains collective permutations and no full-field all-gather.

The interface-crossing gauge wave has amplitude ``0.001`` and final time
``0.25``. Spatial refinement keeps the physical root domain
``[0,1) x [0,0.5) x [0,0.5)`` and child bounds
``[0.25,0.75] x [0.125,0.375] x [0.125,0.375]`` fixed. All three grids use
256 common steps, periodic root faces, and MAD. Errors below are composite,
volume-weighted lapse RMS divided by the wave amplitude; this validation
measure does not change the point-count production constraint norms.

.. list-table:: Spatial measurements
   :header-rows: 1

   * - Root shape
     - Normalized lapse error
     - Observed order from previous grid
   * - ``16 x 8 x 8``
     - ``1.116773e-5``
     - --
   * - ``24 x 12 x 12``
     - ``1.140085e-6``
     - 5.628
   * - ``32 x 16 x 16``
     - ``2.287132e-7``
     - 5.584

These measured rates exceed the fourth-order acceptance threshold for this
wave; they do not establish a higher formal order for general FMR data.

Fixed-grid temporal refinement uses a ``24 x 8 x 8`` root, child coarse
bounds ``(6,2,2)`` through ``(18,5,5)``, the same amplitude/final time, and
20, 40, and 80 steps. Successive normalized lapse differences were
``2.8777354e-7`` and ``4.1344261e-8``, giving observed order **2.7992**.
The frozen pre-migration implementation gives the same order; maximum
tiled/reference lapse discrepancies at the three timesteps are
``8.88e-16``, ``1.11e-15``, and ``1.44e-15``. The fourth-order temporal
hypothesis therefore fails for this tested hierarchy and timestep range in
both implementations. The regression checks agreement with the independent
serial reference and decreasing temporal differences, without claiming an
asymptotic temporal order or changing the inherited synchronization method.

The focused numerical tests are ``tests/FMR/test_tiled_fmr.py`` and
``tests/FMR/test_tiled_fmr_convergence.py``; the latter retains these spatial
and matched temporal measurements. Set ``JAX_PLATFORMS=cpu``,
``JAX_ENABLE_X64=true``, and ``JAX_NUM_CPU_DEVICES=8`` to exercise the
multi-device cases. ``tests/test_fmr_processes.py`` launches two independent
CPU runtimes and requires local loopback communication. These checks do not
qualify GPU performance or long-time nonlinear stability.
