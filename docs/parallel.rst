Parallel Cartesian evolution
============================

``JAX_BSSN.parallel`` evolves collocated Cartesian BSSN fields on the same
device mesh used by PyPIC3D. Each device owns one logical tile. Neighbor slabs
are exchanged with ``jax.lax.ppermute`` in x, y, z order, carrying edge and corner
values through the successive exchanges. Evolution does not assemble a global
field on one device or process.

The current implementation supports a uniform Cartesian grid with periodic
and Sommerfeld faces. It preserves the existing BSSN equations, algebraic
projection at RK stages, and RK4 update. Fixed mesh refinement uses this tile
machinery through ``JAX_BSSN.fmr``, with a separate layout and parent-supplied
ghosts for each patch; see :doc:`fmr`. Cartoon symmetry retains its existing
evolution path.

Layout shared with PyPIC3D
-------------------------

The mesh axes are exactly ``("tile_x", "tile_y", "tile_z")``. With mesh shape
``(tx, ty, tz)``, owned tile widths ``(nx, ny, nz)``, and guard depth ``g``, a
scalar has shape::

    (tx, ty, tz, nx + 2*g, ny + 2*g, nz + 2*g)

A vector prepends ``(3,)`` and a tensor prepends ``(3, 3)``. Tensor components
are replicated; the three tile axes are sharded. When coupling to PyPIC3D,
reuse its actual ``static_parameters.field_mesh`` object and tile widths.
Matching only the device count does not establish the same device ordering.

Use at least four BSSN guards for the standalone evolution interface. Default D4
advection and KO require three neighbors; general MAD upwind derivatives
require four. Mixed derivatives differentiate across distinct axes, so complete
corner halos suffice without an additional exchange of derived tensors.
Resolved tile widths must be at least the guard depth. A globally reduced axis
may have width one on a single tile and must use periodic boundary codes.
Sommerfeld/KO closures require eight
accessible inward physical samples, including an inward neighbor halo when
needed.

``upper_endpoints`` is a three-boolean tuple describing ownership, not a BSSN
boundary condition. A true entry retains PyPIC3D's extra upper collocated C
endpoint on the final tile. The global C shape along that axis is
``tile_width * tile_count + 1``. This endpoint occupies the first nominal upper
guard slot, keeping every tile allocation the same size. The final tile owns
one more sample. For a coupled conducting axis, use the corresponding true
flag; BSSN still applies its own Sommerfeld equations, without electromagnetic
reflection parity.

An extra endpoint on a periodic axis needs **five** allocated guards when
``mad_q != 1``: the endpoint consumes one upper slot, and the MAD upwind stencil
still needs four neighbors beyond it. ``initialize_tiled_bssn`` rejects this
combination with only four guards. Four remain sufficient when that axis uses
Sommerfeld (which forces D4) or when ``mad_q == 1``. Keep the guard depth suitable
for the boundary codes and MAD setting throughout the run.

The existing face codes in ``BSSNParameters`` remain global. The tiled mapper
supplies ``guard_cells``, ``owned_shape``, ``owned_faces`` and padded local
coordinate minima. Local derivatives apply physical closures at owned global
faces only. If either global face is Sommerfeld, MAD retains D4 on that entire
axis, including interior tiles.

Initialize and evolve
---------------------

For one process, expose the desired devices before Python starts. For example,
the following environment makes two CPU devices available for a small local
run::

    JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2 JAX_ENABLE_X64=true python simulation.py

A minimal ``simulation.py`` is:

.. code-block:: python

    import jax
    from JAX_BSSN.bssn.variables import BSSNParameters
    from JAX_BSSN.parallel import (
        make_field_mesh, initialize_tiled_bssn,
        tiled_rk4_step, tiled_constraint_norms,
    )

    params = BSSNParameters(dx=0.1, dt=0.001)
    mesh = make_field_mesh((2, 1, 1))
    layout = dict(
        mesh=mesh, tile_shape=(16, 16, 16), guard_cells=4,
        upper_endpoints=(False, False, False),
    )
    state = initialize_tiled_bssn(None, params, **layout)  # flat spacetime
    state = tiled_rk4_step(state, params, **layout)
    norms = tiled_constraint_norms(state, params, **layout)
    if jax.process_index() == 0:
        print({name: float(value) for name, value in norms.items()})

``initialize_tiled_bssn`` also accepts an existing small global
``BSSNVariables`` tree, a mapping from state field names to native ``.npy``
filenames, or a callback. File input is memory mapped and only addressable
owned tiles are loaded. Omitted or false mapping entries mean literal zero;
use ``None`` for the flat-spacetime default instead of an incomplete mapping.
Passing a large global device array is not the scalable initialization path.

A callback receives broadcast coordinates on its owned samples and local
parameters with ``guard_cells=0``. It must return the full owned tile:

.. code-block:: python

    import jax.numpy as jnp
    from JAX_BSSN.bssn.variables import BSSNVariables

    def initial_data(coordinates, local_params):
        x, y, z = coordinates
        one = jnp.ones(local_params.owned_shape, dtype=jnp.float64)
        zero = jnp.zeros_like(one)
        metric = jnp.eye(3)[:, :, None, None, None] * one
        vector = jnp.zeros((3,) + local_params.owned_shape, dtype=one.dtype)
        lapse = one + 0.001 * jnp.cos(x)
        return BSSNVariables(
            conformal_metric=metric, conformal_factor=one,
            traceless_K=jnp.zeros_like(metric), trace_K=zero,
            conformal_connection=vector, lapse=lapse, shift=vector,
        )

    state = initialize_tiled_bssn(initial_data, params, **layout)

For arbitrary scalar, vector, or tensor initial fields,
``initialize_tiled_field`` has the same callback convention and accepts
``component_shape=()``, ``(3,)`` or ``(3, 3)`` respectively.

Local equation and coupling callbacks
------------------------------------

``map_tiled(function, values, params, **layout)`` calls
``function(local_values, local_params)`` on padded local arrays. Leading
component axes remain present; the three tile axes are removed. The returned
tree must retain the padded spatial shape. Supply ``output_template`` when
the output tree or component ranks differ from the input. The mapper does not
refresh halos itself: use ``refresh_bssn_halos`` or ``refresh_tiled_field``
before callbacks that differentiate or interpolate their input.

``compute_tiled_bssn_rhs`` refreshes the state and evaluates the vacuum RHS.
``compute_tiled_bssn_rhs_with_matter`` additionally receives tiled
``energy_density``, ``momentum_density`` and ``spatial_stress`` with scalar,
vector and tensor component layouts. It refreshes source halos, including
momentum density needed by constraint damping. Matter quantities use the same
physical Eulerian conventions as the serial equations.

``refresh_bssn_halos`` and ``prepare_tiled_state`` accept
``ghost_fill(tiled_state, params)`` as a keyword callback. It runs after the
ordinary neighbor exchange and must return the same tiled shapes. FMR uses
this hook to supply the complete exterior shell from the matching parent
stage, including edges and corners. A callback that depends on another
patch's state must be supplied with that patch's state at the same RK stage;
``fmr_rk4_step`` coordinates this across the hierarchy.

The PyPIC3D coupled path passes its exact field mesh and tile widths to this
interface. Its BSSN guard depth is
``max(4 + int(any(upper_endpoints)), EM_guard_cells + 1)`` so endpoint ownership
and local C-to-Yee interpolation retain the neighbors they need. The extra
endpoint flags follow PyPIC3D's conducting axes. Local metric reconstruction, Yee interpolation
and stress-energy assembly preserve their grid locations, and the existing
coupled RK4/Strang staging is retained.

PyPIC3D's optional tiled gravity-boundary and prescribed-gauge callbacks have
the signature ``callback(local_padded_state, time, *, params=local_parameters)``.
They return the same state shapes and use ``owned_faces`` to identify physical
faces. Their serial signature remains ``callback(state, time)``. Native metric
files retain their global C-array disk layout; workers read only their owned
pieces. Field gathers are unnecessary for the evolution and scalar diagnostics.

Multiple processes and nodes
----------------------------

Every process must initialize distributed JAX **before** importing JAX_BSSN,
constructing arrays, or discovering devices. All processes then construct the
same global mesh and call the same collective operations in the same order.
Only logging and local output handling should be restricted to process zero.

For a standalone MPI-launched BSSN program, place this bootstrap before the
imports in the previous example:

.. code-block:: python

    import jax
    jax.distributed.initialize(
        cluster_detection_method="mpi4py", local_device_ids=[0]
    )

This example assumes one visible GPU per MPI rank and a working ``mpi4py``
installation. The mesh shape must account for all participating ranks and
devices. Explicit coordinator/rank initialization is also available through
``jax.distributed.initialize`` when MPI bootstrap is unavailable; use a
coordinator address reachable from every node, not a loopback address.

Coupled PyPIC3D jobs use its existing launcher and import-time bootstrap. For
example, with one visible GPU per rank:

.. code-block:: bash

    PYPIC_DISTRIBUTED=1 PYPIC_LOCAL_DEVICE_IDS=0 JAX_ENABLE_X64=true \
      PYTHONNOUSERSITE=1 srun --nodes=2 --ntasks-per-node=4 \
      --gpus-per-task=1 --gpu-bind=single:1 \
      python -m PyPIC3D --config coupled.toml

The scheduler allocation, MPI runtime, CUDA/JAX installation and configuration
must match the target cluster. Set the environment before importing PyPIC3D;
``PYPIC_DISTRIBUTED=1`` invokes ``initialize_distributed`` before JAX backend
initialization. Do not initialize the same runtime twice.

Validation and present limits
-----------------------------

The regression tests compare padded derivatives and physical boundaries with
the serial operators, including mixed derivatives, MAD, reduced axes and extra
endpoint ownership. The gauge-wave tests compare against an analytic solution
under spatial refinement and check RK4 temporal refinement separately. Run the
local CPU suites with:

.. code-block:: bash

    JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=8 JAX_ENABLE_X64=true \
      python -m unittest tests.test_parallel tests.test_tiled_boundaries \
      tests.test_parallel_convergence

``python -m unittest tests.test_parallel_processes`` launches
two fresh CPU workers with one device each. It checks encoded periodic halos
across process boundaries, a flat-space vacuum RHS, and globally reduced
constraint norms. This test needs permission to open temporary loopback sockets.

Constraint L2 and Linf diagnostics count each owned sample once, excluding
ghosts, and communicate scalar statistics. They do not gather the fields. The
two-process CPU test validates process separation and communication; it does not
establish multi-node GPU performance, long-time nonlinear stability, or the
accuracy of a particular coupled physical problem. Those require separate
resolution, timestep, long-time evolution and hardware-scale studies. Existing
full-array diagnostics, plotting and checkpoint consumers need their own tiled
data path before use on a distributed state.
