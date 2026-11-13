"""Fresh-process FMR transport worker with one CPU device on each rank."""

import json
import os
import sys


def main():
    address, process_id = sys.argv[1], int(sys.argv[2])
    affinity = os.environ.get("BSSN_TEST_CPU_AFFINITY")
    if affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(cpu) for cpu in affinity.split(",")})

    import jax

    jax.distributed.initialize(
        coordinator_address=address, num_processes=2, process_id=process_id,
        local_device_ids=[0], initialization_timeout=45,
        heartbeat_timeout_seconds=30, shutdown_timeout_seconds=10,
    )
    try:
        import jax.numpy as jnp
        from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
        from JAX_BSSN.fmr import (
            FMRHierarchySpec, FMRPatchSpec, initialize_tiled_fmr,
            fill_tiled_fine_ghosts, restrict_tiled_to_coarse, fmr_rk4_step,
        )
        from JAX_BSSN.parallel import make_field_mesh

        mesh = make_field_mesh((2,1,1))
        spec = FMRPatchSpec((6,5,5), (14,12,12))
        hierarchy = FMRHierarchySpec((spec,))
        params = BSSNParameters(dx=0.1, dt=0.0001)

        def initial(coordinates, p):
            x, y, z = jnp.broadcast_arrays(*coordinates)
            one = jnp.ones_like(x)
            zero = jnp.zeros_like(x)
            metric = jnp.eye(3)[:, :, None, None, None]*one
            vector = jnp.zeros((3,)+one.shape)
            return BSSNVariables(metric, one, jnp.zeros_like(metric), x+2*y+3*z,
                                 vector, one, vector)

        states, parameters, layouts = initialize_tiled_fmr(
            initial, params, (24,20,20), hierarchy, mesh=mesh)
        child = states[1]._replace(trace_K=jnp.full_like(states[1].trace_K, -17.))
        filled = fill_tiled_fine_ghosts(states[0], child, spec,
                                      parent_layout=layouts[0], child_layout=layouts[1])
        assert not filled.trace_K.is_fully_addressable
        assert len(filled.trace_K.addressable_shards) == 1
        shard = filled.trace_K.addressable_shards[0]
        owner = tuple(index.start or 0 for index in shard.index[-6:-3])
        layout = layouts[1]
        width = layout["tile_shape"]
        guard = layout["guard_cells"]
        global_shape = tuple(n*count+int(endpoint) for n,count,endpoint in zip(
            width, mesh.devices.shape, layout["upper_endpoints"]))
        nodes = [owner[d]*width[d]+jnp.arange(width[d]+2*guard)-guard for d in range(3)]
        x, y, z = jnp.meshgrid(*nodes, indexing="ij")
        shell = ((x<0)|(x>=global_shape[0])|(y<0)|(y>=global_shape[1])
                 |(z<0)|(z>=global_shape[2]))
        exact = (spec.coarse_lo[0]*params.dx+x*params.dx/2
                 +2*(spec.coarse_lo[1]*params.dx+y*params.dx/2)
                 +3*(spec.coarse_lo[2]*params.dx+z*params.dx/2))
        values = shard.data[0,0,0]
        halo_error = float(jnp.max(jnp.where(shell, jnp.abs(values-exact), 0)))
        assert halo_error < 1e-11, halo_error
        restricted = restrict_tiled_to_coarse(states[0], filled, spec,
                                             parent_layout=layouts[0], child_layout=layouts[1])
        root_shard = restricted.trace_K.addressable_shards[0]
        root_owner = tuple(index.start or 0 for index in root_shard.index[-6:-3])
        root_layout = layouts[0]
        g = root_layout["guard_cells"]
        root_nodes = [root_owner[d]*root_layout["tile_shape"][d]
                      +jnp.arange(root_layout["tile_shape"][d]) for d in range(3)]
        rx, ry, rz = jnp.meshgrid(*root_nodes, indexing="ij")
        covered = jnp.ones(rx.shape, dtype=bool)
        for node, lo, hi in zip((rx, ry, rz), spec.coarse_lo, spec.coarse_hi):
            covered &= (node >= lo) & (node <= hi)
        owned = tuple(slice(g, g+n) for n in root_layout["tile_shape"])
        root_values = root_shard.data[(0, 0, 0) + owned]
        injection_error = float(jnp.max(jnp.where(covered, jnp.abs(root_values+17.), 0)))
        assert injection_error == 0., injection_error

        flat, flat_params, flat_layouts = initialize_tiled_fmr(
            None, params, (24, 20, 20), hierarchy, mesh=mesh)
        evolved = jax.jit(lambda values: fmr_rk4_step(
            values, flat_params, hierarchy, layouts=flat_layouts))(flat)
        evolution_error = 0.
        for old, new in zip(flat, evolved):
            for before, after in zip(old, new):
                assert not after.is_fully_addressable
                assert len(after.addressable_shards) == 1
                assert after.addressable_shards[0].data.shape[-6:-3] == (1, 1, 1)
                old_local = before.addressable_shards[0].data
                new_local = after.addressable_shards[0].data
                difference = float(jnp.max(jnp.abs(old_local-new_local)))
                evolution_error = max(evolution_error, difference)
        assert evolution_error < 1e-11, evolution_error
        print("BSSN_FMR_DISTRIBUTED_RESULT " + json.dumps(dict(
            process_id=process_id, process_count=jax.process_count(),
            local_device_count=jax.local_device_count(), halo_error=halo_error,
            injection_error=injection_error, evolution_error=evolution_error,
        )), flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
