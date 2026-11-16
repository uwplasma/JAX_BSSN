"""Fresh-process CPU worker for the distributed Cartesian regression test."""

import json
import os
import sys


def main():
    address, process_id = sys.argv[1], int(sys.argv[2])
    affinity = os.environ.get("BSSN_TEST_CPU_AFFINITY")
    if affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(cpu) for cpu in affinity.split(",")})

    # Importing the package can construct JAX arrays. Bootstrap every worker
    # first, before any package import or device discovery initializes a backend.
    import jax

    jax.distributed.initialize(
        coordinator_address=address, num_processes=2, process_id=process_id,
        local_device_ids=[0], initialization_timeout=45,
        heartbeat_timeout_seconds=30, shutdown_timeout_seconds=10,
    )
    try:
        import jax.numpy as jnp
        from JAX_BSSN.bssn.variables import BSSNParameters
        from JAX_BSSN.parallel import (
            compute_tiled_bssn_rhs, initialize_tiled_bssn,
            initialize_tiled_field, make_field_mesh, tiled_constraint_norms,
        )

        assert jax.process_count() == 2
        assert jax.local_device_count() == 1
        mesh = make_field_mesh((2, 1, 1))
        assert {device.process_index for device in mesh.devices.flat} == {0, 1}
        layout = dict(mesh=mesh, tile_shape=(8, 8, 8), guard_cells=4)
        params = BSSNParameters(dx=1.0, dt=0.01)

        def encoded_coordinates(coordinates, local_params):
            x, y, z = coordinates
            return x + 100 * y + 10000 * z

        encoded = initialize_tiled_field(encoded_coordinates, params, **layout)
        assert not encoded.is_fully_addressable
        assert len(encoded.addressable_shards) == 1
        shard = encoded.addressable_shards[0]
        tile_x = shard.index[0].start
        nodes = jnp.arange(16) - 4
        x = ((8 * tile_x + nodes) % 16)[:, None, None]
        y = (nodes % 8)[None, :, None]
        z = (nodes % 8)[None, None, :]
        expected = x + 100 * y + 10000 * z
        halo_error = float(jnp.max(jnp.abs(shard.data[0, 0, 0] - expected)))
        assert halo_error == 0.0, halo_error

        state = initialize_tiled_bssn(None, params, **layout)
        rhs = compute_tiled_bssn_rhs(state, params, **layout)
        rhs_error = 0.0
        for field in rhs:
            assert not field.is_fully_addressable
            assert len(field.addressable_shards) == 1
            local = field.addressable_shards[0].data
            owned = local[(Ellipsis, slice(4, 12), slice(4, 12), slice(4, 12))]
            assert bool(jnp.all(jnp.isfinite(owned)))
            rhs_error = max(rhs_error, float(jnp.max(jnp.abs(owned))))
        assert rhs_error < 1.0e-12, rhs_error

        norms = tiled_constraint_norms(state, params, **layout)
        assert len(norms) == 10
        constraint_error = max(float(value) for value in norms.values())
        assert constraint_error < 1.0e-12, constraint_error
        result = dict(
            process_id=process_id, process_count=jax.process_count(),
            local_device_count=jax.local_device_count(),
            halo_error=halo_error, rhs_error=rhs_error,
            constraint_error=constraint_error,
            state_shape=list(state.conformal_factor.shape),
        )
        print("BSSN_DISTRIBUTED_RESULT " + json.dumps(result), flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
