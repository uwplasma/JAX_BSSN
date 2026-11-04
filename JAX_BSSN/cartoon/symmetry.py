"""Parity tables and field mapping shared by the Cartoon symmetry packages.

Every Cartoon mode stores one authoritative half of a reflection-symmetric grid
and rebuilds the rest from parity.  The reflection sign depends only on a
field's tensor rank and on the mode's vector parity table, so the seven
``BSSNVariables`` entries can be handled by one rank dispatch instead of seven
hand-written cases per operation.
"""

import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNVariables


SCALAR_PARITY = 1.0


def vector_parity(signs, dtype):
    """Return the per-component reflection signs of a Cartesian vector."""

    return jnp.asarray(signs, dtype=dtype)


def tensor_parity(signs, dtype):
    """Return the reflection signs of a rank-two tensor as an outer product."""

    parity = vector_parity(signs, dtype)
    return parity[:, None] * parity[None, :]


def field_parity(field, signs, spatial_ndim=3):
    """Return the reflection parity implied by a field's tensor rank."""

    rank = field.ndim - spatial_ndim
    if rank == 0:
        return jnp.asarray(SCALAR_PARITY, dtype=field.dtype)
    if rank == 1:
        return vector_parity(signs, field.dtype)
    if rank == 2:
        return tensor_parity(signs, field.dtype)
    raise ValueError(f"unsupported tensor rank {rank} for reflection parity")


def map_bssn_fields(vars: BSSNVariables, transform, signs, spatial_ndim=3):
    """Apply ``transform(field, parity)`` to every BSSN field in order."""

    return BSSNVariables(
        *(
            transform(field, field_parity(field, signs, spatial_ndim))
            for field in vars
        )
    )


def reflect(positive, parity, axis=-1, count=None):
    """ Use a parity condition to reflect a tensor across the origin.

    For example, a scalar in spherical symmetry is even under reflection,
    so the reflected grid is simply the mirror image of the positive half
    """

    axis = axis % positive.ndim
    parity = jnp.asarray(parity, dtype=positive.dtype)
    spatial_axes = positive.ndim - parity.ndim
    parity = parity.reshape(parity.shape + (1,) * spatial_axes)

    index = [slice(None)] * positive.ndim
    index[axis] = slice(None, count)
    mirrored = jnp.flip(positive[tuple(index)], axis=axis) * parity

    return jnp.concatenate((mirrored, positive), axis=axis)
