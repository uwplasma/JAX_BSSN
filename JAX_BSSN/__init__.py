import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, NamedTuple
import numpy as np
from functools import partial


def setup_jax_config(enable_x64=True, verbose=True):
    """
    Configure JAX for optimal performance with GPU support.
    
    This function:
    - Enables 64-bit precision (float64) for numerical accuracy
    - Automatically detects and uses GPU if available
    - Logs device information
    
    Args:
        enable_x64 (bool): Enable 64-bit precision (default: True)
        verbose (bool): Print device information (default: True)
    """
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    
    if verbose:
        devices = jax.devices()
        device_names = [str(d) for d in devices]
        backend = jax.default_backend()
        print(f"JAX backend: {backend}")
        print(f"JAX devices available: {device_names}")
        if 'gpu' in backend.lower() or any('GPU' in name for name in device_names):
            print("GPU acceleration enabled")
        else:
            print("WARNING: Using CPU (no GPU detected)")


from . import bssn
from . import boundaries
from . import derivatives
from . import errors
from . import evolve
from . import initialization
from . import plotting
from . import tensor_algebra
