
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from functools import partial
import jax

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables


# NOTE : HAS NOT BEEN FULLY TESTED YET

def save_data(vars: BSSNVariables, t: float, step: int, 
              output_dir: str = "output"):
    """Save simulation data to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save key quantities
    np.savez(f"{output_dir}/data_step_{step:06d}.npz",
             time=t,
             conformal_metric=np.array(vars.conformal_metric),
             conformal_factor=np.array(vars.conformal_factor),
             traceless_K=np.array(vars.traceless_K),
             trace_K=np.array(vars.trace_K),
             lapse=np.array(vars.lapse))

def plot_results(vars: BSSNVariables, t: float, params: BSSNParameters):
    """Create diagnostic plots (energy density plot removed)."""
    ni, nj, nk = vars.conformal_factor.shape

    # Central slices
    i_center = ni // 2
    j_center = nj // 2
    k_center = nk // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'BSSN Evolution at t = {t:.3f}')

    # Conformal factor
    im1 = axes[0, 0].imshow(np.array(vars.conformal_factor[:, :, k_center]),
                            origin='lower', aspect='equal')
    axes[0, 0].set_title('Conformal Factor W (z=0)')
    plt.colorbar(im1, ax=axes[0, 0])

    # Lapse function
    im2 = axes[0, 1].imshow(np.array(vars.lapse[:, :, k_center]),
                            origin='lower', aspect='equal')
    axes[0, 1].set_title('Lapse α (z=0)')
    plt.colorbar(im2, ax=axes[0, 1])

    # Trace of K
    im3 = axes[0, 2].imshow(np.array(vars.trace_K[:, :, k_center]),
                            origin='lower', aspect='equal')
    axes[0, 2].set_title('Trace K (z=0)')
    plt.colorbar(im3, ax=axes[0, 2])

    # Conformal metric component
    im4 = axes[1, 0].imshow(np.array(vars.conformal_metric[0, 0, :, :, k_center]),
                            origin='lower', aspect='equal')
    axes[1, 0].set_title('γ_xx (z=0)')
    plt.colorbar(im4, ax=axes[1, 0])

    # Extrinsic curvature component
    im5 = axes[1, 1].imshow(np.array(vars.traceless_K[0, 0, :, :, k_center]),
                            origin='lower', aspect='equal')
    axes[1, 1].set_title('A_xx (z=0)')
    plt.colorbar(im5, ax=axes[1, 1])

    # The energy density plot was removed; turn off the unused axis
    axes[1, 2].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'evolution_t_{t:.3f}.png', dpi=150)
    plt.show()



def plot_constraint_evolution(constraint_history: List[Tuple[float, dict]]):
    """Plot evolution of constraint violations."""
    times = [entry[0] for entry in constraint_history]
    
    hamiltonian_l2 = [entry[1]['hamiltonian_l2'] for entry in constraint_history]
    momentum_l2 = [entry[1]['momentum_l2'] for entry in constraint_history]
    det_gamma_l2 = [entry[1]['det_gamma_l2'] for entry in constraint_history]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.semilogy(times, hamiltonian_l2, 'b-', label='Hamiltonian')
    plt.xlabel('Time')
    plt.ylabel('L2 Norm')
    plt.title('Hamiltonian Constraint')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(times, momentum_l2, 'r-', label='Momentum')
    plt.xlabel('Time')
    plt.ylabel('L2 Norm')
    plt.title('Momentum Constraint')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogy(times, det_gamma_l2, 'g-', label='det(γ)-1')
    plt.xlabel('Time')
    plt.ylabel('L2 Norm')
    plt.title('det(γ) = 1 Condition')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.loglog(times, hamiltonian_l2, 'b-', label='Hamiltonian')
    plt.loglog(times, momentum_l2, 'r-', label='Momentum')
    plt.loglog(times, det_gamma_l2, 'g-', label='det(γ)-1')
    plt.xlabel('Time')
    plt.ylabel('L2 Norm')
    plt.title('All Constraints')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('constraint_evolution.png', dpi=150)
    plt.show()

@partial(jax.jit, static_argnums=(0))
def write_data(filename, time, data):
    """
    Write the given time and data to a file using JAX's callback mechanism.
    This function is designed to be used with JAX's just-in-time compilation (jit) to optimize performance.

    Args:
        filename (str): The name of the file to write to.
        time (float): The time value to write.
        data (any): The data to write.

    Returns:
        None
    """

    def write_to_file(filename, time, data):
        with open(filename, "a") as f:
            f.write(f"{time}, {data}\n")

    return jax.debug.callback(write_to_file, filename, time, data, ordered=True)
