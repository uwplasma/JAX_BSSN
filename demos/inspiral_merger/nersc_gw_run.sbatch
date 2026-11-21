#!/bin/bash
# Perlmutter batch job: 512^3 inspiral merger on a +/-32 M box for the gravitational-wave probe.
# Overrides the demo's globals for this run (quasi-circular orbit, probes at 12.8, 19.2 and
# 25.6 M, t = 150 M), writes everything to output_gw/, then plots the strain with
# gw_strain_probe.py. Submit from demos/inspiral_merger with: sbatch nersc_gw_run.sbatch
#SBATCH -A m4656_g
#SBATCH -C "gpu&hbm40g"
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 12:00:00
#SBATCH -J inspiral_gw
#SBATCH -o %x-%j.out

module load python
srun --gpus-per-task=1 --gpu-bind=none python -c "$(cat <<'PY'
import inspiral_merger as binary

binary.HALF_WIDTH = 32.0       # probes in the wave zone, boundary reflections arrive after the ringdown
binary.FINAL_TIME = 150.0      # inspiral, merger and several ringdown e-folding times
binary.INFALL_MOMENTUM = 0.0   # quasi-circular orbit for the analytic (BOB) comparison
binary.PROBE_INTERVAL = 0.5    # strain sample spacing in M
state, dx, history = binary.run_inspiral_merger(512, "output_gw", 2000)  # 512^3, dx = M/8, rare full snapshots
if binary.ROOT:
    print(f"Final Hamiltonian L2 = {history[-1, 1]:.4e}, momentum L2 = {history[-1, 2]:.4e}")
PY
)"
python -c "import gw_strain_probe; gw_strain_probe.plot_strain('output_gw')"
