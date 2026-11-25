#!/bin/bash
# Perlmutter batch job: 512^3 inspiral merger on a +/-32 M box for the gravitational-wave probe.
# Solves the initial data on one node's CPU cores, then evolves on all 16.
# Overrides the demo's globals for this run (quasi-circular orbit, probes at 12.8, 19.2 and
# 25.6 M, t = 150 M), writes everything to output_gw/, then plots the strain with
# gw_strain_probe.py. Submit from demos/inspiral_merger with: sbatch nersc_gw_run.sbatch
#SBATCH -A m4656_g
#SBATCH -C "gpu&hbm80g"
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 12:00:00
#SBATCH -J inspiral_gw
#SBATCH -o %x-%j.out

module load python
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95  # the 512^3 RK4 step needs ~56 GiB per A100 on 4 nodes
# demo settings for this run: probes in the wave zone, quasi-circular orbit for the analytic (BOB)
# comparison, inspiral + merger + several ringdown e-folding times, strain sampled every 0.5 M
SETTINGS="import inspiral_merger as b; b.HALF_WIDTH = 32.0; b.FINAL_TIME = 150.0; b.INFALL_MOMENTUM = 0.0; b.PROBE_INTERVAL = 0.5"

srun -N 1 -n 1 --ntasks-per-node=1 -c 128 --cpu-bind=none --gpus-per-task=1     python -c "$SETTINGS; b.prepare_initial_data(512, 'output_gw')"
srun --gpus-per-task=1 --gpu-bind=none     python -c "$SETTINGS; s, dx, h = b.run_inspiral_merger(512, 'output_gw', 2000); b.ROOT and print(f'Final H L2 = {h[-1, 1]:.4e}, M L2 = {h[-1, 2]:.4e}')"
python -c "import gw_strain_probe; gw_strain_probe.plot_strain('output_gw')"
