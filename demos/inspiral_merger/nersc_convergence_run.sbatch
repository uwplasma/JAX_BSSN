#!/bin/bash
# Perlmutter batch job: inspiral merger self-convergence on 128^3, 256^3 and 512^3 (+/-12 M box).
# Solves each grid's initial data on one node's CPU cores, then evolves on all 16.
# Stops at t = 20 M, during the inspiral where the fields are smooth, with the same
# quasi-circular orbit as the GW run, and writes output/convergence/.
# Submit from demos/inspiral_merger with: sbatch nersc_convergence_run.sbatch
#SBATCH -A m4656_g
#SBATCH -C "gpu&hbm40g"
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 06:00:00
#SBATCH -J inspiral_conv
#SBATCH -o %x-%j.out

module load python
# demo settings for this run: stop before the merger, same quasi-circular orbit as the GW run
SETTINGS="import inspiral_merger as b; b.FINAL_TIME = 20.0; b.INFALL_MOMENTUM = 0.0"

for n in 128 256 512; do  # must match GRID_SIZES in inspiral_merger_convergence.py
    srun -N 1 -n 1 --ntasks-per-node=1 -c 128 --cpu-bind=none --gpus-per-task=1         python -c "$SETTINGS; b.prepare_initial_data($n, 'output/convergence/nx_$(printf %04d $n)')"
done
srun --gpus-per-task=1 --gpu-bind=none     python -c "$SETTINGS; import runpy; runpy.run_path('inspiral_merger_convergence.py', run_name='__main__')"
