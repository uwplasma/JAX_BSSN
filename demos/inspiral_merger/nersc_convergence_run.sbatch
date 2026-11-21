#!/bin/bash
# Perlmutter batch job: inspiral merger self-convergence on 128^3, 256^3 and 512^3 (+/-12 M box).
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
srun --gpus-per-task=1 --gpu-bind=none python -c "$(cat <<'PY'
import runpy

import inspiral_merger as binary

binary.FINAL_TIME = 20.0       # before the merger, where self-convergence is well defined
binary.INFALL_MOMENTUM = 0.0   # same quasi-circular orbit as the GW run
runpy.run_path("inspiral_merger_convergence.py", run_name="__main__")  # uses GRID_SIZES = [128, 256, 512]
PY
)"
