#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -n 3                      # number of cores
#SBATCH -t 100                 # time (minutes)
#SBATCH --contiguous
#SBATCH -o logs/parameter_sweep_%a.out        # STDOUT
#SBATCH -e logs/parameter_sweep_%a.err        # STDERR

module load anacondapy/5.3.1
module load elastix/4.8
. activate idisco

#run parameter sweep
xvfb-run python run_parameter_sweep.py 0 ${SLURM_ARRAY_TASK_ID} 
