#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -n 1                      # number of cores
#SBATCH -t 25                 # time (minutes)
#SBATCH -o /scratch/zmd/logs/outmain_param_sweep.out        # STDOUT
#SBATCH -e /scratch/zmd/logs/outmain_param_sweep.err        # STDERR

module load anacondapy/5.3.1
module load elastix/4.8
. activate idisco

#make dict, process images
OUT0=$(sbatch slurm_files/param_sweep_step0.sh) 
echo $OUT0

#run param sweep
OUT1=$(sbatch --dependency=afterany:${OUT0##* } --array=0-15 slurm_files/param_sweep_step1.sh) 
echo $OUT1

# Usage notes:
# after = go once the specified job starts
# afterany = go if the specified job finishes, regardless of success
# afternotok = go if the specified job fails
# afterok = go if the specified job completes successfully
