#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -n 1                      # number of cores
#SBATCH -t 25                 # time (minutes)
#SBATCH -o logs/outmain_tracing.out        # STDOUT
#SBATCH -e logs/outmain_tracing.err        # STDERR

module load anacondapy/5.3.1
module load elastix/4.8
. activate idisco

#set up dictionary and save
OUT0=$(sbatch --array=0 slurm_files/step0.sh) 
echo $OUT0

#process zplns, check that 1000 > zplns/slurmfactor
OUT1=$(sbatch --dependency=afterany:${OUT0##* } --array=0-80 slurm_files/step1.sh) 
echo $OUT1

#run cell detect
OUT2=$(sbatch --dependency=afterany:${OUT1##* } --array=0-50 slurm_files/step4.sh) 
echo $OUT2

#consolidate cell detection
OUT3=$(sbatch --dependency=afterany:${OUT2##* } --array=0 slurm_files/step5.sh) 
echo $OUT3

#Complete Output Analysis
OUT4=$(sbatch --dependency=afterany:${OUT3##* } --array=0 slurm_files/step6.sh) 
echo $OUT4


# Usage notes:
# after = go once the specified job starts
# afterany = go if the specified job finishes, regardless of success
# afternotok = go if the specified job fails
# afterok = go if the specified job completes successfully
