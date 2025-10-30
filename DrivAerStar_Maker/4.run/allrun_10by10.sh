#!/bin/bash


SBATCH_SCRIPT="krun_10.slurm"

# 2000 is (the number of your case )/10 
for NUM in {0..2000}; do
  echo "Submitting job for number $NUM"
  sbatch $SBATCH_SCRIPT $NUM
done
