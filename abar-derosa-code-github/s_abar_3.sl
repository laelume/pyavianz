#!/bin/bash -e
#SBATCH --job-name=rshg_abars
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=44:44:44
#SBATCH --mem-per-cpu 1024M
date
module load R
module load JAGS
R CMD BATCH '--args 3' s_abar

