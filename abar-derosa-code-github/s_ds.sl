#!/bin/bash -e
#SBATCH --job-name=distance_sampling
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=84:44:44
#SBATCH --mem-per-cpu 16384M
date
module load R
module load JAGS
R CMD BATCH smds

