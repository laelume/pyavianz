#!/bin/bash -e
#SBATCH --job-name=northland_noabars
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=44:44:44
#SBATCH --mem-per-cpu 8192M
date
module load R
module load JAGS
R CMD BATCH '--args 4' s_noabar
