#!/bin/bash -e
#SBATCH --job-name=big_all_noabars
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:44:44
#SBATCH --mem-per-cpu 49152M
date
module load R
module load JAGS
R CMD BATCH '--args a' s_noabar
