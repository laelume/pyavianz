#!/bin/bash -e
#SBATCH --job-name=1-2-3-to-4
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=244:44:44
#SBATCH --mem-per-cpu 49152M
date
module load R
module load JAGS
R CMD BATCH '--args 1 2 3' s_source_dest
