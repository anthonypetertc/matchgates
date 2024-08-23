#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000
#SBATCH --exclude=gh1,gh2

#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt
#SBATCH --job-name=matchgate_noisy

# Fail fast on error
set -e

# Activate venv
source /home/chris/experiments/matchgates/.venv/bin/activate

# Run script
python /home/chris/experiments/matchgates/examples/scripts/noisy_circuits.py

