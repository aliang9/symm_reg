#!/bin/bash
# ============================================================
# Launch all symmetry-regularized experiments on SLURM
# ============================================================
#
# Usage:
#   bash experiments/slurm/launch_all.sh
#
# Each experiment is submitted as an independent job.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit.sh"

echo "Submitting all experiments..."
echo ""

# Experiment 1: Regularizer comparison
JOB1=$(sbatch --export=EXPERIMENT_CONFIG=regularizer_comparison \
    --job-name=symm-reg-comp \
    "${SUBMIT_SCRIPT}" | awk '{print $4}')
echo "Regularizer comparison: Job ${JOB1}"

# Experiment 2: Consensus
JOB2=$(sbatch --export=EXPERIMENT_CONFIG=consensus \
    --job-name=symm-reg-cons \
    "${SUBMIT_SCRIPT}" | awk '{print $4}')
echo "Consensus: Job ${JOB2}"

# Experiment 3: Multi-animal (shared dynamics)
JOB3=$(sbatch --export=EXPERIMENT_CONFIG=multi_animal \
    --job-name=symm-reg-multi \
    "${SUBMIT_SCRIPT}" | awk '{print $4}')
echo "Multi-animal: Job ${JOB3}"

# Experiment 4: Sphere SO(3)
JOB4=$(sbatch --export=EXPERIMENT_CONFIG=sphere_so3 \
    --job-name=symm-reg-so3 \
    "${SUBMIT_SCRIPT}" | awk '{print $4}')
echo "Sphere SO(3): Job ${JOB4}"

echo ""
echo "All jobs submitted. Monitor with: squeue --me"
echo "View logs with: tail -f logs/symm_reg_<jobid>.out"
