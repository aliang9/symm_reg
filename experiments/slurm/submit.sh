#!/bin/bash
#SBATCH --job-name=symm-reg
#SBATCH --partition=pi_manoli
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/symm_reg_%j.out
#SBATCH --error=logs/symm_reg_%j.err

# ============================================================
# Symmetry-Regularized Dynamics Experiment
# ============================================================
#
# Submit with:
#   sbatch --export=EXPERIMENT_CONFIG=regularizer_comparison experiments/slurm/submit.sh
#
# With extra Hydra overrides:
#   sbatch --export=EXPERIMENT_CONFIG=consensus,EXTRA_ARGS="training.n_epochs=100" \
#          experiments/slurm/submit.sh
#
# Monitor:
#   squeue --me
#   tail -f logs/symm_reg_<jobid>.out
#
# ============================================================

set -e

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Experiment: ${EXPERIMENT_CONFIG:-base}"
echo "Extra args: ${EXTRA_ARGS:-none}"
echo "Start time: $(date)"
echo "============================================================"

# Create logs directory
mkdir -p logs

# Navigate to project root
cd /orcd/data/manoli/001/om2/artliang/symm_reg

# Activate virtual environment
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

# Verify environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# ============================================================
# Run experiment
# ============================================================

EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-regularizer_comparison}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Running: python experiments/train.py +experiment=${EXPERIMENT_CONFIG} training.device=cuda ${EXTRA_ARGS}"
echo ""

python experiments/train.py \
    +experiment="${EXPERIMENT_CONFIG}" \
    training.device=cuda \
    ${EXTRA_ARGS}

echo ""
echo "============================================================"
echo "Experiment complete!"
echo "End time: $(date)"
echo "============================================================"
