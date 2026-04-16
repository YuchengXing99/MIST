#!/bin/bash
# =============================================================================
#  BCNB Biomarker — Full Experiment Launch
#  Dataset: BCNB (Breast Core Needle Biopsy, 1058 slides)
#  Tasks: ER, PR, HER2 (10-fold CV, per MAMMOTH arXiv:2603.22198)
#  8x H200 GPUs, 2 jobs per GPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
BCNB_ROOT="/path/to/BCNB"
BCNB_PROTO_DIR="/path/to/mol_vis_by_organ/Breast/prototypes_by_k"
RESULTS_DIR_ROOT="results_biomarker_bcnb"
# ====================================================================

GPUS="0 1 2 3 4 5 6 7"
JOBS_PER_GPU=2
SEEDS="1 2 3 4 5"
FOLDS="0 1 2 3 4 5 6 7 8 9"
N_PROTO=16

# MAMMOTH uses class-weighted sampling for imbalanced biomarker tasks
export TRAIN_SAMPLER=weighted

BASELINES="abmil transmil ilra meanmil maxmil clam transformer_mil dsmil"
MIST_VARIANTS="abmil_mist transmil_mist ilra_mist meanmil_mist maxmil_mist clam_mist transformer_mil_mist dsmil_mist"

cd "${SRC_DIR}"

# --- Phase 1: Baselines ---
echo "===== BCNB Phase 1: Baselines ====="
python scripts/classification/biomarker/run_sweep.py \
    --dataset bcnb \
    --configs ${BASELINES} \
    --gpus ${GPUS} \
    --jobs-per-gpu "${JOBS_PER_GPU}" \
    --seeds ${SEEDS} \
    --folds ${FOLDS} \
    --skip-existing \
    --results-dir-root "${RESULTS_DIR_ROOT}" \
    --data-root "${BCNB_ROOT}" \
    || echo "WARN: bcnb baselines had failures."

# --- Phase 2: MIST variants ---
echo "===== BCNB Phase 2: MIST ====="
python scripts/classification/biomarker/run_sweep.py \
    --dataset bcnb \
    --configs ${MIST_VARIANTS} \
    --n-proto ${N_PROTO} \
    --proto-dir "${BCNB_PROTO_DIR}" \
    --gpus ${GPUS} \
    --jobs-per-gpu "${JOBS_PER_GPU}" \
    --seeds ${SEEDS} \
    --folds ${FOLDS} \
    --skip-existing \
    --results-dir-root "${RESULTS_DIR_ROOT}" \
    --data-root "${BCNB_ROOT}" \
    || echo "WARN: bcnb MIST had failures."

echo "===== All BCNB biomarker experiments finished! ====="
