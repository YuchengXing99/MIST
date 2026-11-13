#!/bin/bash
# =============================================================================
#  BRCA Survival — Full Experiment Launch
#  Train/Test: TCGA-BRCA (5-fold CV)
#  8x H200 GPUs, 2 jobs per GPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================= EDIT THESE PATHS =========================
DATA_ROOTS="/path/to/TCGA-BRCA"
PROTO_DIR="/path/to/prototypes"
RESULTS_DIR_ROOT="results_survival_brca"
# ====================================================================

GPUS="0 1 2 3 4 5 6 7"
JOBS_PER_GPU=2
SEEDS="1 2 3 4 5 6 7 8 9 10"
FOLDS="0 1 2 3 4"
N_PROTO="16"

# --- Phase 1: Baselines ---
echo "===== BRCA Phase 1: Baselines ====="
DATASET=brca \
CONFIGS="meanmil maxmil abmil clam transmil ilra transformer_mil dsmil" \
DATA_ROOTS="${DATA_ROOTS}" \
SEEDS="${SEEDS}" FOLDS="${FOLDS}" \
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT}" \
JOBS_PER_GPU="${JOBS_PER_GPU}" \
bash "${SCRIPT_DIR}/run_sweep.sh" ${GPUS}

# --- Phase 2: MIST variants ---
echo "===== BRCA Phase 2: MIST ====="
DATASET=brca \
CONFIGS="meanmil_mist maxmil_mist abmil_mist clam_mist transmil_mist ilra_mist transformer_mil_mist dsmil_mist" \
DATA_ROOTS="${DATA_ROOTS}" \
SEEDS="${SEEDS}" FOLDS="${FOLDS}" \
N_PROTO="${N_PROTO}" PROTO_DIR="${PROTO_DIR}" \
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT}" \
JOBS_PER_GPU="${JOBS_PER_GPU}" \
bash "${SCRIPT_DIR}/run_sweep.sh" ${GPUS}

echo "===== All BRCA experiments finished! ====="
