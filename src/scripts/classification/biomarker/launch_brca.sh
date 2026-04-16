#!/bin/bash
# =============================================================================
#  TCGA-BRCA Biomarker — Full Experiment Launch
#  Tasks: HER2, ER, PR, PIK3CA (5-fold CV)
#  8x H200 GPUs, 2 jobs per GPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
TCGA_BRCA_ROOT="/path/to/TCGA-BRCA"
BRCA_PROTO_DIR="/path/to/mol_vis_by_organ/Breast/prototypes_by_k"
RESULTS_DIR_ROOT="results_biomarker_brca"
# ====================================================================

GPUS="0 1 2 3 4 5 6 7"
JOBS_PER_GPU=2
SEEDS="1 2 3 4 5"
FOLDS="0 1 2 3 4"
N_PROTO=16

BASELINES="abmil transmil ilra meanmil maxmil clam transformer_mil dsmil"
MIST_VARIANTS="abmil_mist transmil_mist ilra_mist meanmil_mist maxmil_mist clam_mist transformer_mil_mist dsmil_mist"

cd "${SRC_DIR}"

# MAMMOTH Appendix A2: "cross-entropy loss with random class-weighted sampling"
export TRAIN_SAMPLER=weighted

# --- Phase 1: Baselines ---
echo "===== TCGA-BRCA Phase 1: Baselines ====="
python scripts/classification/biomarker/run_sweep.py \
    --dataset tcga_brca \
    --configs ${BASELINES} \
    --gpus ${GPUS} \
    --jobs-per-gpu "${JOBS_PER_GPU}" \
    --seeds ${SEEDS} \
    --folds ${FOLDS} \
    --skip-existing \
    --results-dir-root "${RESULTS_DIR_ROOT}" \
    --data-root "${TCGA_BRCA_ROOT}" \
    || echo "WARN: tcga_brca baselines had failures."

# --- Phase 2: MIST variants ---
echo "===== TCGA-BRCA Phase 2: MIST ====="
python scripts/classification/biomarker/run_sweep.py \
    --dataset tcga_brca \
    --configs ${MIST_VARIANTS} \
    --n-proto ${N_PROTO} \
    --proto-dir "${BRCA_PROTO_DIR}" \
    --gpus ${GPUS} \
    --jobs-per-gpu "${JOBS_PER_GPU}" \
    --seeds ${SEEDS} \
    --folds ${FOLDS} \
    --skip-existing \
    --results-dir-root "${RESULTS_DIR_ROOT}" \
    --data-root "${TCGA_BRCA_ROOT}" \
    || echo "WARN: tcga_brca MIST had failures."

echo "===== All TCGA-BRCA biomarker experiments finished! ====="
