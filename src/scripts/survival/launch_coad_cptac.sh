#!/bin/bash
# =============================================================================
#  COAD Survival — TCGA internal + CPTAC external
#  7 GPUs (0-6), 1 job per GPU, 10 seeds, 5 folds
#  All baselines + MIST variants
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
DATA_ROOTS=("/hpc/home/e0950223/MIST_data/survival/TCGA-COAD" "/hpc/home/e0950223/MIST_data/survival/CPTAC-COAD")
PROTO_DIR="/hpc/home/e0950223/MIST_data/mol_vis_by_organ/Bowel/prototypes_by_k"
RESULTS_DIR_ROOT="results_survival_coad"
# ====================================================================

cd "${SRC_DIR}"

# --- Phase 1: Baselines ---
echo "===== COAD Phase 1: Baselines ====="
python scripts/survival/run_sweep.py \
  --dataset coad \
  --gpus 0 1 2 3 4 5 6 \
  --jobs-per-gpu 1 \
  --configs meanmil maxmil abmil clam transmil ilra transformer_mil dsmil \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --folds 0 1 2 3 4 \
  --skip-existing \
  --split-names "train,test_tcgacoad,test_cptaccoad" \
  --results-dir-root "${RESULTS_DIR_ROOT}" \
  --data-root "${DATA_ROOTS[0]}" --data-root "${DATA_ROOTS[1]}" \
  || echo "WARNING: Phase 1 had failures, continuing to Phase 2..."

# --- Phase 2: MIST variants ---
echo "===== COAD Phase 2: MIST ====="
python scripts/survival/run_sweep.py \
  --dataset coad \
  --gpus 0 1 2 3 4 5 6 \
  --jobs-per-gpu 1 \
  --configs meanmil_mist maxmil_mist abmil_mist clam_mist transmil_mist ilra_mist transformer_mil_mist dsmil_mist \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --folds 0 1 2 3 4 \
  --n-proto 32 \
  --skip-existing \
  --proto-dir "${PROTO_DIR}" \
  --split-names "train,test_tcgacoad,test_cptaccoad" \
  --results-dir-root "${RESULTS_DIR_ROOT}" \
  --data-root "${DATA_ROOTS[0]}" --data-root "${DATA_ROOTS[1]}" \
  || echo "WARNING: Phase 2 had failures."

echo "===== All COAD experiments finished! ====="
