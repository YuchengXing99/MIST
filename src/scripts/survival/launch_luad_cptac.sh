#!/bin/bash
# =============================================================================
#  LUAD Survival — TCGA internal + CPTAC external (no NLST)
#  7 GPUs (0-6), 2 jobs per GPU, 10 seeds, 5 folds
#  All baselines + MIST variants
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
DATA_ROOTS=("/path/to/TCGA-LUAD" "/path/to/CPTAC-LUAD")
PROTO_DIR="/path/to/prototypes"
RESULTS_DIR_ROOT="results_survival_luad"
# ====================================================================

cd "${SRC_DIR}"

# --- Phase 1: Baselines ---
echo "===== LUAD Phase 1: Baselines ====="
python scripts/survival/run_sweep.py \
  --dataset luad \
  --gpus 0 1 2 3 4 5 6 \
  --jobs-per-gpu 2 \
  --configs meanmil maxmil abmil clam transmil ilra transformer_mil dsmil \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --folds 0 1 2 3 4 \
  --split-names "train,test_tcgaluad,test_cptacluad" \
  --results-dir-root "${RESULTS_DIR_ROOT}" \
  --data-root "${DATA_ROOTS[0]}" --data-root "${DATA_ROOTS[1]}"

# --- Phase 2: MIST variants ---
echo "===== LUAD Phase 2: MIST ====="
python scripts/survival/run_sweep.py \
  --dataset luad \
  --gpus 0 1 2 3 4 5 6 \
  --jobs-per-gpu 2 \
  --configs meanmil_mist maxmil_mist abmil_mist clam_mist transmil_mist ilra_mist transformer_mil_mist dsmil_mist \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --folds 0 1 2 3 4 \
  --n-proto 16 \
  --proto-dir "${PROTO_DIR}" \
  --split-names "train,test_tcgaluad,test_cptacluad" \
  --results-dir-root "${RESULTS_DIR_ROOT}" \
  --data-root "${DATA_ROOTS[0]}" --data-root "${DATA_ROOTS[1]}"

echo "===== All LUAD experiments finished! ====="
