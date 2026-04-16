#!/bin/bash
# =============================================================================
#  LUAD Survival — NLST inference-only using existing LUAD+CPTAC checkpoints
#  Reuses ckpts trained by launch_luad_cptac.sh; no retraining.
#  7 GPUs (0-6), 3 jobs per GPU, 10 seeds, 5 folds, all baselines + MIST.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
DATA_ROOTS=(
  "/hpc/home/e0950223/MIST_data/survival/TCGA-LUAD"
  "/hpc/home/e0950223/MIST_data/survival/CPTAC-LUAD"
  "/hpc/home/e0950223/MIST_data/survival/NLST"
)
PROTO_DIR="/hpc/home/e0950223/MIST_data/mol_vis_by_organ/Lung/prototypes_by_k"
CKPT_DIR_ROOT="/hpc/home/e0950223/MIST/results_survival_luad"
RESULTS_DIR_ROOT="results_survival_luad_nlst"
# ====================================================================

cd "${SRC_DIR}"

# --- Phase 1: Baselines (eval-only) ---
echo "===== LUAD Phase 1: Baselines (eval-only on NLST) ====="
python scripts/survival/run_sweep.py \
  --dataset luad \
  --gpus 0 1 2 3 4 5 6 \
  --jobs-per-gpu 3 \
  --configs meanmil maxmil abmil clam transmil ilra transformer_mil dsmil \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --folds 0 1 2 3 4 \
  --skip-existing \
  --eval-only \
  --ckpt-dir-root "${CKPT_DIR_ROOT}" \
  --split-names "train,test_tcgaluad,test_cptacluad,test_nlstluad" \
  --results-dir-root "${RESULTS_DIR_ROOT}" \
  --data-root "${DATA_ROOTS[0]}" --data-root "${DATA_ROOTS[1]}" --data-root "${DATA_ROOTS[2]}" \
  || echo "WARNING: Phase 1 had failures, continuing to Phase 2..."

# --- Phase 2: MIST variants (eval-only) ---
echo "===== LUAD Phase 2: MIST (eval-only on NLST) ====="
python scripts/survival/run_sweep.py \
  --dataset luad \
  --gpus 0 1 2 3 4 5 6 \
  --jobs-per-gpu 3 \
  --configs meanmil_mist maxmil_mist abmil_mist clam_mist transmil_mist ilra_mist transformer_mil_mist dsmil_mist \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --folds 0 1 2 3 4 \
  --n-proto 8 \
  --skip-existing \
  --eval-only \
  --ckpt-dir-root "${CKPT_DIR_ROOT}" \
  --proto-dir "${PROTO_DIR}" \
  --split-names "train,test_tcgaluad,test_cptacluad,test_nlstluad" \
  --results-dir-root "${RESULTS_DIR_ROOT}" \
  --data-root "${DATA_ROOTS[0]}" --data-root "${DATA_ROOTS[1]}" --data-root "${DATA_ROOTS[2]}" \
  || echo "WARNING: Phase 2 had failures."

echo "===== All LUAD+NLST eval-only experiments finished! ====="
