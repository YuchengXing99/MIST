#!/usr/bin/env bash
set -euo pipefail

# Molecular-vs-visual analysis for paired ST expression + UNI embeddings.
#
# Mode A (recommended, multi-file paired_h5ad dir):
#   bash run_mol_vis_analysis.sh [PAIRED_H5AD_DIR] [OUT_DIR] [SELECTED_K] [SEED] [SAMPLE_COL] [SPATIAL_KEY] [PAIRED_EMBED_KEY] [GENE_ALIGN]
#
# Example A:
#   bash run_mol_vis_analysis.sh \
#     /data3/yucheng/maple_like_256_v130/embeddings_uni2h/paired_h5ad \
#     /data3/yucheng/maple_like_256_v130/mol_vis_analysis \
#     16 42 sample_id spatial
#
# Mode B (legacy single-file + npy):
#   bash run_mol_vis_analysis.sh [ADATA_H5AD] [EMB_NPY] [EMB_BARCODES] [OUT_DIR] [SELECTED_K] [SEED] [SAMPLE_COL] [SPATIAL_KEY] [EMB_SAMPLE_IDS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${1:-}" || -d "${1:-}" ]]; then
  # Mode A: paired_h5ad directory
  PAIRED_H5AD_DIR="${1:-/data3/yucheng/maple_like_256_v130/embeddings_uni2h/paired_h5ad}"
  OUT_DIR="${2:-/data3/yucheng/maple_like_256_v130/mol_vis_analysis}"
  SELECTED_K="${3:-16}"
  SEED="${4:-42}"
  SAMPLE_COL="${5:-sample_id}"
  SPATIAL_KEY="${6:-spatial}"
  PAIRED_EMBED_KEY="${7:-}"
  GENE_ALIGN="${8:-union}"

  CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_molecular_visual_pairs.py"
    --paired-h5ad-dir "${PAIRED_H5AD_DIR}"
    --gene-align "${GENE_ALIGN}"
    --sample-col "${SAMPLE_COL}"
    --spatial-key "${SPATIAL_KEY}"
    --out-dir "${OUT_DIR}"
    --target-sum 1e4
    --gene-min-spot-frac 0.10
    --hvg-top 2000
    --neighbor-k 6
    --k-list 4,8,16,32,64
    --selected-k "${SELECTED_K}"
    --silhouette-sample 50000
    --umap-sample 50000
    --top-n-genes 10
    --heatmap-top-genes 50
    --seed "${SEED}"
  )

  if [[ -n "${PAIRED_EMBED_KEY}" ]]; then
    CMD+=(--paired-embedding-key "${PAIRED_EMBED_KEY}")
  fi

  "${CMD[@]}"
else
  # Mode B: legacy single-file + npy
  ADATA_H5AD="${1}"
  EMB_NPY="${2}"
  EMB_BARCODES="${3}"
  OUT_DIR="${4:-/data3/yucheng/maple_like_256_v130/mol_vis_analysis}"
  SELECTED_K="${5:-16}"
  SEED="${6:-42}"
  SAMPLE_COL="${7:-sample_id}"
  SPATIAL_KEY="${8:-spatial}"
  EMB_SAMPLE_IDS="${9:-}"

  CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_molecular_visual_pairs.py"
    --adata-h5ad "${ADATA_H5AD}"
    --embedding-npy "${EMB_NPY}"
    --embedding-barcodes "${EMB_BARCODES}"
    --sample-col "${SAMPLE_COL}"
    --spatial-key "${SPATIAL_KEY}"
    --out-dir "${OUT_DIR}"
    --target-sum 1e4
    --gene-min-spot-frac 0.10
    --hvg-top 2000
    --neighbor-k 6
    --k-list 4,8,16,32,64
    --selected-k "${SELECTED_K}"
    --silhouette-sample 50000
    --umap-sample 50000
    --top-n-genes 10
    --heatmap-top-genes 50
    --seed "${SEED}"
  )

  if [[ -n "${EMB_SAMPLE_IDS}" ]]; then
    CMD+=(--embedding-sample-ids "${EMB_SAMPLE_IDS}")
  fi

  "${CMD[@]}"
fi
