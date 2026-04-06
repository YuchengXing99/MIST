# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase is forked from [PANTHER](https://github.com/mahmoodlab/PANTHER) (CVPR 2024). The core addition is **MIST** (Molecularly Informed Spatial Transcriptomics Transform) — a module that injects spatial transcriptomics priors into patch embeddings before MIL aggregation. All `*_mist.py` files are the novel contribution; the base models (ABMIL, TransMIL, ILRA, PANTHER, OT) are largely inherited from the original repo.

## Environment Setup

```bash
conda env create -f env.yaml
conda activate mist
```

Python 3.9, PyTorch 2.3.1, CUDA 12.1. Key dependencies: `faiss-gpu`, `scikit-survival`, `h5py`, `pot`, `nystrom_attention`, `einops`.

## Running Experiments

All training commands are launched from `src/` as the working directory using Python's `-m` flag:

```bash
cd src

# Step 1: Prototype construction (K-means on training patch features)
CUDA_VISIBLE_DEVICES=0 python -m training.main_prototype \
  --mode faiss --data_source FEAT_DIR --split_dir SPLIT_DIR \
  --split_names train --in_dim 1536 --n_proto 16 --n_proto_patches 1000000

# Step 2B: Classification downstream task
CUDA_VISIBLE_DEVICES=0 python -m training.main_classification \
  --data_source FEAT_DIR --split_dir SPLIT_DIR --split_names train,val,test \
  --task TASK_NAME --target_col LABEL_COL \
  --model_type ABMIL_MIST --model_config ABMIL_MIST_default \
  --in_dim 1536 --max_epochs 20 --lr 0.0001 --early_stopping --seed 1

# Step 2B: Survival downstream task
CUDA_VISIBLE_DEVICES=0 python -m training.main_survival \
  --model_type ABMIL_MIST --model_config ABMIL_MIST_default \
  --loss_fn cox ...

# Step 2A: Unsupervised slide embedding (PANTHER/OT only)
CUDA_VISIBLE_DEVICES=0 python -m training.main_embedding ...
```

Convenience shell scripts are in `src/scripts/`. Call them from `src/`:
```bash
./scripts/classification/abmil.sh 0 <task> <target_col> <split_dir> <split_names> <dataroots...>
./scripts/survival/panther.sh 0 <task> <target_col> <split_dir> <split_names> <dataroots...>
./scripts/prototype/clustering.sh 0 ...
```

## Data Organization

**Split CSVs** go in `src/splits/<task_name>/` (or `src/splits/<task_name>_k=<fold>/` for k-fold):
- `train.csv`, `val.csv`, `test.csv` — must contain `slide_id` or `FILENAME` column

**Patch features** must be pre-extracted (e.g., via CLAM/TRIDENT) and organized as:
```
<dataroot>/extracted_mag20x_patch256_fp/<extractor>/feats_h5/   # .h5 format
<dataroot>/extracted_mag20x_patch256_fp/<extractor>/feats_pt/   # .pt format
```
The feature directory path must end with `feats_h5/` or `feats_pt/`.

**Prototypes** are saved to `<split_dir>/prototypes/` after Step 1.
**Embeddings** are saved to `<split_dir>/embeddings/` after Step 2A.

## Architecture

### MIST Module (`src/mil_models/model_abmil_mist.py`)

`MolecularStainingTransform` is the core MIST layer, shared across all three MIST model variants:

```
Input: (B, N, D_in)
  → shared_proj → x_low (B, N, D_out)  +  prototypes → proto (K, D_out)
  → affinity = sigmoid(x_low · proto^T), then L1-normalize over K
  → per-prototype low-rank views: V(x_low) → U_k (for each k)
  → stained = Σ_k affinity_k * (x_low + U_k(V(x_low)))
  → x_out = x_low + stained  (skip connection)
Output: (B, N, D_out), affinity (B, N, K)
```

`init_prototypes(st_prototypes)` loads K ST-derived prototype vectors (shape `(K, in_dim)`) into the learnable `self.prototypes` parameter.

### MIST Model Variants

| Model | File | Config class | Config dir |
|-------|------|-------------|------------|
| `ABMIL_MIST` | `model_abmil_mist.py` | `ABMIL_MISTConfig` | `configs/ABMIL_MIST_default/` |
| `TransMIL_MIST` | `model_transmil_mist.py` | `TransMIL_MISTConfig` | `configs/TransMIL_MIST_default/` |
| `ILRA_MIST` | `model_ilra_mist.py` | `ILRA_MISTConfig` | `configs/ILRA_MIST_default/` |

MIST replaces the first linear projection in each base model. The `staining_affinity` tensor `(B, N, K)` is returned in the output dict for interpretability.

### Config System

Each model has a JSON config under `src/configs/<MODEL>_default/config.json`. Configs are loaded via `PretrainedConfig.from_pretrained(path, update_dict)`, where `update_dict` overrides keys from CLI args (e.g., `in_dim`, `n_classes`). MIST-specific config fields are `n_prototypes` (K) and `mist_rank` (low-rank dim r).

### Model Factory

`src/mil_models/model_factory.py` has two factory functions:
- `create_downstream_model(args, mode)` — for ABMIL/TransMIL/ILRA variants (classification/survival)
- `create_embedding_model(args, mode)` — for PANTHER/OT/H2T/ProtoCount (unsupervised embeddings)

Model selection is driven by `args.model_type` (string, e.g. `'ABMIL_MIST'`) and `args.model_config` (config dir name, e.g. `'ABMIL_MIST_default'`).

### Training Loop

`src/training/trainer.py:train()` handles both classification and survival modes. It calls `prepare_emb()` for prototype-based models (PANTHER/OT/H2T/ProtoCount) to build slide representations first, then trains a downstream head. MIL models (ABMIL, TransMIL, ILRA and their MIST variants) operate directly on patch features.

### Adding a New MIST Variant

1. Create `src/mil_models/model_<NAME>_mist.py` — import `MolecularStainingTransform` from `model_abmil_mist.py`, apply it at the start of `forward_no_loss()`
2. Add `<NAME>_MISTConfig` dataclass in `model_configs.py` with `n_prototypes` and `mist_rank` fields
3. Create `src/configs/<NAME>_MIST_default/config.json`
4. Register in `model_factory.py:create_downstream_model()` and in `src/mil_models/__init__.py`
