#!/usr/bin/env python
"""Molecular-vs-visual analysis for paired ST expression and UNI embeddings.

Pipeline in this script (aligned to user request):
- Gene/spot filtering on raw expression
- Total-count normalization (target_sum=1e4) + log1p
- Seurat v3 HVG selection (top 2000 by default)
- Local smoothing with Visium-like 6-neighbor averaging within each sample
- K-means scan on molecular expression and silhouette selection
- Differential genes, heatmap, UMAP on embeddings, ARI, prototypes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze paired molecular + visual spot data.")
    p.add_argument(
        "--paired-h5ad-dir",
        type=Path,
        default=None,
        help="Directory containing many per-sample paired .h5ad files (e.g., embeddings_uni2h/paired_h5ad).",
    )
    p.add_argument("--paired-embedding-key", type=str, default=None, help="Optional obsm key for embeddings inside paired h5ad.")
    p.add_argument(
        "--gene-align",
        type=str,
        default="union",
        choices=["union", "intersection"],
        help="How to align gene axes across paired_h5ad files.",
    )

    p.add_argument("--adata-h5ad", type=Path, default=None, help="Path to expression .h5ad (single-file mode).")
    p.add_argument("--embedding-npy", type=Path, default=None, help="Path to embedding .npy (N x D, single-file mode).")
    p.add_argument(
        "--embedding-barcodes",
        type=Path,
        default=None,
        help="Path to embedding barcodes (.npy/.csv/.tsv/.txt/.parquet, single-file mode).",
    )
    p.add_argument(
        "--embedding-sample-ids",
        type=Path,
        default=None,
        help="Optional sample_id file for embeddings (same row order as embedding-npy).",
    )
    p.add_argument(
        "--expr-barcode-col",
        type=str,
        default=None,
        help="obs column for expression barcode. Default uses obs_names.",
    )
    p.add_argument(
        "--sample-col",
        type=str,
        default="sample_id",
        help="obs column used as sample id for smoothing and optional alignment key.",
    )
    p.add_argument(
        "--spatial-key",
        type=str,
        default="spatial",
        help="obsm key containing Nx2 spot coordinates.",
    )
    p.add_argument("--array-row-col", type=str, default="array_row,array_col", help="Fallback obs columns for coordinates.")

    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")

    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--gene-min-spot-frac", type=float, default=0.10, help="Keep genes expressed in >= this fraction of spots.")
    p.add_argument("--hvg-top", type=int, default=2000)
    p.add_argument(
        "--hvg-flavor",
        type=str,
        default="seurat_v3",
        choices=["seurat_v3", "seurat", "cell_ranger"],
        help="Primary HVG method. Keep seurat_v3 for SEAL-like setting.",
    )
    p.add_argument(
        "--hvg-fallback-flavor",
        type=str,
        default="cell_ranger",
        choices=["none", "seurat", "cell_ranger"],
        help="Fallback HVG flavor when primary method fails (e.g. missing scikit-misc for seurat_v3).",
    )
    p.add_argument("--neighbor-k", type=int, default=6, help="Local smoothing neighbor count (Visium six-neighborhood).")

    p.add_argument("--k-list", type=str, default="4,8,16,32,64")
    p.add_argument("--selected-k", type=int, default=16, help="If <=0, pick best silhouette K.")
    p.add_argument("--top-n-genes", type=int, default=10, help="Top upregulated genes per cluster.")
    p.add_argument("--heatmap-top-genes", type=int, default=50)

    p.add_argument("--silhouette-sample", type=int, default=50000)
    p.add_argument("--umap-sample", type=int, default=50000)

    p.add_argument("--kmeans-batch-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_1d_values(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return np.asarray(arr).astype(str).reshape(-1)
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] < 1:
            raise ValueError(f"No columns found in {path}")
        return df.iloc[:, 0].astype(str).to_numpy()
    if suffix == ".txt":
        vals = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return np.asarray(vals, dtype=str)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if df.shape[1] < 1:
            raise ValueError(f"No columns found in {path}")
        return df.iloc[:, 0].astype(str).to_numpy()
    raise ValueError(f"Unsupported file format: {path}")


def resolve_input_mode(args: argparse.Namespace) -> str:
    has_paired_dir = args.paired_h5ad_dir is not None
    has_single_mode_arg = any(
        x is not None for x in [args.adata_h5ad, args.embedding_npy, args.embedding_barcodes]
    )
    if has_paired_dir and has_single_mode_arg:
        raise ValueError(
            "Choose one input mode only: either --paired-h5ad-dir OR "
            "--adata-h5ad + --embedding-npy + --embedding-barcodes."
        )
    if has_paired_dir:
        if not args.paired_h5ad_dir.exists():
            raise FileNotFoundError(f"paired_h5ad directory not found: {args.paired_h5ad_dir}")
        return "paired_dir"

    if args.adata_h5ad is None or args.embedding_npy is None or args.embedding_barcodes is None:
        raise ValueError(
            "Single-file mode requires --adata-h5ad, --embedding-npy and --embedding-barcodes."
        )
    return "single"


def infer_embedding_key(adata: sc.AnnData, preferred_key: str | None) -> str:
    if preferred_key:
        if preferred_key not in adata.obsm:
            raise KeyError(f"Embedding key '{preferred_key}' not found in adata.obsm.")
        return preferred_key

    if "embedding_key" in adata.uns:
        k = str(adata.uns["embedding_key"])
        if k in adata.obsm:
            return k

    for k in adata.obsm.keys():
        arr = adata.obsm[k]
        if getattr(arr, "ndim", None) == 2 and arr.shape[0] == adata.n_obs:
            return str(k)

    raise KeyError("Cannot infer embedding key from obsm. Pass --paired-embedding-key explicitly.")


def infer_sample_id_from_filename(path: Path) -> str:
    stem = path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        suffix = parts[1].lower()
        if len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix):
            return parts[0]
    return stem


def load_from_paired_h5ad_dir(
    paired_dir: Path,
    out_dir: Path,
    sample_col: str,
    spatial_key: str,
    array_row_col: Tuple[str, str],
    expr_barcode_col: str | None,
    paired_embedding_key: str | None,
    gene_align: str,
) -> Tuple[sc.AnnData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    files = sorted(paired_dir.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No .h5ad files found under {paired_dir}")

    print(f"[INFO] paired_h5ad files: {len(files)}")

    n_total = 0
    emb_dim = None
    first_gene_order = None
    intersection_set = None
    union_order = []
    union_seen = set()

    for fp in tqdm(files, desc="scan_paired_h5ad"):
        a = sc.read_h5ad(fp, backed="r")
        try:
            n_total += int(a.n_obs)

            genes_raw = a.var_names.astype(str).to_numpy()
            # De-duplicate by first occurrence to avoid ambiguous gene columns.
            genes = pd.Index(genes_raw).drop_duplicates(keep="first").to_numpy(dtype=str)
            gset = set(genes.tolist())

            if first_gene_order is None:
                first_gene_order = genes
                intersection_set = set(genes.tolist())
            else:
                intersection_set &= gset

            for g in genes:
                if g not in union_seen:
                    union_seen.add(g)
                    union_order.append(g)

            ekey = infer_embedding_key(a, paired_embedding_key)
            shape = tuple(a.obsm[ekey].shape)
            if len(shape) != 2 or shape[0] != a.n_obs:
                raise ValueError(
                    f"Embedding matrix shape mismatch in {fp}: {shape}, expected ({a.n_obs}, D)."
                )
            if emb_dim is None:
                emb_dim = int(shape[1])
            elif emb_dim != int(shape[1]):
                raise ValueError(f"Embedding dim mismatch in {fp}: {shape[1]} vs {emb_dim}")
        finally:
            if getattr(a, "isbacked", False):
                a.file.close()

    if emb_dim is None or first_gene_order is None:
        raise RuntimeError("Failed to infer genes/embedding dim from paired_h5ad files.")
    if gene_align == "union":
        target_genes = np.asarray(union_order, dtype=str)
    elif gene_align == "intersection":
        target_genes = np.asarray([g for g in first_gene_order if g in intersection_set], dtype=str)
    else:
        raise ValueError(f"Unsupported gene_align: {gene_align}")

    if len(target_genes) == 0:
        raise ValueError(
            "Gene alignment produced an empty gene set. "
            "Try --gene-align union."
        )
    print(f"[INFO] gene_align={gene_align}, target genes={len(target_genes)}")

    tmp_emb_path = out_dir / "_tmp_embeddings_from_paired_h5ad.npy"
    emb_mem = np.lib.format.open_memmap(
        str(tmp_emb_path),
        mode="w+",
        dtype="float32",
        shape=(n_total, emb_dim),
    )

    x_blocks = []
    coords_blocks = []
    sample_blocks = []
    barcode_blocks = []
    gene_to_col = {g: i for i, g in enumerate(target_genes)}

    ptr = 0
    for fp in tqdm(files, desc="load_paired_h5ad"):
        a = sc.read_h5ad(fp)

        genes_raw = a.var_names.astype(str).to_numpy()
        keep_unique = ~pd.Index(genes_raw).duplicated(keep="first")
        if not np.all(keep_unique):
            dropped = int((~keep_unique).sum())
            print(f"[WARN] {fp.name}: dropped {dropped} duplicated genes by keeping first occurrence.")
            a = a[:, keep_unique].copy()
        genes = a.var_names.astype(str).to_numpy()

        x_blk = a.X
        if sparse.issparse(x_blk):
            x_blk = x_blk.tocsr().astype(np.float32)
        else:
            x_blk = sparse.csr_matrix(np.asarray(x_blk, dtype=np.float32))

        if len(genes) == len(target_genes) and np.array_equal(genes, target_genes):
            x_aligned = x_blk
        else:
            keep_cols = np.asarray([g in gene_to_col for g in genes], dtype=bool)
            if keep_cols.any():
                x_keep = x_blk[:, keep_cols]
                genes_keep = genes[keep_cols]
                new_cols = np.asarray([gene_to_col[g] for g in genes_keep], dtype=np.int64)
                coo = x_keep.tocoo()
                remapped_cols = new_cols[coo.col]
                x_aligned = sparse.csr_matrix(
                    (coo.data, (coo.row, remapped_cols)),
                    shape=(x_blk.shape[0], len(target_genes)),
                    dtype=np.float32,
                )
            else:
                x_aligned = sparse.csr_matrix((x_blk.shape[0], len(target_genes)), dtype=np.float32)

        coords_blk = get_coords(a, spatial_key, array_row_col).astype(np.float32)

        if expr_barcode_col and expr_barcode_col in a.obs.columns:
            bc = a.obs[expr_barcode_col].astype(str).to_numpy()
        elif "barcode" in a.obs.columns:
            bc = a.obs["barcode"].astype(str).to_numpy()
        else:
            bc = a.obs_names.astype(str).to_numpy()

        if sample_col in a.obs.columns:
            sid = a.obs[sample_col].astype(str).to_numpy()
        else:
            sid = np.asarray([infer_sample_id_from_filename(fp)] * a.n_obs, dtype=str)

        ekey = infer_embedding_key(a, paired_embedding_key)
        e_blk = np.asarray(a.obsm[ekey], dtype=np.float32)

        n = int(a.n_obs)
        emb_mem[ptr: ptr + n] = e_blk
        ptr += n

        x_blocks.append(x_aligned)
        coords_blocks.append(coords_blk)
        sample_blocks.append(sid)
        barcode_blocks.append(np.asarray(bc, dtype=str))

    x_all = sparse.vstack(x_blocks, format="csr")
    coords_all = np.vstack(coords_blocks).astype(np.float32)
    sample_all = np.concatenate(sample_blocks).astype(str)
    barcode_all = np.concatenate(barcode_blocks).astype(str)

    adata = sc.AnnData(X=x_all)
    adata.var_names = target_genes
    adata.obs[sample_col] = sample_all
    adata.obs["barcode"] = barcode_all
    if expr_barcode_col:
        adata.obs[expr_barcode_col] = barcode_all
    adata.obs_names = np.char.add(np.char.add(sample_all, "||"), barcode_all)
    adata.obsm[spatial_key] = coords_all

    emb_idx = np.arange(n_total, dtype=np.int64)
    return adata, emb_mem, emb_idx, barcode_all, sample_all, str(tmp_emb_path)


def get_expr_barcodes_and_samples(adata: sc.AnnData, barcode_col: str | None, sample_col: str) -> Tuple[np.ndarray, np.ndarray]:
    if barcode_col and barcode_col in adata.obs.columns:
        barcodes = adata.obs[barcode_col].astype(str).to_numpy()
    else:
        barcodes = adata.obs_names.astype(str).to_numpy()

    if sample_col in adata.obs.columns:
        sample_ids = adata.obs[sample_col].astype(str).to_numpy()
    else:
        sample_ids = np.asarray(["sample0"] * adata.n_obs, dtype=str)
    return barcodes, sample_ids


def get_coords(adata: sc.AnnData, spatial_key: str, array_row_col: Tuple[str, str]) -> np.ndarray:
    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"obsm['{spatial_key}'] must be Nx2+, got shape {coords.shape}")
        return coords[:, :2]

    if "patch_x" in adata.obs.columns and "patch_y" in adata.obs.columns:
        return adata.obs[["patch_x", "patch_y"]].to_numpy(dtype=np.float32)

    row_col_0, row_col_1 = array_row_col
    if row_col_0 in adata.obs.columns and row_col_1 in adata.obs.columns:
        return adata.obs[[row_col_0, row_col_1]].to_numpy(dtype=np.float32)

    raise KeyError(
        f"Cannot find coordinates: neither obsm['{spatial_key}'] nor obs columns {array_row_col} are available."
    )


def maybe_make_keys(barcodes: np.ndarray, sample_ids: np.ndarray | None) -> np.ndarray:
    if sample_ids is None:
        return barcodes.astype(str)
    return np.char.add(np.char.add(sample_ids.astype(str), "||"), barcodes.astype(str))


def has_duplicates(arr: np.ndarray) -> bool:
    return len(arr) != len(np.unique(arr))


def align_indices(
    expr_barcodes: np.ndarray,
    expr_samples: np.ndarray,
    emb_barcodes: np.ndarray,
    emb_samples: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefer sample+barcode composite alignment when possible.
    use_composite = emb_samples is not None
    if use_composite:
        expr_keys = maybe_make_keys(expr_barcodes, expr_samples)
        emb_keys = maybe_make_keys(emb_barcodes, emb_samples)
    else:
        expr_keys = expr_barcodes
        emb_keys = emb_barcodes

    if has_duplicates(expr_keys):
        mode = "sample+barcode" if use_composite else "barcode"
        raise ValueError(
            f"Expression keys are not unique under {mode}. "
            "Provide --embedding-sample-ids and ensure sample_id exists in h5ad, "
            "or make barcode globally unique."
        )
    if has_duplicates(emb_keys):
        mode = "sample+barcode" if use_composite else "barcode"
        raise ValueError(f"Embedding keys are not unique under {mode}.")

    emb_pos = {k: i for i, k in enumerate(emb_keys)}
    expr_keep = []
    emb_idx = []
    for i, k in enumerate(expr_keys):
        j = emb_pos.get(k)
        if j is not None:
            expr_keep.append(i)
            emb_idx.append(j)

    if not expr_keep:
        raise RuntimeError("No overlap found between expression and embedding keys.")

    expr_keep = np.asarray(expr_keep, dtype=np.int64)
    emb_idx = np.asarray(emb_idx, dtype=np.int64)
    matched_barcodes = expr_barcodes[expr_keep]
    return expr_keep, emb_idx, matched_barcodes


def preprocess_expression(
    adata: sc.AnnData,
    target_sum: float,
    gene_min_spot_frac: float,
    hvg_top: int,
    hvg_flavor: str,
    hvg_fallback_flavor: str,
) -> Tuple[sc.AnnData, np.ndarray]:
    # Filter empty spots (total count == 0) on raw matrix.
    spot_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    keep_idx = np.where(spot_counts > 0)[0]
    keep_spots = np.zeros(adata.n_obs, dtype=bool)
    keep_spots[keep_idx] = True
    adata = adata[keep_idx].copy()

    min_cells = int(np.ceil(gene_min_spot_frac * adata.n_obs))
    min_cells = max(min_cells, 1)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Gene filtering can create new zero-count spots; drop them before normalization.
    spot_counts_after_gene_filter = np.asarray(adata.X.sum(axis=1)).ravel()
    keep_spots_after_gene_filter = spot_counts_after_gene_filter > 0
    if not np.all(keep_spots_after_gene_filter):
        dropped = int((~keep_spots_after_gene_filter).sum())
        print(f"[WARN] Dropped {dropped} zero-count spots after gene filtering.")
        adata = adata[keep_spots_after_gene_filter].copy()
        keep_idx = keep_idx[keep_spots_after_gene_filter]
        keep_spots = np.zeros(keep_spots.shape[0], dtype=bool)
        keep_spots[keep_idx] = True

    # Keep raw counts for Seurat-v3 HVG selection.
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    n_top = min(int(hvg_top), int(adata.n_vars))
    if n_top < int(hvg_top):
        print(
            f"[WARN] Requested top {hvg_top} HVGs but only {adata.n_vars} genes remain after filtering. "
            f"Using n_top_genes={n_top}."
        )

    def _run_hvg(flavor: str) -> None:
        kwargs = dict(
            n_top_genes=n_top,
            subset=True,
        )
        if flavor == "seurat_v3":
            kwargs["layer"] = "counts"
        sc.pp.highly_variable_genes(
            adata,
            flavor=flavor,
            **kwargs,
        )

    try:
        _run_hvg(hvg_flavor)
        print(f"[INFO] HVG flavor used: {hvg_flavor}")
    except Exception as exc:
        can_fallback = (
            hvg_fallback_flavor != "none"
            and hvg_fallback_flavor != hvg_flavor
        )
        if not can_fallback:
            raise RuntimeError(
                f"HVG selection failed with flavor={hvg_flavor}. "
                "If using seurat_v3, install `scikit-misc` (pip install scikit-misc), "
                "or set --hvg-fallback-flavor cell_ranger."
            ) from exc

        print(
            f"[WARN] HVG flavor '{hvg_flavor}' failed ({exc.__class__.__name__}: {exc}). "
            f"Falling back to '{hvg_fallback_flavor}'."
        )
        try:
            _run_hvg(hvg_fallback_flavor)
            print(f"[INFO] HVG flavor used: {hvg_fallback_flavor} (fallback)")
        except Exception as exc2:
            raise RuntimeError(
                f"HVG selection failed for both primary ({hvg_flavor}) and fallback "
                f"({hvg_fallback_flavor})."
            ) from exc2

    return adata, keep_spots


def smooth_within_sample(
    x: sparse.csr_matrix,
    coords: np.ndarray,
    sample_ids: np.ndarray,
    neighbor_k: int,
) -> sparse.csr_matrix:
    n = x.shape[0]
    order = np.argsort(sample_ids, kind="mergesort")
    x_ord = x[order]
    c_ord = coords[order]
    s_ord = sample_ids[order]

    blocks = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and s_ord[end] == s_ord[start]:
            end += 1

        x_blk = x_ord[start:end]
        c_blk = c_ord[start:end]
        m = end - start

        if m <= 1:
            blocks.append(x_blk)
            start = end
            continue

        k = min(neighbor_k + 1, m)  # include self
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(c_blk)
        neigh = nn.kneighbors(return_distance=False)

        rows = np.repeat(np.arange(m), k)
        cols = neigh.reshape(-1)
        data = np.full(rows.shape[0], 1.0 / float(k), dtype=np.float32)
        s_mat = sparse.csr_matrix((data, (rows, cols)), shape=(m, m))

        x_sm = s_mat @ x_blk
        blocks.append(x_sm.tocsr())
        start = end

    x_sm_ord = sparse.vstack(blocks, format="csr")
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)
    x_sm = x_sm_ord[inv]
    return x_sm.tocsr()


def cluster_means(x: sparse.csr_matrix, labels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    g = x.shape[1]
    means = np.zeros((k, g), dtype=np.float32)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        means[c] = np.asarray(x[idx].mean(axis=0)).ravel().astype(np.float32)
    return means, counts


def differential_up_genes(means: np.ndarray, counts: np.ndarray, genes: np.ndarray, top_n: int) -> Tuple[pd.DataFrame, np.ndarray]:
    k, g = means.shape
    total_n = counts.sum()
    weighted_sum = (means * counts[:, None]).sum(axis=0)
    delta = np.zeros_like(means, dtype=np.float32)

    rows = []
    for c in range(k):
        n_in = counts[c]
        n_out = total_n - n_in
        if n_in <= 0 or n_out <= 0:
            continue
        mean_out = (weighted_sum - means[c] * n_in) / max(n_out, 1)
        d = means[c] - mean_out
        delta[c] = d

        rank = np.argsort(d)[::-1][:top_n]
        for r, gi in enumerate(rank, start=1):
            rows.append(
                {
                    "cluster": int(c),
                    "rank": int(r),
                    "gene": str(genes[gi]),
                    "delta_mean": float(d[gi]),
                    "mean_in": float(means[c, gi]),
                    "mean_out": float(mean_out[gi]),
                }
            )

    return pd.DataFrame(rows), delta


def draw_silhouette_curve(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(df["k"], df["silhouette"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.title("Silhouette scan on molecular expression")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def draw_heatmap(
    means: np.ndarray,
    genes: np.ndarray,
    delta: np.ndarray,
    top_n: int,
    out_png: Path,
    out_matrix_csv: Path,
) -> None:
    score = delta.max(axis=0)
    top_idx = np.argsort(score)[::-1][:top_n]

    mat = means[:, top_idx]
    sel_genes = genes[top_idx]

    # z-score by gene for visibility
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True) + 1e-8
    z = (mat - mu) / sd

    row_order = np.arange(z.shape[0])
    col_order = np.arange(z.shape[1])

    if z.shape[0] > 1:
        row_order = leaves_list(linkage(z, method="average", metric="euclidean"))
    if z.shape[1] > 1:
        col_order = leaves_list(linkage(z.T, method="average", metric="euclidean"))

    z_ord = z[row_order][:, col_order]
    genes_ord = sel_genes[col_order]

    plot_df = pd.DataFrame(
        mat[row_order][:, col_order],
        index=[f"C{i}" for i in row_order],
        columns=genes_ord,
    )
    plot_df.to_csv(out_matrix_csv)

    fig_w = max(12, top_n * 0.28)
    fig_h = max(4, z.shape[0] * 0.35)
    plt.figure(figsize=(fig_w, fig_h), dpi=150)
    im = plt.imshow(z_ord, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, fraction=0.025, pad=0.01, label="z-score")
    plt.xticks(np.arange(len(genes_ord)), genes_ord, rotation=90, fontsize=8)
    plt.yticks(np.arange(z_ord.shape[0]), [f"C{i}" for i in row_order], fontsize=9)
    plt.title(f"Cluster x top {top_n} differential genes (hierarchical order)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def iter_embedding_batches(emb_mem: np.ndarray, emb_idx: np.ndarray, bs: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    n = len(emb_idx)
    for start in range(0, n, bs):
        end = min(start + bs, n)
        pos = np.arange(start, end, dtype=np.int64)
        arr = np.asarray(emb_mem[emb_idx[start:end]], dtype=np.float32)
        yield pos, arr


def fit_predict_embedding_kmeans(emb_mem: np.ndarray, emb_idx: np.ndarray, k: int, seed: int, bs: int) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=bs, n_init=10)
    for _, batch in tqdm(iter_embedding_batches(emb_mem, emb_idx, bs), desc="fit_visual_kmeans"):
        km.partial_fit(batch)

    y = np.empty(len(emb_idx), dtype=np.int32)
    for pos, batch in tqdm(iter_embedding_batches(emb_mem, emb_idx, bs), desc="predict_visual_kmeans"):
        y[pos] = km.predict(batch)
    return y


def compute_embedding_prototypes(emb_mem: np.ndarray, emb_idx: np.ndarray, labels: np.ndarray, k: int, bs: int) -> np.ndarray:
    dim = int(emb_mem.shape[1])
    sums = np.zeros((k, dim), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)

    for pos, batch in tqdm(iter_embedding_batches(emb_mem, emb_idx, bs), desc="compute_prototypes"):
        y = labels[pos]
        for c in np.unique(y):
            m = y == c
            sums[c] += batch[m].sum(axis=0)
            counts[c] += int(m.sum())

    if np.any(counts == 0):
        miss = np.where(counts == 0)[0].tolist()
        raise RuntimeError(f"Empty clusters in molecular labels: {miss}")

    return (sums / counts[:, None]).astype(np.float32)


def draw_umap(emb: np.ndarray, labels: np.ndarray, out_png: Path, seed: int) -> None:
    try:
        import umap
    except Exception as exc:
        raise ImportError("Please install umap-learn: pip install umap-learn") from exc

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=seed)
    z = reducer.fit_transform(emb)

    plt.figure(figsize=(7, 6), dpi=150)
    sca = plt.scatter(z[:, 0], z[:, 1], c=labels, s=2, cmap="tab20", alpha=0.8, linewidths=0)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("Patch embedding UMAP colored by molecular clusters")
    plt.colorbar(sca, fraction=0.03, pad=0.01, label="Molecular cluster")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    if not k_values:
        raise ValueError("k-list is empty")

    array_row_col = tuple(x.strip() for x in args.array_row_col.split(","))
    if len(array_row_col) != 2:
        raise ValueError("--array-row-col must contain exactly two comma-separated column names")

    input_mode = resolve_input_mode(args)
    tmp_emb_path = None

    if input_mode == "paired_dir":
        print("[INFO] Loading paired_h5ad directory...")
        adata_raw, emb_mem, emb_idx, matched_barcodes, expr_samples, tmp_emb_path = load_from_paired_h5ad_dir(
            paired_dir=args.paired_h5ad_dir,
            out_dir=out_dir,
            sample_col=args.sample_col,
            spatial_key=args.spatial_key,
            array_row_col=array_row_col,
            expr_barcode_col=args.expr_barcode_col,
            paired_embedding_key=args.paired_embedding_key,
            gene_align=args.gene_align,
        )
    else:
        print("[INFO] Loading expression h5ad...")
        adata_raw = sc.read_h5ad(args.adata_h5ad)
        expr_barcodes_raw, expr_samples_raw = get_expr_barcodes_and_samples(
            adata_raw, args.expr_barcode_col, args.sample_col
        )

        print("[INFO] Loading embeddings...")
        emb_mem = np.load(args.embedding_npy, mmap_mode="r")
        if emb_mem.ndim != 2:
            raise ValueError(f"Embedding must be 2D, got shape {emb_mem.shape}")

        emb_barcodes = load_1d_values(args.embedding_barcodes)
        emb_samples = load_1d_values(args.embedding_sample_ids) if args.embedding_sample_ids else None

        if emb_mem.shape[0] != len(emb_barcodes):
            raise ValueError(
                f"embedding rows ({emb_mem.shape[0]}) != embedding barcodes ({len(emb_barcodes)})"
            )
        if emb_samples is not None and len(emb_samples) != emb_mem.shape[0]:
            raise ValueError(
                f"embedding sample_id rows ({len(emb_samples)}) != embedding rows ({emb_mem.shape[0]})"
            )

        expr_keep, emb_idx, matched_barcodes = align_indices(
            expr_barcodes=expr_barcodes_raw,
            expr_samples=expr_samples_raw,
            emb_barcodes=emb_barcodes,
            emb_samples=emb_samples,
        )
        adata_raw = adata_raw[expr_keep].copy()
        expr_samples = expr_samples_raw[expr_keep]

    adata, keep_spots = preprocess_expression(
        adata=adata_raw,
        target_sum=args.target_sum,
        gene_min_spot_frac=args.gene_min_spot_frac,
        hvg_top=args.hvg_top,
        hvg_flavor=args.hvg_flavor,
        hvg_fallback_flavor=args.hvg_fallback_flavor,
    )

    emb_idx = emb_idx[keep_spots]
    matched_barcodes = matched_barcodes[keep_spots]
    expr_samples = expr_samples[keep_spots]
    coords = get_coords(adata, args.spatial_key, array_row_col)
    genes = adata.var_names.astype(str).to_numpy()

    print(f"[INFO] Aligned spots after preprocessing: {adata.n_obs}")

    x = adata.X
    if sparse.issparse(x):
        x = x.tocsr().astype(np.float32)
    else:
        x = sparse.csr_matrix(np.asarray(x, dtype=np.float32))

    print("[INFO] Local smoothing with 6-neighborhood within sample...")
    x_sm = smooth_within_sample(
        x=x,
        coords=coords,
        sample_ids=expr_samples,
        neighbor_k=args.neighbor_k,
    )

    print("[INFO] K scan on molecular expression...")
    rng = np.random.default_rng(args.seed)
    n = x_sm.shape[0]
    sil_n = min(args.silhouette_sample, n)
    sil_idx = rng.choice(n, size=sil_n, replace=False)

    scan_rows = []
    labels_cache = {}
    for k in k_values:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=args.seed,
            batch_size=args.kmeans_batch_size,
            n_init=10,
        )
        labels = km.fit_predict(x_sm).astype(np.int32)
        sil = float(silhouette_score(x_sm[sil_idx], labels[sil_idx], metric="euclidean"))
        scan_rows.append({"k": int(k), "silhouette": sil, "n_silhouette": int(sil_n)})
        labels_cache[k] = labels
        print(f"[INFO] K={k} silhouette={sil:.6f}")

    sil_df = pd.DataFrame(scan_rows).sort_values("k")
    sil_df.to_csv(out_dir / "silhouette_scan.csv", index=False)
    draw_silhouette_curve(sil_df, out_dir / "silhouette_scan.png")

    best_k = int(sil_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
    selected_k = int(args.selected_k if args.selected_k > 0 else best_k)

    if selected_k in labels_cache:
        mol_labels = labels_cache[selected_k]
    else:
        km = MiniBatchKMeans(
            n_clusters=selected_k,
            random_state=args.seed,
            batch_size=args.kmeans_batch_size,
            n_init=10,
        )
        mol_labels = km.fit_predict(x_sm).astype(np.int32)

    means, counts = cluster_means(x_sm, mol_labels, selected_k)
    labels_cache[selected_k] = mol_labels

    cluster_sizes = pd.DataFrame({"cluster": np.arange(selected_k, dtype=int), "n_spots": counts.astype(int)})
    cluster_sizes.to_csv(out_dir / "cluster_sizes.csv", index=False)

    top_markers, delta = differential_up_genes(means, counts, genes, top_n=args.top_n_genes)
    top_markers.to_csv(out_dir / "top10_upregulated_genes_by_cluster.csv", index=False)

    draw_heatmap(
        means=means,
        genes=genes,
        delta=delta,
        top_n=args.heatmap_top_genes,
        out_png=out_dir / "cluster_gene_heatmap_top50.png",
        out_matrix_csv=out_dir / "cluster_gene_heatmap_top50_matrix.csv",
    )

    print("[INFO] UMAP on sampled embeddings...")
    umap_n = min(args.umap_sample, n)
    umap_idx = rng.choice(n, size=umap_n, replace=False)
    umap_emb = np.asarray(emb_mem[emb_idx[umap_idx]], dtype=np.float32)
    draw_umap(umap_emb, mol_labels[umap_idx], out_dir / "embedding_umap_molecular_labels.png", args.seed)

    print("[INFO] Visual K-means + ARI...")
    vis_labels = fit_predict_embedding_kmeans(
        emb_mem=emb_mem,
        emb_idx=emb_idx,
        k=selected_k,
        seed=args.seed,
        bs=args.kmeans_batch_size,
    )
    ari = float(adjusted_rand_score(mol_labels, vis_labels))
    pd.DataFrame([{"selected_k": selected_k, "ari": ari}]).to_csv(out_dir / "ari_molecular_vs_visual.csv", index=False)

    pd.DataFrame(
        {
            "barcode": matched_barcodes,
            "sample_id": expr_samples,
            "molecular_cluster": mol_labels,
            "visual_cluster": vis_labels,
        }
    ).to_csv(out_dir / "spot_cluster_labels.csv", index=False)

    print("[INFO] Prototypes and centroids...")
    prototypes = compute_embedding_prototypes(
        emb_mem=emb_mem,
        emb_idx=emb_idx,
        labels=mol_labels,
        k=selected_k,
        bs=args.kmeans_batch_size,
    )
    np.save(out_dir / "prototypes.npy", prototypes)
    np.save(out_dir / "prototype_cluster_ids.npy", np.arange(selected_k, dtype=np.int32))

    np.save(out_dir / "gene_expression_centroids.npy", means.astype(np.float32))
    np.save(out_dir / "gene_expression_centroid_genes.npy", genes.astype(str))

    # Save prototype/centroid artifacts for every K in k-list (and selected_k if not in k-list).
    all_k = sorted(set([int(k) for k in k_values] + [int(selected_k)]))
    all_k_dir = out_dir / "prototypes_by_k"
    all_k_dir.mkdir(parents=True, exist_ok=True)
    np.save(all_k_dir / "gene_expression_centroid_genes.npy", genes.astype(str))

    k_rows = []
    for k in all_k:
        if k in labels_cache:
            labels_k = labels_cache[k]
        else:
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=args.seed,
                batch_size=args.kmeans_batch_size,
                n_init=10,
            )
            labels_k = km.fit_predict(x_sm).astype(np.int32)
            labels_cache[k] = labels_k

        if k == selected_k:
            means_k = means
            counts_k = counts
            prototypes_k = prototypes
        else:
            means_k, counts_k = cluster_means(x_sm, labels_k, k)
            prototypes_k = compute_embedding_prototypes(
                emb_mem=emb_mem,
                emb_idx=emb_idx,
                labels=labels_k,
                k=k,
                bs=args.kmeans_batch_size,
            )

        np.save(all_k_dir / f"prototypes_k{k}.npy", prototypes_k.astype(np.float32))
        np.save(all_k_dir / f"prototype_cluster_ids_k{k}.npy", np.arange(k, dtype=np.int32))
        np.save(all_k_dir / f"gene_expression_centroids_k{k}.npy", means_k.astype(np.float32))
        pd.DataFrame({"cluster": np.arange(k, dtype=int), "n_spots": counts_k.astype(int)}).to_csv(
            all_k_dir / f"cluster_sizes_k{k}.csv", index=False
        )
        k_rows.append(
            {
                "k": int(k),
                "n_spots": int(len(labels_k)),
                "n_clusters": int(k),
                "prototypes_path": str(all_k_dir / f"prototypes_k{k}.npy"),
                "gene_centroids_path": str(all_k_dir / f"gene_expression_centroids_k{k}.npy"),
            }
        )

    pd.DataFrame(k_rows).sort_values("k").to_csv(all_k_dir / "k_prototype_manifest.csv", index=False)

    run_info = {
        "input_mode": input_mode,
        "tmp_embedding_memmap": tmp_emb_path if tmp_emb_path else "",
        "n_spots_aligned": int(n),
        "n_genes_after_filter": int(adata.n_vars),
        "k_scan": k_values,
        "best_k": best_k,
        "selected_k": selected_k,
        "saved_prototype_ks": all_k,
        "silhouette_sample": int(sil_n),
        "umap_sample": int(umap_n),
        "ari_molecular_vs_visual": ari,
        "notes": [
            "Preprocessing: empty spot filtering + gene>=10% spots + normalize_total(target_sum=1e4) + log1p + Seurat-v3 top HVGs.",
            "Local smoothing: per-sample KNN(6 neighbors + self) averaging using spatial coordinates.",
        ],
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with (out_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(f"[INFO] wrote outputs to {out_dir}")
    print("[INFO] done")


if __name__ == "__main__":
    main()
