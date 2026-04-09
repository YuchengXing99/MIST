"""
Tool 2: Prototype Semantic Interpretation

Connect trained MIST prototypes back to gene expression by finding
the nearest ST spots to each prototype in embedding space, then
running differential gene analysis per prototype.

Usage (from src/):
    python -m visualization.prototype_semantics \
        --ckpt results/ABMIL_MIST/s_checkpoint.pth \
        --paired-h5ad-dir /path/to/paired_h5ad/ \
        --out-dir viz_out/semantics
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import sparse
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ._shared import (
    load_model_from_ckpt, extract_mist, save_fig, ensure_dir,
    DPI, CMAP_DIV,
)


# ── Data loading ────────────────────────────────────────────────────────────
def load_paired_h5ad(paired_dir: str, embedding_key: str = None):
    """
    Load paired h5ad files (ST expression + visual embeddings).

    Follows the pattern from analyze_molecular_visual_pairs.py.

    Returns:
        expression: sparse CSR matrix (M, G)
        embeddings: np.ndarray (M, D)
        genes: np.ndarray (G,) gene names
        sample_ids: np.ndarray (M,)
    """
    import scanpy as sc

    paired_path = Path(paired_dir)
    files = sorted(paired_path.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No .h5ad files in {paired_dir}")

    print(f"Found {len(files)} h5ad files")

    # First pass: determine shared genes and embedding key
    intersection_genes = None
    emb_key = None

    for fp in files:
        a = sc.read_h5ad(fp, backed='r')
        try:
            genes = a.var_names.astype(str).to_numpy()
            gene_set = set(genes.tolist())
            if intersection_genes is None:
                intersection_genes = gene_set
            else:
                intersection_genes &= gene_set

            # Detect embedding key
            if emb_key is None:
                if embedding_key:
                    emb_key = embedding_key
                else:
                    for candidate in ['X_embed', 'X_embedding', 'embedding',
                                      'X_features', 'paired_embedding']:
                        if candidate in a.obsm:
                            emb_key = candidate
                            break
                    if emb_key is None:
                        # Use first obsm key that looks like embeddings
                        for k in a.obsm.keys():
                            if a.obsm[k].ndim == 2 and a.obsm[k].shape[1] > 10:
                                emb_key = k
                                break
                    if emb_key is None:
                        raise ValueError(
                            f"Cannot detect embedding key in {fp}. "
                            f"Available obsm keys: {list(a.obsm.keys())}. "
                            f"Specify --embedding-key explicitly.")
        finally:
            if getattr(a, 'isbacked', False):
                a.file.close()

    target_genes = np.array(sorted(intersection_genes))
    gene_to_col = {g: i for i, g in enumerate(target_genes)}
    print(f"Shared genes: {len(target_genes)}, Embedding key: {emb_key}")

    # Second pass: load data
    x_blocks = []
    emb_blocks = []
    sample_blocks = []

    for fp in tqdm(files, desc="Loading h5ad"):
        a = sc.read_h5ad(fp)
        genes = a.var_names.astype(str).to_numpy()

        # Extract expression
        x = a.X
        if sparse.issparse(x):
            x = x.tocsr().astype(np.float32)
        else:
            x = sparse.csr_matrix(np.asarray(x, dtype=np.float32))

        # Align genes
        keep = np.array([g in gene_to_col for g in genes])
        if keep.any():
            x_keep = x[:, keep]
            genes_keep = genes[keep]
            new_cols = np.array([gene_to_col[g] for g in genes_keep])
            coo = x_keep.tocoo()
            x_aligned = sparse.csr_matrix(
                (coo.data, (coo.row, new_cols[coo.col])),
                shape=(x.shape[0], len(target_genes)),
                dtype=np.float32,
            )
        else:
            x_aligned = sparse.csr_matrix(
                (x.shape[0], len(target_genes)), dtype=np.float32)

        # Embeddings
        emb = np.asarray(a.obsm[emb_key], dtype=np.float32)

        # Sample ID
        sid = np.array([fp.stem] * a.n_obs, dtype=str)

        x_blocks.append(x_aligned)
        emb_blocks.append(emb)
        sample_blocks.append(sid)

    expression = sparse.vstack(x_blocks, format='csr')
    embeddings = np.vstack(emb_blocks)
    sample_ids = np.concatenate(sample_blocks)

    return expression, embeddings, target_genes, sample_ids


# ── Prototype projection & matching ────────────────────────────────────────
@torch.no_grad()
def project_prototypes(mist_module, device) -> np.ndarray:
    """Project prototypes through shared_proj → (K, out_dim)."""
    proto = mist_module.shared_proj(mist_module.prototypes.to(device))
    return proto.cpu().numpy()


@torch.no_grad()
def project_embeddings(mist_module, embeddings: np.ndarray,
                       device, batch_size: int = 4096) -> np.ndarray:
    """Project ST spot embeddings through shared_proj → (M, out_dim)."""
    n = len(embeddings)
    out_parts = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(embeddings[start:end]).float().to(device)
        proj = mist_module.shared_proj(x)
        out_parts.append(proj.cpu().numpy())
    return np.vstack(out_parts)


def find_nearest_spots(proto_vectors: np.ndarray, spot_vectors: np.ndarray,
                       top_n: int = 100) -> np.ndarray:
    """
    For each prototype, find top_n nearest spots by cosine similarity.

    Returns: (K, top_n) indices into spot_vectors.
    """
    sim = cosine_similarity(proto_vectors, spot_vectors)  # (K, M)
    K = sim.shape[0]
    indices = np.zeros((K, top_n), dtype=np.int64)
    for k in range(K):
        indices[k] = np.argsort(sim[k])[::-1][:top_n]
    return indices


def assign_spots_to_prototypes(proto_vectors: np.ndarray,
                               spot_vectors: np.ndarray) -> np.ndarray:
    """Assign each spot to its nearest prototype. Returns (M,) labels."""
    sim = cosine_similarity(spot_vectors, proto_vectors)  # (M, K)
    return sim.argmax(axis=1)


# ── Gene analysis ──────────────────────────────────────────────────────────
def aggregate_expression(expression, nearest_indices: np.ndarray,
                         genes: np.ndarray):
    """
    Mean gene expression per prototype based on nearest spots.

    Returns: means (K, G), counts (K,)
    """
    K = nearest_indices.shape[0]
    G = expression.shape[1]
    means = np.zeros((K, G), dtype=np.float32)
    counts = np.zeros(K, dtype=np.int64)

    for k in range(K):
        idx = nearest_indices[k]
        counts[k] = len(idx)
        means[k] = np.asarray(expression[idx].mean(axis=0)).ravel()

    return means, counts


def differential_up_genes(means: np.ndarray, counts: np.ndarray,
                          genes: np.ndarray,
                          top_n: int = 10):
    """
    Find top upregulated genes per prototype (delta-mean method).

    Replicates analyze_molecular_visual_pairs.differential_up_genes.
    """
    K, G = means.shape
    total_n = counts.sum()
    weighted_sum = (means * counts[:, None]).sum(axis=0)
    delta = np.zeros_like(means, dtype=np.float32)

    rows = []
    for k in range(K):
        n_in = counts[k]
        n_out = total_n - n_in
        if n_in <= 0 or n_out <= 0:
            continue
        mean_out = (weighted_sum - means[k] * n_in) / max(n_out, 1)
        d = means[k] - mean_out
        delta[k] = d

        rank = np.argsort(d)[::-1][:top_n]
        for r, gi in enumerate(rank, start=1):
            rows.append({
                'prototype': int(k),
                'rank': int(r),
                'gene': str(genes[gi]),
                'delta_mean': float(d[gi]),
                'mean_in': float(means[k, gi]),
                'mean_out': float(mean_out[gi]),
            })

    return pd.DataFrame(rows), delta


# ── Plotting ────────────────────────────────────────────────────────────────
def draw_gene_heatmap(means: np.ndarray, genes: np.ndarray,
                      delta: np.ndarray, top_n: int,
                      out_png: str, out_csv: str):
    """
    Hierarchical-clustering heatmap of top differential genes per prototype.

    Follows the style from analyze_molecular_visual_pairs.draw_heatmap.
    """
    score = delta.max(axis=0)
    top_idx = np.argsort(score)[::-1][:top_n]

    mat = means[:, top_idx]
    sel_genes = genes[top_idx]

    # Z-score per gene
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True) + 1e-8
    z = (mat - mu) / sd

    row_order = np.arange(z.shape[0])
    col_order = np.arange(z.shape[1])
    if z.shape[0] > 1:
        row_order = leaves_list(linkage(z, method='average', metric='euclidean'))
    if z.shape[1] > 1:
        col_order = leaves_list(linkage(z.T, method='average', metric='euclidean'))

    z_ord = z[row_order][:, col_order]
    genes_ord = sel_genes[col_order]

    # Save matrix
    plot_df = pd.DataFrame(
        mat[row_order][:, col_order],
        index=[f'Proto{i}' for i in row_order],
        columns=genes_ord,
    )
    plot_df.to_csv(out_csv)

    # Plot
    fig_w = max(12, top_n * 0.28)
    fig_h = max(4, z.shape[0] * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(z_ord, aspect='auto', cmap=CMAP_DIV)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label='z-score')
    ax.set_xticks(range(len(genes_ord)))
    ax.set_xticklabels(genes_ord, rotation=90, fontsize=8)
    ax.set_yticks(range(z_ord.shape[0]))
    ax.set_yticklabels([f'Proto{i}' for i in row_order], fontsize=9)
    ax.set_title(f'Prototype x Top-{top_n} Differential Genes')
    save_fig(fig, out_png)


def draw_prototype_gene_bar(top_genes_df: pd.DataFrame, out_path: str, K: int):
    """Bar chart of top genes per prototype."""
    n_show = min(K, 8)
    fig, axes = plt.subplots(2, (n_show + 1) // 2,
                             figsize=(5 * ((n_show + 1) // 2), 8))
    axes = axes.ravel()

    for k in range(n_show):
        ax = axes[k]
        sub = top_genes_df[top_genes_df['prototype'] == k].head(10)
        if len(sub) == 0:
            ax.set_title(f'Proto {k} (empty)')
            continue
        ax.barh(range(len(sub)), sub['delta_mean'].values, color='#4C72B0')
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub['gene'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'Proto {k}', fontsize=10)
        ax.set_xlabel('Delta Mean', fontsize=8)

    for k in range(n_show, len(axes)):
        axes[k].axis('off')

    save_fig(fig, out_path)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='MIST Prototype Semantic Interpretation')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--paired-h5ad-dir', type=str, required=True,
                        help='Directory with paired h5ad files (ST + embeddings)')
    parser.add_argument('--embedding-key', type=str, default=None,
                        help='obsm key for embeddings (auto-detect if omitted)')
    parser.add_argument('--config-dir', type=str, default='./configs')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'survival'])
    parser.add_argument('--top-n-genes', type=int, default=10,
                        help='Top genes per prototype in table')
    parser.add_argument('--heatmap-top', type=int, default=50,
                        help='Total genes in heatmap')
    parser.add_argument('--nearest-spots', type=int, default=200,
                        help='Nearest spots per prototype for aggregation')
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # Load model
    model, cfg, device = load_model_from_ckpt(
        args.ckpt, args.config_dir, mode=args.mode, device_str=args.device)
    mist = extract_mist(model)
    K = mist.K
    print(f"Model loaded: {K} prototypes")

    # Load ST data
    print("\nLoading paired h5ad files...")
    expression, embeddings, genes, sample_ids = load_paired_h5ad(
        args.paired_h5ad_dir, args.embedding_key)
    print(f"Spots: {expression.shape[0]}, Genes: {len(genes)}, "
          f"Embedding dim: {embeddings.shape[1]}")

    # Project prototypes and spot embeddings into shared space
    print("\nProjecting to shared space...")
    proto_proj = project_prototypes(mist, device)
    spot_proj = project_embeddings(mist, embeddings, device)

    # Find nearest spots per prototype
    print("Finding nearest spots...")
    nearest_idx = find_nearest_spots(proto_proj, spot_proj,
                                     top_n=args.nearest_spots)

    # Aggregate expression
    means, counts = aggregate_expression(expression, nearest_idx, genes)
    print(f"Spots per prototype: {counts}")

    # Differential gene analysis
    print("\nDifferential gene analysis...")
    top_genes_df, delta = differential_up_genes(means, counts, genes,
                                                top_n=args.top_n_genes)
    top_genes_df.to_csv(f'{args.out_dir}/top_genes_per_prototype.csv', index=False)

    # Plots
    draw_gene_heatmap(means, genes, delta, args.heatmap_top,
                      f'{args.out_dir}/gene_heatmap.png',
                      f'{args.out_dir}/gene_heatmap_matrix.csv')
    draw_prototype_gene_bar(top_genes_df, f'{args.out_dir}/top_genes_bar.png', K)

    # Also save the full mean expression per prototype
    means_df = pd.DataFrame(means, columns=genes,
                            index=[f'Proto{k}' for k in range(K)])
    means_df.to_csv(f'{args.out_dir}/mean_expression_per_prototype.csv')

    # Assignment-based analysis (assign all spots to nearest prototype)
    print("\nAssigning all spots to prototypes...")
    labels = assign_spots_to_prototypes(proto_proj, spot_proj)
    from collections import Counter
    label_counts = Counter(labels.tolist())
    print(f"Spot assignment distribution: {dict(sorted(label_counts.items()))}")

    # Full cluster-based differential genes
    full_means, full_counts = np.zeros((K, len(genes)), dtype=np.float32), np.zeros(K, dtype=np.int64)
    for k in range(K):
        idx = np.where(labels == k)[0]
        full_counts[k] = len(idx)
        if len(idx) > 0:
            full_means[k] = np.asarray(expression[idx].mean(axis=0)).ravel()

    full_top_df, full_delta = differential_up_genes(full_means, full_counts, genes,
                                                    top_n=args.top_n_genes)
    full_top_df.to_csv(f'{args.out_dir}/top_genes_full_assignment.csv', index=False)
    draw_gene_heatmap(full_means, genes, full_delta, args.heatmap_top,
                      f'{args.out_dir}/gene_heatmap_full.png',
                      f'{args.out_dir}/gene_heatmap_full_matrix.csv')

    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
