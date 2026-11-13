"""
Tool 4: Affinity vs Attention Comparison

Compare MIL attention weights with MIST staining affinity to understand
whether MIST provides complementary spatial information.

Supported models: ABMIL, CLAM, DSMIL (models that return 'attn' in output).

Usage (from src/):
    python -m visualization.affinity_vs_attention \
        --ckpt results/ABMIL_MIST/s_checkpoint.pth \
        --h5 /path/to/slide.h5 \
        --out-dir viz_out/attn_vs_affinity
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from scipy.stats import spearmanr, pearsonr

from ._shared import (
    load_model_from_ckpt, load_h5, run_inference, save_fig, ensure_dir,
    DPI, CMAP_DIV,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _extract_attention_and_affinity(out: dict):
    """
    Extract attention (N,) and affinity (N, K) from model output.

    ABMIL/CLAM/DSMIL return attn as (1, N) or (N,).
    """
    if 'attn' not in out or out['attn'] is None:
        raise ValueError(
            "Model does not return attention weights. "
            "This tool only supports ABMIL, CLAM, and DSMIL.")

    attn = out['attn']
    if attn.ndim == 2:
        attn = attn.squeeze(0)  # (1, N) -> (N,)

    if 'staining_affinity' not in out:
        raise ValueError("Model did not return staining_affinity (is use_mist=True?)")
    affinity = out['staining_affinity']  # (N, K)

    return attn, affinity


def compute_correlations(attention: np.ndarray, affinity: np.ndarray) -> dict:
    """
    Spearman and Pearson correlation between attention and each prototype.

    Returns dict with 'spearman' (K,), 'pearson' (K,), 'spearman_p' (K,), 'pearson_p' (K,).
    """
    K = affinity.shape[1]
    spear = np.zeros(K)
    pear = np.zeros(K)
    spear_p = np.zeros(K)
    pear_p = np.zeros(K)

    for k in range(K):
        sr, sp = spearmanr(attention, affinity[:, k])
        pr, pp = pearsonr(attention, affinity[:, k])
        spear[k], spear_p[k] = sr, sp
        pear[k], pear_p[k] = pr, pp

    return {'spearman': spear, 'pearson': pear,
            'spearman_p': spear_p, 'pearson_p': pear_p}


# ── Plotting ────────────────────────────────────────────────────────────────
def _heatmap_panel(ax, coords, values, patch_size, cmap, title, vmin=None, vmax=None):
    """Single heatmap panel on given axis."""
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    rects = []
    colors = []
    for i in range(len(coords)):
        x, y = coords[i, 0], coords[i, 1]
        rects.append(Rectangle((x, y), patch_size, patch_size))
        colors.append(cmap_obj(norm(values[i])))

    pc = PatchCollection(rects, facecolors=colors, edgecolors='none')
    ax.add_collection(pc)

    margin = patch_size
    ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + patch_size + margin)
    ax.set_ylim(coords[:, 1].max() + patch_size + margin, coords[:, 1].min() - margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    return sm


def plot_spatial_comparison(coords, attention, affinity, out_path,
                            patch_size=256, max_protos=4):
    """Side-by-side: attention + top correlated prototypes."""
    K = affinity.shape[1]
    n_show = min(K, max_protos)
    ncols = 1 + n_show

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    # Attention panel
    sm = _heatmap_panel(axes[0], coords, attention, patch_size,
                        'Reds', 'Attention')
    fig.colorbar(sm, ax=axes[0], fraction=0.03, pad=0.01)

    # Top prototypes by absolute correlation with attention
    corrs = compute_correlations(attention, affinity)
    top_k = np.argsort(np.abs(corrs['spearman']))[::-1][:n_show]

    for i, k in enumerate(top_k):
        sm = _heatmap_panel(
            axes[1 + i], coords, affinity[:, k], patch_size,
            CMAP_DIV, f'Proto {k} (r={corrs["spearman"][k]:.2f})')
        fig.colorbar(sm, ax=axes[1 + i], fraction=0.03, pad=0.01)

    save_fig(fig, out_path)


def plot_correlation_bar(corrs: dict, out_path: str):
    """Bar chart of Spearman and Pearson correlation per prototype."""
    K = len(corrs['spearman'])
    x = np.arange(K)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 4))
    ax.bar(x - w/2, corrs['spearman'], w, label='Spearman', color='#4C72B0')
    ax.bar(x + w/2, corrs['pearson'], w, label='Pearson', color='#DD8452')
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.set_xlabel('Prototype Index')
    ax.set_ylabel('Correlation with Attention')
    ax.set_xticks(x)
    ax.set_title('Affinity–Attention Correlation')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    save_fig(fig, out_path)


def plot_scatter_top(attention, affinity, corrs, out_path, top_n=4):
    """Scatter plots: attention vs top-correlated prototype affinities."""
    K = affinity.shape[1]
    n_show = min(K, top_n)
    top_k = np.argsort(np.abs(corrs['spearman']))[::-1][:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3.5))
    if n_show == 1:
        axes = [axes]

    for i, k in enumerate(top_k):
        ax = axes[i]
        ax.scatter(attention, affinity[:, k], s=2, alpha=0.4, color='#4C72B0')
        ax.set_xlabel('Attention', fontsize=9)
        ax.set_ylabel(f'Proto {k} Affinity', fontsize=9)
        ax.set_title(f'r_s={corrs["spearman"][k]:.3f}', fontsize=9)
        ax.grid(alpha=0.3)

    save_fig(fig, out_path)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='MIST Affinity vs Attention Comparison')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--h5', type=str, required=True)
    parser.add_argument('--config-dir', type=str, default='./configs')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'survival'])
    parser.add_argument('--patch-size', type=int, default=256)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # Load
    model, cfg, device = load_model_from_ckpt(
        args.ckpt, args.config_dir, mode=args.mode, device_str=args.device)
    features, coords = load_h5(args.h5)

    if len(coords) == 0:
        raise ValueError("H5 file does not contain patch coordinates")

    # Inference
    out = run_inference(model, features, device)
    attention, affinity = _extract_attention_and_affinity(out)

    print(f"Patches: {len(attention)}, Prototypes: {affinity.shape[1]}")

    # Correlations
    corrs = compute_correlations(attention, affinity)

    # Plots
    plot_spatial_comparison(coords, attention, affinity,
                            f'{args.out_dir}/spatial_comparison.png',
                            patch_size=args.patch_size)
    plot_correlation_bar(corrs, f'{args.out_dir}/correlation_bar.png')
    plot_scatter_top(attention, affinity, corrs,
                     f'{args.out_dir}/scatter_top.png')

    # Save numeric results
    import pandas as pd
    df = pd.DataFrame({
        'prototype': range(len(corrs['spearman'])),
        'spearman_r': corrs['spearman'],
        'spearman_p': corrs['spearman_p'],
        'pearson_r': corrs['pearson'],
        'pearson_p': corrs['pearson_p'],
    })
    df.to_csv(f'{args.out_dir}/correlations.csv', index=False)
    print(f"\nResults saved to {args.out_dir}/")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
