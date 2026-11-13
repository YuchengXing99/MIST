"""
Tool 5: Prototype Drift Analysis

Compare ST-initialised prototypes with trained prototypes to measure
how much MIST adapted each prototype during task-specific training.

Usage (from src/):
    python -m visualization.prototype_drift \
        --ckpt results/ABMIL_MIST/s_checkpoint.pth \
        --init-proto splits/task/prototypes/prototypes.npy \
        --out-dir viz_out/drift
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from ._shared import (
    load_model_from_ckpt, extract_mist, save_fig, ensure_dir,
    DPI, CMAP_DIV,
)


# ── Metric computation ──────────────────────────────────────────────────────
def compute_drift(init_protos: np.ndarray, trained_protos: np.ndarray) -> dict:
    """
    Per-prototype drift metrics between initial and trained prototypes.

    Args:
        init_protos:    (K, D) initial prototype vectors
        trained_protos: (K, D) trained prototype vectors

    Returns:
        dict with 'cosine_sim' (K,), 'l2_dist' (K,), 'angular_dist' (K,)
    """
    K = init_protos.shape[0]
    cos = np.array([
        cosine_similarity(init_protos[k:k+1], trained_protos[k:k+1])[0, 0]
        for k in range(K)
    ])
    l2 = np.linalg.norm(init_protos - trained_protos, axis=1)
    angular = np.arccos(np.clip(cos, -1, 1)) / np.pi  # normalised to [0, 1]
    return {'cosine_sim': cos, 'l2_dist': l2, 'angular_dist': angular}


def project_through_shared_proj(mist_module, protos: np.ndarray, device) -> np.ndarray:
    """Project prototypes through the shared_proj linear layer."""
    with torch.no_grad():
        t = torch.from_numpy(protos).float().to(device)
        projected = mist_module.shared_proj(t)
    return projected.cpu().numpy()


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_drift_bar(metrics: dict, out_path: str, title_suffix: str = ''):
    """Grouped bar chart: cosine similarity and L2 distance per prototype."""
    K = len(metrics['cosine_sim'])
    x = np.arange(K)
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(max(6, K * 0.6), 4))

    color1 = '#4C72B0'
    color2 = '#DD8452'
    ax1.bar(x - w/2, metrics['cosine_sim'], w, label='Cosine Similarity', color=color1)
    ax1.set_ylabel('Cosine Similarity', color=color1)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.bar(x + w/2, metrics['l2_dist'], w, label='L2 Distance', color=color2)
    ax2.set_ylabel('L2 Distance', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_xlabel('Prototype Index')
    ax1.set_xticks(x)
    ax1.set_title(f'Prototype Drift{title_suffix}')
    ax1.grid(axis='y', alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    save_fig(fig, out_path)


def plot_pairwise_similarity(protos: np.ndarray, out_path: str, title: str = ''):
    """(K, K) cosine similarity heatmap of prototypes."""
    sim = cosine_similarity(protos)
    K = sim.shape[0]

    fig, ax = plt.subplots(figsize=(max(5, K * 0.45), max(4, K * 0.4)))
    im = ax.imshow(sim, cmap=CMAP_DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xlabel('Prototype')
    ax.set_ylabel('Prototype')
    ax.set_title(title or 'Pairwise Cosine Similarity')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_fig(fig, out_path)


def plot_drift_comparison(init_sim: np.ndarray, trained_sim: np.ndarray,
                          out_path: str):
    """Side-by-side pairwise similarity: init vs trained."""
    K = init_sim.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, K * 0.8), max(4, K * 0.4)))

    for ax, sim, title in [(ax1, init_sim, 'Initial Prototypes'),
                            (ax2, trained_sim, 'Trained Prototypes')]:
        im = ax.imshow(sim, cmap=CMAP_DIV, vmin=-1, vmax=1)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_title(title)
        ax.set_xlabel('Prototype')
        ax.set_ylabel('Prototype')
    fig.colorbar(im, ax=[ax1, ax2], fraction=0.025, pad=0.04)

    save_fig(fig, out_path)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MIST Prototype Drift Analysis')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--init-proto', type=str, required=True,
                        help='Path to initial ST prototypes (.npy)')
    parser.add_argument('--config-dir', type=str, default='./configs')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'survival'])
    parser.add_argument('--compare-projected', action='store_true',
                        help='Also compare in shared_proj projected space')
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # Load
    model, cfg, device = load_model_from_ckpt(
        args.ckpt, args.config_dir, mode=args.mode, device_str=args.device)
    mist = extract_mist(model)
    init_protos = np.load(args.init_proto)
    trained_protos = mist.prototypes.detach().cpu().numpy()

    assert init_protos.shape == trained_protos.shape, \
        f"Shape mismatch: init {init_protos.shape} vs trained {trained_protos.shape}"

    # Raw-space drift
    metrics = compute_drift(init_protos, trained_protos)
    plot_drift_bar(metrics, f'{args.out_dir}/drift_raw.png', ' (Raw Space)')

    # Pairwise similarity comparison
    init_sim = cosine_similarity(init_protos)
    trained_sim = cosine_similarity(trained_protos)
    plot_drift_comparison(init_sim, trained_sim, f'{args.out_dir}/pairwise_comparison.png')
    plot_pairwise_similarity(trained_protos, f'{args.out_dir}/pairwise_trained.png',
                             'Trained Prototypes Pairwise Similarity')

    # Projected-space drift
    if args.compare_projected:
        init_proj = project_through_shared_proj(mist, init_protos, device)
        trained_proj = project_through_shared_proj(mist, trained_protos, device)
        metrics_proj = compute_drift(init_proj, trained_proj)
        plot_drift_bar(metrics_proj, f'{args.out_dir}/drift_projected.png',
                       ' (Projected Space)')

    # Save numeric results
    import pandas as pd
    df = pd.DataFrame({
        'prototype': range(len(metrics['cosine_sim'])),
        'cosine_sim': metrics['cosine_sim'],
        'l2_dist': metrics['l2_dist'],
        'angular_dist': metrics['angular_dist'],
    })
    df.to_csv(f'{args.out_dir}/drift_metrics.csv', index=False)
    print(f"Results saved to {args.out_dir}/")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
