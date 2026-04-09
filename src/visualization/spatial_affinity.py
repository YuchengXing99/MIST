"""
Tool 1: Spatial Affinity Heatmaps

Map per-patch staining affinity back to WSI coordinates to visualise
which tissue regions each molecular prototype activates.

Usage (from src/):
    python -m visualization.spatial_affinity \
        --ckpt results/ABMIL_MIST/s_checkpoint.pth \
        --h5 /path/to/slide.h5 \
        --out-dir viz_out/spatial
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

from ._shared import (
    load_model_from_ckpt, load_h5, run_inference, save_fig, ensure_dir,
    DPI, CMAP_DIV,
)


# ── Plotting ────────────────────────────────────────────────────────────────
def _make_patch_heatmap(ax, coords, values, patch_size, cmap, vmin, vmax, title):
    """Draw rectangles at patch coords coloured by values."""
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

    # Set axis limits
    margin = patch_size
    ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + patch_size + margin)
    ax.set_ylim(coords[:, 1].max() + patch_size + margin, coords[:, 1].min() - margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.axis('off')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    return sm


def plot_single_prototype(coords, affinity_k, proto_idx, out_path,
                          patch_size=256, cmap=CMAP_DIV):
    """Generate a spatial heatmap for one prototype."""
    fig, ax = plt.subplots(figsize=(8, 8))
    sm = _make_patch_heatmap(
        ax, coords, affinity_k, patch_size, cmap,
        vmin=0, vmax=affinity_k.max(),
        title=f'Prototype {proto_idx}')
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    save_fig(fig, out_path)


def plot_combined_panel(coords, affinity, out_path, patch_size=256,
                        ncols=4, proto_indices=None, cmap=CMAP_DIV):
    """Multi-panel figure with all (or selected) prototypes."""
    K = affinity.shape[1]
    indices = proto_indices if proto_indices is not None else list(range(K))
    n = len(indices)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, k in enumerate(indices):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sm = _make_patch_heatmap(
            ax, coords, affinity[:, k], patch_size, cmap,
            vmin=0, vmax=affinity[:, k].max(),
            title=f'Proto {k}')

    # Turn off unused axes
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis('off')

    # Shared colorbar
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    save_fig(fig, out_path)


def plot_max_prototype_map(coords, affinity, out_path, patch_size=256):
    """Colour each patch by its dominant (argmax) prototype."""
    dominant = affinity.argmax(axis=1)  # (N,)
    K = affinity.shape[1]

    cmap = plt.get_cmap('tab20', K)
    fig, ax = plt.subplots(figsize=(8, 8))

    rects = []
    colors = []
    for i in range(len(coords)):
        x, y = coords[i, 0], coords[i, 1]
        rects.append(Rectangle((x, y), patch_size, patch_size))
        colors.append(cmap(dominant[i]))

    pc = PatchCollection(rects, facecolors=colors, edgecolors='none')
    ax.add_collection(pc)

    margin = patch_size
    ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + patch_size + margin)
    ax.set_ylim(coords[:, 1].max() + patch_size + margin, coords[:, 1].min() - margin)
    ax.set_aspect('equal')
    ax.set_title('Dominant Prototype per Patch')
    ax.axis('off')

    # Discrete legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=cmap(k), label=f'Proto {k}') for k in range(K)]
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              ncol=max(1, K // 8), framealpha=0.8)

    save_fig(fig, out_path)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MIST Spatial Affinity Heatmaps')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--h5', type=str, required=True,
                        help='Path to a single slide H5 file')
    parser.add_argument('--config-dir', type=str, default='./configs')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'survival'])
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--ncols', type=int, default=4)
    parser.add_argument('--proto-indices', type=str, default=None,
                        help='Comma-separated prototype indices to plot (default: all)')
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    proto_indices = None
    if args.proto_indices:
        proto_indices = [int(x) for x in args.proto_indices.split(',')]

    # Load model and data
    model, cfg, device = load_model_from_ckpt(
        args.ckpt, args.config_dir, mode=args.mode, device_str=args.device)
    features, coords = load_h5(args.h5)

    if len(coords) == 0:
        raise ValueError("H5 file does not contain patch coordinates ('coords' key)")

    # Inference
    out = run_inference(model, features, device)
    if 'staining_affinity' not in out:
        raise ValueError("Model did not return staining_affinity (is use_mist=True?)")
    affinity = out['staining_affinity']  # (N, K)

    print(f"Slide: {args.h5}")
    print(f"Patches: {affinity.shape[0]}, Prototypes: {affinity.shape[1]}")

    # Per-prototype heatmaps
    indices = proto_indices if proto_indices else list(range(affinity.shape[1]))
    for k in indices:
        plot_single_prototype(coords, affinity[:, k], k,
                              f'{args.out_dir}/proto_{k}.png',
                              patch_size=args.patch_size)

    # Combined panel
    plot_combined_panel(coords, affinity, f'{args.out_dir}/combined.png',
                        patch_size=args.patch_size, ncols=args.ncols,
                        proto_indices=proto_indices)

    # Dominant prototype map
    plot_max_prototype_map(coords, affinity, f'{args.out_dir}/dominant.png',
                           patch_size=args.patch_size)

    print(f"Saved {len(indices) + 2} figures to {args.out_dir}/")


if __name__ == '__main__':
    main()
