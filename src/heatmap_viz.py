"""
LAHIS Heatmap Visualiser — reproduces Figure 1 from the paper.

For each language, loads the saved importance matrix (.pth) and plots a
colour-coded (layers × heads) heatmap.  All languages can be plotted in a
single figure (one row per language) or saved as individual files.

Usage (from LAHIS/src/):
    # Multi-language grid (saves to ../results/llama2/heatmaps/all_languages.png)
    python3 heatmap_viz.py --model llama2 --languages en fr es zh ru de

    # Single language quick-check
    python3 heatmap_viz.py --model llama2 --languages en --no-grid

    # Also overlay the top-2% heads as markers
    python3 heatmap_viz.py --model llama2 --languages en fr zh --mark-topk 0.02
"""

import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

MODEL_CHOICES = ["llama2", "aya", "mistral", "llama3"]


def load_importance_matrix(model_name: str, lan: str, results_dir: str) -> torch.Tensor:
    path = os.path.join(results_dir, f"{model_name}_{lan}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Importance matrix not found: {path}\n"
            f"Run attn_matrix_ted.py --lan {lan} first."
        )
    return torch.load(path, map_location="cpu").float()


def plot_single_heatmap(
    ax,
    matrix: torch.Tensor,
    title: str,
    vmin: float = None,
    vmax: float = None,
    topk_p: float = None,
    cmap: str = "YlOrRd",
):
    """
    Draw one (layers × heads) heatmap on `ax`.

    Args:
        matrix:  [num_layers, num_heads]
        title:   axis title (language name / code)
        vmin/vmax: shared colour scale (pass the same values for all languages
                   so importance values are visually comparable)
        topk_p:  if set, mark the top-p fraction of heads with a dot
        cmap:    matplotlib colormap name
    """
    data = matrix.numpy()
    num_layers, num_heads = data.shape

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=vmin if vmin is not None else data.min(),
        vmax=vmax if vmax is not None else data.max(),
        origin="upper",
    )

    if topk_p is not None:
        n = max(1, int(num_layers * num_heads * topk_p))
        flat = matrix.view(-1)
        _, topk_idx = torch.topk(flat, k=n)
        for idx in topk_idx.tolist():
            l, h = divmod(idx, num_heads)
            ax.plot(h, l, "b.", markersize=3, alpha=0.6)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Head", fontsize=8)
    ax.set_ylabel("Layer", fontsize=8)
    ax.set_xticks(range(0, num_heads, max(1, num_heads // 8)))
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 8)))
    ax.tick_params(labelsize=7)

    return im


def plot_all_languages(
    model_name: str,
    languages: list,
    results_dir: str,
    output_path: str,
    topk_p: float = None,
    shared_scale: bool = True,
    cmap: str = "YlOrRd",
):
    """
    Plot a grid of heatmaps, one per language (paper Fig 1 style).
    """
    matrices = {}
    for lan in languages:
        try:
            matrices[lan] = load_importance_matrix(model_name, lan, results_dir)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if not matrices:
        print("No matrices found. Run attn_matrix_ted.py first.")
        return

    n = len(matrices)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    if shared_scale:
        all_vals = torch.cat([m.view(-1) for m in matrices.values()])
        vmin, vmax = all_vals.min().item(), all_vals.max().item()
    else:
        vmin = vmax = None

    fig = plt.figure(figsize=(ncols * 4.5, nrows * 3.5 + 0.8))
    gs = gridspec.GridSpec(nrows, ncols + 1,
                           width_ratios=[1] * ncols + [0.07],
                           hspace=0.45, wspace=0.3)

    last_im = None
    for i, (lan, matrix) in enumerate(matrices.items()):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        last_im = plot_single_heatmap(
            ax, matrix, title=lan.upper(),
            vmin=vmin, vmax=vmax,
            topk_p=topk_p, cmap=cmap,
        )

    # Shared colorbar
    if last_im is not None:
        cbar_ax = fig.add_subplot(gs[:, ncols])
        fig.colorbar(last_im, cax=cbar_ax, label="LAHIS importance")

    # Hide unused axes
    for j in range(len(matrices), nrows * ncols):
        row, col = divmod(j, ncols)
        fig.add_subplot(gs[row, col]).axis("off")

    suptitle = (
        f"LAHIS Head Importance — {model_name.upper()}\n"
        f"({model_name}: {list(matrices.values())[0].shape[0]} layers × "
        f"{list(matrices.values())[0].shape[1]} heads)"
    )
    fig.suptitle(suptitle, fontsize=13, y=1.01)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved grid heatmap -> {output_path}")
    plt.close(fig)


def plot_top_head_distribution(
    model_name: str,
    languages: list,
    results_dir: str,
    output_path: str,
    topk_p: float = 0.02,
):
    """
    Layer-by-layer bar chart showing where the top-k% heads cluster per language
    (paper-style distribution plot, companion to the heatmaps).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for lan in languages:
        try:
            matrix = load_importance_matrix(model_name, lan, results_dir)
        except FileNotFoundError:
            continue

        num_layers, num_heads = matrix.shape
        n = max(1, int(num_layers * num_heads * topk_p))
        flat = matrix.view(-1)
        _, topk_idx = torch.topk(flat, k=n)

        # Count top-k heads per layer
        layer_counts = torch.zeros(num_layers, dtype=torch.int)
        for idx in topk_idx.tolist():
            l, _ = divmod(idx, num_heads)
            layer_counts[l] += 1

        pct = (layer_counts.float() / layer_counts.sum() * 100).numpy()
        ax.plot(range(num_layers), pct, "-o", markersize=4, label=lan.upper())

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(f"Top-{topk_p*100:.0f}% head % per layer", fontsize=11)
    ax.set_title(f"Important Head Distribution — {model_name.upper()}", fontsize=12)
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved distribution plot -> {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise LAHIS head importance matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama2", choices=MODEL_CHOICES)
    parser.add_argument("--languages", nargs="+",
                        default=["en", "fr", "es", "zh", "ru", "de"],
                        help="Languages to plot")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with .pth files (default: ../results/{model})")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ../results/{model}/heatmaps)")
    parser.add_argument("--mark-topk", type=float, default=None,
                        help="Mark top-k fraction of heads (e.g. 0.02 for top 2%%)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Save individual heatmaps instead of a grid")
    parser.add_argument("--cmap", default="YlOrRd",
                        help="Matplotlib colormap (e.g. YlOrRd, viridis, plasma)")
    args = parser.parse_args()

    results_dir = args.results_dir or f"../results/{args.model}"
    output_dir  = args.output_dir  or f"../results/{args.model}/heatmaps"

    if args.no_grid:
        for lan in args.languages:
            try:
                matrix = load_importance_matrix(args.model, lan, results_dir)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            im = plot_single_heatmap(ax, matrix, title=lan.upper(),
                                     topk_p=args.mark_topk, cmap=args.cmap)
            fig.colorbar(im, ax=ax, label="LAHIS importance")
            out = os.path.join(output_dir, f"heatmap_{lan}.png")
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved {out}")
            plt.close(fig)
    else:
        plot_all_languages(
            model_name=args.model,
            languages=args.languages,
            results_dir=results_dir,
            output_path=os.path.join(output_dir, "all_languages.png"),
            topk_p=args.mark_topk,
            cmap=args.cmap,
        )

    # Also produce the distribution plot
    plot_top_head_distribution(
        model_name=args.model,
        languages=args.languages,
        results_dir=results_dir,
        output_path=os.path.join(output_dir, "head_distribution.png"),
        topk_p=args.mark_topk or 0.02,
    )
