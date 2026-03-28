"""
visualization.py
----------------
Reusable plotting helpers for the image enhancement assignment.

All functions return a matplotlib Figure (not shown/saved automatically)
so the caller can save or display as needed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from pathlib import Path


# ── Generic comparison ────────────────────────────────────────────────────────

def plot_comparison(images: list, titles: list,
                    cmap: str = None,
                    suptitle: str = None,
                    figsize_per: tuple = (3.5, 3.5)) -> plt.Figure:
    """
    Display N images side-by-side in a single row.

    Parameters
    ----------
    images       : list of np.ndarray
    titles       : list of str — one per image
    cmap         : colormap for grayscale images (default 'gray' if 2D)
    suptitle     : overall figure title
    figsize_per  : (w, h) per subplot in inches
    """
    n = len(images)
    fig, axes = plt.subplots(1, n,
                             figsize=(figsize_per[0] * n, figsize_per[1]),
                             squeeze=False)
    for ax, img, title in zip(axes[0], images, titles):
        used_cmap = cmap
        if used_cmap is None and img.ndim == 2:
            used_cmap = "gray"
        ax.imshow(np.clip(img, 0, 1), cmap=used_cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ── Histogram comparison ──────────────────────────────────────────────────────

def plot_histogram_comparison(img_before: np.ndarray,
                               img_after: np.ndarray,
                               title_before: str = "Before",
                               title_after: str = "After",
                               suptitle: str = None) -> plt.Figure:
    """
    2×2 grid: image before/after (top row) + histograms (bottom row).
    Works for both grayscale and RGB images (histogram uses luminance for RGB).
    """
    def _lum(img):
        return img if img.ndim == 2 else 0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]

    cmap = "gray" if img_before.ndim == 2 else None

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # Images
    axes[0, 0].imshow(np.clip(img_before, 0, 1), cmap=cmap, vmin=0, vmax=1)
    axes[0, 0].set_title(title_before, fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.clip(img_after, 0, 1), cmap=cmap, vmin=0, vmax=1)
    axes[0, 1].set_title(title_after, fontsize=10)
    axes[0, 1].axis("off")

    # Histograms
    for ax, img, title in [(axes[1, 0], img_before, title_before),
                            (axes[1, 1], img_after,  title_after)]:
        lum = _lum(img).ravel()
        ax.hist(lum, bins=256, range=(0, 1), color="steelblue", alpha=0.75)
        ax.set_xlim(0, 1)
        ax.set_title(f"Histogram — {title}", fontsize=9)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


# ── Pipeline strip ────────────────────────────────────────────────────────────

def plot_pipeline_strip(steps: list,
                        suptitle: str = "Pipeline Steps") -> plt.Figure:
    """
    Horizontal strip showing each pipeline step.

    Parameters
    ----------
    steps : list of (np.ndarray, str)  — (image, label) pairs
    """
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), squeeze=False)
    for ax, (img, label) in zip(axes[0], steps):
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(np.clip(img, 0, 1), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


# ── Metric bar chart ──────────────────────────────────────────────────────────

def plot_metric_bars(df: pd.DataFrame,
                     metric: str = "PSNR",
                     group_col: str = "image",
                     hue_col: str = "stage",
                     title: str = None) -> plt.Figure:
    """
    Grouped bar chart comparing a metric across images and pipeline stages.

    Parameters
    ----------
    df         : DataFrame with columns [group_col, hue_col, metric]
    metric     : name of the metric column to plot
    group_col  : x-axis grouping (e.g. 'image')
    hue_col    : bar colour grouping (e.g. 'stage')
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x=group_col, y=metric, hue=hue_col, ax=ax)
    ax.set_title(title or f"{metric} by Image and Stage")
    ax.set_xlabel("Image")
    ax.set_ylabel(metric)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ── SSIM heatmap ──────────────────────────────────────────────────────────────

def plot_ssim_heatmap(df: pd.DataFrame,
                      index_col: str = "image",
                      columns_col: str = "stage",
                      value_col: str = "SSIM") -> plt.Figure:
    """
    Seaborn heatmap of SSIM values: rows = images, columns = pipeline stages.
    """
    pivot = df.pivot(index=index_col, columns=columns_col, values=value_col)
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5),
                                    max(4, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("SSIM Heatmap — Images × Pipeline Stage")
    fig.tight_layout()
    return fig


# ── Transfer function curve ───────────────────────────────────────────────────

def plot_transfer_function(control_points: list,
                            title: str = "Piecewise Linear Transfer Function") -> plt.Figure:
    """
    Plot the intensity mapping T(r) defined by control_points.

    Parameters
    ----------
    control_points : list of (r, s) pairs in [0,1]
    """
    r_pts = [p[0] for p in control_points]
    s_pts = [p[1] for p in control_points]
    r_range = np.linspace(0, 1, 256)
    s_range = np.interp(r_range, r_pts, s_pts)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r_range, s_range, "b-", linewidth=2, label="T(r)")
    ax.plot(r_pts, s_pts, "ro", markersize=8, label="Control points")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Identity")
    ax.set_xlabel("Input intensity  r")
    ax.set_ylabel("Output intensity  s = T(r)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Zoomed crop comparison ───────────────────────────────────────────────────

def plot_zoom_comparison(img_a: np.ndarray, img_b: np.ndarray,
                          roi: tuple = None,
                          labels: tuple = ("Before", "After"),
                          suptitle: str = "Edge Detail Comparison") -> plt.Figure:
    """
    Side-by-side full image + zoomed ROI to show edge preservation.

    Parameters
    ----------
    roi : (row_start, row_end, col_start, col_end) or None (auto-center crop)
    """
    h, w = img_a.shape[:2]
    if roi is None:
        r0, r1 = h // 4, 3 * h // 4
        c0, c1 = w // 4, 3 * w // 4
    else:
        r0, r1, c0, c1 = roi

    cmap = "gray" if img_a.ndim == 2 else None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, (img, lbl) in enumerate([(img_a, labels[0]), (img_b, labels[1])]):
        axes[0, col].imshow(np.clip(img, 0, 1), cmap=cmap, vmin=0, vmax=1)
        axes[0, col].set_title(lbl, fontsize=10)
        axes[0, col].axis("off")
        # Draw ROI rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((c0, r0), c1 - c0, r1 - r0,
                          linewidth=2, edgecolor="red", facecolor="none")
        axes[0, col].add_patch(rect)

        crop = img[r0:r1, c0:c1]
        axes[1, col].imshow(np.clip(crop, 0, 1), cmap=cmap, vmin=0, vmax=1)
        axes[1, col].set_title(f"{lbl} — crop", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


# ── Difference image ─────────────────────────────────────────────────────────

def plot_difference(ref: np.ndarray, enhanced: np.ndarray,
                    scale: float = 5.0,
                    title: str = "Difference (scaled ×5)") -> plt.Figure:
    """
    Show (reference, enhanced, |difference|*scale) in a 3-panel figure.
    """
    diff = np.abs(ref - enhanced)
    cmap = "gray" if ref.ndim == 2 else None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(ref, 0, 1),              cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Reference");  axes[0].axis("off")
    axes[1].imshow(np.clip(enhanced, 0, 1),          cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Enhanced");   axes[1].axis("off")
    axes[2].imshow(np.clip(diff * scale, 0, 1),      cmap="hot",  vmin=0, vmax=1)
    axes[2].set_title(title);        axes[2].axis("off")
    fig.tight_layout()
    return fig


# ── Utility ───────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, path: str, dpi: int = 150):
    """Save figure to disk and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
