"""
metrics.py
----------
Objective image quality metrics: MSE, PSNR, SSIM.

All functions expect float64 images in [0, 1].
For color images the metrics are computed on the luminance (L*) channel
so that the score reflects perceptual quality rather than raw pixel differences.
"""

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as _ssim
from skimage import color as skcolor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB to luminance float64 [0,1]; pass-through for grayscale."""
    if img.ndim == 3:
        lab = skcolor.rgb2lab(img)
        return lab[:, :, 0] / 100.0
    return img


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_mse(ref: np.ndarray, test: np.ndarray) -> float:
    """
    Mean Squared Error:
        MSE = (1/MN) * sum_{x,y} (I_ref(x,y) - I_test(x,y))^2
    """
    r = _to_gray(ref)
    t = _to_gray(test)
    return float(np.mean((r - t) ** 2))


def compute_psnr(ref: np.ndarray, test: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB):
        PSNR = 10 * log10(MAX^2 / MSE)

    Higher is better.  Returns inf if images are identical.
    """
    mse = compute_mse(ref, test)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) in [−1, 1]; 1 = perfect match.

    Uses skimage.metrics.structural_similarity with data_range=1.0.
    """
    r = _to_gray(ref)
    t = _to_gray(test)
    return float(_ssim(r, t, data_range=1.0))


def compute_all(ref: np.ndarray, test: np.ndarray) -> dict:
    """Return a dict with MSE, PSNR, and SSIM for the (ref, test) pair."""
    return {
        "MSE":  compute_mse(ref, test),
        "PSNR": compute_psnr(ref, test),
        "SSIM": compute_ssim(ref, test),
    }


# ── Table builders ────────────────────────────────────────────────────────────

def build_metrics_table(records: list) -> pd.DataFrame:
    """
    Build a tidy DataFrame from a list of metric records.

    Each record should be a dict with at minimum:
        {"image": str, "stage": str, "MSE": float, "PSNR": float, "SSIM": float}
    """
    return pd.DataFrame(records)


def format_latex_table(df: pd.DataFrame) -> str:
    """Return a LaTeX-formatted table string (booktabs style)."""
    return df.to_latex(index=False, float_format="%.4f",
                       caption="Objective quality metrics (MSE ↓, PSNR ↑, SSIM ↑)",
                       label="tab:metrics")
