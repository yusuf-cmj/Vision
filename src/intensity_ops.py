"""
intensity_ops.py
----------------
Intensity and histogram-based enhancement operations.

All functions accept float64 images in [0, 1] and return float64 [0, 1].
For color (RGB) images the caller should pass color=True where supported;
the function then operates only on the L* channel of CIE Lab colorspace
to preserve hue and saturation.
"""

import numpy as np
from skimage import exposure, color as skcolor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rgb_to_lab_L(img_rgb: np.ndarray):
    """Split RGB into (lab, L_norm) where L_norm ∈ [0,1]."""
    lab = skcolor.rgb2lab(img_rgb)
    L_norm = lab[:, :, 0] / 100.0
    return lab, L_norm


def _lab_merge_rgb(lab: np.ndarray, L_norm: np.ndarray) -> np.ndarray:
    """Reconstruct RGB from lab array with updated L (in [0,1])."""
    lab_out = lab.copy()
    lab_out[:, :, 0] = np.clip(L_norm, 0.0, 1.0) * 100.0
    return np.clip(skcolor.lab2rgb(lab_out), 0.0, 1.0)


# ── 1. Gamma Correction ───────────────────────────────────────────────────────

def gamma_correction(img: np.ndarray, gamma: float = 2.2, color: bool = False) -> np.ndarray:
    """
    Power-law intensity mapping:
        I_out = I_in ^ (1 / gamma)

    gamma > 1  →  brightens the image (correction for underexposure).
    gamma < 1  →  darkens.

    Parameters
    ----------
    gamma : float
        Encoding gamma of the degraded image (e.g. 2.2 for sRGB, 2.5 for dark).
    color : bool
        If True and img is RGB, apply only to L* channel.
    """
    if color and img.ndim == 3:
        lab, L = _rgb_to_lab_L(img)
        return _lab_merge_rgb(lab, np.power(np.clip(L, 1e-8, 1.0), 1.0 / gamma))
    return np.power(np.clip(img, 1e-8, 1.0), 1.0 / gamma)


# ── 2. Contrast Stretching ────────────────────────────────────────────────────

def contrast_stretch(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0,
                     color: bool = False) -> np.ndarray:
    """
    Percentile-based linear contrast stretching:
        I_out = clip((I - p_low%) / (p_high% - p_low%), 0, 1)

    Robust to outliers compared to min-max normalisation.

    Parameters
    ----------
    p_low, p_high : float
        Lower and upper percentiles used as stretch limits (default 2 / 98).
    """
    if color and img.ndim == 3:
        lab, L = _rgb_to_lab_L(img)
        lo, hi = np.percentile(L, [p_low, p_high])
        denom = hi - lo if hi > lo else 1e-8
        return _lab_merge_rgb(lab, np.clip((L - lo) / denom, 0.0, 1.0))

    if img.ndim == 3:
        # Per-channel stretch for direct RGB comparison (used in failure case)
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            lo, hi = np.percentile(img[:, :, c], [p_low, p_high])
            denom = hi - lo if hi > lo else 1e-8
            out[:, :, c] = np.clip((img[:, :, c] - lo) / denom, 0.0, 1.0)
        return out

    lo, hi = np.percentile(img, [p_low, p_high])
    denom = hi - lo if hi > lo else 1e-8
    return np.clip((img - lo) / denom, 0.0, 1.0)


# ── 3. Histogram Equalisation ─────────────────────────────────────────────────

def histogram_equalize(img: np.ndarray, color: bool = False) -> np.ndarray:
    """
    Global histogram equalisation.

    Discrete mapping:
        s_k = T(r_k) = (L-1) * sum_{j=0}^{k} p_r(r_j)

    For color images (color=True) operates only on the L* channel;
    naive per-channel RGB equalisation is intentionally NOT the default
    because it shifts hue.

    Parameters
    ----------
    color : bool
        If True and img is 3-channel, use Lab L channel only.
    """
    if color and img.ndim == 3:
        lab, L = _rgb_to_lab_L(img)
        L_eq = exposure.equalize_hist(L)
        return _lab_merge_rgb(lab, L_eq)

    if img.ndim == 3:
        # Naïve per-channel (for failure-case demonstration)
        out = np.stack(
            [exposure.equalize_hist(img[:, :, c]) for c in range(img.shape[2])],
            axis=2,
        )
        return out

    return exposure.equalize_hist(img)


# ── 4. CLAHE ─────────────────────────────────────────────────────────────────

def clahe(img: np.ndarray, tile_size: int = 64, clip_limit: float = 0.03,
          color: bool = False) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Divides the image into non-overlapping tiles of `tile_size x tile_size`.
    Within each tile the histogram is clipped at:
        clip_thresh = clip_limit * tile_area / num_bins
    The excess is redistributed uniformly.  Tile borders are blended via
    bilinear interpolation.

    Uses skimage.exposure.equalize_adapthist under the hood.

    Parameters
    ----------
    tile_size : int
        Square tile side length in pixels.
    clip_limit : float
        Normalised clip limit in [0, 1].  Higher → stronger contrast.
    color : bool
        If True and img is RGB, apply to L* channel only.
    """
    if color and img.ndim == 3:
        lab, L = _rgb_to_lab_L(img)
        L_clahe = exposure.equalize_adapthist(L, kernel_size=tile_size,
                                              clip_limit=clip_limit)
        return _lab_merge_rgb(lab, L_clahe)

    if img.ndim == 3:
        lab, L = _rgb_to_lab_L(img)
        L_clahe = exposure.equalize_adapthist(L, kernel_size=tile_size,
                                              clip_limit=clip_limit)
        return _lab_merge_rgb(lab, L_clahe)

    return exposure.equalize_adapthist(img, kernel_size=tile_size,
                                       clip_limit=clip_limit)


# ── 5. Piecewise Linear Transform ─────────────────────────────────────────────

def piecewise_linear(img: np.ndarray,
                     control_points: list = None,
                     color: bool = False) -> np.ndarray:
    """
    Piecewise linear intensity mapping defined by a set of (r, s) control points.

    Between consecutive control points the mapping is linear; values are
    interpolated via numpy.interp, which implicitly clips at the boundary points.

    Default control points aggressively brighten shadows while compressing
    highlights:
        (0, 0) → (0.30, 0.60) → (0.70, 0.90) → (1, 1)

    Parameters
    ----------
    control_points : list of (float, float)
        Input-output pairs in [0,1] x [0,1], must start at (0,*) and end at (1,*).
    color : bool
        If True and img is RGB, apply to L* channel only.
    """
    if control_points is None:
        control_points = [(0.0, 0.0), (0.30, 0.60), (0.70, 0.90), (1.0, 1.0)]

    r_pts = [p[0] for p in control_points]
    s_pts = [p[1] for p in control_points]

    if color and img.ndim == 3:
        lab, L = _rgb_to_lab_L(img)
        return _lab_merge_rgb(lab, np.interp(L, r_pts, s_pts))

    return np.interp(img, r_pts, s_pts).reshape(img.shape)
