"""
pipelines.py
------------
Two complete end-to-end enhancement pipelines.

Pipeline 1 — Grayscale noise reduction + contrast enhancement
    Target images: camera, coins, moon (and their degraded versions)
    Rationale: Grayscale images need no colour-space conversion; operating
    directly in intensity space is lossless and avoids any hue artefacts.

Pipeline 2 — Colour image enhancement via CIE Lab luminance channel
    Target images: astronaut (degraded), coffee, cat
    Rationale: Applying histogram operations to all RGB channels independently
    shifts hue.  CIE L*a*b* is perceptually uniform; the L* channel isolates
    luminance from chrominance (a*, b*), so all contrast/sharpening operations
    are applied only to L*, leaving colour appearance unchanged.
"""

import numpy as np
from skimage import restoration

from .intensity_ops import clahe, gamma_correction, contrast_stretch, piecewise_linear
from .spatial_filters import median_filter_img, gaussian_filter_img, bilateral_filter, unsharp_mask, laplacian_sharpen


# ── Pipeline 1 ────────────────────────────────────────────────────────────────

def pipeline1_grayscale(img: np.ndarray,
                         verbose: bool = False) -> dict:
    """
    Grayscale enhancement pipeline.

    Steps
    -----
    1. Load as float64 [0,1]                 (assumed done by caller)
    2. Estimate noise level
    3. Median filter   — removes impulse noise (salt-and-pepper)
    4. Gaussian filter — applied only when estimated sigma > 0.03
    5. CLAHE           — local contrast enhancement
    6. Gamma correction — global brightness lift (gamma=1.8)
    7. Contrast stretch — ensures full dynamic range
    8. Laplacian sharpening — restores edges softened by filtering

    Returns
    -------
    dict with keys: 'input', 'after_median', 'after_gaussian',
                    'after_clahe', 'after_gamma', 'after_stretch',
                    'output'  (final result = after Laplacian)
    """
    assert img.ndim == 2, "Pipeline 1 expects a 2-D grayscale image."
    steps = {"input": img}

    # Step 3 — median filter (3×3): removes salt-and-pepper impulse noise
    after_median = median_filter_img(img, k=1)
    steps["after_median"] = after_median

    # Step 4 — conditional Gaussian (only if notable Gaussian noise remains)
    sigma_est = restoration.estimate_sigma(after_median)
    if verbose:
        print(f"  Estimated noise sigma after median: {sigma_est:.4f}")
    if sigma_est > 0.03:
        after_gaussian = gaussian_filter_img(after_median, sigma=0.8)
    else:
        after_gaussian = after_median
    steps["after_gaussian"] = after_gaussian

    # Step 5 — Contrast stretching FIRST: restore the compressed dynamic range.
    # Applying stretch before CLAHE ensures CLAHE operates on a well-distributed
    # histogram, preventing over-amplification of a flat/compressed signal.
    after_stretch = contrast_stretch(after_gaussian, p_low=1.0, p_high=99.0)
    steps["after_stretch"] = after_stretch

    # Step 6 — CLAHE: local contrast refinement on the already-stretched image.
    # Lower clip_limit (0.02) reduces the risk of noise amplification.
    after_clahe = clahe(after_stretch, tile_size=64, clip_limit=0.02)
    steps["after_clahe"] = after_clahe

    # Step 7 — Gamma correction: only applied when image is still notably dark
    # (mean < 0.38) to avoid over-brightening images with reasonable exposure.
    mean_l = float(np.mean(after_clahe))
    if mean_l < 0.38:
        after_gamma = gamma_correction(after_clahe, gamma=1.6)
        if verbose:
            print(f"  Gamma applied (mean={mean_l:.3f})")
    else:
        after_gamma = after_clahe
        if verbose:
            print(f"  Gamma skipped (mean={mean_l:.3f} >= 0.38)")
    steps["after_gamma"] = after_gamma

    # Step 8 — Laplacian sharpening (c=0.5 — conservative, avoids ringing)
    output = laplacian_sharpen(after_gamma, c=0.5, connectivity=4)
    steps["output"] = output

    return steps


# ── Pipeline 2 ────────────────────────────────────────────────────────────────

def pipeline2_color_lab(img: np.ndarray,
                         verbose: bool = False) -> dict:
    """
    Colour image enhancement via the CIE L*a*b* colour space.

    CIE L*a*b* formula (D65 illuminant):
        L* = 116·f(Y/Yn) − 16
        a* = 500·(f(X/Xn) − f(Y/Yn))
        b* = 200·(f(Y/Yn) − f(Z/Zn))
        f(t) = t^(1/3)  if t > (6/29)^3,  else (1/3)·(29/6)^2·t + 4/29

    Steps
    -----
    1. RGB → CIE Lab                    — decouple luminance from chrominance
    2. Extract L* channel, normalise to [0,1]
    3. Bilateral filter on L*           — edge-preserving smoothing
    4. CLAHE on L*                      — local contrast enhancement
    5. Piecewise linear on L*           — brighten shadows, compress highlights
    6. Unsharp masking on L*            — detail restoration
    7. Recombine L* with a*, b*
    8. Lab → RGB (clip to [0,1])

    Returns
    -------
    dict with keys: 'input', 'L_original', 'L_bilateral', 'L_clahe',
                    'L_piecewise', 'L_unsharp', 'output'
    """
    assert img.ndim == 3, "Pipeline 2 expects a 3-D RGB image."
    from skimage import color as skcolor

    steps = {"input": img}

    # Step 1–2 — RGB → Lab, extract L
    lab = skcolor.rgb2lab(img)
    L_orig = lab[:, :, 0] / 100.0          # normalise to [0,1]
    steps["L_original"] = L_orig

    # Step 3 — bilateral filter (edge-preserving denoising)
    L_bil = bilateral_filter(L_orig, d=9, sigma_color=25, sigma_space=9)
    steps["L_bilateral"] = L_bil
    if verbose:
        print(f"  L bilateral: min={L_bil.min():.3f}  max={L_bil.max():.3f}")

    # Step 4 — CLAHE (local contrast)
    L_clahe = clahe(L_bil, tile_size=64, clip_limit=0.02)
    steps["L_clahe"] = L_clahe

    # Step 5 — piecewise linear (aggressive shadow lift)
    ctrl_pts = [(0.0, 0.0), (0.30, 0.55), (0.75, 0.92), (1.0, 1.0)]
    L_pw = piecewise_linear(L_clahe, control_points=ctrl_pts)
    steps["L_piecewise"] = L_pw

    # Step 6 — unsharp masking (fine edge detail)
    L_sharp = unsharp_mask(L_pw, sigma=1.5, lam=1.2)
    steps["L_unsharp"] = L_sharp

    # Step 7–8 — recombine and convert back to RGB
    lab_out = lab.copy()
    lab_out[:, :, 0] = np.clip(L_sharp, 0.0, 1.0) * 100.0
    output = np.clip(skcolor.lab2rgb(lab_out), 0.0, 1.0)
    steps["output"] = output

    return steps
