"""
spatial_filters.py
------------------
Spatial filtering operations: smoothing and sharpening.

All functions accept float64 images in [0, 1] and return float64 [0, 1].
Color images (ndim == 3) are filtered channel-by-channel unless otherwise noted.
"""

import numpy as np
import cv2
from scipy.ndimage import (
    uniform_filter,
    gaussian_filter,
    median_filter as _scipy_median,
    laplace,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_per_channel(fn, img, **kwargs):
    """Apply a single-channel function to each channel of an RGB image."""
    if img.ndim == 2:
        return fn(img, **kwargs)
    return np.stack([fn(img[:, :, c], **kwargs) for c in range(img.shape[2])], axis=2)


# ── 1. Mean (Box) Filter ─────────────────────────────────────────────────────

def mean_filter(img: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Averaging filter with a square (2k+1) x (2k+1) kernel.

        I_out(x,y) = 1/((2k+1)^2) * sum_{s,t in [-k,k]} I(x+s, y+t)

    Parameters
    ----------
    k : int
        Half-size of the kernel.  k=1 → 3x3,  k=2 → 5x5.
    """
    size = 2 * k + 1
    return np.clip(_apply_per_channel(uniform_filter, img, size=size), 0.0, 1.0)


# ── 2. Gaussian Filter ────────────────────────────────────────────────────────

def gaussian_filter_img(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian smoothing kernel:

        h(x,y) = 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))

    Kernel is truncated at ±3*sigma.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian in pixels.
    """
    return np.clip(_apply_per_channel(gaussian_filter, img, sigma=sigma), 0.0, 1.0)


# ── 3. Median Filter ─────────────────────────────────────────────────────────

def median_filter_img(img: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Non-linear median filter.  The output at each pixel is the median of the
    (2k+1)x(2k+1) neighbourhood.

        I_out(x,y) = median{ I(x+s, y+t) : s,t in [-k,k] }

    Highly effective for salt-and-pepper noise because the median is
    insensitive to extreme (impulse) outliers.

    Parameters
    ----------
    k : int
        Half-size of the neighbourhood.  k=1 → 3x3 window.
    """
    size = 2 * k + 1
    return np.clip(_apply_per_channel(_scipy_median, img, size=size), 0.0, 1.0)


# ── 4. Bilateral Filter ───────────────────────────────────────────────────────

def bilateral_filter(img: np.ndarray, d: int = 9,
                     sigma_color: float = 25.0,
                     sigma_space: float = 9.0) -> np.ndarray:
    """
    Edge-preserving bilateral filter.

    The output is a weighted average where weights depend on both
    spatial proximity (g_s) and radiometric similarity (f_r):

        I_out(x,y) = (1/W) * sum_{(s,t) in N} I(s,t) * f_r(...) * g_s(...)

        f_r(delta) = exp(-delta^2 / (2*sigma_r^2))   radiometric kernel
        g_s(d)     = exp(-d^2    / (2*sigma_s^2))    spatial kernel
        W          = normalisation constant

    OpenCV's implementation uses uint8 scale [0,255] internally.
    sigma_color = 25  corresponds to sigma_r ≈ 0.10 in [0,1] float scale.

    Parameters
    ----------
    d            : int   Filter diameter (neighbourhood size in pixels).
    sigma_color  : float Photometric bandwidth (OpenCV uint8 scale, default 25).
    sigma_space  : float Spatial bandwidth in pixels.
    """
    def _bilateral_channel(ch):
        ch_u8 = (np.clip(ch, 0.0, 1.0) * 255).astype(np.uint8)
        out_u8 = cv2.bilateralFilter(ch_u8, d=d,
                                     sigmaColor=sigma_color,
                                     sigmaSpace=sigma_space)
        return out_u8.astype(np.float64) / 255.0

    if img.ndim == 2:
        return _bilateral_channel(img)
    return np.stack([_bilateral_channel(img[:, :, c])
                     for c in range(img.shape[2])], axis=2)


# ── 5. Unsharp Masking ────────────────────────────────────────────────────────

def unsharp_mask(img: np.ndarray, sigma: float = 1.5,
                 lam: float = 1.5) -> np.ndarray:
    """
    Unsharp masking — adds a scaled high-frequency residual back to the image:

        I_sharp = I + lambda * (I - G_sigma(I))
                = (1 + lambda) * I - lambda * G_sigma(I)

    The term (I - G_sigma(I)) is a high-pass (detail) signal.

    Parameters
    ----------
    sigma : float  Standard deviation of the Gaussian low-pass blur.
    lam   : float  Sharpening strength (lambda).  lam=0 → no sharpening.
    """
    blur = gaussian_filter_img(img, sigma=sigma)
    return np.clip(img + lam * (img - blur), 0.0, 1.0)


# ── 6. Laplacian Sharpening ───────────────────────────────────────────────────

def laplacian_sharpen(img: np.ndarray, c: float = 1.0,
                      connectivity: int = 4) -> np.ndarray:
    """
    Laplacian-based sharpening using the isotropic discrete Laplacian operator.

    4-connectivity kernel (used by scipy.ndimage.laplace):
        L4 = [[0,  1, 0],
              [1, -4, 1],
              [0,  1, 0]]

    8-connectivity kernel (stronger, includes diagonals):
        L8 = [[1,  1, 1],
              [1, -8, 1],
              [1,  1, 1]]

    Sharpened image:
        I_sharp = I - c * (L * I)

    The subtraction adds edges because L*I is negative at peaks.

    Parameters
    ----------
    c            : float  Sharpening coefficient (default 1.0).
    connectivity : int    4 (scipy default) or 8 (manual kernel).
    """
    if connectivity == 4:
        def _lap(ch):
            return laplace(ch)
    else:
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
        from scipy.ndimage import convolve
        def _lap(ch):
            return convolve(ch, kernel)

    return np.clip(_apply_per_channel(
        lambda ch: ch - c * _lap(ch), img
    ), 0.0, 1.0)
