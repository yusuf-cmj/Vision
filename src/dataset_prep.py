"""
dataset_prep.py
---------------
Load scikit-image built-in images and apply controlled degradations
to produce reference/degraded pairs for quantitative evaluation.
"""

import numpy as np
from skimage import data, img_as_float, io, color
from pathlib import Path


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_builtin_images() -> dict:
    """Return all 8 built-in images as float64 [0,1] arrays."""
    images = {
        "camera":    img_as_float(data.camera()),
        "coins":     img_as_float(data.coins()),
        "moon":      img_as_float(data.moon()),
        "brick":     img_as_float(data.brick()),
        "astronaut": img_as_float(data.astronaut()),
        "coffee":    img_as_float(data.coffee()),
        "cat":       img_as_float(data.cat()),
        "chelsea":   img_as_float(data.chelsea()),
    }
    return images


def save_raw_images(images: dict, out_dir: str):
    """Save all built-in images to data/raw/ as PNG."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        io.imsave(str(out / f"{name}.png"), (np.clip(img, 0, 1) * 255).astype(np.uint8))


# ── Degradation functions ─────────────────────────────────────────────────────

def degrade_low_contrast(img: np.ndarray, alpha: float = 0.4, beta: float = 0.3) -> np.ndarray:
    """
    Linear contrast compression: maps [0,1] → [beta, alpha+beta].

    I_degraded = clip(alpha * I + beta, 0, 1)
    """
    return np.clip(alpha * img + beta, 0.0, 1.0)


def degrade_gaussian_noise(img: np.ndarray, sigma: float = 0.08, rng=None) -> np.ndarray:
    """
    Additive zero-mean Gaussian noise.

    I_degraded = clip(I + N(0, sigma^2), 0, 1)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0.0, 1.0)


def degrade_salt_pepper(img: np.ndarray, p: float = 0.05, rng=None) -> np.ndarray:
    """
    Salt-and-pepper noise: each pixel independently becomes 0, 1, or unchanged.

    P(salt) = p/2,  P(pepper) = p/2,  P(unchanged) = 1 - p
    """
    if rng is None:
        rng = np.random.default_rng(42)
    out = img.copy()
    mask = rng.random(img.shape)
    out[mask < p / 2] = 0.0
    out[(mask >= p / 2) & (mask < p)] = 1.0
    return out


def degrade_gamma_darkening(img: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    """
    Simulate underexposure via power-law darkening.

    I_degraded = I^gamma   (gamma > 1 → darker)
    """
    return np.power(np.clip(img, 0.0, 1.0), gamma)


def degrade_vignette(img: np.ndarray, sigma_ratio: float = 0.6) -> np.ndarray:
    """
    Uneven illumination via a 2-D Gaussian vignette mask.

    V(x,y) = exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma_v^2))
    I_degraded = I * V  (V normalised to peak at 1)
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    sigma_v = max(h, w) * sigma_ratio
    yy, xx = np.mgrid[0:h, 0:w]
    V = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_v ** 2))
    V = V / V.max()
    if img.ndim == 3:
        V = V[:, :, np.newaxis]
    return np.clip(img * V, 0.0, 1.0)


def degrade_combined(img: np.ndarray, rng=None) -> np.ndarray:
    """
    Multi-step degradation mimicking a real bad photograph:
      1. Gamma darkening (gamma=2.0)
      2. Gaussian noise  (sigma=0.08)
      3. Contrast compression (alpha=0.6, beta=0.1)
    Used for the astronaut controlled experiment.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    out = degrade_gamma_darkening(img, gamma=2.0)
    out = degrade_gaussian_noise(out, sigma=0.08, rng=rng)
    out = degrade_low_contrast(out, alpha=0.6, beta=0.1)
    return out


# ── Controlled experiment pairs ───────────────────────────────────────────────

def build_degraded_pairs(images: dict, rng=None) -> dict:
    """
    Build the 4 controlled (reference, degraded) pairs required by the rubric.

    Returns a dict:
        {name: {"ref": ndarray, "degraded": ndarray, "method": str}}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pairs = {
        "astronaut_combined": {
            "ref":      images["astronaut"],
            "degraded": degrade_combined(images["astronaut"], rng=rng),
            "method":   "gamma darkening(2.0) + Gaussian noise(σ=0.08) + contrast compression(α=0.6, β=0.1)",
        },
        "camera_lowcontrast": {
            "ref":      images["camera"],
            "degraded": degrade_low_contrast(images["camera"], alpha=0.4, beta=0.3),
            "method":   "Low-contrast compression (α=0.4, β=0.3)",
        },
        "moon_saltpepper": {
            "ref":      images["moon"],
            "degraded": degrade_salt_pepper(images["moon"], p=0.05, rng=rng),
            "method":   "Salt-and-pepper noise (p=0.05)",
        },
        "coins_vignette": {
            "ref":      images["coins"],
            "degraded": degrade_vignette(
                degrade_gamma_darkening(images["coins"], gamma=2.0)
            ),
            "method":   "Vignette + gamma darkening(2.0)",
        },
    }
    return pairs
