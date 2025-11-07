"""
featurize.py — Convert a raw time-series signal into numeric features

PURPOSE (expanded once):
  Turn each 1-D signal array (length T) into a compact feature vector for
  simple, fast models (like Logistic Regression). This keeps training quick
  and supports embedded/edge constraints later.

FEATURES we compute (kept small and general-purpose):
  - mean:       average signal level
  - std:        standard deviation (variability)
  - ptp:        peak-to-peak (max - min)
  - slope:      best-fit line slope across the series (trend)
  - fft_band_1, fft_band_2, fft_band_3: coarse frequency-band energies from the
                magnitude of the real FFT (Fast Fourier Transform)

NOTES:
  - We avoid dependencies beyond NumPy to keep the pipeline light.
  - These features are deliberately simple; you can extend with wavelets,
    more bands, or domain-specific descriptors once your dry-lab stabilizes.
"""

from __future__ import annotations
import numpy as np


def features_from_signal(x: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D signal into a small, fixed-size feature vector.

    Args:
      x: np.ndarray, shape (T,), the time-series (float)

    Returns:
      np.ndarray, shape (7,), ordered as:
        [mean, std, ptp, slope, fft_band_1, fft_band_2, fft_band_3]
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        # pad with zeros for degenerate inputs
        return np.zeros(7, dtype=float)

    # ---- basic stats ----
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    ptp = float(np.max(x) - np.min(x))

    # ---- linear trend (least-squares slope) ----
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones(n)]).T  # design matrix for y = m*t + b
    slope, _intercept = np.linalg.lstsq(A, x, rcond=None)[0]
    slope = float(slope)

    # ---- coarse FFT band energies ----
    # Real FFT magnitudes
    X = np.fft.rfft(x)
    mag = np.abs(X)

    # Split the spectrum magnitude into 3 contiguous chunks and sum power
    thirds = np.array_split(mag, 3)
    band_energies = [float(np.sum(b**2)) for b in thirds]  # power ~ |X|^2

    return np.array
