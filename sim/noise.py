"""
noise.py — Noise, drift, and quantization utilities

PURPOSE:
    Add realism to the clean sensor signal:
      - Gaussian noise (thermal-like)
      - 1/f noise ("flicker", low-frequency drift-like component)
      - AR(1) slow drift
      - ADC quantization (Analog-to-Digital Converter)

EXPANDED TERMS:
    - 1/f noise: power spectral density ∝ 1/frequency (more low-frequency content).
    - AR(1): Autoregressive model of order 1 → x_t = ρ x_{t-1} + ε_t (|ρ|<1).
    - ADC: Analog-to-Digital Converter (quantizes to a finite number of levels).
"""

from __future__ import annotations
import numpy as np


def gaussian_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean Gaussian noise with standard deviation sigma."""
    return rng.normal(loc=0.0, scale=float(sigma), size=int(n))


def one_over_f_noise(n: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """
    Very simple 1/f-shaped noise:
        - Start with white noise in frequency domain.
        - Scale amplitudes ∝ 1/sqrt(f) (avoid f=0 bin).
        - Transform back to time domain and normalize to 'strength'.

    NOTE:
        This is a convenient approximation for dry-lab simulation, not a rigorous generator.
    """
    n = int(n)
    white = rng.normal(0.0, 1.0, size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)  # [0..0.5] for normalized sampling rate

    scale = np.ones_like(freqs)
    # Avoid division by zero at DC (freq=0); keep it as 1.0
    idx = freqs > 0
    scale[idx] = 1.0 / np.sqrt(freqs[idx])

    shaped = np.fft.irfft(spectrum * scale, n=n)
    # Normalize to requested 'strength' (as approximate std)
    shaped = shaped * (float(strength) / (np.std(shaped) + 1e-12))
    return shaped


def ar1_drift(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    AR(1) slow drift:
        x_t = ρ x_{t-1} + ε_t,  with ε_t ~ N(0, sigma^2)

    Args:
        n: length
        rho: persistence (0.99..0.999 is "very slow" drift)
        sigma: innovation noise (std)

    Returns:
        x: drift series of length n
    """
    n = int(n)
    x = np.zeros(n, dtype=float)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0.0, float(sigma))
    return x


def quantize_adc(x: np.ndarray, bits: int, vmin: float, vmax: float) -> np.ndarray:
    """
    Uniform ADC quantization into 2^bits levels spanning [vmin, vmax].

    Args:
        x: analog signal
        bits: number of bits (e.g., 12 means 4096 levels)
        vmin, vmax: input range the ADC can represent

    Returns:
        x_q: quantized signal
    """
    x = x.astype(float, copy=False)
    levels = 2 ** int(bits)
    x = np.clip(x, vmin, vmax)
    # Map to [0, levels-1], round, then map back to [vmin, vmax]
    q = np.round((x - vmin) / (vmax - vmin) * (levels - 1))
    return vmin + q * (vmax - vmin) / (levels - 1)
