"""
electronics.py — Minimal electronics chain

PURPOSE:
    Emulate two common front-end effects:
      1) Low-pass filtering (anti-alias / bandwidth limit) via a simple single-pole RC-style filter.
      2) ADC (Analog-to-Digital Converter) quantization using noise.quantize_adc().

EXPANDED TERMS:
    - RC low-pass: simple infinite impulse response (IIR) filter y[i] = α x[i] + (1-α) y[i-1].
    - α (alpha): smoothing factor in (0,1); smaller α = stronger smoothing (lower cutoff).

USAGE:
    x = rc_lowpass(x, alpha=0.6)
    x_q = apply_adc(x, bits=12, vmin=-1.0, vmax=1.0)
"""

from __future__ import annotations
import numpy as np
from .noise import quantize_adc


def rc_lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Single-pole IIR low-pass filter.

    Args:
        x: input signal
        alpha: smoothing factor in (0,1). Smaller → heavier smoothing.

    Returns:
        y: filtered signal
    """
    x = x.astype(float, copy=False)
    y = np.zeros_like(x)
    y[0] = x[0]
    a = float(alpha)
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def apply_adc(
    x: np.ndarray, bits: int = 12, vmin: float = -1.0, vmax: float = 1.0
) -> np.ndarray:
    """
    Convenience wrapper around uniform ADC quantization.

    Args:
        x: input signal
        bits: number of ADC bits (levels = 2^bits)
        vmin, vmax: input range

    Returns:
        quantized signal
    """
    return quantize_adc(x, bits=int(bits), vmin=float(vmin), vmax=float(vmax))
