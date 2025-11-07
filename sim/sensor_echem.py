"""
sensor_echem.py — Map occupancy θ(t) to a sensor signal S(t) (electrochemical surrogate)

PURPOSE:
    Convert θ(t) from binding.py into a measurable signal S(t) before noise/electronics.

EXPANDED TERMS:
    - S: sensor signal (arbitrary units).
    - S0: baseline signal (offset).
    - α (alpha), β (beta): sensitivity coefficients.

MODEL (low-complexity polynomial):
    S = S0 + α·θ + β·θ^2

WHY SIMPLE?
    We want a controllable surrogate. You can later replace this with
    a more realistic mapping (e.g., impedance model, voltammetry shape, or optical spectra).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EchemMap:
    """Electrochemical-style mapping from occupancy θ to signal S."""

    S0: float = 0.0
    alpha: float = 1.0
    beta: float = 0.0

    def map(self, theta: np.ndarray) -> np.ndarray:
        """
        Args:
            theta: occupancy array (0..1), shape (T,)

        Returns:
            S: signal array, same shape as θ.
        """
        theta = theta.astype(float, copy=False)
        return self.S0 + self.alpha * theta + self.beta * (theta**2)
