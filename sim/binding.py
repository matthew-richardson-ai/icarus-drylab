"""
binding.py — Simple binding kinetics for occupancy θ(t)

PURPOSE:
    Simulate how target molecules bind/unbind to a sensor's surface receptors over time.
    Output is θ(t), the fraction of occupied binding sites (0..1).

EXPANDED TERMS:
    - ODE: Ordinary Differential Equation.
    - θ (theta): fraction of binding sites that are occupied (dimensionless).
    - k_on: association (on-rate) constant.
    - k_off: dissociation (off-rate) constant.

MODEL (well-mixed, first-order):
    dθ/dt = k_on * C * (1 - θ) - k_off * θ
    where C is analyte concentration (relative units here).

NUMERICS:
    We use a simple Forward-Euler integrator: θ_{t+1} = θ_t + dt * f(θ_t).
    For small dt this is stable enough for our dry-lab use.

NOTE:
    Units are "relative" because we simulate; later you will calibrate real hardware.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Kinetics:
    """Kinetic parameters for one class/species."""

    k_on: float  # association rate constant (relative units)
    k_off: float  # dissociation rate constant (relative units)


def simulate_theta(
    C: float,
    kin: Kinetics,
    dt: float,
    steps: int,
    theta0: float = 0.0,
) -> np.ndarray:
    """
    Integrate the ODE for θ(t) over 'steps' time steps of size 'dt'.

    Args:
        C: concentration (relative units).
        kin: Kinetics(k_on, k_off).
        dt: time step in seconds.
        steps: number of samples in the output trajectory.
        theta0: initial occupancy at t=0 (0..1).

    Returns:
        theta: np.ndarray shape (steps,) with values in [0,1].
    """
    theta = np.zeros(steps, dtype=float)
    theta[0] = float(np.clip(theta0, 0.0, 1.0))

    for t in range(1, steps):
        dtheta = kin.k_on * C * (1.0 - theta[t - 1]) - kin.k_off * theta[t - 1]
        theta[t] = theta[t - 1] + dt * dtheta
        # Physically, occupancy cannot go below 0 or above 1:
        if theta[t] < 0.0:
            theta[t] = 0.0
        elif theta[t] > 1.0:
            theta[t] = 1.0

    return theta
