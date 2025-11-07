"""
digital_twin.py — End-to-end synthetic data generator for the Icarus dry-lab

PURPOSE (what this file does):
    1) Loads a "profile" (a YAML config) that defines a target use-case
       (e.g., BSI = Bloodstream Infection, or VAP = Ventilator-Associated Pneumonia).
    2) Simulates the full chain:
          physics (binding kinetics) → sensor mapping → noise/drift →
          simple electronics (filter + ADC) → labeled signals on disk.
    3) Writes a metadata CSV that points to each saved signal .npy file.
       Downstream training/eval scripts read this CSV.

MAJOR CONCEPTS (expanded once):
    - ODE: Ordinary Differential Equation.
    - θ (theta): Fraction of receptor/binding sites that are occupied (0..1).
    - k_on / k_off: Association / dissociation rate constants in simple binding kinetics.
    - DOE: Design Of Experiments; the grid of concentrations, temperatures, etc.
    - ADC: Analog-to-Digital Converter (quantizes analog signal to a fixed number of bits).
    - AR(1): Autoregressive (order 1) drift process: x_t = ρ·x_{t-1} + ε_t.
    - LoD: Limit of Detection (used later in evaluation, not here).
    - YAML: Human-readable config file format (our profiles live in validation/profiles/*.yaml).

FILE OUTPUTS:
    data/synthetic/<profile_name>/sig_000123.npy   # signal arrays
    data/synthetic/<profile_name>/metadata.csv     # per-sample labels & conditions

USAGE:
    python sim/digital_twin.py --profile validation/profiles/bsi.yaml
    python sim/digital_twin.py --profile validation/profiles/vap.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yaml

# --- Local module imports (these live in sim/ alongside this file) ---
# binding.py     -> binding kinetics ODE simulator
# sensor_echem.py-> simple electrochemical-style mapping θ → signal S
# noise.py       -> gaussian, 1/f noise, AR(1) drift, ADC quantization helpers
# electronics.py -> simple RC low-pass and ADC application
from .binding import Kinetics, simulate_theta
from .sensor_echem import EchemMap
from .noise import gaussian_noise, one_over_f_noise, ar1_drift
from .electronics import rc_lowpass, apply_adc


# -----------------------
# Config / Profile schema
# -----------------------
@dataclass
class ClassSpec:
    """One biological class in the dataset (e.g., NEGATIVE vs PATHOGEN_MIX).
    In the profile we store approximate kinetics to emulate different affinities.
    """

    label: str
    kin: Kinetics
    prior: float  # Not used in the DOE grid (we generate both classes per cell), but kept for future sampling.


@dataclass
class NoiseSpec:
    gaussian_sigma_base: float = 0.01
    one_over_f_strength: float = 0.005
    spike_probability: float = 0.0  # chance per sample to inject a few spikes (0..1)


@dataclass
class DriftSpec:
    ar1_rho: float = 0.995
    ar1_sigma: float = 1e-4


@dataclass
class ElectronicsSpec:
    adc_bits: int = 12
    vmin: float = -1.0
    vmax: float = 1.0
    lowpass_alpha: float = 0.6
    saturation_clip: bool = (
        False  # if True, clip before ADC to emulate sensor saturation
    )


@dataclass
class Profile:
    name: str
    concentrations: List[float]
    temperatures_C: List[float]
    replicates_per_cell: int
    classes: List[ClassSpec]
    noise: NoiseSpec
    drift: DriftSpec
    electronics: ElectronicsSpec
    # Simulation runtime knobs
    steps: int = 512  # number of time steps per signal
    dt_s: float = 0.1  # seconds per step
    mapper: EchemMap = EchemMap(S0=0.0, alpha=1.2, beta=0.1)  # θ → signal mapping


def load_profile(path: Path) -> Profile:
    """Parse a YAML profile into strongly-typed objects."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Classes (convert dicts → ClassSpec)
    classes = []
    for c in raw["classes"]:
        kin = Kinetics(k_on=float(c["kin"]["k_on"]), k_off=float(c["kin"]["k_off"]))
        classes.append(
            ClassSpec(label=str(c["label"]), kin=kin, prior=float(c.get("prior", 0.5)))
        )

    # Noise / drift / electronics
    nraw = raw.get("noise", {})
    draw = raw.get("drift", {})
    eraw = raw.get("electronics", {})

    profile = Profile(
        name=str(raw["name"]),
        concentrations=list(map(float, raw["concentrations"])),
        temperatures_C=list(map(float, raw["temperatures_C"])),
        replicates_per_cell=int(raw["replicates_per_cell"]),
        classes=classes,
        noise=NoiseSpec(
            gaussian_sigma_base=float(nraw.get("gaussian_sigma_base", 0.01)),
            one_over_f_strength=float(nraw.get("one_over_f_strength", 0.005)),
            spike_probability=float(nraw.get("spike_probability", 0.0)),
        ),
        drift=DriftSpec(
            ar1_rho=float(draw.get("ar1_rho", 0.995)),
            ar1_sigma=float(draw.get("ar1_sigma", 1e-4)),
        ),
        electronics=ElectronicsSpec(
            adc_bits=int(eraw.get("adc_bits", 12)),
            vmin=float(eraw.get("vmin", -1.0)),
            vmax=float(eraw.get("vmax", 1.0)),
            lowpass_alpha=float(eraw.get("lowpass_alpha", 0.6)),
            saturation_clip=bool(
                eraw.get(
                    "saturation_clip",
                    False
                    or raw.get("eval", {})
                    .get("stress", {})
                    .get("saturation_clip", False),
                )
            ),
        ),
        steps=int(raw.get("steps", 512)),
        dt_s=float(raw.get("dt_s", 0.1)),
        mapper=EchemMap(  # You can make this configurable later if needed
            S0=0.0, alpha=1.2, beta=0.1
        ),
    )
    return profile


# -------------------------
# Synthetic signal building
# -------------------------
def add_spikes(
    x: np.ndarray,
    rng: np.random.Generator,
    n_spikes_range: Tuple[int, int] = (1, 3),
    amp_range: Tuple[float, float] = (0.05, 0.15),
) -> np.ndarray:
    """Inject a few impulsive artifacts (to mimic line draw hiccups, motion, etc.)."""
    x = x.copy()
    n_spikes = rng.integers(n_spikes_range[0], n_spikes_range[1] + 1)
    N = len(x)
    for _ in range(n_spikes):
        i = int(rng.integers(0, N))
        amp = float(rng.uniform(amp_range[0], amp_range[1])) * (
            1 if rng.random() < 0.5 else -1
        )
        x[i] += amp
    return x


def synth_signal_for_cell(
    *,
    klass: ClassSpec,
    concentration: float,
    temperature_C: float,
    profile: Profile,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a single synthetic signal for one DOE cell and class.

    PIPELINE (what happens and why):
        1) Binding kinetics integration:
           dθ/dt = k_on * C * (1 - θ) - k_off * θ
           This yields θ(t), the occupancy over time.
        2) Sensor mapping:
           S_raw = S0 + α·θ + β·θ^2  (EchemMap)
           Converts occupancy into an instrument signal (arbitrary units).
        3) Noise:
           Add Gaussian (thermal), 1/f (flicker) components, and optional spikes.
           We increase Gaussian sigma slightly with temperature to emulate thermal effects.
        4) Drift:
           Add AR(1) slow drift to capture baseline wander over time.
        5) Electronics:
           Apply a simple low-pass (RC-like) filter, then quantize via ADC to emulate digitization.
    """
    # 1) Occupancy θ(t) from binding kinetics (ODE integrated via Forward Euler inside simulate_theta).
    theta = simulate_theta(
        C=concentration,
        kin=klass.kin,
        dt=profile.dt_s,
        steps=profile.steps,
        theta0=0.0,
    )

    # 2) Map θ(t) → signal S(t) via simple polynomial (S0 + αθ + βθ^2).
    s = profile.mapper.map(theta)

    # 3) Add noise terms.
    N = len(s)
    # Gaussian noise baseline, with a mild temperature dependence (hotter → slightly noisier).
    temp_sigma = profile.noise.gaussian_sigma_base * (
        1.0 + 0.01 * (temperature_C - 25.0)
    )
    s = s + gaussian_noise(N, sigma=temp_sigma, rng=rng)

    # 1/f noise for low-frequency wander (simulates flicker noise).
    s = s + one_over_f_noise(N, strength=profile.noise.one_over_f_strength, rng=rng)

    # Optional impulsive spikes (rare artifacts).
    if rng.random() < profile.noise.spike_probability:
        s = add_spikes(s, rng)

    # 4) Add slow AR(1) drift (baseline creep).
    s = s + ar1_drift(
        N, rho=profile.drift.ar1_rho, sigma=profile.drift.ar1_sigma, rng=rng
    )

    # 5) Simple electronics chain.
    #    a) Low-pass filter (crudely mimics anti-aliasing / bandwidth limit).
    s = rc_lowpass(s, alpha=profile.electronics.lowpass_alpha)

    #    b) Optional "sensor saturation" before ADC, simulating frontend rails.
    if profile.electronics.saturation_clip:
        s = np.clip(s, profile.electronics.vmin, profile.electronics.vmax)

    #    c) ADC quantization to N bits within [vmin, vmax].
    s = apply_adc(
        s,
        bits=profile.electronics.adc_bits,
        vmin=profile.electronics.vmin,
        vmax=profile.electronics.vmax,
    )

    return s.astype(np.float32)


# -----------------
# Dataset generation
# -----------------
def ensure_out_dirs(profile_name: str) -> Path:
    """Create (if needed) the output directory for this profile."""
    out_dir = Path("data") / "synthetic" / profile_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def generate_dataset(profile: Profile, seed: int = 42) -> pd.DataFrame:
    """
    Generate the full DOE grid for the given profile.

    We loop over:
        - concentrations (log-spaced in profile)
        - temperatures (°C)
        - classes (NEGATIVE vs PATHOGEN_MIX, etc.)
        - replicates_per_cell (repeat to build distribution)

    For each combination, we synthesize a signal and write it to disk,
    returning a metadata table with columns that downstream code can join on.
    """
    rng = np.random.default_rng(seed)
    out_dir = ensure_out_dirs(profile.name)

    rows: List[Dict[str, Any]] = []
    sample_id = 0

    # Helpful print so you see the DOE size before running
    total = (
        len(profile.concentrations)
        * len(profile.temperatures_C)
        * len(profile.classes)
        * profile.replicates_per_cell
    )
    print(
        f"[digital_twin] Profile '{profile.name}' → generating {total} samples "
        f"({len(profile.concentrations)} conc × {len(profile.temperatures_C)} temp × "
        f"{len(profile.classes)} classes × {profile.replicates_per_cell} reps)"
    )

    for C in profile.concentrations:
        for T in profile.temperatures_C:
            for klass in profile.classes:
                for rep in range(profile.replicates_per_cell):
                    s = synth_signal_for_cell(
                        klass=klass,
                        concentration=C,
                        temperature_C=T,
                        profile=profile,
                        rng=rng,
                    )

                    # Save the signal to a .npy file (compact and fast to load).
                    sig_path = out_dir / f"sig_{sample_id:06d}.npy"
                    np.save(sig_path, s)

                    # Assign split: a simple round-robin to get a small test set.
                    split = "test" if (sample_id % 10 == 0) else "train"

                    # Record all relevant metadata for downstream training/evaluation.
                    rows.append(
                        {
                            "id": sample_id,
                            "profile": profile.name,
                            "class": klass.label,
                            "concentration": C,
                            "temperature_C": T,
                            "steps": profile.steps,
                            "dt_s": profile.dt_s,
                            "adc_bits": profile.electronics.adc_bits,
                            "vmin": profile.electronics.vmin,
                            "vmax": profile.electronics.vmax,
                            "lowpass_alpha": profile.electronics.lowpass_alpha,
                            "drift_rho": profile.drift.ar1_rho,
                            "drift_sigma": profile.drift.ar1_sigma,
                            "gaussian_sigma_base": profile.noise.gaussian_sigma_base,
                            "one_over_f_strength": profile.noise.one_over_f_strength,
                            "spike_probability": profile.noise.spike_probability,
                            "signal_path": str(sig_path.as_posix()),
                            "split": split,
                            "seed": seed,
                        }
                    )
                    sample_id += 1

    df = pd.DataFrame(rows)
    meta_path = out_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)
    print(
        f"[digital_twin] Wrote {len(df)} samples and metadata to: {meta_path.as_posix()}"
    )
    return df


# ----
# CLI
# ----
def main():
    ap = argparse.ArgumentParser(
        description="Icarus digital twin synthetic data generator"
    )
    ap.add_argument(
        "--profile",
        type=str,
        default="validation/profiles/bsi.yaml",
        help="Path to YAML profile (e.g., validation/profiles/bsi.yaml or vap.yaml)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = ap.parse_args()

    profile_path = Path(args.profile)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    profile = load_profile(profile_path)

    # Show a compact summary of what we loaded (great for debugging).
    summary = {
        "name": profile.name,
        "concentrations": profile.concentrations,
        "temperatures_C": profile.temperatures_C,
        "replicates_per_cell": profile.replicates_per_cell,
        "classes": [c.label for c in profile.classes],
        "noise": {
            "gaussian_sigma_base": profile.noise.gaussian_sigma_base,
            "one_over_f_strength": profile.noise.one_over_f_strength,
            "spike_probability": profile.noise.spike_probability,
        },
        "drift": {
            "ar1_rho": profile.drift.ar1_rho,
            "ar1_sigma": profile.drift.ar1_sigma,
        },
        "electronics": {
            "adc_bits": profile.electronics.adc_bits,
            "vmin": profile.electronics.vmin,
            "vmax": profile.electronics.vmax,
            "lowpass_alpha": profile.electronics.lowpass_alpha,
            "saturation_clip": profile.electronics.saturation_clip,
        },
        "steps": profile.steps,
        "dt_s": profile.dt_s,
    }
    print("[digital_twin] Loaded profile:\n" + json.dumps(summary, indent=2))

    # Generate!
    generate_dataset(profile=profile, seed=args.seed)


if __name__ == "__main__":
    main()
