"""
eval.py — Evaluate a trained model for a given profile (BSI, VAP, etc.)

WHAT THIS FILE DOES (expanded once):
  - Loads test data metadata from:
        data/synthetic/<profile>/metadata.csv
  - Loads the trained model from:
        models/<profile>/logreg.joblib
  - Computes:
        * ROC-AUC (Receiver Operating Characteristic, area-under-curve)
        * PR-AUC  (Precision–Recall, area-under-curve)
        * A point-estimate LoD (Limit of Detection) using a fixed operating point:
              choose a threshold where FPR ≈ --lod_fpr (default 0.05),
              then find the lowest concentration with sensitivity (TPR) ≥ --lod_tpr (default 0.95)
        * Per-temperature metrics (ROC-AUC by temperature)
  - Saves:
        models/<profile>/eval_overall.json
        models/<profile>/per_temperature.csv
        models/<profile>/lod_summary.json
        (and optional plots: fig_roc.png, fig_pr.png if matplotlib is present)

TERMS (expanded once):
  - ROC: Receiver Operating Characteristic (TPR vs FPR as threshold varies)
  - PR:  Precision–Recall curve (precision vs recall)
  - TPR: True Positive Rate (sensitivity)
  - FPR: False Positive Rate (1 - specificity)
  - LoD: Limit of Detection (lowest concentration achieving target sensitivity at a chosen FPR)

USAGE:
  python ml/eval.py --profile bsi
  python ml/eval.py --profile vap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

# Feature extractor
from .featurize import features_from_signal


# ----------------
# Helper functions
# ----------------
def profile_paths(profile: str) -> Tuple[Path, Path, Path]:
    """
    Returns:
      meta_path:  data/synthetic/<profile>/metadata.csv
      model_path: models/<profile>/logreg.joblib
      out_dir:    models/<profile>  (for saving reports/plots)
    """
    meta_path = Path("data") / "synthetic" / profile / "metadata.csv"
    out_dir = Path("models") / profile
    model_path = out_dir / "logreg.joblib"
    return meta_path, model_path, out_dir


def is_positive(label: str) -> int:
    """Convention: any label not starting with 'NEGATIVE' is positive."""
    return 0 if str(label).upper().startswith("NEGATIVE") else 1


def load_test_split(meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X, y, and an array of per-row temperatures for the 'test' split.
    """
    df = meta[meta["split"] == "test"].copy()
    if df.empty:
        raise ValueError("No 'test' rows found. Re-generate data or check splits.")
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    temps: List[float] = []
    for _, r in df.iterrows():
        sig = np.load(r["signal_path"]).astype(float)
        feat = features_from_signal(sig)
        X_list.append(feat)
        y_list.append(is_positive(r["class"]))
        temps.append(float(r["temperature_C"]))
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    temps = np.array(temps, dtype=float)
    return X, y, temps


def lod_point_estimate(
    y_true: np.ndarray,
    scores: np.ndarray,
    meta_test: pd.DataFrame,
    target_tpr: float = 0.95,
    fpr_at: float = 0.05,
) -> Dict[str, float]:
    """
    Compute a simple LoD (Limit of Detection) point estimate:
      1) Pick threshold τ such that global FPR ≈ fpr_at (choose closest point on ROC).
      2) For each concentration level, compute TPR at τ using only positive-class samples.
      3) Return the smallest concentration whose TPR ≥ target_tpr.

    Returns a dict with:
      lod_concentration, threshold_tau, fpr_selected, tpr_at_tau
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    idx = int(np.argmin(np.abs(fpr - fpr_at)))
    tau = float(thr[idx])
    fpr_sel = float(fpr[idx])

    # Merge scores back to metadata rows (test split only)
    df = meta_test.copy()
    df = df.assign(score=scores, y=y_true)

    # Compute per-concentration TPR at threshold tau
    results = []
    for C, dC in df.groupby("concentration"):
        # positives in this concentration
        mask_pos = dC["y"].values == 1
        if mask_pos.any():
            tprC = float((dC.loc[mask_pos, "score"].values >= tau).mean())
        else:
            tprC = np.nan
        results.append((float(C), tprC))
    results.sort(key=lambda z: z[0])

    lod = np.nan
    for C, tprC in results:
        if np.isnan(tprC):
            continue
        if tprC >= target_tpr:
            lod = C
            break

    # Also compute TPR/FPR at tau for the whole test set for reference
    tpr_at_tau = (
        float((scores[y_true == 1] >= tau).mean())
        if (y_true == 1).any()
        else float("nan")
    )

    return {
        "lod_concentration": float(lod) if not np.isnan(lod) else float("nan"),
        "threshold_tau": tau,
        "fpr_selected": fpr_sel,
        "tpr_at_tau_overall": tpr_at_tau,
    }


def try_save_plots(out_dir: Path, y: np.ndarray, scores: np.ndarray) -> None:
    """
    Save ROC and PR plots if matplotlib is available.
    (We don't specify colors; defaults are fine.)
    """
    try:
        import matplotlib.pyplot as plt  # optional
    except Exception:
        print("[eval] matplotlib not installed; skipping plots.")
        return

    # ROC
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_roc.png", dpi=150)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, scores)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_pr.png", dpi=150)
    plt.close()


# -----
#  Main
# -----
def main():
    ap = argparse.ArgumentParser(description="Evaluate model for a given profile")
    ap.add_argument(
        "--profile", type=str, default="bsi", help="Profile name (e.g., 'bsi' or 'vap')"
    )
    ap.add_argument(
        "--lod_tpr", type=float, default=0.95, help="Target sensitivity (TPR) for LoD"
    )
    ap.add_argument(
        "--lod_fpr",
        type=float,
        default=0.05,
        help="FPR operating point at which to select threshold",
    )
    ap.add_argument(
        "--save_plots",
        action="store_true",
        help="If set, save ROC/PR plots to models/<profile>/",
    )
    args = ap.parse_args()

    # Resolve paths
    meta_path, model_path, out_dir = profile_paths(args.profile)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} (train first)")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata & test split
    meta = pd.read_csv(meta_path)
    meta_test = meta[meta["split"] == "test"].copy()
    Xte, yte, temps = load_test_split(meta)

    # Load model and score
    model = load(model_path)
    scores = model.predict_proba(Xte)[:, 1]

    # Overall metrics
    roc_auc = float(roc_auc_score(yte, scores))
    pr_auc = float(average_precision_score(yte, scores))
    print(f"[eval] Profile={args.profile}  ROC-AUC={roc_auc:.3f} | PR-AUC={pr_auc:.3f}")

    # LoD point estimate
    lod = lod_point_estimate(
        y_true=yte,
        scores=scores,
        meta_test=meta_test,
        target_tpr=args.lod_tpr,
        fpr_at=args.lod_fpr,
    )
    print(
        f"[eval] LoD estimate @ TPR>={args.lod_tpr:.2f} & FPR≈{args.lod_fpr:.2f}: "
        f"concentration ≈ {lod['lod_concentration']}"
    )

    # Per-temperature ROC-AUC table (diagnostics for robustness)
    per_temp_rows = []
    for T in sorted(np.unique(temps)):
        m = temps == T
        if m.sum() >= 5 and len(np.unique(yte[m])) > 1:
            roc_t = roc_auc_score(yte[m], scores[m])
        else:
            roc_t = np.nan
        per_temp_rows.append({"temperature_C": float(T), "roc_auc": float(roc_t)})

    per_temp_df = pd.DataFrame(per_temp_rows)
    print("[eval] Per-temperature ROC-AUC:")
    print(per_temp_df.to_string(index=False))

    # Optional plots
    if args.save_plots:
        try_save_plots(out_dir, yte, scores)

    # Save reports
    with open(out_dir / "eval_overall.json", "w") as f:
        json.dump(
            {
                "profile": args.profile,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "n_test": int(len(yte)),
            },
            f,
            indent=2,
        )

    per_temp_df.to_csv(out_dir / "per_temperature.csv", index=False)
    with open(out_dir / "lod_summary.json", "w") as f:
        json.dump(lod, f, indent=2)

    print(f"[eval] Saved → {out_dir / 'eval_overall.json'}")
    print(f"[eval] Saved → {out_dir / 'per_temperature.csv'}")
    print(f"[eval] Saved → {out_dir / 'lod_summary.json'}")
    if args.save_plots:
        print(f"[eval] Saved → {out_dir / 'fig_roc.png'} and 'fig_pr.png'")


if __name__ == "__main__":
    main()
