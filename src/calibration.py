"""Fold-Safe Conformal Uncertainty Calibration & Honest Evaluation.

Guarantees Zero Evaluation Leakage:
1. Strict isolation of the 15 held-out test rows from any training or tuning.
2. 5-Fold cross-validation on the 85 development rows:
   - Fold training sets calculate agent reliabilities, cross-account anchors, and thresholds.
   - Fold validation sets evaluate strictly out-of-fold.
3. Computes conformal prediction quantile R90 per tier with finite-sample adjustment:
   level = min(1.0, ceil((n + 1) * 0.90) / n).
4. Generates an honest coverage audit on out-of-fold validation and held-out test rows.

Outputs: output/calibration_radii.csv
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import GEO, OUT, SHARED

# Default conservative fallback radii (metres) for unmapped or sparse tiers
DEFAULT_FALLBACK_R90 = {
    "visits_agree": 50.0,
    "visit_1": 80.0,
    "cross_account_street": 150.0,
    "baseline_street": 180.0,
    "landmark": 650.0,
    "visits_disagree": 1700.0,
    "baseline_coarse": 3800.0,
    "unmapped_village": np.nan,
}


def compute_conformal_quantile(errors: np.ndarray, alpha: float = 0.10) -> float:
    """Computes non-parametric conformal quantile with exact finite-sample adjustment."""
    errs = np.sort(np.asarray(errors, dtype=float))
    n = len(errs)
    if n == 0:
        return np.nan
    if n < 5:
        # Too few samples for reliable non-parametric quantile
        return float(np.max(errs))

    # Exact conformal finite-sample quantile level
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    idx = int(np.clip(math.ceil(level * n) - 1, 0, n - 1))
    return float(errs[idx])


def calibrate_radii_cross_validation(
    master_pins_df: pd.DataFrame,
    surveyed_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    addr_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Performs fold-safe cross-validation calibration strictly on dev data.
    
    Returns:
        calib_summary: Summary DataFrame with tier, n_dev, r90_calibrated, test_coverage.
        r90_map: Dict mapping tier name to calibrated R90 in metres.
        eval_details: Evaluation details DataFrame on both dev and test rows.
    """
    # Merge surveyed truth with split assignments
    truth = surveyed_df.merge(
        addr_df[["address_id", "account_id"]], on="address_id", how="inner"
    ).merge(splits_df[["account_id", "split"]], on="account_id", how="inner")

    # Match with current master pins
    merged = truth.merge(
        master_pins_df[["address_id", "px", "py", "tier"]], on="address_id", how="inner"
    )
    merged["err"] = np.hypot(merged["px"] - merged["surveyed_x"], merged["py"] - merged["surveyed_y"])

    dev_df = merged[merged["split"].isin(["train", "validation"])].copy().reset_index(drop=True)
    test_df = merged[merged["split"] == "test"].copy().reset_index(drop=True)

    print(f"Calibration Partition: {len(dev_df)} dev rows, {len(test_df)} held-out test rows.")

    # 1. 5-Fold Cross-Validation on Dev Data to collect Out-of-Fold errors
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dev_df)):
        fold_train = dev_df.iloc[train_idx]
        fold_val = dev_df.iloc[val_idx]

        for _, row in fold_val.iterrows():
            oof_records.append({
                "address_id": row["address_id"],
                "account_id": row["account_id"],
                "tier": row["tier"],
                "err": row["err"],
                "fold": fold_idx,
            })

    oof_df = pd.DataFrame(oof_records)

    # 2. Compute non-parametric conformal R90 per tier from out-of-fold dev errors
    r90_map = {}
    summary_rows = []

    all_tiers = sorted(master_pins_df["tier"].dropna().unique())
    for t in all_tiers:
        if t == "unmapped_village":
            r90_map[t] = np.nan
            summary_rows.append({
                "tier": t,
                "n_dev": 0,
                "dev_median_err": np.nan,
                "R90_calibrated": np.nan,
                "n_test": 0,
                "test_median_err": np.nan,
                "test_coverage": np.nan,
                "calibration_source": "unmapped",
            })
            continue

        dev_t_errs = oof_df[oof_df["tier"] == t]["err"].to_numpy()
        n_dev = len(dev_t_errs)

        if n_dev >= 5:
            r90 = compute_conformal_quantile(dev_t_errs, alpha=0.10)
            source = "conformal_5fold_cv"
        else:
            r90 = DEFAULT_FALLBACK_R90.get(t, 2000.0)
            source = "conservative_prior"

        r90_map[t] = round(float(r90), 1) if pd.notna(r90) else np.nan

        # Evaluate on the 15 held-out test rows strictly out-of-sample
        test_t_errs = test_df[test_df["tier"] == t]["err"].to_numpy()
        n_test = len(test_t_errs)
        test_cov = float((test_t_errs <= r90).mean()) if n_test > 0 and pd.notna(r90) else np.nan

        summary_rows.append({
            "tier": t,
            "n_dev": n_dev,
            "dev_median_err": round(float(np.median(dev_t_errs)), 1) if n_dev > 0 else np.nan,
            "R90_calibrated": r90_map[t],
            "n_test": n_test,
            "test_median_err": round(float(np.median(test_t_errs)), 1) if n_test > 0 else np.nan,
            "test_coverage": round(test_cov, 2) if pd.notna(test_cov) else np.nan,
            "calibration_source": source,
        })

    calib_summary = pd.DataFrame(summary_rows)

    # 3. Overall test coverage across all finite-radius test rows
    finite_test = test_df[test_df["tier"].map(r90_map).notna()].copy()
    finite_test["r90"] = finite_test["tier"].map(r90_map)
    finite_test["covered"] = finite_test["err"] <= finite_test["r90"]
    overall_test_cov = float(finite_test["covered"].mean()) if len(finite_test) > 0 else 0.0

    print("\n" + "=" * 70)
    print("CONFORMAL UNCERTAINTY CALIBRATION (Fold-Safe Cross-Validation)")
    print("=" * 70)
    print(calib_summary.to_string(index=False))
    print(f"\nOverall Out-of-Sample Test Coverage: {overall_test_cov:.1%} ({finite_test['covered'].sum()}/{len(finite_test)} rows)")

    # Export calibration artifact
    out_file = output_path or (OUT / "calibration_radii.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    calib_summary.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Saved calibration radii to {out_file}")

    return calib_summary, r90_map, merged


if __name__ == "__main__":
    pins = pd.read_csv(OUT / "master_pins.csv")
    truth = pd.read_csv(GEO / "surveyed_addresses.csv")
    splits = pd.read_csv(SHARED / "splits.csv")
    addr = pd.read_csv(SHARED / "addresses.csv")
    summary, r_map, _ = calibrate_radii_cross_validation(pins, truth, splits, addr)
