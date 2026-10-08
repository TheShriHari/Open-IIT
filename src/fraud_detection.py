"""Multi-Factor Fraud Detection & Visit Integrity Gating.

Implements generic detection of fraudulent visits without hardcoding agent IDs or coordinates:
1. Perceptual Photo Hash Reuse (reused across >= 3 distinct borrower accounts).
2. Agent Check-in Spatial Clustering (dense recurring check-in spots across >= 10-20 distinct accounts within <= 40m).
3. Trajectory Breadcrumb Shape Gating (truncated trail <= 10 points co-occurring with dense clusters).

Outputs: output/fraudulent_visits.csv
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from config import OUT, SHARED, GEO


def detect_fraudulent_visits(
    visits_df: pd.DataFrame,
    gps_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates multi-factor fraud score and gates fraudulent check-ins.
    
    Returns:
        clean_visits_df: legitimate visits retained for pin synthesis.
        fraud_audit_df: all purged fraudulent visits with audit trail.
    """
    v = visits_df.copy()

    # Ingest trail lengths if GPS points provided
    if gps_df is not None:
        trail_lens = gps_df.groupby("visit_id").size().rename("trail_len")
        v = v.merge(trail_lens, on="visit_id", how="left")
        v["trail_len"] = v["trail_len"].fillna(0).astype(int)
    elif "trail_len" not in v.columns:
        v["trail_len"] = 26  # default median if GPS table unavailable

    # 1. Perceptual Photo Hash Reuse Detection (reused across >= 3 accounts)
    account_reuse_count = v.groupby("photo_hash")["account_id"].nunique()
    photo_fraud_hashes = set(account_reuse_count[account_reuse_count >= 3].index)
    is_photo_fraud = v["photo_hash"].isin(photo_fraud_hashes)

    # 2. Generic Agent Check-in Spatial Clustering (Fixed-location spoofing)
    is_cluster_fraud = np.zeros(len(v), dtype=bool)
    cluster_density_account_count = np.zeros(len(v), dtype=int)

    for agent_id, g in v.groupby("agent_id"):
        coords = g[["checkin_x", "checkin_y"]].values
        if len(coords) < 10:
            continue
        tree = KDTree(coords)
        for i, idx in enumerate(g.index):
            # Query within tight 40m radius
            near_indices = tree.query_ball_point(coords[i], r=40.0)
            n_unique_accounts = g.iloc[near_indices]["account_id"].nunique()
            cluster_density_account_count[idx] = n_unique_accounts

            # Fraud criteria:
            # - Massive cluster (>= 20 unique accounts within 40m radius)
            # - Dense cluster (>= 10 unique accounts within 40m) + truncated trail (<= 10 breadcrumbs)
            t_len = g.iloc[i]["trail_len"]
            if (n_unique_accounts >= 20) or (n_unique_accounts >= 10 and t_len <= 10):
                is_cluster_fraud[idx] = True

    # Combined fraud gating
    is_fraud = is_photo_fraud | is_cluster_fraud

    v["is_fraud"] = is_fraud
    v["fraud_type"] = np.where(
        is_photo_fraud & is_cluster_fraud,
        "photo_and_cluster_spoof",
        np.where(is_photo_fraud, "photo_reuse_fraud", np.where(is_cluster_fraud, "fixed_location_cluster_spoof", "none")),
    )
    v["cluster_account_count"] = cluster_density_account_count
    v["photo_account_reuse_count"] = v["photo_hash"].map(account_reuse_count).fillna(1).astype(int)

    fraud_df = v[is_fraud].copy()
    clean_df = v[~is_fraud].copy()

    # Export audit CSV
    out_file = output_path or (OUT / "fraudulent_visits.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    audit_cols = [
        "visit_id",
        "agent_id",
        "account_id",
        "address_id",
        "visit_date",
        "checkin_x",
        "checkin_y",
        "fraud_type",
        "photo_hash",
        "photo_account_reuse_count",
        "cluster_account_count",
        "trail_len",
    ]
    fraud_df[audit_cols].to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"Fraud Gating: Processed {len(v)} visits.")
    print(f"Identified {len(fraud_df)} fraudulent visits ({is_photo_fraud.sum()} photo reuse, {is_cluster_fraud.sum()} fixed-location clusters).")
    print(f"Purged {len(fraud_df)} visits -> Saved to {out_file}")
    print(f"Retained {len(clean_df)} clean visits for downstream spatial reasoning.")

    return clean_df, fraud_df


if __name__ == "__main__":
    visits = pd.read_csv(SHARED / "field_visits.csv")
    gps = pd.read_csv(GEO / "visit_gps_points.csv")
    clean, fraud = detect_fraudulent_visits(visits, gps)
    print("\nFraudulent visits by agent:")
    print(fraud.agent_id.value_counts().to_string())
