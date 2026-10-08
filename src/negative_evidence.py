"""Negative Spatial Evidence with Search-Quality Scoring.

Ingests failed field visits ('address_not_traceable') and their GPS trails.
Calculates search quality based on:
search_quality = trail duration * spatial coverage * movement evidence * GPS quality.

Constructs continuous Gaussian repulsion fields to penalize candidate landmark POIs
or localities where field agents thoroughly searched but could not trace the borrower.

Outputs: output/negative_exclusion_zones.csv
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import GEO, OUT, SHARED

R_VISUAL_M = 40.0


def compute_search_quality(
    g_points: pd.DataFrame,
) -> float:
    """Computes search-quality score in [0.05, 1.0] for a failed visit trajectory.
    
    Factors:
    - Duration: up to 600s (10 mins)
    - Spatial coverage: bounding box diagonal up to 300m
    - Movement evidence: points count (up to 40) and path distance (up to 500m)
    - GPS quality: inverse mean accuracy
    """
    n_pts = len(g_points)
    if n_pts == 0:
        return 0.05

    # Duration
    ts = pd.to_datetime(g_points["point_ts"])
    duration_s = max(1.0, (ts.iloc[-1] - ts.iloc[0]).total_seconds()) if n_pts >= 2 else 10.0
    f_duration = min(duration_s / 600.0, 1.0)

    # Spatial coverage (bounding box diagonal)
    dx_span = g_points["x"].max() - g_points["x"].min()
    dy_span = g_points["y"].max() - g_points["y"].min()
    bbox_diag = float(np.hypot(dx_span, dy_span))
    f_coverage = min(bbox_diag / 300.0, 1.0)

    # Movement distance along trajectory
    dxs = np.diff(g_points["x"].to_numpy())
    dys = np.diff(g_points["y"].to_numpy())
    total_path = float(np.sum(np.hypot(dxs, dys))) if n_pts >= 2 else 0.0
    f_movement = (min(n_pts / 40.0, 1.0) + min(total_path / 500.0, 1.0)) / 2.0

    # GPS accuracy factor
    mean_acc = float(g_points["accuracy_m"].mean())
    f_gps = max(0.2, min(1.0, 1.0 - (mean_acc / 50.0)))

    # Weighted search quality score
    quality = 0.35 * f_duration + 0.25 * f_coverage + 0.25 * f_movement + 0.15 * f_gps
    return float(np.clip(quality, 0.05, 1.0))


def build_negative_exclusion_zones(
    visits_df: pd.DataFrame,
    gps_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Tuple[Dict[str, List[Dict]], pd.DataFrame]:
    """Extracts failed visits, computes search-quality scores, and models exclusion zones.
    
    Returns:
        addr_fail_trajectories: dict mapping address_id -> list of trajectory point dicts with search_quality.
        neg_audit_df: DataFrame saved to output/negative_exclusion_zones.csv.
    """
    fail_visits = visits_df[visits_df["outcome"] == "address_not_traceable"].copy()
    fail_vids = set(fail_visits["visit_id"])

    fail_gps = gps_df[gps_df["visit_id"].isin(fail_vids)].merge(
        fail_visits[["visit_id", "address_id"]], on="visit_id", how="inner"
    )

    addr_fail_trajectories = {}
    audit_records = []

    for aid, g_addr in fail_gps.groupby("address_id"):
        addr_pts_list = []
        for vid, g_vis in g_addr.groupby("visit_id"):
            sq = compute_search_quality(g_vis)
            xs = g_vis["x"].to_numpy(dtype=float)
            ys = g_vis["y"].to_numpy(dtype=float)
            accs = g_vis["accuracy_m"].to_numpy(dtype=float)
            denoms = 2.0 * ((accs + R_VISUAL_M) ** 2)

            addr_pts_list.append({
                "visit_id": vid,
                "search_quality": sq,
                "x": xs,
                "y": ys,
                "denom": denoms,
            })

            audit_records.append({
                "address_id": aid,
                "visit_id": vid,
                "search_quality_score": round(sq, 3),
                "point_count": len(g_vis),
                "min_x": round(float(xs.min()), 1),
                "max_x": round(float(xs.max()), 1),
                "min_y": round(float(ys.min()), 1),
                "max_y": round(float(ys.max()), 1),
                "mean_accuracy_m": round(float(accs.mean()), 1),
            })

        addr_fail_trajectories[aid] = addr_pts_list

    neg_df = pd.DataFrame(audit_records)
    out_file = output_path or (OUT / "negative_exclusion_zones.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    neg_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"Negative Spatial Evidence: Modeled {len(fail_visits)} failed search visits across {len(addr_fail_trajectories)} addresses.")
    if not neg_df.empty:
        print(f"Search Quality Scores: mean={neg_df['search_quality_score'].mean():.2f}, min={neg_df['search_quality_score'].min():.2f}, max={neg_df['search_quality_score'].max():.2f}")
    print(f"Saved audit log to {out_file}")

    return addr_fail_trajectories, neg_df


def compute_repulsion_penalty(
    candidate_x: float,
    candidate_y: float,
    failed_trails_for_addr: List[Dict],
) -> float:
    """Computes the continuous Gaussian repulsion penalty for a candidate point."""
    if not failed_trails_for_addr:
        return 0.0

    total_repulsion = 0.0
    for trail in failed_trails_for_addr:
        sq = trail["search_quality"]
        xs = trail["x"]
        ys = trail["y"]
        denoms = trail["denom"]

        d2 = (candidate_x - xs) ** 2 + (candidate_y - ys) ** 2
        repulsion_trail = float(np.sum(np.exp(-d2 / denoms)))
        total_repulsion += sq * repulsion_trail

    return total_repulsion


if __name__ == "__main__":
    visits = pd.read_csv(SHARED / "field_visits.csv")
    gps = pd.read_csv(GEO / "visit_gps_points.csv")
    trajectories, df = build_negative_exclusion_zones(visits, gps)
