"""Visit Pin Synthesis with Canonical Dwell Coordinates & Learned Agent Reliability.

Implements:
1. Gating of multi-factor fraudulent visits.
2. Canonical doorstep dwell coordinate mapping.
3. Expanded outcome acceptance: cash_collected, met_borrower, met_family, locked_premises, neighbour_says_shifted.
4. Replaces inverted 1/accuracy^2 weighting with Outcome Weight * Agent Reliability * Linguistic Remark Confidence.
5. Weighted median pin synthesis and agreement spread calculation.

Outputs: output/visit_derived_pins.csv
"""

from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd

from config import GEO, OUT, SHARED
from fraud_detection import detect_fraudulent_visits
from dwell_kinematics import apply_dwell_corrections
from indic_address_understanding import process_single_remark

# Accepted outcome weights (highest for verified physical transactions/meetings)
OUTCOME_BASE_WEIGHTS = {
    "cash_collected": 2.5,
    "met_borrower": 1.4,
    "met_family": 1.3,
    "locked_premises": 1.0,
    "neighbour_says_shifted": 0.7,
}

AGREE_SPREAD_M = 100.0


def weighted_median(vals: np.ndarray, weights: np.ndarray) -> float:
    """Computes the 1D weighted median."""
    v = np.asarray(vals, float)
    w = np.asarray(weights, float)
    if len(v) == 0:
        return np.nan
    if len(v) == 1:
        return float(v[0])
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cumsum = np.cumsum(w_sorted)
    cutoff = 0.5 * cumsum[-1]
    idx = np.searchsorted(cumsum, cutoff)
    return float(v_sorted[min(idx, len(v_sorted) - 1)])


def single_linkage_clusters(x: np.ndarray, y: np.ndarray, cluster_m: float = 100.0) -> np.ndarray:
    """Computes single-linkage spatial cluster labels for 2D points at distance cluster_m."""
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=int)
    dists = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    labels = -np.ones(n, dtype=int)
    k = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = k
        while stack:
            curr = stack.pop()
            neighbors = np.where((dists[curr] <= cluster_m) & (labels < 0))[0]
            for nb in neighbors:
                labels[nb] = k
                stack.append(nb)
        k += 1
    return labels


def cap_agent_weights(w: np.ndarray, agents: np.ndarray, cap: float = 0.8) -> np.ndarray:
    """Scales down any single agent holding > cap share of the weight when other agents exist."""
    w = w.copy()
    unique_agents = np.unique(agents)
    if len(unique_agents) < 2:
        return w
    total_w = w.sum()
    for ag in unique_agents:
        mask = (agents == ag)
        ag_w = w[mask].sum()
        rest_w = total_w - ag_w
        if ag_w > cap * total_w and rest_w > 0:
            target = rest_w * cap / (1.0 - cap)
            w[mask] *= (target / ag_w)
            total_w = rest_w + target
    return w


def learn_agent_reliability(
    clean_visits: pd.DataFrame,
    surveyed_training_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Computes empirical agent reliability weights fold-safely with Bayesian shrinkage."""
    reliabilities = {}
    survey_err_map = {}

    if surveyed_training_df is not None and not surveyed_training_df.empty:
        # Compute ground truth error per agent on training fold only
        v_surv = clean_visits.merge(surveyed_training_df, on="address_id")
        if not v_surv.empty:
            v_surv["err"] = np.hypot(v_surv["checkin_x"] - v_surv["surveyed_x"], v_surv["checkin_y"] - v_surv["surveyed_y"])
            for ag, g in v_surv.groupby("agent_id"):
                survey_err_map[ag] = float(g["err"].median())

    for ag, g in clean_visits.groupby("agent_id"):
        # Dwell rate: fraction of visits with confirmed stationary dwell
        dwell_rate = float(g["dwell_applied"].mean()) if "dwell_applied" in g.columns else 0.5
        
        # Base factor from dwell adherence
        rel = 0.8 + 0.4 * dwell_rate  # range [0.8, 1.2]

        # Survey error adjustment if available
        if ag in survey_err_map:
            med_err = survey_err_map[ag]
            if med_err <= 25.0:
                rel += 0.20
            elif med_err >= 150.0:
                rel -= 0.30

        # Bayesian shrinkage regularization: pull towards 1.0 based on sample size n
        n_visits = len(g)
        shrinkage = n_visits / (n_visits + 10.0)
        final_rel = 1.0 + shrinkage * (rel - 1.0)
        reliabilities[ag] = float(np.clip(final_rel, 0.5, 1.5))

    return reliabilities


def synthesize_visit_pins(
    visits_df: pd.DataFrame,
    gps_df: Optional[pd.DataFrame] = None,
    pois_df: Optional[pd.DataFrame] = None,
    addr_df: Optional[pd.DataFrame] = None,
    surveyed_training_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    cluster_m: float = 100.0,
    tie_ratio: float = 0.80,
    trim_sigmas: float = 3.0,
    enable_clustering: bool = True,
    enable_trimming: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates authoritative visit-derived pins using canonical dwell coordinates,
    learned reliability, dominant cluster selection, and 3-sigma outlier trimming.
    
    Execution Order:
    raw GPS -> dwell/kinematic correction -> fraud filtering -> composite weights
    -> spatial clustering -> dominant cluster -> 3σ trimming -> weighted-median pin.
    """
    # 1. Trajectory kinematics & dwell correction on raw GPS
    if gps_df is not None:
        can_visits, dwell_audit = apply_dwell_corrections(visits_df, gps_df)
    else:
        can_visits = visits_df.copy()
        can_visits["dwell_applied"] = False

    # 2. Multi-factor fraud detection & integrity filtering (purging fake check-ins)
    clean_visits, fraud_visits = detect_fraudulent_visits(can_visits, gps_df)

    # 3. Filter to accepted outcomes
    accepted_v = clean_visits[clean_visits["outcome"].isin(OUTCOME_BASE_WEIGHTS.keys())].copy()

    # 4. Attach town_id and baseline prior if available
    if addr_df is not None:
        merge_cols = [c for c in ["address_id", "town_id", "geocoder_x", "geocoder_y"] if c in addr_df.columns]
        accepted_v = accepted_v.merge(addr_df[merge_cols], on="address_id", how="left")
    elif "town_id" not in accepted_v.columns:
        accepted_v["town_id"] = None

    # 5. Ingest remark confidences fold-safely / via Indic NLP module
    rem_conf = []
    rem_status = []
    for row in accepted_v.itertuples(index=False):
        parsed = process_single_remark(
            remark_raw=row.remark,
            town_id=getattr(row, "town_id", None),
            pois_df=pois_df,
            checkin_x=row.checkin_x,
            checkin_y=row.checkin_y,
        )
        rem_conf.append(parsed["remark_confidence"])
        rem_status.append(parsed["remark_status"])

    accepted_v["remark_confidence"] = rem_conf
    accepted_v["remark_status"] = rem_status

    # 6. Compute learned agent reliabilities
    agent_rel = learn_agent_reliability(can_visits, surveyed_training_df)
    accepted_v["agent_reliability"] = accepted_v["agent_id"].map(agent_rel).fillna(1.0)

    # 7. Calculate composite visit weights
    base_w = accepted_v["outcome"].map(OUTCOME_BASE_WEIGHTS).fillna(1.0)
    agent_w = accepted_v["agent_reliability"]

    # Remark multiplier
    rem_factor = np.ones(len(accepted_v), dtype=float)
    for i, (st, c) in enumerate(zip(accepted_v["remark_status"], accepted_v["remark_confidence"])):
        if st == "STRONG_CORRECTION":
            rem_factor[i] = 1.0 + 1.5 * c
        elif st == "USEFUL_SPATIAL_CUE":
            rem_factor[i] = 1.0 + 0.8 * c
        elif st == "CONFLICTING":
            rem_factor[i] = 0.25

    raw_w = (base_w * agent_w * rem_factor).to_numpy()
    accepted_v["w"] = raw_w

    # Prior reference coordinates map for tie-breaking
    prior_ref_map = {}
    if addr_df is not None and "geocoder_x" in addr_df.columns and "geocoder_y" in addr_df.columns:
        for _, r in addr_df[["address_id", "geocoder_x", "geocoder_y"]].dropna().iterrows():
            prior_ref_map[r["address_id"]] = (float(r["geocoder_x"]), float(r["geocoder_y"]))

    # 8. Synthesize pins with robust cluster selection and 3-sigma trimming
    pin_rows = []
    accepted_v_with_audit = []

    for aid, g in accepted_v.groupby("address_id"):
        xs = g["checkin_x"].to_numpy(dtype=float)
        ys = g["checkin_y"].to_numpy(dtype=float)
        ws = g["w"].to_numpy(dtype=float)
        agents = g["agent_id"].to_numpy()
        n_orig = len(g)

        # Cap agent weight concentration
        ws = cap_agent_weights(ws, agents, cap=0.8)

        if not enable_clustering or n_orig <= 1:
            # Baseline unclustered
            x_kept = xs
            y_kept = ys
            w_kept = ws
            disagree = False
            dominant_cluster_size = n_orig
            discarded_clusters_count = 0
            trimmed_points_count = 0
        else:
            # Step A: 100m (or cluster_m) single-linkage spatial clustering
            labels = single_linkage_clusters(xs, ys, cluster_m=cluster_m)
            n_clusters = labels.max() + 1
            disagree = (n_clusters > 1)

            if n_clusters == 1:
                best_cluster = 0
                discarded_clusters_count = 0
            else:
                # Step B: Dominant cluster selection with tie-breaker
                c_weights = np.array([ws[labels == k].sum() for k in range(n_clusters)])
                top_clusters = np.argsort(-c_weights)
                c0 = int(top_clusters[0])
                c1 = int(top_clusters[1])

                if c_weights[c1] >= tie_ratio * c_weights[c0]:
                    # Near-tie: compare robust center of both clusters to text prior
                    c0_x = weighted_median(xs[labels == c0], ws[labels == c0])
                    c0_y = weighted_median(ys[labels == c0], ws[labels == c0])
                    c1_x = weighted_median(xs[labels == c1], ws[labels == c1])
                    c1_y = weighted_median(ys[labels == c1], ws[labels == c1])

                    ref_x, ref_y = prior_ref_map.get(aid, (c0_x, c0_y))
                    d0 = np.hypot(c0_x - ref_x, c0_y - ref_y)
                    d1 = np.hypot(c1_x - ref_x, c1_y - ref_y)
                    best_cluster = c0 if d0 <= d1 else c1
                else:
                    best_cluster = c0

                discarded_clusters_count = int(np.sum(labels != best_cluster))

            keep_mask = (labels == best_cluster)
            dominant_cluster_size = int(keep_mask.sum())
            x_kept = xs[keep_mask]
            y_kept = ys[keep_mask]
            w_kept = ws[keep_mask]

            # Step C: 3-sigma outlier trimming within dominant cluster
            trimmed_points_count = 0
            if enable_trimming and len(x_kept) >= 3:
                for _ in range(2):
                    cx = weighted_median(x_kept, w_kept)
                    cy = weighted_median(y_kept, w_kept)
                    d = np.hypot(x_kept - cx, y_kept - cy)
                    med_d = weighted_median(d, w_kept)
                    sigma = max(15.0, 1.4826 * med_d)
                    inliers = (d <= trim_sigmas * sigma)
                    if inliers.any() and not inliers.all():
                        trimmed_points_count += int((~inliers).sum())
                        x_kept = x_kept[inliers]
                        y_kept = y_kept[inliers]
                        w_kept = w_kept[inliers]

        # Step D: Final weighted median from kept visits
        wx = weighted_median(x_kept, w_kept)
        wy = weighted_median(y_kept, w_kept)
        n_final = len(x_kept)
        spread = float(np.hypot(x_kept - wx, y_kept - wy).max()) if n_final >= 2 else 0.0

        if n_final == 1:
            tier = "visit_1"
        elif disagree or spread > AGREE_SPREAD_M:
            tier = "visits_disagree"
        else:
            tier = "visits_agree"

        max_rem_c = float(g["remark_confidence"].max()) if len(g) > 0 else 0.0

        pin_rows.append({
            "address_id": aid,
            "visit_px": round(wx, 1),
            "visit_py": round(wy, 1),
            "n_good_visits": n_final,
            "visit_tier": tier,
            "remark_confidence": round(max_rem_c, 3),
            "visit_spread_m": round(spread, 1),
            "dominant_cluster_size": dominant_cluster_size,
            "discarded_clusters_count": discarded_clusters_count,
            "trimmed_outliers_count": trimmed_points_count,
        })

    pins_df = pd.DataFrame(pin_rows)
    out_file = output_path or (OUT / "visit_derived_pins.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pins_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"Pin Synthesis: Generated {len(pins_df)} visit-derived address pins.")
    print(pins_df["visit_tier"].value_counts().to_string())
    print(f"Saved pins to {out_file}")

    return pins_df, accepted_v


if __name__ == "__main__":
    visits = pd.read_csv(SHARED / "field_visits.csv")
    gps = pd.read_csv(GEO / "visit_gps_points.csv")
    pois = pd.read_csv(GEO / "landmarks_poi.csv")
    addr = pd.read_csv(SHARED / "addresses.csv")
    pins, _ = synthesize_visit_pins(visits, gps, pois, addr)
