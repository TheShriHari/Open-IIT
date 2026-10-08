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


def learn_agent_reliability(
    clean_visits: pd.DataFrame,
    surveyed_training_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Computes empirical agent reliability weights fold-safely.
    
    Factors:
    - Consistency rate across multi-visit addresses
    - Dwell frequency
    - Error on surveyed training accounts (if provided)
    """
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

        reliabilities[ag] = float(np.clip(rel, 0.4, 1.5))

    return reliabilities


def synthesize_visit_pins(
    visits_df: pd.DataFrame,
    gps_df: Optional[pd.DataFrame] = None,
    pois_df: Optional[pd.DataFrame] = None,
    addr_df: Optional[pd.DataFrame] = None,
    surveyed_training_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates authoritative visit-derived pins from canonical dwell coordinates and learned weights.
    
    Returns:
        visit_pins_df: DataFrame of address pins (address_id, visit_px, visit_py, n_good_visits, visit_tier, remark_confidence, spread_m)
        canonical_visits_df: Clean visits with dwell coordinates and assigned weights.
    """
    # 1. Multi-factor fraud detection
    clean_visits, fraud_visits = detect_fraudulent_visits(visits_df, gps_df)

    # 2. Canonical trajectory kinematics & dwell correction
    if gps_df is not None:
        can_visits, dwell_audit = apply_dwell_corrections(clean_visits, gps_df)
    else:
        can_visits = clean_visits.copy()
        can_visits["dwell_applied"] = False

    # 3. Filter to accepted outcomes
    accepted_v = can_visits[can_visits["outcome"].isin(OUTCOME_BASE_WEIGHTS.keys())].copy()

    # 4. Attach town_id if available
    if addr_df is not None and "town_id" in addr_df.columns:
        accepted_v = accepted_v.merge(addr_df[["address_id", "town_id"]], on="address_id", how="left")
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

    accepted_v["w"] = base_w * agent_w * rem_factor

    # 8. Synthesize pins via weighted median
    pin_rows = []
    for aid, g in accepted_v.groupby("address_id"):
        wx = weighted_median(g["checkin_x"].to_numpy(), g["w"].to_numpy())
        wy = weighted_median(g["checkin_y"].to_numpy(), g["w"].to_numpy())
        n = len(g)
        spread = float(np.hypot(g["checkin_x"] - wx, g["checkin_y"] - wy).max()) if n >= 2 else 0.0
        tier = "visit_1" if n == 1 else "visits_agree" if spread <= AGREE_SPREAD_M else "visits_disagree"
        max_rem_c = float(g["remark_confidence"].max()) if len(g) > 0 else 0.0

        pin_rows.append({
            "address_id": aid,
            "visit_px": round(wx, 1),
            "visit_py": round(wy, 1),
            "n_good_visits": n,
            "visit_tier": tier,
            "remark_confidence": round(max_rem_c, 3),
            "visit_spread_m": round(spread, 1),
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
