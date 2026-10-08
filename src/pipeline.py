"""CreditNirvana Problem Statement 3: Unified Self-Correcting Geocoder Pipeline.

Single Canonical Entrypoint:
    python -m src.pipeline
    or python src/pipeline.py

Pipeline Architecture:
1. Address NLP: Multilingual structuring & entity extraction (phase1_parse)
2. Fraud & Integrity: Photo hash + generic spatial clustering + trail shape gating (fraud_detection)
3. Trajectory Kinematics: Canonical doorstep dwell centroid correction (dwell_kinematics)
4. Negative Spatial Evidence: Search-quality-weighted continuous repulsion field (negative_evidence)
5. Indic Remark Reasoning: Multilingual normalization & fuzzy logic inference (indic_address_understanding)
6. Pin Synthesis: Canonical dwell coordinates & learned agent reliability (pin_synthesis)
7. Fallback & Cross-Account Resolution: Empirical street anchors & negative repulsion POI selection (cross_account)
8. Conformal Uncertainty: Fold-safe 5-fold cross-validation R90 calibration (calibration)
9. Dispatch Output: Auditable 18-column contract with exactly 3,117 rows including 237 OUT addresses (CANNOT_GEOCODE).

Authoritative Outputs:
- output/geocoder_output.csv (3,117 rows)
- output/master_pins.csv (3,117 rows)
- output/visit_derived_pins.csv
- output/fraudulent_visits.csv
- output/dwell_corrected_visits.csv
- output/negative_exclusion_zones.csv
- output/cross_account_promotions.csv
- output/remark_extracted_corrections.csv
- output/calibration_radii.csv
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Add src to path if executed directly
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import GEO, OUT, SHARED
from fraud_detection import detect_fraudulent_visits
from dwell_kinematics import apply_dwell_corrections
from negative_evidence import build_negative_exclusion_zones, compute_repulsion_penalty
from indic_address_understanding import generate_upgraded_audit_trail
from fuzzy_remarks import get_address_level_remark_evidence
from pin_synthesis import synthesize_visit_pins
from cross_account import build_street_anchors, resolve_cross_account_streets
from calibration import calibrate_radii_cross_validation, compute_conformal_quantile

import phase1_parse

LANDMARK_MAX_FROM_CENTROID = 800.0
HINT_MAX_DIST = 200.0
AT_LANDMARK_M = 25.0


def run_pipeline() -> pd.DataFrame:
    """Executes the entire self-correcting geocoding pipeline end-to-end."""
    print("=" * 80)
    print("CREDITNIRVANA PS3: UNIFIED SELF-CORRECTING ADDRESS GEOCODER")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 0. INGEST RAW DATA
    # --------------------------------------------------------------------------
    print("\n[Step 0] Ingesting raw datasets...")
    addr = pd.read_csv(SHARED / "addresses.csv")
    visits = pd.read_csv(SHARED / "field_visits.csv")
    splits = pd.read_csv(SHARED / "splits.csv")
    base = pd.read_csv(GEO / "baseline_geocodes.csv")
    pois = pd.read_csv(GEO / "landmarks_poi.csv")
    loc = pd.read_csv(GEO / "localities.csv")
    towns = pd.read_csv(GEO / "towns.csv")
    surveyed = pd.read_csv(GEO / "surveyed_addresses.csv")
    gps = pd.read_csv(GEO / "visit_gps_points.csv")

    total_addresses = len(addr)
    print(f"Loaded {total_addresses} total addresses, {len(visits)} field visits, {len(surveyed)} ground-truth survey points.")

    # --------------------------------------------------------------------------
    # 1. ADDRESS NLP & STRUCTURING
    # --------------------------------------------------------------------------
    print("\n[Step 1] Executing multilingual address parsing and structuring...")
    st = phase1_parse.parse_addresses(addr, pois=pois, loc=loc, towns=towns)
    st.to_csv(OUT / "structured_addresses.csv", index=False)
    print(f"Address NLP complete: {len(st)} structured addresses.")

    # --------------------------------------------------------------------------
    # 2. FRAUD DETECTION & TRAJECTORY KINEMATICS
    # --------------------------------------------------------------------------
    print("\n[Step 2] Executing fraud detection & trajectory dwell kinematics...")
    clean_visits, fraud_audit = detect_fraudulent_visits(visits, gps)
    can_visits, dwell_audit = apply_dwell_corrections(clean_visits, gps)

    # --------------------------------------------------------------------------
    # 3. NEGATIVE SPATIAL EVIDENCE
    # --------------------------------------------------------------------------
    print("\n[Step 3] Constructing search-quality negative exclusion zones...")
    fail_trajectories, neg_audit = build_negative_exclusion_zones(visits, gps)

    # --------------------------------------------------------------------------
    # 4. INDIC NLP & FUZZY REMARK MINING
    # --------------------------------------------------------------------------
    print("\n[Step 4] Running Indic NLP & fuzzy logic remark reasoning...")
    audit_remarks_df = generate_upgraded_audit_trail(visits_path=SHARED / "field_visits.csv")
    addr_remarks = get_address_level_remark_evidence(audit_remarks_df)

    # --------------------------------------------------------------------------
    # 5. VISIT PIN SYNTHESIS (CANONICAL DWELL + LEARNED RELIABILITY)
    # --------------------------------------------------------------------------
    print("\n[Step 5] Synthesizing visit pins with canonical dwell coordinates...")
    visit_pins, accepted_visits_weighted = synthesize_visit_pins(
        visits_df=visits,
        gps_df=gps,
        pois_df=pois,
        addr_df=addr,
    )

    # --------------------------------------------------------------------------
    # 6. SPATIAL FALLBACK, NEGATIVE POI SELECTION & STREET KNOWLEDGE GRAPH
    # --------------------------------------------------------------------------
    print("\n[Step 6] Running fallback fusion, negative POI gating & cross-account resolution...")
    town_xy = {t: (g.centroid_x.mean(), g.centroid_y.mean()) for t, g in loc.groupby("town_id")}

    # Merge structured text, baseline, visit pins, and remark evidence cleanly without duplicate columns
    st_cols = [c for c in st.columns if c not in addr.columns or c == "address_id"]
    base_cols = [c for c in base.columns if c not in addr.columns or c == "address_id"]
    vis_cols = [c for c in visit_pins.columns if c not in addr.columns or c == "address_id"]
    rem_cols = [c for c in addr_remarks.columns if c not in addr.columns or c == "address_id"]

    df = addr.merge(st[st_cols], on="address_id", how="left")
    df = df.merge(base[base_cols], on="address_id", how="left")
    df = df.merge(visit_pins[vis_cols], on="address_id", how="left")
    df = df.merge(addr_remarks[rem_cols], on="address_id", how="left")

    master_records = []
    poi_by_town = {tid: g for tid, g in pois.groupby("town_id")}

    for r in df.itertuples(index=False):
        aid = r.address_id
        acc_id = r.account_id
        tid = r.town_id

        # 6a. OUT Village handling
        if tid == "OUT":
            master_records.append({
                "address_id": aid,
                "account_id": acc_id,
                "town_id": tid,
                "px": np.nan,
                "py": np.nan,
                "tier": "unmapped_village",
                "can_geocode": False,
                "n_good_visits": 0,
                "landmark_hint": "",
                "reason": "Village outside mapped urban town coverage; zero baseline geocode and zero field visits",
                "best_evidence_source": "unmapped_village",
                "negative_evidence_score": 0.0,
                "clean_street": getattr(r, "street_info", "") if pd.notna(getattr(r, "street_info", None)) else "",
                "clean_locality": getattr(r, "locality_name", "") if pd.notna(getattr(r, "locality_name", None)) else "",
                "clean_pincode": getattr(r, "pincode", "") if pd.notna(getattr(r, "pincode", None)) else "",
                "remark_confidence": 0.0,
            })
            continue

        rem_conf = float(r.remark_confidence) if pd.notna(r.remark_confidence) else (
            float(r.best_remark_conf) if hasattr(r, "best_remark_conf") and pd.notna(r.best_remark_conf) else 0.0
        )

        neg_score = 0.0

        # Tier 1: Ground Visit Pin
        if pd.notna(r.visit_px):
            px, py, tier = r.visit_px, r.visit_py, r.visit_tier
            n_v = int(r.n_good_visits)
            source = "field_visit_dwell"
            reason = f"Synthesized from {n_v} verified field visit dwell centroids ({tier})"
        else:
            n_v = 0
            # Tier 2: Baseline Street / Rooftop
            if r.precision in ("street", "rooftop") and pd.notna(r.geocoder_x):
                px, py, tier = r.geocoder_x, r.geocoder_y, "baseline_street"
                source = "baseline_high_precision"
                reason = f"High-precision baseline geocoder ({r.precision})"
            else:
                # Default Tier 4: Baseline Coarse
                px, py, tier = r.geocoder_x if pd.notna(r.geocoder_x) else np.nan, r.geocoder_y if pd.notna(r.geocoder_y) else np.nan, "baseline_coarse"
                source = "baseline_coarse"
                reason = "Coarse baseline locality/ward centroid"

                # Check landmark fallback
                lm_to_search = r.landmark_type if isinstance(r.landmark_type, str) else None
                if lm_to_search is None and hasattr(r, "best_remark_status") and r.best_remark_status == "STRONG_CORRECTION" and isinstance(r.remark_landmark_name, str):
                    lm_to_search = r.remark_landmark_name.lower().replace(" ", "_")

                if lm_to_search and tid in poi_by_town:
                    cand_pois = poi_by_town[tid]
                    cand_pois = cand_pois[cand_pois["landmark_type"].str.contains(lm_to_search, regex=False)]
                    if not cand_pois.empty:
                        # Locality centroid
                        c_loc = loc[(loc["town_id"] == tid) & (loc["locality_name"] == getattr(r, "locality_name", ""))]
                        if c_loc.empty and pd.notna(getattr(r, "pincode", None)):
                            c_loc = loc[(loc["town_id"] == tid) & (loc["pincode"].astype(str) == str(r.pincode))]
                        cx, cy = (c_loc["centroid_x"].mean(), c_loc["centroid_y"].mean()) if not c_loc.empty else town_xy.get(tid, (0.0, 0.0))

                        # Evaluate candidate POIs with Negative Spatial Repulsion
                        cand_xs = cand_pois["x"].to_numpy(dtype=float)
                        cand_ys = cand_pois["y"].to_numpy(dtype=float)
                        dists_to_c = np.hypot(cand_xs - cx, cand_ys - cy)

                        # Negative repulsion penalties
                        failed_trails = fail_trajectories.get(aid, [])
                        repulsions = np.zeros(len(cand_pois))
                        for j in range(len(cand_pois)):
                            repulsions[j] = compute_repulsion_penalty(cand_xs[j], cand_ys[j], failed_trails)

                        # Score combines distance to locality centroid + repulsion penalty
                        effective_scores = dists_to_c + 200.0 * repulsions
                        best_cand_idx = int(np.argmin(effective_scores))

                        if dists_to_c[best_cand_idx] <= LANDMARK_MAX_FROM_CENTROID:
                            px, py = cand_xs[best_cand_idx], cand_ys[best_cand_idx]
                            tier = "landmark"
                            source = "landmark_poi_fallback"
                            neg_score = repulsions[best_cand_idx]
                            pname = cand_pois.iloc[best_cand_idx]["name"]
                            reason = f"Promoted to nearest un-repelled landmark POI '{pname}'"

        # Generate navigational hints
        hint = ""
        if pd.notna(px) and hasattr(r, "landmark_type") and isinstance(r.landmark_type, str) and tid in poi_by_town:
            c_pois = poi_by_town[tid]
            c_matches = c_pois[c_pois["landmark_type"].str.contains(r.landmark_type, regex=False)]
            if not c_matches.empty:
                ds = np.hypot(c_matches["x"].to_numpy() - px, c_matches["y"].to_numpy() - py)
                i_best = int(np.argmin(ds))
                if ds[i_best] <= HINT_MAX_DIST:
                    rel = str(r.spatial_relation).title() if isinstance(r.spatial_relation, str) else "Near"
                    pname = c_matches.iloc[i_best]["name"]
                    if tier == "landmark":
                        hint = f"{rel} {pname} (pin is a guess near this landmark)"
                    elif ds[i_best] < AT_LANDMARK_M:
                        hint = f"{rel} {pname} (pin is at this landmark)"
                    else:
                        hint = f"{rel} {pname} (about {round(ds[i_best], -1):.0f} m from pin)"

        # Enrich with agent remark spatial cue
        if hasattr(r, "best_remark_conf") and pd.notna(r.best_remark_conf) and r.best_remark_conf >= 0.60:
            rem_rel = str(r.remark_relation).title() if pd.notna(r.remark_relation) else "Near"
            rem_lm = str(r.remark_landmark_name) if pd.notna(r.remark_landmark_name) else ""
            if rem_lm:
                if hint:
                    hint += f" | Remark Cue: {rem_rel} {rem_lm} (conf={r.best_remark_conf:.2f})"
                else:
                    hint = f"Field Remark: {rem_rel} {rem_lm} (conf={r.best_remark_conf:.2f})"

        master_records.append({
            "address_id": aid,
            "account_id": acc_id,
            "town_id": tid,
            "px": round(px, 1) if pd.notna(px) else np.nan,
            "py": round(py, 1) if pd.notna(py) else np.nan,
            "tier": tier,
            "can_geocode": pd.notna(px),
            "n_good_visits": n_v,
            "landmark_hint": hint,
            "reason": reason,
            "best_evidence_source": source,
            "negative_evidence_score": round(neg_score, 2),
            "clean_street": getattr(r, "street_info", "") if pd.notna(getattr(r, "street_info", None)) else "",
            "clean_locality": getattr(r, "locality_name", "") if pd.notna(getattr(r, "locality_name", None)) else "",
            "clean_pincode": getattr(r, "pincode", "") if pd.notna(getattr(r, "pincode", None)) else "",
            "remark_confidence": round(rem_conf, 3),
        })

    m_df = pd.DataFrame(master_records)

    # 6b. Cross-Account Street-Level Knowledge Graph
    # Build anchors fold-safely (excluding held-out test accounts from anchor compilation)
    test_acc_ids = set(splits[splits["split"] == "test"]["account_id"])
    test_addr_ids = set(addr[addr["account_id"].isin(test_acc_ids)]["address_id"])

    loc_anchors, gen_anchors = build_street_anchors(m_df, excluded_address_ids=test_addr_ids)
    m_df, promo_audit = resolve_cross_account_streets(m_df, loc_anchors, gen_anchors)

    # Save master pins
    m_df.to_csv(OUT / "master_pins.csv", index=False, encoding="utf-8-sig")
    print(f"Master Pins generated: {len(m_df)} rows.")
    print(m_df["tier"].value_counts().to_string())

    # --------------------------------------------------------------------------
    # 7. CONFORMAL UNCERTAINTY CALIBRATION (R90 via Fold-Safe 5-Fold CV)
    # --------------------------------------------------------------------------
    print("\n[Step 7] Calibrating conformal uncertainty radii via fold-safe 5-fold cross-validation...")
    calib_summary, r90_map, _ = calibrate_radii_cross_validation(
        master_pins_df=m_df,
        surveyed_df=surveyed,
        splits_df=splits,
        addr_df=addr,
        n_splits=5,
    )

    # Attach R90 to master pins
    m_df["R90_meters"] = m_df["tier"].map(r90_map)

    # --------------------------------------------------------------------------
    # 8. DISPATCH ACTION ALLOCATION & 18-COLUMN CANONICAL OUTPUT
    # --------------------------------------------------------------------------
    print("\n[Step 8] Allocating dispatch actions & compiling 18-column canonical output...")

    # Action allocation
    actions = []
    for _, row in m_df.iterrows():
        t = row["tier"]
        r90 = row["R90_meters"]

        if t == "unmapped_village" or not row["can_geocode"]:
            actions.append("CANNOT_GEOCODE")
        elif pd.notna(r90) and r90 <= 100.0:
            actions.append("DIRECT_VISIT")
        elif pd.notna(r90) and r90 <= 500.0:
            actions.append("VISIT_WITH_HINT")
        else:
            actions.append("VERIFY_FIRST")

    m_df["action"] = actions

    # Cross-account support count
    promo_support_map = promo_audit.set_index("address_id")["anchor_support_accounts"].to_dict() if len(promo_audit) else {}
    m_df["cross_account_support"] = m_df["address_id"].map(promo_support_map).fillna(0).astype(int)

    # Fraud flag for address
    fraud_addr_ids = set(fraud_audit["address_id"])
    m_df["fraud_flag"] = m_df["address_id"].isin(fraud_addr_ids)

    # Reliable visit count
    m_df["visit_evidence_count"] = m_df["n_good_visits"]
    m_df["reliable_visit_count"] = np.where(m_df["tier"].isin(["visits_agree", "visit_1"]), m_df["n_good_visits"], 0)

    # Final 18 canonical audit columns
    final_cols = [
        "address_id",
        "account_id",
        "town_id",
        "px",
        "py",
        "R90_meters",
        "tier",
        "action",
        "can_geocode",
        "landmark_hint",
        "reason",
        "visit_evidence_count",
        "reliable_visit_count",
        "best_evidence_source",
        "fraud_flag",
        "negative_evidence_score",
        "cross_account_support",
        "remark_confidence",
    ]

    final_df = m_df[final_cols].copy()
    final_df["px"] = final_df["px"].round(1)
    final_df["py"] = final_df["py"].round(1)
    final_df["R90_meters"] = final_df["R90_meters"].round(1)

    out_geocoder = OUT / "geocoder_output.csv"
    final_df.to_csv(out_geocoder, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Output saved to {out_geocoder} with EXACTLY {len(final_df)} rows (expected {total_addresses}).")
    print(f"Action Breakdown:\n{final_df['action'].value_counts().to_string()}")
    print(f"\nOUT Village Addresses: {(final_df['town_id'] == 'OUT').sum()} rows correctly tagged CANNOT_GEOCODE.")
    print("=" * 80)

    return final_df


if __name__ == "__main__":
    run_pipeline()
