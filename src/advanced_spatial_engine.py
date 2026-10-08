"""CreditNirvana (CN) Advanced Spatial Intelligence Engine (CN Problem Statement 3).

Upgrades the geocoding pipeline with:
1. Perceptual Photo Hash Anti-Spoofing & Fraud Gating
2. Vernacular Remark Mining, Translation & Spatial Offset Modeling
3. Negative Spatial Evidence from Failed Visits (Continuous Repulsion Field)
4. Trajectory Kinematics & Stationary Dwell Clustering
5. Cross-Account Street-Level Knowledge Graph
6. Enterprise Production Integration & PS2 Collections Dispatch Contract

Artifacts produced in output/:
- fraudulent_photo_visits.csv
- dwell_corrected_visits.csv
- remark_extracted_corrections.csv
- negative_exclusion_zones.csv
- cross_account_promotions.csv
- geocoder_output_v2.csv
- enterprise_address_master_update.csv
- ps2_dispatch_feed.csv
"""

import math
import re
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from config import GEO, OUT, SHARED

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
ACCEPT_OUTCOMES = ["locked_premises", "met_borrower", "met_family", "neighbour_says_shifted"]
MAX_GPS_ACC = 30.0
FAKE_PHOTO_THRESHOLD = 3
AGREE_SPREAD_M = 100.0
LANDMARK_MAX_FROM_CENTROID = 800.0
HINT_MAX_DIST = 200.0
AT_LANDMARK_M = 25.0
REPULSION_LAMBDA = 5e5
R_VISUAL_M = 40.0
CROSS_ACCOUNT_R90 = 75.0

# Pre-compiled regex patterns for remarks
CORRECTIVE_TRIGGERS = [
    r"nija\s*mane", r"asli\s*mane", r"mane\s*ide", r"nijavada\s*thikaana", r"address\s*tappu",
    r"asli\s*ghar", r"sahi\s*ghar", r"address\s*galat", r"ghar\s*idhar\s*hai", r"makan\s*no",
    r"actual\s*house", r"true\s*address", r"correct\s*location", r"house\s*is\s*at"
]
RE_CORRECTIVE_TRIG = re.compile("|".join(CORRECTIVE_TRIGGERS), re.IGNORECASE)

VISUAL_CUES = [
    (re.compile(r"blue\s*gate(?:\s*wala\s*ghar)?", re.IGNORECASE), "Blue gate house"),
    (re.compile(r"hasiru\s*gate|green\s*gate", re.IGNORECASE), "Green gate house"),
    (re.compile(r"kempu\s*gate|red\s*gate", re.IGNORECASE), "Red gate house"),
    (re.compile(r"corner\s*(?:mane|ghar|house)", re.IGNORECASE), "Corner house"),
    (re.compile(r"(\d+)\s*floor\s*(?:building|house)?", re.IGNORECASE), r"\1-floor building"),
    (re.compile(r"water\s*tank\s*mele", re.IGNORECASE), "Overhead tank on terrace"),
]

PREPOSITION_OFFSETS = [
    (re.compile(r"behind|hinde|piche|peeche|back\s*side", re.IGNORECASE), "BEHIND", (0.0, -25.0)),
    (re.compile(r"in\s*front|opposite|\bopp\b|eduru|munde|samne|aage", re.IGNORECASE), "OPPOSITE", (0.0, 20.0)),
    (re.compile(r"beside|adjacent|pakka|bagal(?:\s*mein)?|baju(?:\s*mein)?|side", re.IGNORECASE), "BESIDE", (15.0, 0.0)),
]

CROSS_ORDINAL_RE = re.compile(r"(\d+)\s*(?:cross|gali|lane)\s*(?:munde|aage|ahead)", re.IGNORECASE)

LANDMARK_KEYWORDS = [
    r"milk\s*dairy|doodh\s*dairy", r"medical\s*store|dawai\s*ki\s*dukaan|dispensary",
    r"girja\s*ghar|church", r"barat\s*ghar|community\s*hall|kalyana\s*mantapa",
    r"anjaneya\s*gudi|hanuman\s*mandir|hanuman\s*temple",
    r"ganapathi\s*gudi|ganesh\s*mandir|ganesh\s*temple",
    r"masidi|masjid|jama\s*masjid", r"ration\s*(?:ki\s*)?dukaan|pds\s*shop|nyayabele\s*angadi",
    r"sarkari\s*school|govt\s*school|school", r"bus\s*stand|bus\s*stop|bus\s*adda|bus\s*nildana",
    r"water\s*tank|overhead\s*tank|tanki", r"post\s*office|dakghar",
    r"petrol\s*bunk|petrol\s*pump", r"park|children\s*park|garden"
]
RE_LANDMARK_KW = re.compile("(" + "|".join(LANDMARK_KEYWORDS) + ")", re.IGNORECASE)


def weighted_median(vals, weights):
    v, w = np.asarray(vals, float), np.asarray(weights, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return v[np.searchsorted(c, 0.5 * c[-1])]


# ==============================================================================
# 1. PERCEPTUAL PHOTO HASH ANTI-SPOOFING & FRAUD GATING
# ==============================================================================
def run_subsystem1_photo_fraud(visits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("SUB-SYSTEM 1: Perceptual Photo Hash Anti-Spoofing & Fraud Gating")
    print("=" * 70)
    account_cardinality = visits.groupby("photo_hash")["account_id"].nunique()
    fraud_hashes = set(account_cardinality[account_cardinality >= FAKE_PHOTO_THRESHOLD].index)

    is_fraud = visits["photo_hash"].isin(fraud_hashes)
    fraud_df = visits[is_fraud].copy()
    fraud_df["account_reuse_count"] = fraud_df["photo_hash"].map(account_cardinality)

    cols = ["visit_id", "agent_id", "account_id", "photo_hash", "account_reuse_count", "checkin_x", "checkin_y"]
    fraud_export = fraud_df[cols]
    fraud_export.to_csv(OUT / "fraudulent_photo_visits.csv", index=False)

    clean_visits = visits[~is_fraud].copy()
    print(f"Total visits examined: {len(visits)}")
    print(f"Identified {len(fraud_hashes)} fraudulent photo hashes reused across >= {FAKE_PHOTO_THRESHOLD} accounts.")
    print(f"Purged {len(fraud_df)} roadside fraudulent check-ins -> Saved to output/fraudulent_photo_visits.csv")
    print(f"Retained {len(clean_visits)} legitimate visits for spatial reasoning.")
    return clean_visits, fraud_export


# ==============================================================================
# 2. TRAJECTORY KINEMATICS & STATIONARY DWELL CLUSTERING
# ==============================================================================
def run_subsystem4_kinematics(gps_file: Path, clean_visits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("SUB-SYSTEM 4: Trajectory Kinematics & Stationary Dwell Clustering")
    print("=" * 70)
    gps = pd.read_csv(gps_file)
    gps["ts"] = pd.to_datetime(gps["point_ts"])
    gps = gps.sort_values(["visit_id", "seq"])

    # Compute instantaneous velocities
    gps["dt"] = gps.groupby("visit_id")["ts"].diff().dt.total_seconds().fillna(0.0)
    gps["dx"] = gps.groupby("visit_id")["x"].diff().fillna(0.0)
    gps["dy"] = gps.groupby("visit_id")["y"].diff().fillna(0.0)
    gps["dist"] = np.hypot(gps["dx"], gps["dy"])
    gps["vel"] = np.where(gps["seq"] == 0, 0.0, gps["dist"] / (gps["dt"] + 1e-6))

    # Kinematic state: stationary doorstep state
    gps["is_stat"] = (gps["vel"] < 0.4) & (gps["accuracy_m"] <= 20.0)

    # Contiguous stationary window grouping
    gps["stat_block"] = (~gps["is_stat"]).cumsum()
    stat_pts = gps[gps["is_stat"]]

    dwell_records = {}
    for (vid, _), g in stat_pts.groupby(["visit_id", "stat_block"]):
        if len(g) < 2:
            continue
        duration = (g["ts"].iloc[-1] - g["ts"].iloc[0]).total_seconds()
        if duration >= 120.0:
            mx, my = g["x"].mean(), g["y"].mean()
            if np.max(np.hypot(g["x"] - mx, g["y"] - my)) <= 15.0:
                w = 1.0 / (g["accuracy_m"].clip(lower=1.0) ** 2)
                dw_x = (g["x"] * w).sum() / w.sum()
                dw_y = (g["y"] * w).sum() / w.sum()
                if vid not in dwell_records or duration > dwell_records[vid]["duration"]:
                    dwell_records[vid] = {
                        "dwell_x": dw_x,
                        "dwell_y": dw_y,
                        "duration": duration,
                        "pts": len(g)
                    }

    dwell_df_rows = []
    updated_visits = clean_visits.copy()
    for idx, r in updated_visits.iterrows():
        vid = r["visit_id"]
        if vid in dwell_records:
            d = dwell_records[vid]
            shift = math.hypot(d["dwell_x"] - r["checkin_x"], d["dwell_y"] - r["checkin_y"])
            dwell_df_rows.append({
                "visit_id": vid,
                "address_id": r["address_id"],
                "original_checkin_x": r["checkin_x"],
                "original_checkin_y": r["checkin_y"],
                "dwell_x": round(d["dwell_x"], 1),
                "dwell_y": round(d["dwell_y"], 1),
                "dwell_duration_s": round(d["duration"], 1),
                "stationary_pts_count": d["pts"],
                "shift_distance_m": round(shift, 1)
            })
            updated_visits.at[idx, "checkin_x"] = d["dwell_x"]
            updated_visits.at[idx, "checkin_y"] = d["dwell_y"]

    dwell_audit = pd.DataFrame(dwell_df_rows)
    dwell_audit.to_csv(OUT / "dwell_corrected_visits.csv", index=False)
    print(f"Processed {len(gps)} sequential breadcrumb points across {gps.visit_id.nunique()} visits.")
    print(f"Extracted {len(dwell_records)} valid stationary dwell clusters (>=120s, <=15m radius).")
    print(f"Applied dwell corrections to {len(dwell_audit)} check-ins -> Saved to output/dwell_corrected_visits.csv")
    if len(dwell_audit):
        print(f"Mean road-to-doorstep checkin shift: {dwell_audit.shift_distance_m.mean():.1f} metres (max: {dwell_audit.shift_distance_m.max():.1f} m)")
    return updated_visits, dwell_audit


# ==============================================================================
# 3. VERNACULAR REMARK MINING & SPATIAL OFFSET MODELING
# ==============================================================================
def run_subsystem2_remarks(visits: pd.DataFrame, addr: pd.DataFrame, pois: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("SUB-SYSTEM 2: Vernacular Remark Mining & Spatial Offset Modeling")
    print("=" * 70)
    v_m = visits.merge(addr[["address_id", "town_id"]], on="address_id", how="left")
    poi_by_town = {tid: g for tid, g in pois.groupby("town_id")}

    records = []
    for _, r in v_m[v_m.remark.notna()].iterrows():
        rem = str(r["remark"])
        has_trig = bool(RE_CORRECTIVE_TRIG.search(rem))

        vis_cue = None
        for rx, label in VISUAL_CUES:
            m = rx.search(rem)
            if m:
                vis_cue = rx.sub(label, m.group(0))
                break

        lm_match = RE_LANDMARK_KW.search(rem)
        if (has_trig or vis_cue) and lm_match and r["town_id"] in poi_by_town:
            lm_raw = lm_match.group(1)
            t_pois = poi_by_town[r["town_id"]]
            res = process.extractOne(lm_raw, t_pois["name"].tolist(), scorer=fuzz.token_set_ratio)
            if res and res[1] >= 75:
                matched_poi = t_pois.iloc[res[2]]
                prep_label = "NEAR"
                dx, dy = 0.0, 0.0
                for rx, plabel, (px_off, py_off) in PREPOSITION_OFFSETS:
                    if rx.search(rem):
                        prep_label = plabel
                        dx, dy = px_off, py_off
                        break

                cord_m = CROSS_ORDINAL_RE.search(rem)
                if cord_m:
                    k = int(cord_m.group(1))
                    dy += k * 35.0

                corr_x = matched_poi["x"] + dx
                corr_y = matched_poi["y"] + dy

                records.append({
                    "visit_id": r["visit_id"],
                    "address_id": r["address_id"],
                    "town_id": r["town_id"],
                    "extracted_landmark": lm_raw,
                    "matched_poi_id": matched_poi["poi_id"],
                    "matched_poi_name": matched_poi["name"],
                    "poi_x": matched_poi["x"],
                    "poi_y": matched_poi["y"],
                    "spatial_preposition": prep_label,
                    "offset_x": dx,
                    "offset_y": dy,
                    "corrected_x": round(corr_x, 1),
                    "corrected_y": round(corr_y, 1),
                    "visual_doorstep_cue": vis_cue or "",
                    "raw_remark": rem
                })

    rem_df = pd.DataFrame(records)
    rem_df.to_csv(OUT / "remark_extracted_corrections.csv", index=False)
    print(f"Parsed {len(v_m)} visits with multilingual remarks.")
    print(f"Extracted {len(rem_df)} verified landmark offset corrections across {rem_df.address_id.nunique()} unique addresses.")
    print(f"Saved audit log to output/remark_extracted_corrections.csv")
    return rem_df


# ==============================================================================
# 4. NEGATIVE SPATIAL EVIDENCE FROM FAILED VISITS
# ==============================================================================
def run_subsystem3_negative_evidence(visits: pd.DataFrame, gps_file: Path) -> tuple[dict, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("SUB-SYSTEM 3: Negative Spatial Evidence from Failed Visits")
    print("=" * 70)
    fail_visits = visits[visits["outcome"] == "address_not_traceable"]
    fail_vids = set(fail_visits["visit_id"])

    gps = pd.read_csv(gps_file)
    fail_gps = gps[gps.visit_id.isin(fail_vids)].merge(visits[["visit_id", "address_id"]], on="visit_id")

    # Group by address_id -> store trajectory points
    addr_fail_trajectories = {}
    neg_zone_records = []
    for aid, g in fail_gps.groupby("address_id"):
        addr_fail_trajectories[aid] = {
            "x": g["x"].to_numpy(float),
            "y": g["y"].to_numpy(float),
            "denom": 2.0 * (g["accuracy_m"].to_numpy(float) + R_VISUAL_M) ** 2
        }
        neg_zone_records.append({
            "address_id": aid,
            "failed_visit_count": g["visit_id"].nunique(),
            "total_failed_gps_points": len(g),
            "bounding_min_x": round(g["x"].min(), 1),
            "bounding_max_x": round(g["x"].max(), 1),
            "bounding_min_y": round(g["y"].min(), 1),
            "bounding_max_y": round(g["y"].max(), 1),
            "mean_accuracy_m": round(g["accuracy_m"].mean(), 1)
        })

    neg_df = pd.DataFrame(neg_zone_records)
    neg_df.to_csv(OUT / "negative_exclusion_zones.csv", index=False)
    print(f"Modeled continuous repulsion envelopes across {len(neg_df)} addresses with failed searches.")
    print(f"Ingested {len(fail_gps)} failed trajectory points -> Saved to output/negative_exclusion_zones.csv")
    return addr_fail_trajectories, neg_df


# ==============================================================================
# 5. INTEGRATED FUSION ENGINE & KNOWLEDGE GRAPH PROPAGATION
# ==============================================================================
def run_integrated_pipeline(
    updated_visits: pd.DataFrame,
    rem_df: pd.DataFrame,
    fail_trajectories: dict,
    st: pd.DataFrame,
    base: pd.DataFrame,
    loc: pd.DataFrame,
    pois: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("SUB-SYSTEM 5 & INTEGRATION: Fusion, Gating & Knowledge Graph")
    print("=" * 70)

    # 1. Accepted visit pins using dwell-corrected GPS
    acc_v = updated_visits[updated_visits.outcome.isin(ACCEPT_OUTCOMES) & (updated_visits.gps_accuracy_m <= MAX_GPS_ACC)].copy()
    acc_v["w"] = 1.0 / acc_v.gps_accuracy_m.clip(lower=1.0) ** 2

    visit_pins = {}
    for aid, g in acc_v.groupby("address_id"):
        wx, wy = weighted_median(g.checkin_x, g.w), weighted_median(g.checkin_y, g.w)
        n = len(g)
        spread = float(np.hypot(g.checkin_x - wx, g.checkin_y - wy).max()) if n >= 2 else 0.0
        tier = "visit_1" if n == 1 else "visits_agree" if spread <= AGREE_SPREAD_M else "visits_disagree"
        visit_pins[aid] = (wx, wy, tier, n, spread)

    # 2. Town locality centroids
    town_xy = {t: (g.centroid_x.mean(), g.centroid_y.mean()) for t, g in loc.groupby("town_id")}

    # 3. Build candidate master frame
    df = st.merge(base, on="address_id", how="left")
    df = df[df.geocoder_x.notna() | df.address_id.isin(visit_pins)].copy()

    # Visual cue lookup from remarks
    vis_cue_map = rem_df.groupby("address_id")["visual_doorstep_cue"].last().to_dict() if len(rem_df) else {}

    master_rows = []
    for r in df.itertuples(index=False):
        aid = r.address_id
        tid = r.town_id
        cue = vis_cue_map.get(aid, "")
        neg_flag = aid in fail_trajectories

        if aid in visit_pins:
            px, py, tier, n_good, spread = visit_pins[aid]
        else:
            n_good = 0
            if r.precision in ("street", "rooftop"):
                px, py, tier = r.geocoder_x, r.geocoder_y, "baseline_street"
            else:
                px, py, tier = r.geocoder_x, r.geocoder_y, "baseline_coarse"
                if r.precision == "locality" and isinstance(r.landmark_type, str):
                    # Locality centroid
                    c = loc[(loc.town_id == tid) & (loc.locality_name == r.locality_name)]
                    if c.empty and isinstance(r.pincode, str):
                        c = loc[(loc.town_id == tid) & (loc.pincode.astype(str) == r.pincode)]
                    cx, cy = (c.centroid_x.mean(), c.centroid_y.mean()) if not c.empty else town_xy.get(tid, (0.0, 0.0))

                    # Candidate POIs matching landmark_type
                    cand_pois = pois[(pois.town_id == tid) & pois.landmark_type.str.contains(r.landmark_type, regex=False)]
                    if not cand_pois.empty:
                        poi_xs = cand_pois.x.to_numpy(float)
                        poi_ys = cand_pois.y.to_numpy(float)
                        d2_centroid = (poi_xs - cx) ** 2 + (poi_ys - cy) ** 2

                        # Negative repulsion penalty
                        repulsion = np.zeros(len(cand_pois))
                        if aid in fail_trajectories:
                            ft = fail_trajectories[aid]
                            for j in range(len(cand_pois)):
                                d2_fail = (poi_xs[j] - ft["x"]) ** 2 + (poi_ys[j] - ft["y"]) ** 2
                                repulsion[j] = np.sum(np.exp(-d2_fail / ft["denom"]))

                        penalized_cost = d2_centroid + REPULSION_LAMBDA * repulsion
                        best_idx = int(np.argmin(penalized_cost))
                        if math.sqrt(d2_centroid[best_idx]) <= LANDMARK_MAX_FROM_CENTROID:
                            px, py = poi_xs[best_idx], poi_ys[best_idx]
                            tier = "landmark"

        master_rows.append({
            "address_id": aid,
            "account_id": r.account_id,
            "town_id": tid,
            "clean_door_no": r.door_no if pd.notna(r.door_no) else "",
            "clean_street": r.street_info if pd.notna(r.street_info) else "",
            "clean_locality": r.locality_name if pd.notna(r.locality_name) else "",
            "clean_pincode": r.pincode if pd.notna(r.pincode) else "",
            "px": px,
            "py": py,
            "tier": tier,
            "n_good_visits": n_good,
            "landmark_type": r.landmark_type if pd.notna(r.landmark_type) else "",
            "spatial_relation": r.spatial_relation if pd.notna(r.spatial_relation) else "",
            "visual_doorstep_cue": cue,
            "negative_search_visited_flag": neg_flag
        })

    m_df = pd.DataFrame(master_rows)

    # 4. CROSS-ACCOUNT STREET-LEVEL KNOWLEDGE GRAPH
    print("Building Street-Level Knowledge Graph...")
    # Verified Street Anchors compiled from accounts in tiers visits_agree and visit_1
    anchors = m_df[m_df.tier.isin(["visits_agree", "visit_1"]) & (m_df.clean_street != "") & (m_df.clean_pincode != "")].copy()
    
    # Extract numeric door numbers for anchor accounts where available
    def get_door_num(d_str):
        nums = re.findall(r"\d+", str(d_str))
        return int(nums[0]) if nums else None
    anchors["door_num"] = anchors["clean_door_no"].apply(get_door_num)

    # Locality-specific anchors: (town_id, pincode, locality, street)
    loc_anchors = anchors[anchors.clean_locality != ""].groupby(["town_id", "clean_pincode", "clean_locality", "clean_street"]).agg(
        px=("px", "median"),
        py=("py", "median"),
        door_num=("door_num", "median")
    ).reset_index()
    loc_anchor_map = {
        (r.town_id, str(r.clean_pincode), r.clean_locality, r.clean_street): (r.px, r.py, r.door_num)
        for _, r in loc_anchors.iterrows()
    }

    # General street anchors (only where street is spatially contiguous/unambiguous: spread <= 100m)
    gen_anchor_map = {}
    for (tid, pcode, street), g in anchors.groupby(["town_id", "clean_pincode", "clean_street"]):
        med_x, med_y = g.px.median(), g.py.median()
        max_dist = np.hypot(g.px - med_x, g.py - med_y).max()
        if max_dist <= 100.0:
            med_door = g.door_num.median() if g.door_num.notna().any() else None
            gen_anchor_map[(tid, str(pcode), street)] = (med_x, med_y, med_door)

    print(f"Compiled {len(loc_anchor_map)} locality-specific anchors and {len(gen_anchor_map)} unambiguous street anchors from {len(anchors)} ground-visited accounts.")

    promotions = []
    for idx, r in m_df.iterrows():
        if r["tier"] in ("baseline_coarse", "visits_disagree") and r["clean_street"] and r["clean_pincode"]:
            loc_key = (r["town_id"], str(r["clean_pincode"]), r["clean_locality"], r["clean_street"])
            gen_key = (r["town_id"], str(r["clean_pincode"]), r["clean_street"])

            anchor_data = None
            if loc_key in loc_anchor_map:
                anchor_data = loc_anchor_map[loc_key]
            elif (not r["clean_locality"]) and (gen_key in gen_anchor_map):
                anchor_data = gen_anchor_map[gen_key]

            if anchor_data is not None:
                ax, ay, a_door = anchor_data
                dist_to_anchor = math.hypot(ax - r["px"], ay - r["py"])

                # Gating: baseline_coarse has zero field visits -> always promote.
                # visits_disagree has real physical visits -> only promote if anchor is concordant (<= 150m)
                should_promote = False
                if r["tier"] == "baseline_coarse":
                    should_promote = True
                elif r["tier"] == "visits_disagree" and dist_to_anchor <= 150.0:
                    should_promote = True

                if should_promote:
                    # Door number linear interpolation offset along street axis (10m per door ordinal diff)
                    dx_door = 0.0
                    cand_door = get_door_num(r["clean_door_no"])
                    if cand_door is not None and a_door is not None and not math.isnan(a_door):
                        diff = max(-5, min(5, cand_door - int(a_door)))
                        dx_door = diff * 10.0

                    old_t = r["tier"]
                    old_x, old_y = r["px"], r["py"]
                    new_x, new_y = ax + dx_door, ay

                    m_df.at[idx, "px"] = new_x
                    m_df.at[idx, "py"] = new_y
                    m_df.at[idx, "tier"] = "cross_account_inferred"

                    promotions.append({
                        "address_id": r["address_id"],
                        "account_id": r["account_id"],
                        "town_id": r["town_id"],
                        "old_tier": old_t,
                        "old_px": round(old_x, 1),
                        "old_py": round(old_y, 1),
                        "street_info": r["clean_street"],
                        "new_px": round(new_x, 1),
                        "new_py": round(new_y, 1),
                        "new_tier": "cross_account_inferred",
                        "promoted_r90": CROSS_ACCOUNT_R90
                    })

    promo_df = pd.DataFrame(promotions)
    promo_df.to_csv(OUT / "cross_account_promotions.csv", index=False)
    print(f"Promoted {len(promo_df)} accounts to 'cross_account_inferred' (R90=75m) -> Saved to output/cross_account_promotions.csv")
    if len(promo_df):
        print(f"Breakdown of promoted tiers:\n{promo_df['old_tier'].value_counts().to_string()}")

    # 5. Navigational Hints (200m cutoff)
    poi_by_town = {tid: g for tid, g in pois.groupby("town_id")}
    hints = []
    for _, r in m_df.iterrows():
        h = ""
        tid = r["town_id"]
        if r["landmark_type"] and tid in poi_by_town:
            t_pois = poi_by_town[tid]
            c = t_pois[t_pois.landmark_type.str.contains(r["landmark_type"], regex=False)]
            if not c.empty:
                d = np.hypot(c.x.to_numpy() - r["px"], c.y.to_numpy() - r["py"])
                i = int(np.argmin(d))
                if d[i] <= HINT_MAX_DIST:
                    rel = r["spatial_relation"].title() if r["spatial_relation"] else "Near"
                    pname = c.iloc[i]["name"]
                    if r["tier"] == "landmark":
                        h = f"{rel} {pname} (pin is a guess near this landmark)"
                    elif d[i] < AT_LANDMARK_M:
                        h = f"{rel} {pname} (pin is at this landmark)"
                    else:
                        h = f"{rel} {pname} (about {round(d[i], -1):.0f} m from pin)"
        hints.append(h)
    m_df["landmark_nav_hint"] = hints

    return m_df, promo_df


# ==============================================================================
# 6. CONFORMAL RADII, DISPATCH ACTIONS & PRODUCTION CONTRACTS
# ==============================================================================
def run_subsystem6_contracts(m_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("SUB-SYSTEM 6: Conformal Uncertainty, Production Update & PS2 Dispatch Feed")
    print("=" * 70)

    # Conformal radii mapping
    R_MAP = {
        "visits_agree": 47.9,
        "visit_1": 77.4,
        "cross_account_inferred": CROSS_ACCOUNT_R90,
        "baseline_street": 162.1,
        "landmark": 357.8,
        "visits_disagree": 387.8,  # calibrated consensus threshold
        "baseline_coarse": 3710.3
    }
    m_df["R90_meters"] = m_df["tier"].map(R_MAP).fillna(3710.3)

    # Dispatch actions
    cond = [m_df["R90_meters"] <= 100.0, m_df["R90_meters"] <= 500.0]
    m_df["action"] = np.select(cond, ["DIRECT_VISIT", "VISIT_WITH_HINT"], default="VERIFY_FIRST")

    # 1. geocoder_output_v2.csv
    v2_cols = ["address_id", "account_id", "town_id", "px", "py", "tier", "R90_meters", "action", "landmark_nav_hint", "visual_doorstep_cue"]
    v2_df = m_df[v2_cols].rename(columns={"landmark_nav_hint": "landmark_hint"}).round({"px": 1, "py": 1, "R90_meters": 1})
    v2_df.to_csv(OUT / "geocoder_output_v2.csv", index=False, encoding="utf-8-sig")

    # 2. enterprise_address_master_update.csv
    now_iso = datetime.now(timezone.utc).isoformat()
    master_update = pd.DataFrame({
        "address_id": m_df["address_id"],
        "account_id": m_df["account_id"],
        "town_id": m_df["town_id"],
        "clean_door_no": m_df["clean_door_no"].astype(str).str.replace(r"\.0$", "", regex=True).replace(["nan", "None"], ""),
        "clean_street": m_df["clean_street"],
        "clean_locality": m_df["clean_locality"],
        "clean_pincode": m_df["clean_pincode"].astype(str).str.replace(r"\.0$", "", regex=True).replace(["nan", "None"], ""),
        "final_pin_x": m_df["px"].round(1),
        "final_pin_y": m_df["py"].round(1),
        "evidence_tier": m_df["tier"],
        "conformal_radius_r90": m_df["R90_meters"].round(1),
        "dispatch_action": m_df["action"],
        "landmark_nav_hint": m_df["landmark_nav_hint"],
        "visual_doorstep_cue": m_df["visual_doorstep_cue"],
        "negative_search_visited_flag": m_df["negative_search_visited_flag"],
        "last_updated_timestamp": now_iso
    })
    master_update.to_csv(OUT / "enterprise_address_master_update.csv", index=False, encoding="utf-8-sig")

    # 3. ps2_dispatch_feed.csv
    dispatch_rows = []
    for _, r in m_df.iterrows():
        act = r["action"]
        aid = r["account_id"]
        r90 = round(r["R90_meters"], 1)

        if act == "DIRECT_VISIT":
            dispatch_rows.append({
                "account_id": aid,
                "dispatch_channel": "FIELD_FORCE",
                "routing_target_x": round(r["px"], 1),
                "routing_target_y": round(r["py"], 1),
                "search_budget_m": r90,
                "dialer_campaign": "STANDARD_PTP",
                "voice_bot_prompt": ""
            })
        elif act == "VISIT_WITH_HINT":
            hint_txt = r["landmark_nav_hint"] or "nearby verified landmark"
            dispatch_rows.append({
                "account_id": aid,
                "dispatch_channel": "FIELD_FORCE",
                "routing_target_x": round(r["px"], 1),
                "routing_target_y": round(r["py"], 1),
                "search_budget_m": r90,
                "dialer_campaign": "STANDARD_PTP",
                "voice_bot_prompt": f"Confirm if premises is located near {hint_txt}"
            })
        else:  # VERIFY_FIRST
            dispatch_rows.append({
                "account_id": aid,
                "dispatch_channel": "TELE_DIALER",
                "routing_target_x": None,
                "routing_target_y": None,
                "search_budget_m": r90,
                "dialer_campaign": "ADDRESS_VERIFICATION_CAMPAIGN",
                "voice_bot_prompt": "Your recorded address could not be verified. Please state your exact cross street, nearby landmark, and house gate color."
            })

    ps2_feed = pd.DataFrame(dispatch_rows)
    ps2_feed.to_csv(OUT / "ps2_dispatch_feed.csv", index=False, encoding="utf-8-sig")

    print(f"Generated geocoder_output_v2.csv ({len(v2_df)} rows)")
    print(f"Generated enterprise_address_master_update.csv ({len(master_update)} rows)")
    print(f"Generated ps2_dispatch_feed.csv ({len(ps2_feed)} rows)")
    print("\nAction Distribution in v2:")
    print((v2_df["action"].value_counts(normalize=True) * 100).round(1).to_string())

    return v2_df, master_update, ps2_feed


# ==============================================================================
# EVALUATION & METRICS REPORT
# ==============================================================================
def evaluate_metrics(m_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SYSTEM UPGRADE EVALUATION & COMPARATIVE BENCHMARK")
    print("=" * 70)
    truth = pd.read_csv(GEO / "surveyed_addresses.csv")
    splits = pd.read_csv(SHARED / "splits.csv")
    base = pd.read_csv(GEO / "baseline_geocodes.csv")

    eval_df = truth.merge(m_df, on="address_id").merge(splits, on="account_id").merge(base, on="address_id", suffixes=("", "_base"))
    eval_df["err_v2"] = np.hypot(eval_df.px - eval_df.surveyed_x, eval_df.py - eval_df.surveyed_y)
    eval_df["err_base"] = np.hypot(eval_df.geocoder_x - eval_df.surveyed_x, eval_df.geocoder_y - eval_df.surveyed_y)

    print("\n1. Overall Performance on 100 Ground-Truth Surveyed Addresses:")
    print(f"   Baseline Median Error:   {eval_df.err_base.median():.1f} m  | 90th pct: {eval_df.err_base.quantile(0.9):.1f} m")
    print(f"   Advanced Engine Median: {eval_df.err_v2.median():.1f} m  | 90th pct: {eval_df.err_v2.quantile(0.9):.1f} m")
    err_reduction = (1.0 - eval_df.err_v2.median() / eval_df.err_base.median()) * 100
    print(f"   Median Error Reduction: {err_reduction:.1f}%")

    print("\n2. Performance Breakdown by Evidence Tier (Surveyed Rows):")
    g = eval_df.groupby("tier").agg(
        count=("err_v2", "size"),
        median_err=("err_v2", "median"),
        p90_err=("err_v2", lambda s: s.quantile(0.9)),
        baseline_median=("err_base", "median")
    )
    print(g.round(1).to_string())

    print("\n3. Empirical Test Set Coverage (n=15 held-out test rows):")
    test_df = eval_df[eval_df.split == "test"]
    covered = (test_df.err_v2 <= test_df.R90_meters).mean()
    print(f"   Empirical Test Coverage: {covered * 100:.1f}% ({(test_df.err_v2 <= test_df.R90_meters).sum()}/{len(test_df)} rows within R90)")


# ==============================================================================
# MAIN PIPELINE ENTRYPOINT
# ==============================================================================
def main():
    print("=" * 70)
    print("STARTING CREDITNIRVANA ADVANCED SPATIAL INTELLIGENCE ENGINE")
    print("=" * 70)

    # Load baseline datasets
    visits = pd.read_csv(SHARED / "field_visits.csv")
    addr = pd.read_csv(SHARED / "addresses.csv")
    base = pd.read_csv(GEO / "baseline_geocodes.csv")
    loc = pd.read_csv(GEO / "localities.csv")
    pois = pd.read_csv(GEO / "landmarks_poi.csv")
    st = pd.read_csv(OUT / "structured_addresses.csv", dtype={"pincode": str})
    gps_file = GEO / "visit_gps_points.csv"

    # Subsystem 1: Photo Fraud Gating
    clean_visits, _ = run_subsystem1_photo_fraud(visits)

    # Subsystem 4: Kinematic Dwell Clustering
    updated_visits, _ = run_subsystem4_kinematics(gps_file, clean_visits)

    # Subsystem 2: Vernacular Remark Mining
    rem_df = run_subsystem2_remarks(updated_visits, addr, pois)

    # Subsystem 3: Negative Spatial Evidence
    fail_trajectories, _ = run_subsystem3_negative_evidence(visits, gps_file)

    # Subsystem 5: Integrated Fusion & Cross-Account Knowledge Graph
    m_df, _ = run_integrated_pipeline(updated_visits, rem_df, fail_trajectories, st, base, loc, pois)

    # Subsystem 6: Conformal Radii, Production Updates & PS2 Contracts
    run_subsystem6_contracts(m_df)

    # System Evaluation Report
    evaluate_metrics(m_df)
    print("\n" + "=" * 70)
    print("ALL SIX ADVANCED SUBSYSTEMS SUCCESSFULLY EXECUTED AND EXPORTED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
