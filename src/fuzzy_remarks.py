"""Fuzzy Logic Layer for Understanding and Weighting Field Agent Remarks.

CreditNirvana Problem Statement 3: Address Geocoder That Learns from Field Visits.

Key Design Principles:
1. Lightweight, deterministic, explainable, and fully auditable (No black-box ML/LLMs).
2. Keeps existing RapidFuzz string matching for lexical tolerance.
3. Uses interpretable triangular/trapezoidal fuzzy membership functions and visible fuzzy inference rules.
4. Converts noisy multilingual remarks into structured spatial constraints with confidence scores.
5. Emits output/remark_extracted_corrections.csv with detailed reasoning for each remark.
"""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from config import GEO, OUT, SHARED


# ==============================================================================
# 1. FUZZY MEMBERSHIP FUNCTIONS (Lightweight pure-Python / NumPy)
# ==============================================================================

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangular fuzzy membership function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal fuzzy membership function."""
    if x <= a or x >= d:
        return 0.0
    if a <= x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    if b <= x <= c:
        return 1.0
    if c <= x <= d:
        return (d - x) / (d - c) if d > c else 1.0
    return 0.0


def fuzzify_score(score: float) -> Dict[str, float]:
    """Fuzzifies a continuous [0.0, 1.0] score into linguistic membership degrees.
    
    Linguistic Terms:
    - LOW:       trapmf(score, -0.1, 0.0, 0.35, 0.60)
    - MEDIUM:    trimf(score, 0.40, 0.60, 0.80)
    - HIGH:      trimf(score, 0.65, 0.80, 0.95)
    - VERY_HIGH: trapmf(score, 0.80, 0.92, 1.0, 1.1)
    """
    x = float(np.clip(score, 0.0, 1.0))
    return {
        "low": trapmf(x, -0.1, 0.0, 0.35, 0.60),
        "med": trimf(x, 0.40, 0.60, 0.80),
        "high": trimf(x, 0.65, 0.80, 0.95),
        "vhigh": trapmf(x, 0.80, 0.92, 1.0, 1.1),
    }


# ==============================================================================
# 2. CANONICAL MULTILINGUAL DICTIONARY (Kannada, Hindi, English)
# ==============================================================================

# Explicit correction indicators with baseline prior weights
CANONICAL_CORRECTIONS = [
    ("nija mane ide", 1.0),
    ("nija mane", 1.0),
    ("nijavada thikaana", 1.0),
    ("asli ghar", 1.0),
    ("sahi ghar", 1.0),
    ("address galat likha hai", 1.0),
    ("address galat", 1.0),
    ("address tappu", 1.0),
    ("address wrong", 1.0),
    ("ghar idhar hai", 0.9),
    ("actual house", 1.0),
    ("true address", 1.0),
    ("correct location", 1.0),
    ("blue gate wala ghar", 0.75),
    ("shift ho gaye", 0.6),
    ("house vacated", 0.6),
    ("padosi ne bola", 0.5),
]

# Canonical spatial relations with aliases across English, Kannada, and Hindi
CANONICAL_RELATIONS = [
    ("BEHIND", "hinde", 1.0),
    ("BEHIND", "hindhe", 0.95),
    ("BEHIND", "peeche", 1.0),
    ("BEHIND", "pichhe", 0.95),
    ("BEHIND", "behind", 1.0),
    ("BEHIND", "back side", 0.95),
    ("OPPOSITE", "eduru road nalli", 1.0),
    ("OPPOSITE", "eduru", 1.0),
    ("OPPOSITE", "edurru", 0.95),
    ("OPPOSITE", "ke saamne wali gali mein", 1.0),
    ("OPPOSITE", "saamne", 1.0),
    ("OPPOSITE", "samne", 0.95),
    ("OPPOSITE", "opposite", 1.0),
    ("OPPOSITE", "front", 0.9),
    ("BESIDE", "se right side", 1.0),
    ("BESIDE", "right side", 0.95),
    ("BESIDE", "pakka", 0.9),
    ("BESIDE", "bagal", 0.9),
    ("BESIDE", "baju", 0.9),
    ("BESIDE", "beside", 1.0),
    ("BESIDE", "next to", 0.95),
    ("AHEAD", "cross munde", 1.0),
    ("AHEAD", "gali aage", 1.0),
    ("AHEAD", "lanes ahead", 1.0),
    ("AHEAD", "munde", 0.9),
    ("AHEAD", "aage", 0.9),
    ("AHEAD", "ahead", 0.9),
    ("NEAR", "hattira ide", 1.0),
    ("NEAR", "hattira", 0.95),
    ("NEAR", "ke paas hai", 1.0),
    ("NEAR", "ke paas", 0.95),
    ("NEAR", "paas", 0.9),
    ("NEAR", "near", 1.0),
    ("NEAR", "close to", 0.95),
]

# 14 Canonical Landmark Categories matching landmarks_poi.csv
CANONICAL_LANDMARKS = [
    ("ganesha_temple", "ganesh mandir"),
    ("ganesha_temple", "ganesh temple"),
    ("ganesha_temple", "ganapathi gudi"),
    ("ganesha_temple", "ganapathi temple"),
    ("hanuman_temple", "hanuman mandir"),
    ("hanuman_temple", "hanuman temple"),
    ("hanuman_temple", "anjaneya gudi"),
    ("hanuman_temple", "anjaneya swamy temple"),
    ("ration_shop", "ration shop"),
    ("ration_shop", "ration dukan"),
    ("ration_shop", "ration ki dukaan"),
    ("ration_shop", "ration angadi"),
    ("ration_shop", "nyayabele angadi"),
    ("ration_shop", "pds shop"),
    ("church", "church"),
    ("church", "girja ghar"),
    ("masjid", "masjid"),
    ("masjid", "jama masjid"),
    ("masjid", "mosque"),
    ("masjid", "masidi"),
    ("bus_stop", "bus stop"),
    ("bus_stop", "bus stand"),
    ("bus_stop", "bus adda"),
    ("bus_stop", "bus nildana"),
    ("community_hall", "community hall"),
    ("community_hall", "barat ghar"),
    ("community_hall", "kalyana mantapa"),
    ("community_hall", "samudaya bhavana"),
    ("milk_dairy", "milk dairy"),
    ("milk_dairy", "doodh dairy"),
    ("milk_dairy", "haalina dairy"),
    ("milk_dairy", "milk booth"),
    ("govt_school", "govt school"),
    ("govt_school", "government school"),
    ("govt_school", "sarkari school"),
    ("govt_school", "sarkari shaale"),
    ("medical_store", "medical store"),
    ("medical_store", "dawai ki dukaan"),
    ("medical_store", "medical shop"),
    ("medical_store", "medicals"),
    ("medical_store", "pharmacy"),
    ("water_tank", "water tank"),
    ("water_tank", "overhead tank"),
    ("water_tank", "paani ki tanki"),
    ("water_tank", "tanki"),
    ("water_tank", "neerina tank"),
    ("post_office", "post office"),
    ("post_office", "dak ghar"),
    ("post_office", "anche kacheri"),
    ("petrol_bunk", "petrol bunk"),
    ("petrol_bunk", "petrol pump"),
    ("park", "park"),
    ("park", "children park"),
    ("park", "bagicha"),
    ("park", "udyanavana"),
]


# ==============================================================================
# 3. FUZZY INFERENCE RULES (Interpretable & Auditable)
# ==============================================================================

def evaluate_fuzzy_rules(
    lexical_score: float,
    context_score: float,
    correction_score: float,
    landmark_score: float,
    direction_score: float,
) -> float:
    """Evaluates transparent fuzzy inference rules and defuzzifies to crisp confidence.
    
    Rules:
    - R1 (Strong Explicit Spatial Correction):
        IF lexical is HIGH/VHIGH AND context is HIGH/VHIGH AND correction is HIGH/VHIGH AND landmark is HIGH/VHIGH
        THEN confidence is VERY_HIGH (0.95)
    - R2 (Landmark Spatial Cue without explicit correction):
        IF lexical is HIGH/VHIGH AND landmark is HIGH/VHIGH AND direction is HIGH/VHIGH
        THEN confidence is HIGH (0.85)
    - R3 (Displacement Correction without Landmark):
        IF lexical is HIGH/VHIGH AND correction is HIGH/VHIGH AND direction is HIGH/VHIGH
        THEN confidence is HIGH (0.80)
    - R4 (Medium Lexical with High Context):
        IF lexical is MED AND context is HIGH/VHIGH
        THEN confidence is MEDIUM_HIGH (0.75)
    - R5 (Correction detected with landmark or direction):
        IF correction is HIGH/VHIGH AND (landmark is HIGH/VHIGH OR direction is HIGH/VHIGH)
        THEN confidence is MEDIUM_HIGH (0.72)
    - R6 (Weak Landmark/Direction with Low Correction):
        IF correction is LOW AND (landmark is LOW OR direction is LOW)
        THEN confidence is LOW (0.30)
    - R7 (Uninformative / Non-spatial operational remark):
        IF lexical is LOW AND context is LOW
        THEN confidence is VERY_LOW (0.10)
    """
    mf_lex = fuzzify_score(lexical_score)
    mf_ctx = fuzzify_score(context_score)
    mf_cor = fuzzify_score(correction_score)
    mf_lm  = fuzzify_score(landmark_score)
    mf_dir = fuzzify_score(direction_score)

    r1 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_ctx["high"], mf_ctx["vhigh"]),
             max(mf_cor["high"], mf_cor["vhigh"]), max(mf_lm["high"], mf_lm["vhigh"]))
    r2 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_lm["high"], mf_lm["vhigh"]),
             max(mf_dir["high"], mf_dir["vhigh"]))
    r3 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_cor["high"], mf_cor["vhigh"]),
             max(mf_dir["high"], mf_dir["vhigh"]))
    r4 = min(mf_lex["med"], max(mf_ctx["high"], mf_ctx["vhigh"]))
    r5 = min(max(mf_cor["high"], mf_cor["vhigh"]),
             max(mf_lm["high"], mf_lm["vhigh"], mf_dir["high"], mf_dir["vhigh"]))
    r6 = min(mf_cor["low"], max(mf_lm["low"], mf_dir["low"]))
    r7 = min(mf_lex["low"], mf_ctx["low"])

    rule_weights = [r1, r2, r3, r4, r5, r6, r7]
    rule_consequents = [0.95, 0.85, 0.80, 0.75, 0.72, 0.30, 0.10]

    sum_w = sum(rule_weights)
    if sum_w > 1e-6:
        crisp_conf = float(np.dot(rule_weights, rule_consequents) / sum_w)
    else:
        crisp_conf = 0.20
    return float(np.clip(crisp_conf, 0.0, 1.0))


# ==============================================================================
# 4. STRUCTURED REMARK PARSING & EVIDENCE EXTRACTION
# ==============================================================================

def parse_agent_remark(
    remark_text: str,
    town_id: Optional[str] = None,
    pois_df: Optional[pd.DataFrame] = None,
    checkin_x: Optional[float] = None,
    checkin_y: Optional[float] = None,
) -> Dict[str, Any]:
    """Parses a noisy agent remark into structured spatial evidence with fuzzy confidence.
    
    Returns a dictionary matching the required schema:
    remark_raw, remark_normalized, correction_detected, landmark_type, landmark_name,
    relation, cross_count, token_match_score, landmark_match_score, correction_confidence,
    direction_confidence, context_confidence, landmark_confidence, remark_confidence,
    remark_status, reason.
    """
    if not isinstance(remark_text, str) or not remark_text.strip():
        return {
            "remark_raw": "",
            "remark_normalized": "",
            "correction_detected": False,
            "landmark_type": None,
            "landmark_name": None,
            "relation": None,
            "cross_count": np.nan,
            "token_match_score": 0.0,
            "landmark_match_score": 0.0,
            "correction_confidence": 0.0,
            "direction_confidence": 0.0,
            "context_confidence": 0.0,
            "landmark_confidence": 0.0,
            "remark_confidence": 0.0,
            "remark_status": "UNRESOLVED",
            "reason": "Empty remark",
        }

    raw = remark_text.strip()
    norm = re.sub(r"\s+", " ", raw.lower().replace(";", " ; "))

    # If the remark has a status prefix before semicolon, isolate the spatial cue segment
    cue_text = norm
    if ";" in norm:
        parts = [p.strip() for p in norm.split(";") if p.strip()]
        for p in parts[1:]:
            cue_text = p
            break

    # 1. Fuzzy match correction indicators
    corr_found = False
    corr_score = 0.0
    matched_corr_phrase = None
    for phrase, weight in CANONICAL_CORRECTIONS:
        if phrase in norm:
            corr_found = True
            corr_score = max(corr_score, weight)
            matched_corr_phrase = phrase
            break
        sim = fuzz.partial_ratio(phrase, norm) / 100.0
        if sim >= 0.88:
            corr_found = True
            corr_score = max(corr_score, sim * weight)
            matched_corr_phrase = phrase
            break

    # 2. Fuzzy match landmark categories and POI aliases
    lm_type, lm_name, lm_score = None, None, 0.0
    for ltype, alias in CANONICAL_LANDMARKS:
        if alias in cue_text:
            lm_type = ltype
            lm_name = alias.title()
            lm_score = 1.0
            break
        sim = fuzz.partial_ratio(alias, cue_text) / 100.0
        if sim >= 0.85 and sim > lm_score:
            lm_type = ltype
            lm_name = alias.title()
            lm_score = sim

    # 3. Fuzzy match spatial relations
    rel_found, rel_score = None, 0.0
    for rel, alias, weight in CANONICAL_RELATIONS:
        if alias in cue_text:
            rel_found = rel
            rel_score = max(rel_score, weight)
            break
        sim = fuzz.partial_ratio(alias, cue_text) / 100.0
        if sim >= 0.85 and sim * weight > rel_score:
            rel_found = rel
            rel_score = sim * weight

    # 4. Extract cross count / lane displacement modifier
    cross_match = re.search(r"\b(\d+)\s*(?:cross|gali|lanes?|road)\b", cue_text)
    cross_count = int(cross_match.group(1)) if cross_match else np.nan

    token_match_score = float(max(corr_score, lm_score, rel_score))

    # 5. Evaluate Context Confidence
    has_lm = lm_type is not None
    has_rel = rel_found is not None
    has_mod = not np.isnan(cross_count)
    if has_lm and has_rel and has_mod:
        context_score = 1.0
    elif has_lm and has_rel:
        context_score = 0.85
    elif has_lm or (has_rel and has_mod):
        context_score = 0.65
    elif has_rel:
        context_score = 0.45
    else:
        context_score = 0.10

    # 6. Direction Confidence
    direction_score = float(rel_score)

    # 7. Correction Confidence
    correction_score = float(corr_score) if corr_found else (0.40 if (has_lm and has_rel) else 0.10)

    # 8. Landmark Confidence against landmarks_poi.csv and consistency
    landmark_confidence = float(lm_score)
    poi_dist = np.nan
    poi_in_town = False
    if has_lm and pois_df is not None and not pois_df.empty:
        cand = pois_df[pois_df.landmark_type == lm_type]
        if town_id:
            cand_town = cand[cand.town_id == town_id]
            if not cand_town.empty:
                cand = cand_town
                poi_in_town = True

        if not cand.empty and pd.notna(checkin_x) and pd.notna(checkin_y):
            dists = np.hypot(cand.x - checkin_x, cand.y - checkin_y)
            poi_dist = float(dists.min())
            if poi_dist <= 350.0:
                landmark_confidence = min(1.0, landmark_confidence * 1.10)
            elif poi_dist > 1500.0:
                landmark_confidence = landmark_confidence * 0.70

    # 9. Evaluate Fuzzy Inference Rules
    raw_fuzzy_conf = evaluate_fuzzy_rules(
        lexical_score=token_match_score,
        context_score=context_score,
        correction_score=correction_score,
        landmark_score=landmark_confidence,
        direction_score=direction_score,
    )

    # Consistency Check: detect strong contradiction (>2500m from candidate POI)
    is_conflicting = False
    if has_lm and not np.isnan(poi_dist) and poi_dist > 2500.0:
        is_conflicting = True
        remark_conf = raw_fuzzy_conf * 0.50
    else:
        remark_conf = raw_fuzzy_conf

    # Assign remark status
    if not has_lm and not has_rel and not corr_found:
        status = "UNRESOLVED"
    elif is_conflicting:
        status = "CONFLICTING"
    elif corr_found and remark_conf >= 0.70:
        status = "STRONG_CORRECTION"
    elif (has_lm or has_rel) and remark_conf >= 0.55:
        status = "USEFUL_SPATIAL_CUE"
    else:
        status = "WEAK_CUE"

    # Human-readable audit reason
    reasons = []
    if corr_found:
        reasons.append(f"Detected correction phrase '{matched_corr_phrase}' (conf={corr_score:.2f})")
    if rel_found:
        reasons.append(f"Fuzzy matched relation '{rel_found}' (conf={rel_score:.2f})")
    if lm_type:
        reasons.append(f"Landmark '{lm_name}' resolved to category {lm_type} (score={lm_score:.2f})")
    if not np.isnan(cross_count):
        reasons.append(f"Displacement constraint: {cross_count} cross/gali/lanes ahead")
    if is_conflicting:
        reasons.append(f"Spatial conflict: candidate POI is {poi_dist:.0f}m from checkin GPS (>2500m)")

    reason_str = "; ".join(reasons) if reasons else "No spatial or correction cues recognized"

    return {
        "remark_raw": raw,
        "remark_normalized": norm,
        "correction_detected": bool(corr_found),
        "landmark_type": lm_type,
        "landmark_name": lm_name,
        "relation": rel_found,
        "cross_count": cross_count,
        "token_match_score": round(token_match_score, 3),
        "landmark_match_score": round(lm_score, 3),
        "correction_confidence": round(correction_score, 3),
        "direction_confidence": round(direction_score, 3),
        "context_confidence": round(context_score, 3),
        "landmark_confidence": round(landmark_confidence, 3),
        "final_remark_confidence": round(remark_conf, 3),
        "remark_confidence": round(remark_conf, 3),
        "remark_status": status,
        "reason": reason_str,
    }


# ==============================================================================
# 5. AUDIT TRAIL GENERATION & BATCH EXTRACTION
# ==============================================================================

def extract_and_audit_all_remarks(
    visits_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Processes all remarks in field_visits.csv and creates output/remark_extracted_corrections.csv."""
    v_path = visits_path or (SHARED / "field_visits.csv")
    out_path = output_path or (OUT / "remark_extracted_corrections.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    visits = pd.read_csv(v_path)
    addr = pd.read_csv(SHARED / "addresses.csv")
    pois = pd.read_csv(GEO / "landmarks_poi.csv")

    df_merged = visits.merge(addr[["address_id", "town_id"]], on="address_id", how="left")

    parsed_records = []
    for row in df_merged.itertuples(index=False):
        res = parse_agent_remark(
            remark_text=row.remark,
            town_id=row.town_id,
            pois_df=pois,
            checkin_x=row.checkin_x,
            checkin_y=row.checkin_y,
        )
        res["address_id"] = row.address_id
        res["visit_id"] = row.visit_id
        res["normalized_text"] = res["remark_normalized"]
        res["detected_relation"] = res["relation"]
        res["detected_landmark"] = res["landmark_name"]
        parsed_records.append(res)

    audit_df = pd.DataFrame(parsed_records)

    # Reorder columns as specified in Section 10
    cols = [
        "address_id",
        "visit_id",
        "remark_raw",
        "normalized_text",
        "detected_relation",
        "detected_landmark",
        "token_match_score",
        "correction_confidence",
        "direction_confidence",
        "landmark_confidence",
        "final_remark_confidence",
        "remark_status",
        "reason",
    ]
    audit_df[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Created audit trail: {out_path} ({len(audit_df)} records)")
    return audit_df


# ==============================================================================
# 6. ADDRESS-LEVEL REMARK AGGREGATION & SPATIAL EVIDENCE
# ==============================================================================

def get_address_level_remark_evidence(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates remark evidence to the address level for fusion with pins and fallback."""
    # Filter to non-empty cues
    cues = audit_df[audit_df.remark_status != "UNRESOLVED"].copy()
    if cues.empty:
        return pd.DataFrame(columns=["address_id", "best_remark_conf", "best_remark_status",
                                     "remark_landmark_type", "remark_relation", "has_strong_correction"])

    # Rank cues by final_remark_confidence
    cues = cues.sort_values(["address_id", "final_remark_confidence"], ascending=[True, False])
    best = cues.groupby("address_id").first().reset_index()

    addr_rows = []
    for aid, g in cues.groupby("address_id"):
        top = g.iloc[0]
        has_strong = (g.remark_status == "STRONG_CORRECTION").any()
        best_conf = float(g.final_remark_confidence.max())
        lm_type = top["detected_landmark"]
        rel = top["detected_relation"]
        status = top["remark_status"]
        addr_rows.append({
            "address_id": aid,
            "best_remark_conf": round(best_conf, 3),
            "best_remark_status": status,
            "remark_landmark_name": lm_type,
            "remark_relation": rel,
            "has_strong_correction": has_strong,
            "remark_reason": top["reason"],
        })

    return pd.DataFrame(addr_rows)


if __name__ == "__main__":
    df_audit = extract_and_audit_all_remarks()
    print("Remark Status Breakdown:")
    print(df_audit.remark_status.value_counts().to_string())
