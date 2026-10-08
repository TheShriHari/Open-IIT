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
    """Parses a noisy agent remark using the upgraded Indic NLP + Transliteration + Fuzzy layer."""
    from indic_address_understanding import process_single_remark
    res = process_single_remark(
        remark_raw=remark_text,
        town_id=town_id,
        pois_df=pois_df,
        checkin_x=checkin_x,
        checkin_y=checkin_y,
    )
    res["relation"] = res["spatial_relation"]
    res["cross_count"] = res["cross_number"]
    res["final_remark_confidence"] = res["remark_confidence"]
    res["landmark_confidence"] = res["landmark_match_score"]
    return res


# ==============================================================================
# 5. AUDIT TRAIL GENERATION & BATCH EXTRACTION
# ==============================================================================

def extract_and_audit_all_remarks(
    visits_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Processes all remarks in field_visits.csv and creates output/remark_extracted_corrections.csv."""
    from indic_address_understanding import generate_upgraded_audit_trail
    return generate_upgraded_audit_trail(visits_path=visits_path, output_path=output_path)


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
    conf_col = "remark_confidence" if "remark_confidence" in cues.columns else "final_remark_confidence"
    lm_col = "landmark_name" if "landmark_name" in cues.columns else "detected_landmark"
    rel_col = "spatial_relation" if "spatial_relation" in cues.columns else "detected_relation"

    cues = cues.sort_values(["address_id", conf_col], ascending=[True, False])

    addr_rows = []
    for aid, g in cues.groupby("address_id"):
        top = g.iloc[0]
        has_strong = (g.remark_status == "STRONG_CORRECTION").any()
        best_conf = float(g[conf_col].max())
        lm_type = top[lm_col]
        rel = top[rel_col]
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
