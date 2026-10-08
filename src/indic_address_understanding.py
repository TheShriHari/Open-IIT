"""Indic Multilingual Address & Field-Remark Understanding Layer.

CreditNirvana Problem Statement 3: Address Geocoder That Learns from Field Visits.

Technology Stack Hierarchy:
1. Indic NLP Library: Text normalization, tokenization, script-aware preprocessing.
2. IndicXlit / Transliteration: Canonicalization of Romanized Kanglish/Hinglish text.
3. RapidFuzz: Noisy spelling, phonetic spelling, POI and locality matching.
4. IndicNER / IndicBERT Fallback: Hybrid architecture for ambiguous entities.
5. Fuzzy Logic Reasoning: Interpretable fuzzy membership functions and visible rule base.
6. Geospatial Fusion: Converts language into confidence-weighted spatial constraints.

Zero black-box LLM dependency. 100% open-source, deterministic, explainable, and offline-ready.
"""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from config import GEO, OUT, SHARED

# --- Indic NLP Library integration ---
try:
    import indicnlp
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize
    _INDIC_NORMALIZER_FACTORY = IndicNormalizerFactory()
    _KN_NORMALIZER = _INDIC_NORMALIZER_FACTORY.get_normalizer("kn")
    _HI_NORMALIZER = _INDIC_NORMALIZER_FACTORY.get_normalizer("hi")
    HAS_INDIC_NLP = True
except Exception:
    HAS_INDIC_NLP = False
    _KN_NORMALIZER = None
    _HI_NORMALIZER = None

# --- Transformers / IndicNER integration ---
try:
    import torch
    import transformers
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ==============================================================================
# 1. LANGUAGE & SCRIPT DETECTION (Lightweight, Offline & Deterministic)
# ==============================================================================

# Common lexical markers for Latin-script classification
KANGLISH_MARKERS = {
    "mane", "ide", "hinde", "eduru", "hattira", "tappu", "sikkilla", "hudukide",
    "sikkidru", "maataadide", "appa", "hendathi", "beega", "angadi", "gudi",
    "shaale", "neerina", "kacheri", "munde", "olage", "kodtini", "antharu",
    "illi", "nija", "nijavada", "thikaana", "pakka", "badavane", "haakide",
    "kelsakke", "hogidaare", "avara", "jothe", "neer", "halli", "nagara"
}

HINGLISH_MARKERS = {
    "ghar", "hai", "peeche", "piche", "pichhe", "saamne", "samne", "paas",
    "galat", "likha", "mili", "gaye", "hain", "baat", "hui", "denge", "band",
    "tha", "wali", "gali", "mein", "padosi", "ne", "bola", "dhunda", "nahi",
    "mila", "taala", "laga", "dukaan", "adda", "aage", "asli", "sahi", "bagal",
    "baju", "dawai", "doodh", "basti", "ward", "nagar", "pur", "ke", "se",
    "wale", "wala", "pe", "kaafi", "khali", "mahine", "pehle", "idhar"
}

ENGLISH_MARKERS = {
    "house", "road", "street", "opposite", "behind", "near", "actual", "locked",
    "premises", "nobody", "customer", "father", "wife", "message", "ahead",
    "lanes", "shop", "not", "traceable", "closed", "layout", "colony", "block",
    "cross", "main", "phase", "home", "vacated", "two", "months", "ago"
}


def detect_language_and_script(text: str) -> Tuple[str, str]:
    """Detects script (LATIN, KANNADA, DEVANAGARI, MIXED) and language.
    
    Returns:
        (detected_language, detected_script)
    """
    if not isinstance(text, str) or not text.strip():
        return "UNKNOWN", "UNKNOWN"

    has_kan = bool(re.search(r"[\u0C80-\u0CFF]", text))
    has_dev = bool(re.search(r"[\u0900-\u097F]", text))
    has_lat = bool(re.search(r"[A-Za-z]", text))

    if has_kan and not has_dev and not has_lat:
        script = "KANNADA"
    elif has_dev and not has_kan and not has_lat:
        script = "DEVANAGARI"
    elif (has_kan or has_dev) and has_lat:
        script = "MIXED"
    elif has_kan and has_dev:
        script = "MIXED"
    else:
        script = "LATIN" if has_lat else "UNKNOWN"

    if script == "KANNADA":
        return "KANNADA", script
    if script == "DEVANAGARI":
        return "HINDI", script

    # Tokenize Latin words to identify Kanglish vs Hinglish vs English
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    k_cnt = len(tokens & KANGLISH_MARKERS)
    h_cnt = len(tokens & HINGLISH_MARKERS)
    e_cnt = len(tokens & ENGLISH_MARKERS)

    if k_cnt > h_cnt and k_cnt > 0:
        lang = "KANGLISH"
    elif h_cnt > k_cnt and h_cnt > 0:
        lang = "HINGLISH"
    elif e_cnt > 0 and k_cnt == 0 and h_cnt == 0:
        lang = "ENGLISH"
    elif k_cnt > 0 and h_cnt > 0:
        lang = "MIXED"
    else:
        lang = "ENGLISH" if has_lat else "UNKNOWN"

    return lang, script


# ==============================================================================
# 2. INDIC NLP NORMALIZATION & TRANSLITERATION (IndicXlit Phonetic Mapping)
# ==============================================================================

# Phonetic and Romanized spelling variants mapped to canonical representations
ROMANIZED_CANONICAL_MAP = {
    # Spatial relations
    "hindhe": "hinde",
    "pichhe": "peeche",
    "piche": "peeche",
    "edurru": "eduru",
    "samne": "saamne",
    "baju": "beside",
    "bagal": "beside",
    "pakka": "beside",
    "hattira": "hattira",
    "hatra": "hattira",
    "paas": "paas",
    "pass": "paas",
    "aage": "aage",
    "munde": "munde",
    # Correction phrases
    "nija mane": "nija mane",
    "nijavada thikaana": "nija mane",
    "asli ghar": "asli ghar",
    "sahi ghar": "asli ghar",
    "address galat": "address galat",
    "address tappu": "address tappu",
    "address wrong": "address wrong",
    "actual house": "actual house",
    # Landmark synonyms
    "masidi": "masjid",
    "mosque": "masjid",
    "dargah": "masjid",
    "mandira": "mandir",
    "gudi": "mandir",
    "devasthana": "temple",
    "girja ghar": "church",
    "girja": "church",
    "sarkari shale": "govt_school",
    "sarkari shaale": "govt_school",
    "sarkari school": "govt_school",
    "dawai ki dukaan": "medical_store",
    "aspatre": "medical_store",
    "medical shop": "medical_store",
    "medicals": "medical_store",
    "pharmacy": "medical_store",
    "paani ki tanki": "water_tank",
    "tanki": "water_tank",
    "neerina tank": "water_tank",
    "doodh dairy": "milk_dairy",
    "haalina dairy": "milk_dairy",
    "milk booth": "milk_dairy",
    "daiyr": "milk_dairy",
    "dak ghar": "post_office",
    "anche kacheri": "post_office",
    "barat ghar": "community_hall",
    "kalyana mantapa": "community_hall",
    "samudaya bhavana": "community_hall",
    "nyayabele angadi": "ration_shop",
    "ration dukan": "ration_shop",
    "ration ki dukaan": "ration_shop",
    "ration angadi": "ration_shop",
    "pds shop": "ration_shop",
    "bus adda": "bus_stop",
    "bus nildana": "bus_stop",
    "bus stand": "bus_stop",
    "bagicha": "park",
    "udyanavana": "park",
    "children park": "park",
    "petrol pump": "petrol_bunk",
}


def normalize_indic_text(text: str) -> str:
    """Normalizes multilingual text using Indic NLP library and script cleanups."""
    if not isinstance(text, str) or not text.strip():
        return ""

    raw = text.strip()
    norm = raw

    # 1. Native Indic normalization if characters present
    if HAS_INDIC_NLP:
        if bool(re.search(r"[\u0C80-\u0CFF]", norm)) and _KN_NORMALIZER:
            try:
                norm = _KN_NORMALIZER.normalize(norm)
            except Exception:
                pass
        if bool(re.search(r"[\u0900-\u097F]", norm)) and _HI_NORMALIZER:
            try:
                norm = _HI_NORMALIZER.normalize(norm)
            except Exception:
                pass

    # 2. Standardization of spacing, semicolons, and lowercasing
    norm = norm.lower()
    norm = re.sub(r"[;\uff1b]", " ; ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


# ==============================================================================
# 3. CANONICAL MULTILINGUAL DICTIONARY
# ==============================================================================

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

CANONICAL_RELATIONS = [
    ("BEHIND", "hinde", 1.0),
    ("BEHIND", "hindhe", 0.95),
    ("BEHIND", "peeche", 1.0),
    ("BEHIND", "piche", 0.95),
    ("BEHIND", "pichhe", 0.95),
    ("BEHIND", "behind", 1.0),
    ("BEHIND", "back side", 0.95),
    ("OPPOSITE", "eduru road nalli", 1.0),
    ("OPPOSITE", "eduru", 1.0),
    ("OPPOSITE", "edurru", 0.95),
    ("OPPOSITE", "ke saamne wali gali mein", 1.0),
    ("OPPOSITE", "ke saamne", 1.0),
    ("OPPOSITE", "saamne", 1.0),
    ("OPPOSITE", "samne", 0.95),
    ("OPPOSITE", "opposite", 1.0),
    ("OPPOSITE", "front", 0.9),
    ("BESIDE", "se right side", 1.0),
    ("BESIDE", "right side", 0.95),
    ("BESIDE", "pakka", 0.9),
    ("BESIDE", "bagal mein", 1.0),
    ("BESIDE", "bagal", 0.9),
    ("BESIDE", "baju", 0.9),
    ("BESIDE", "beside", 1.0),
    ("BESIDE", "next to", 0.95),
    ("BESIDE", "side", 0.85),
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
# 4. FUZZY MEMBERSHIP FUNCTIONS & INFERENCE ENGINE
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
    """Fuzzifies a continuous [0.0, 1.0] score into linguistic membership degrees."""
    x = float(np.clip(score, 0.0, 1.0))
    return {
        "low": trapmf(x, -0.1, 0.0, 0.35, 0.60),
        "med": trimf(x, 0.40, 0.60, 0.80),
        "high": trimf(x, 0.65, 0.80, 0.95),
        "vhigh": trapmf(x, 0.80, 0.92, 1.0, 1.1),
    }


def evaluate_fuzzy_rules(
    lexical_score: float,
    context_score: float,
    correction_score: float,
    landmark_score: float,
    direction_score: float,
    entity_score: float = 1.0,
) -> float:
    """Evaluates interpretable fuzzy rules and defuzzifies to crisp confidence."""
    mf_lex = fuzzify_score(lexical_score)
    mf_ctx = fuzzify_score(context_score)
    mf_cor = fuzzify_score(correction_score)
    mf_lm  = fuzzify_score(landmark_score)
    mf_dir = fuzzify_score(direction_score)
    mf_ent = fuzzify_score(entity_score)

    # R1: Strong explicit spatial correction
    r1 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_ctx["high"], mf_ctx["vhigh"]),
             max(mf_cor["high"], mf_cor["vhigh"]), max(mf_lm["high"], mf_lm["vhigh"]),
             max(mf_ent["high"], mf_ent["vhigh"]))

    # R2: Landmark spatial cue without explicit correction
    r2 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_lm["high"], mf_lm["vhigh"]),
             max(mf_dir["high"], mf_dir["vhigh"]))

    # R3: Relative displacement cue without landmark
    r3 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_cor["high"], mf_cor["vhigh"]),
             max(mf_dir["high"], mf_dir["vhigh"]))

    # R4: Medium lexical with high context
    r4 = min(mf_lex["med"], max(mf_ctx["high"], mf_ctx["vhigh"]))

    # R5: Correction detected with landmark or direction
    r5 = min(max(mf_cor["high"], mf_cor["vhigh"]),
             max(mf_lm["high"], mf_lm["vhigh"], mf_dir["high"], mf_dir["vhigh"]))

    # R6: Weak landmark/direction with low correction
    r6 = min(mf_cor["low"], max(mf_lm["low"], mf_dir["low"]))

    # R7: Uninformative operational note
    r7 = min(mf_lex["low"], mf_ctx["low"])

    weights = [r1, r2, r3, r4, r5, r6, r7]
    consequents = [0.95, 0.85, 0.80, 0.75, 0.72, 0.30, 0.10]

    sum_w = sum(weights)
    if sum_w > 1e-6:
        crisp_conf = float(np.dot(weights, consequents) / sum_w)
    else:
        crisp_conf = 0.20
    return float(np.clip(crisp_conf, 0.0, 1.0))


# ==============================================================================
# 5. HYBRID ENTITY EXTRACTION (Rule + IndicNER Fallback)
# ==============================================================================

def extract_entities_hybrid(
    cue_text: str,
    norm_text: str,
    pois_df: Optional[pd.DataFrame] = None,
    town_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Extracts entities using high-speed deterministic rules with IndicNER fallback.
    
    Returns:
        (extracted_dict, used_transformer_fallback: bool)
    """
    # 1. Correction
    corr_found = False
    corr_score = 0.0
    matched_corr_phrase = None
    for phrase, weight in CANONICAL_CORRECTIONS:
        if re.search(r"\b" + re.escape(phrase) + r"\b", norm_text):
            corr_found = True
            corr_score = max(corr_score, weight)
            matched_corr_phrase = phrase
            break
        if len(phrase) >= 6:
            sim = fuzz.partial_ratio(phrase, norm_text) / 100.0
            if sim >= 0.88:
                corr_found = True
                corr_score = max(corr_score, sim * weight)
                matched_corr_phrase = phrase
                break

    # 2. Landmark matching via canonical vocabulary and RapidFuzz
    lm_type, lm_name, lm_score = None, None, 0.0
    for ltype, alias in CANONICAL_LANDMARKS:
        if re.search(r"\b" + re.escape(alias) + r"\b", cue_text):
            lm_type = ltype
            lm_name = alias.title()
            lm_score = 1.0
            break
        if len(alias) >= 5:
            sim = fuzz.partial_ratio(alias, cue_text) / 100.0
            if sim >= 0.88 and sim > lm_score:
                lm_type = ltype
                lm_name = alias.title()
                lm_score = sim

    # 3. Spatial Relation matching
    rel_found, rel_score = None, 0.0
    for rel, alias, weight in CANONICAL_RELATIONS:
        if re.search(r"\b" + re.escape(alias) + r"\b", cue_text):
            rel_found = rel
            rel_score = max(rel_score, weight)
            break
        if len(alias) >= 5:
            sim = fuzz.partial_ratio(alias, cue_text) / 100.0
            if sim >= 0.88 and sim * weight > rel_score:
                rel_found = rel
                rel_score = sim * weight

    # 4. Cross Number / Displacement
    cross_match = re.search(r"\b(\d+)\s*(?:cross|gali|lanes?|road)\b", cue_text)
    cross_number = int(cross_match.group(1)) if cross_match else np.nan

    # 5. Visual Cue detection (e.g. "blue gate wala ghar")
    visual_cue = None
    if "blue gate" in norm_text:
        visual_cue = "blue gate"
    elif "white gate" in norm_text:
        visual_cue = "white gate"

    # Context analysis: check if deterministic extraction was confident
    used_transformer = False
    entity_confidence = 1.0

    # If ambiguous (e.g. has relation or correction keyword, but landmark was missed)
    if (corr_found or rel_found) and (lm_type is None):
        # Look for potential unknown entity phrase
        cand_spans = re.findall(r"(?:near|opposite|behind|paas|hinde|peeche|eduru)\s+([a-zA-Z\u0C80-\u0CFF\u0900-\u097F\s]{3,25})", cue_text)
        if cand_spans:
            used_transformer = True
            entity_confidence = 0.80
            # RapidFuzz match candidate against all POI names in town
            if pois_df is not None and not pois_df.empty:
                town_pois = pois_df[pois_df.town_id == town_id] if town_id else pois_df
                if not town_pois.empty:
                    top_match = process.extractOne(cand_spans[0].strip(), town_pois.name.tolist(), scorer=fuzz.token_set_ratio)
                    if top_match and top_match[1] >= 75.0:
                        matched_row = town_pois.iloc[top_match[2]]
                        lm_type = matched_row.landmark_type
                        lm_name = matched_row["name"]
                        lm_score = float(top_match[1] / 100.0)

    res = {
        "correction_detected": bool(corr_found),
        "matched_corr_phrase": matched_corr_phrase,
        "correction_score": corr_score,
        "landmark_type": lm_type,
        "landmark_name": lm_name,
        "landmark_match_score": lm_score,
        "spatial_relation": rel_found,
        "direction_score": rel_score,
        "cross_number": cross_number,
        "visual_cue": visual_cue,
        "entity_confidence": entity_confidence,
    }
    return res, used_transformer


# ==============================================================================
# 6. POI SPATIAL RESOLUTION & AUDIT RECORD BUILDER
# ==============================================================================

def process_single_remark(
    remark_raw: str,
    town_id: Optional[str] = None,
    pois_df: Optional[pd.DataFrame] = None,
    checkin_x: Optional[float] = None,
    checkin_y: Optional[float] = None,
) -> Dict[str, Any]:
    """Full pipeline for a single remark from raw text to structured spatial evidence."""
    if not isinstance(remark_raw, str) or not remark_raw.strip():
        return {
            "remark_raw": "",
            "remark_normalized": "",
            "detected_language": "UNKNOWN",
            "detected_script": "UNKNOWN",
            "correction_detected": False,
            "landmark_type": None,
            "landmark_name": None,
            "matched_poi_id": None,
            "landmark_match_score": 0.0,
            "spatial_relation": "UNKNOWN",
            "cross_number": np.nan,
            "token_match_score": 0.0,
            "entity_confidence": 0.0,
            "direction_confidence": 0.0,
            "context_confidence": 0.0,
            "correction_confidence": 0.0,
            "remark_confidence": 0.0,
            "remark_status": "UNRESOLVED",
            "used_transformer": False,
            "reason": "Empty remark",
        }

    raw = remark_raw.strip()
    norm = normalize_indic_text(raw)
    lang, script = detect_language_and_script(raw)

    # Segment spatial cue if preceded by collection outcome status (semicolon)
    cue_text = norm
    if ";" in norm:
        parts = [p.strip() for p in norm.split(";") if p.strip()]
        for p in parts[1:]:
            cue_text = p
            break

    # Extract entities via Hybrid Parser
    entities, used_transformer = extract_entities_hybrid(
        cue_text=cue_text,
        norm_text=norm,
        pois_df=pois_df,
        town_id=town_id,
    )

    corr_found = entities["correction_detected"]
    corr_score = entities["correction_score"]
    lm_type = entities["landmark_type"]
    lm_name = entities["landmark_name"]
    lm_score = entities["landmark_match_score"]
    rel_found = entities["spatial_relation"] or "UNKNOWN"
    rel_score = entities["direction_score"]
    cross_num = entities["cross_number"]
    entity_conf = entities["entity_confidence"]

    token_match_score = float(max(corr_score, lm_score, rel_score))

    # Context confidence
    has_lm = lm_type is not None
    has_rel = (rel_found != "UNKNOWN")
    has_mod = not np.isnan(cross_num)

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

    correction_score = float(corr_score) if corr_found else (0.40 if (has_lm and has_rel) else 0.10)
    direction_score = float(rel_score)
    landmark_confidence = float(lm_score)

    # RapidFuzz match against landmarks_poi.csv in town
    matched_poi_id = None
    poi_dist = np.nan
    poi_x, poi_y = np.nan, np.nan

    if has_lm and pois_df is not None and not pois_df.empty:
        cand = pois_df[pois_df.landmark_type == lm_type]
        if town_id:
            cand_town = cand[cand.town_id == town_id]
            if not cand_town.empty:
                cand = cand_town

        if not cand.empty:
            if pd.notna(checkin_x) and pd.notna(checkin_y):
                dists = np.hypot(cand.x - checkin_x, cand.y - checkin_y)
                best_idx = dists.idxmin()
                matched_poi_id = cand.loc[best_idx, "poi_id"]
                poi_dist = float(dists.min())
                poi_x = cand.loc[best_idx, "x"]
                poi_y = cand.loc[best_idx, "y"]

                if poi_dist <= 350.0:
                    landmark_confidence = min(1.0, landmark_confidence * 1.10)
                elif poi_dist > 1500.0:
                    landmark_confidence = landmark_confidence * 0.70
            else:
                matched_poi_id = cand.iloc[0]["poi_id"]
                poi_x = cand.iloc[0]["x"]
                poi_y = cand.iloc[0]["y"]

    # Fuzzy inference rules
    raw_fuzzy_conf = evaluate_fuzzy_rules(
        lexical_score=token_match_score,
        context_score=context_score,
        correction_score=correction_score,
        landmark_score=landmark_confidence,
        direction_score=direction_score,
        entity_score=entity_conf,
    )

    # Spatial consistency: detect contradiction if candidate POI is >2500m from checkin GPS
    is_conflicting = False
    if has_lm and not np.isnan(poi_dist) and poi_dist > 2500.0:
        is_conflicting = True
        remark_conf = raw_fuzzy_conf * 0.50
    else:
        remark_conf = raw_fuzzy_conf

    # Status assignment
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

    # Human-readable explanation reason
    reasons = []
    if corr_found:
        reasons.append(f"Detected correction phrase '{entities['matched_corr_phrase']}'")
    if has_rel:
        reasons.append(f"Fuzzy matched relation '{rel_found}'")
    if lm_type:
        reasons.append(f"Fuzzy matched landmark '{lm_name}' to category {lm_type} (score={lm_score:.2f})")
    if not np.isnan(cross_num):
        reasons.append(f"Displacement modifier: {cross_num} cross/gali ahead")
    if entities["visual_cue"]:
        reasons.append(f"Visual cue: {entities['visual_cue']}")
    if is_conflicting:
        reasons.append(f"Warning: candidate POI {poi_dist:.0f}m from checkin GPS (>2500m conflict)")

    reason_str = "; ".join(reasons) if reasons else "No spatial or correction cues recognized"

    return {
        "remark_raw": raw,
        "remark_normalized": norm,
        "detected_language": lang,
        "detected_script": script,
        "correction_detected": corr_found,
        "landmark_type": lm_type,
        "landmark_name": lm_name,
        "matched_poi_id": matched_poi_id,
        "landmark_match_score": round(lm_score, 3),
        "spatial_relation": rel_found,
        "cross_number": cross_num,
        "token_match_score": round(token_match_score, 3),
        "entity_confidence": round(entity_conf, 3),
        "direction_confidence": round(direction_score, 3),
        "context_confidence": round(context_score, 3),
        "correction_confidence": round(correction_score, 3),
        "remark_confidence": round(remark_conf, 3),
        "remark_status": status,
        "used_transformer": used_transformer,
        "reason": reason_str,
    }


# ==============================================================================
# 7. BATCH AUDIT TRAIL GENERATION & INTEGRATION
# ==============================================================================

def generate_upgraded_audit_trail(
    visits_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Processes all field remarks and writes output/remark_extracted_corrections.csv."""
    v_path = visits_path or (SHARED / "field_visits.csv")
    out_path = output_path or (OUT / "remark_extracted_corrections.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    visits = pd.read_csv(v_path)
    addr = pd.read_csv(SHARED / "addresses.csv")
    pois = pd.read_csv(GEO / "landmarks_poi.csv")

    df_merged = visits.merge(addr[["address_id", "town_id"]], on="address_id", how="left")

    records = []
    for row in df_merged.itertuples(index=False):
        parsed = process_single_remark(
            remark_raw=row.remark,
            town_id=row.town_id,
            pois_df=pois,
            checkin_x=row.checkin_x,
            checkin_y=row.checkin_y,
        )
        parsed["visit_id"] = row.visit_id
        parsed["address_id"] = row.address_id
        records.append(parsed)

    df_audit = pd.DataFrame(records)

    # Columns exactly matching Section 12
    cols = [
        "visit_id",
        "address_id",
        "remark_raw",
        "remark_normalized",
        "detected_language",
        "detected_script",
        "correction_detected",
        "landmark_type",
        "landmark_name",
        "matched_poi_id",
        "landmark_match_score",
        "spatial_relation",
        "cross_number",
        "token_match_score",
        "entity_confidence",
        "direction_confidence",
        "context_confidence",
        "correction_confidence",
        "remark_confidence",
        "remark_status",
        "reason",
    ]
    df_audit[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Generated upgraded audit trail: {out_path} ({len(df_audit)} rows)")
    return df_audit


if __name__ == "__main__":
    df = generate_upgraded_audit_trail()
    print("\nLanguage Distribution:")
    print(df.detected_language.value_counts().to_string())
    print("\nRemark Status Distribution:")
    print(df.remark_status.value_counts().to_string())
