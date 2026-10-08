import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# --- Fuzzy Membership Functions ---
def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif x < b:
        return (x - a) / (b - a) if b > a else 1.0
    else:
        return (c - x) / (c - b) if c > b else 1.0

def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a <= x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    elif b <= x <= c:
        return 1.0
    elif c <= x <= d:
        return (d - x) / (d - c) if d > c else 1.0
    return 0.0

def fuzzify_score(x):
    """Fuzzifies a [0, 1] score into (low, medium, high, very_high) memberships."""
    x = float(np.clip(x, 0.0, 1.0))
    low = trapmf(x, -0.1, 0.0, 0.40, 0.65)
    med = trimf(x, 0.45, 0.65, 0.82)
    high = trimf(x, 0.70, 0.85, 0.95)
    vhigh = trapmf(x, 0.85, 0.94, 1.0, 1.1)
    return {"low": low, "med": med, "high": high, "vhigh": vhigh}

# Dictionary definitions
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
    ("padosi ne bola", 0.5)
]

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

def parse_remark(remark_text: str, town_id: str = None, town_pois: pd.DataFrame = None, checkin_xy = None):
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
            "remark_confidence": 0.0,
            "remark_status": "UNRESOLVED",
            "reason": "Empty remark"
        }
    
    raw = remark_text.strip()
    norm = re.sub(r"\s+", " ", raw.lower().replace(";", " ; "))
    
    # Check if there is a spatial cue (especially after semicolon)
    cue_text = norm
    if ";" in norm:
        parts = [p.strip() for p in norm.split(";") if p.strip()]
        for p in parts[1:]:
            # if second part has any landmark or relation keywords, use it as primary cue
            cue_text = p
            break
    
    # 1. Match correction indicators
    corr_found = False
    corr_score = 0.0
    matched_corr_phrase = None
    for phrase, weight in CANONICAL_CORRECTIONS:
        if phrase in norm:
            corr_found = True
            corr_score = max(corr_score, weight)
            matched_corr_phrase = phrase
            break
        else:
            sim = fuzz.partial_ratio(phrase, norm) / 100.0
            if sim >= 0.88:
                corr_found = True
                corr_score = max(corr_score, sim * weight)
                matched_corr_phrase = phrase
                break
    
    # 2. Match landmark
    lm_type, lm_name, lm_score = None, None, 0.0
    for ltype, alias in CANONICAL_LANDMARKS:
        if alias in cue_text:
            lm_type = ltype
            lm_name = alias.title()
            lm_score = 1.0
            break
        else:
            sim = fuzz.partial_ratio(alias, cue_text) / 100.0
            if sim >= 0.85 and sim > lm_score:
                lm_type = ltype
                lm_name = alias.title()
                lm_score = sim

    # 3. Match relation
    rel_found, rel_score = None, 0.0
    for rel, alias, weight in CANONICAL_RELATIONS:
        if alias in cue_text:
            rel_found = rel
            rel_score = max(rel_score, weight)
            break
        else:
            sim = fuzz.partial_ratio(alias, cue_text) / 100.0
            if sim >= 0.85 and sim * weight > rel_score:
                rel_found = rel
                rel_score = sim * weight

    # 4. Extract cross count (e.g. "2 cross munde", "2 gali aage", "2 lanes ahead")
    cross_match = re.search(r"\b(\d+)\s*(?:cross|gali|lanes?|road)\b", cue_text)
    cross_count = int(cross_match.group(1)) if cross_match else np.nan

    token_match_score = float(max(corr_score, lm_score, rel_score))
    
    # 5. Evaluate Context Confidence
    # High context = has relation + landmark + modifier
    has_lm = lm_type is not None
    has_rel = rel_found is not None
    has_mod = not np.isnan(cross_count)
    if has_lm and has_rel and has_mod:
        context_score = 1.0
    elif has_lm and has_rel:
        context_score = 0.85
    elif has_lm or has_rel:
        context_score = 0.45
    else:
        context_score = 0.10

    # 6. Direction confidence
    direction_score = float(rel_score)

    # 7. Correction confidence
    correction_score = float(corr_score) if corr_found else (0.4 if (has_lm and has_rel) else 0.1)

    # 8. Landmark confidence in town
    landmark_confidence = float(lm_score)
    poi_dist = np.nan
    if has_lm and town_pois is not None and not town_pois.empty and checkin_xy is not None:
        cand = town_pois[town_pois.landmark_type == lm_type]
        if not cand.empty:
            dists = np.hypot(cand.x - checkin_xy[0], cand.y - checkin_xy[1])
            poi_dist = float(dists.min())
            if poi_dist <= 300.0:
                landmark_confidence = min(1.0, landmark_confidence * 1.1)
            elif poi_dist > 1500.0:
                landmark_confidence = landmark_confidence * 0.7

    # --- FUZZY INFERENCE ENGINE ---
    # Membership values
    mf_lex = fuzzify_score(token_match_score)
    mf_ctx = fuzzify_score(context_score)
    mf_cor = fuzzify_score(correction_score)
    mf_lm  = fuzzify_score(landmark_confidence)
    mf_dir = fuzzify_score(direction_score)

    # Fuzzy Rules:
    # R1: IF lexical is HIGH/VHIGH AND context is HIGH/VHIGH AND correction is HIGH/VHIGH -> remark_confidence is VERY_HIGH (0.95)
    r1 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_ctx["high"], mf_ctx["vhigh"]), max(mf_cor["high"], mf_cor["vhigh"]))
    
    # R2: IF lexical is HIGH/VHIGH AND context is HIGH/VHIGH AND correction is NOT HIGH -> remark_confidence is HIGH (0.80)
    r2 = min(max(mf_lex["high"], mf_lex["vhigh"]), max(mf_ctx["high"], mf_ctx["vhigh"]), max(mf_cor["low"], mf_cor["med"]))

    # R3: IF lexical is MED AND context is HIGH/VHIGH -> remark_confidence is HIGH (0.75)
    r3 = min(mf_lex["med"], max(mf_ctx["high"], mf_ctx["vhigh"]))

    # R4: IF landmark is HIGH/VHIGH AND direction is HIGH/VHIGH -> spatial is HIGH (0.85)
    r4 = min(max(mf_lm["high"], mf_lm["vhigh"]), max(mf_dir["high"], mf_dir["vhigh"]))

    # R5: IF correction is LOW AND only weak landmark match -> LOW (0.25)
    r5 = min(mf_cor["low"], max(mf_lm["low"], mf_lm["med"]))

    # R6: IF lexical is LOW -> LOW (0.15)
    r6 = mf_lex["low"]

    # Defuzzification (Sugeno weighted average)
    rule_weights = [r1, r2, r3, r4, r5, r6]
    rule_outputs = [0.95, 0.80, 0.75, 0.85, 0.25, 0.15]
    total_w = sum(rule_weights)
    if total_w > 1e-6:
        remark_conf = float(np.dot(rule_weights, rule_outputs) / total_w)
    else:
        remark_conf = float(0.20)

    # Check consistency / conflict
    is_conflicting = False
    if has_lm and not np.isnan(poi_dist) and poi_dist > 2500.0:
        is_conflicting = True
        remark_conf *= 0.5

    # Determine status
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

    # Human readable explanation reason
    reasons = []
    if corr_found:
        reasons.append(f"Correction indicator '{matched_corr_phrase}' detected (conf={corr_score:.2f})")
    if rel_found:
        reasons.append(f"Relation matched to {rel_found} (conf={rel_score:.2f})")
    if lm_type:
        reasons.append(f"Landmark category {lm_type} ('{lm_name}', score={lm_score:.2f})")
    if not np.isnan(cross_count):
        reasons.append(f"Displacement modifier: {cross_count} cross/gali/lanes ahead")
    if is_conflicting:
        reasons.append(f"Warning: candidate POI {poi_dist:.0f}m from checkin GPS (>2500m conflict)")
    
    reason_str = "; ".join(reasons) if reasons else "No spatial or correction cues recognized"

    return {
        "remark_raw": raw,
        "remark_normalized": norm,
        "correction_detected": corr_found,
        "landmark_type": lm_type,
        "landmark_name": lm_name,
        "relation": rel_found,
        "cross_count": cross_count,
        "token_match_score": round(token_match_score, 3),
        "landmark_match_score": round(lm_score, 3),
        "correction_confidence": round(correction_score, 3),
        "direction_confidence": round(direction_score, 3),
        "context_confidence": round(context_score, 3),
        "remark_confidence": round(remark_conf, 3),
        "remark_status": status,
        "reason": reason_str
    }

# Test on the sample remarks given in the user prompt!
test_samples = [
    "nija mane 2 gali aage ide",
    "asli ghar masjid ke peeche",
    "yaaru illa, beega haakide; nija mane Park hinde ide, 2 cross munde",
    "father se baat hui, message de denge; asli ghar Sarkari School ke peeche hai, 2 gali aage",
    "customer se mila, baat hui; address galat likha hai, ghar Ganesh Mandir ke paas hai",
    "avara appa jothe maataadide; mane Neerina Tank eduru road nalli ide",
    "premises locked, nobody home; actual house behind Church, 2 lanes ahead",
    "ghar band tha; blue gate wala ghar, Dawai ki Dukaan se right side",
    "pakkadavaru shift aagidaare antharu; address tappu, mane Sarkari Shaale hattira ide",
    "customer maneli sikkidru",
    "is address pe ghar nahi mila"
]

print("--- Testing user examples & dataset samples ---")
for s in test_samples:
    res = parse_remark(s)
    print(f"\nRemark: {s}")
    print(f"  Correction: {res['correction_detected']} | Rel: {res['relation']} | LM: {res['landmark_name']} ({res['landmark_type']}) | Cross: {res['cross_count']}")
    print(f"  Conf: {res['remark_confidence']} | Status: {res['remark_status']}")
    print(f"  Reason: {res['reason']}")
