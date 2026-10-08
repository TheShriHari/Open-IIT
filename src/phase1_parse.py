"""Phase 1 - Address parsing & structuring (blueprint section 3).

Deterministic: compiled regex + rapidfuzz. No ML.
Outputs: output/structured_addresses.csv, output/phase1_audit.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import GEO, LOCALITY_THRESHOLD, OUT, POI_FUZZY_THRESHOLD, SHARED


# ---- Tier 1 lexicon: canonical type == landmarks_poi.landmark_type (most specific first) ----
LEXICON = [
    ("ganesha_temple", r"ganesh|ganapathi|ಗಣಪತಿ|गणेश"),
    ("hanuman_temple", r"hanuman|anjaneya|ಆಂಜನೇಯ|हनुमान"),
    ("temple", r"temple|mandir|devasthana|ದೇವಸ್ಥಾನ|मंदिर|ಗುಡಿ|\bgudi\b"),
    ("ration_shop", r"\bration\b|\bpds\b|fair price|ಪಡಿತರ|ರೇಷನ್|ನ್ಯಾಯಬೆಲೆ|राशन"),
    ("church", r"church|ಚರ್ಚ್|चर्च|girja"),
    ("masjid", r"masjid|mosque|dargah|ಮಸೀದಿ|मस्जिद|दरगाह"),
    ("bus_stop", r"\bbus (?:stop|stand)\b|\bbus\b|ಬಸ್|बस स्टॉप|बस स्टैंड"),
    ("community_hall", r"community hall|\bhall\b|kalyana|mantapa|ಕಲ್ಯಾಣ|ಹಾಲ್|हॉल|बारात घर"),
    ("milk_dairy", r"dairy|\bmilk\b|daiyr|ಡೈರಿ|डेयरी|doodh"),
    ("govt_school", r"school|vidyalaya|\bshale\b|ಶಾಲೆ|स्कूल|विद्यालय"),
    ("medical_store", r"medical|hospital|clinic|dispensary|aspatre|ಮೆಡಿಕಲ್|ಆಸ್ಪತ್ರೆ|मेडिकल|अस्पताल"),
    ("water_tank", r"water tank|overhead tank|\btank\b|\btanki\b|ಟ್ಯಾಂಕ್|टंकी"),
    ("post_office", r"post office|ಅಂಚೆ|डाकघर"),
    ("petrol_bunk", r"petrol|\bbunk\b|\bpump\b|ಪೆಟ್ರೋಲ್|पेट्रोल"),
    ("park", r"\bpark\b|ಉದ್ಯಾನವನ|पार्क"),
]
LEXICON = [(t, re.compile(p, re.IGNORECASE)) for t, p in LEXICON]

RELATIONS = [
    ("NEAR", r"\bnear\b|\bnr\b|close to|adj\.?|pakka|\bpaas\b|hattira|ಹತ್ತಿರ|पास"),
    ("OPPOSITE", r"\bopp\b|opposite|eduru|edurru|samne|saamne|ಎದುರು|सामने"),
    ("BEHIND", r"behind|hinde|peeche|pichhe|ಹಿಂದೆ|पीछे"),
]
RELATIONS = [(l, re.compile(p, re.IGNORECASE)) for l, p in RELATIONS]

DOOR_A = re.compile(
    r"\b(?:no\.?|#|h\.?no\.?|house\s*(?:no\.?)?|flat\s*(?:no\.?)?|plot\s*(?:no\.?)?)\s*([a-z0-9/-]+)")
DOOR_B = re.compile(r"^\s*([a-z0-9]+[/-][a-z0-9]+|\d{1,4}[a-z]?)\b")
ORDINAL = re.compile(r"(st|nd|rd|th)$")
STREET = re.compile(
    r"\b(\d+(?:st|nd|rd|th)?\s*(?:cross|main|road|rd|x|gali|lane)|gali\s*(?:no\.?)?\s*\d+"
    r"|block\s*[a-z0-9]+|[a-z]\s*block)\b")


def door_no(t: str):
    m = DOOR_A.search(t)
    if m:
        return m.group(1)
    m = DOOR_B.search(t[:35])
    if m:
        v = m.group(1)
        if len(v) <= 5 and not ORDINAL.search(v):
            return v
    return np.nan


def street_info(t: str):
    found = list(dict.fromkeys(m.group(1) for m in STREET.finditer(t)))
    return "; ".join(found) if found else np.nan


def extract_street_keys(t: str, raw_norm: str = ""):
    """Extracts hierarchical Indian street key components from address text.
    
    Hierarchy:
    1. cross_main: cross_no + main_no
    2. block_road: block + road_no
    3. block: block identifier
    4. gali: gali_no
    5. fallback: locality + street_info (computed in fusion/cross-account)
    """
    def num(pat, src):
        m = re.search(pat, src, re.IGNORECASE)
        if not m:
            return None
        for g in m.groups():
            if g is not None:
                return g
        return None

    # Cross & Main numbers
    cross_no = num(r"\b(\d+)(?:st|nd|rd|th)?\s*(?:cross|x)\b", t)
    main_no = num(r"\b(\d+)(?:st|nd|rd|th)?\s*main\b", t)
    cross_main = f"{cross_no}_{main_no}" if cross_no and main_no else None

    # Block & Road numbers (handling 'block a', 'b block', 'b blk', 'blk b')
    blk_match = re.search(r"\b(?:block|blk)\.?\s*([a-z0-9]+)\b|\b([a-z0-9]+)\s*(?:block|blk)\b", t, re.IGNORECASE)
    block = None
    if blk_match:
        b_cand = blk_match.group(1) or blk_match.group(2)
        if len(b_cand) <= 3:
            block = b_cand.lower()

    road_no = num(r"\b(?:road|rd)\.?\s*(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s*(?:road|rd)\b", t)
    block_road = f"{block}_{road_no}" if block and road_no else None

    # Gali number (Latin + Devanagari)
    gali_no = num(r"\bgali\s*(?:no\.?|n\.?|#)?\s*(\d+)\b", t)
    if not gali_no and raw_norm:
        gali_no = num(r"गली\s*(?:नं\.?|नंबर)?\s*(\d+)", raw_norm)
    gali = str(gali_no) if gali_no else None

    # Determine highest-priority street key available
    if cross_main:
        key = f"cross_main:{cross_main}"
    elif block_road:
        key = f"block_road:{block_road}"
    elif block:
        key = f"block:{block}"
    elif gali:
        key = f"gali:{gali}"
    else:
        key = None

    return {
        "cross_no": cross_no,
        "main_no": main_no,
        "cross_main": cross_main,
        "block": block,
        "road_no": road_no,
        "block_road": block_road,
        "gali_no": gali_no,
        "gali": gali,
        "street_key": key,
    }




def tier1_landmark(t: str):
    for typ, rx in LEXICON:
        m = rx.search(t)
        if m:
            return typ, m.span()
    return None, None


def build_poi_lookup(lm):
    return {
        tid: {"types": g.landmark_type.tolist(), "names": g.name.str.lower().tolist()}
        for tid, g in lm.groupby("town_id")
    }


def tier2_landmark(t: str, town_id, lookup):
    """Fuzzy match 2-3 word spans against POI names of the same town (token_set_ratio)."""
    cand = lookup.get(town_id)
    if not cand:
        return None, None, np.nan
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[^\W\d_]+", t)]
    best = (None, None, 0.0)
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            span_txt = " ".join(w[0] for w in words[i:i + n])
            r = process.extractOne(span_txt, cand["names"], scorer=fuzz.token_set_ratio)
            if r and r[1] >= POI_FUZZY_THRESHOLD and r[1] > best[2]:
                best = (cand["types"][r[2]], (words[i][1], words[i + n - 1][2]), r[1])
    return best


def relation(t: str, lm_span):
    if lm_span is None:
        return np.nan
    mid = (lm_span[0] + lm_span[1]) / 2.0
    best, dmin = np.nan, 1e9
    for label, rx in RELATIONS:
        for m in rx.finditer(t):
            d = abs((m.start() + m.end()) / 2.0 - mid)
            if d < dmin:
                best, dmin = label, d
    return best


def resolve_locality(t: str, town_id, pincode, loc):
    c = loc[loc.town_id == town_id]
    if isinstance(pincode, str):
        c2 = c[c.pincode.astype(str) == pincode]
        if len(c2):
            c = c2
    if c.empty:
        return np.nan
    r = process.extractOne(t, c.locality_name.str.lower().tolist(), scorer=fuzz.partial_ratio)
    return c.locality_name.iloc[r[2]] if r and r[1] >= LOCALITY_THRESHOLD else np.nan


def parse_addresses(addr_df, pois=None, loc=None, towns=None):
    addr = addr_df.copy()
    if loc is None:
        loc = pd.read_csv(GEO / "localities.csv")
    if pois is None:
        pois = pd.read_csv(GEO / "landmarks_poi.csv")
    lookup = build_poi_lookup(pois)

    addr["clean_text_lower"] = addr.address_text.fillna("").str.lower().str.replace(r"\s+", " ", regex=True)
    # vectorised fields
    addr["pincode"] = addr.clean_text_lower.str.extract(r"\b([1-9][0-9]{5})\b")[0]
    addr["door_no"] = addr.clean_text_lower.map(door_no)
    addr["street_info"] = addr.clean_text_lower.map(street_info)

    # Tier 1 landmark
    t1 = addr.clean_text_lower.map(tier1_landmark)
    addr["landmark_type"] = [x[0] for x in t1]
    spans = [x[1] for x in t1]
    addr["landmark_source"] = np.where(addr.landmark_type.notna(), "regex_lexicon", None)
    addr["landmark_match_score"] = np.where(addr.landmark_type.notna(), 100.0, np.nan)

    # Tier 2 only where Tier 1 missed (in-town rows)
    miss = addr.index[addr.landmark_type.isna() & addr.town_id.ne("OUT")]
    for i in miss:
        typ, span, score = tier2_landmark(addr.at[i, "clean_text_lower"], addr.at[i, "town_id"], lookup)
        if typ:
            addr.at[i, "landmark_type"], spans[i] = typ, span
            addr.at[i, "landmark_source"], addr.at[i, "landmark_match_score"] = "poi_token_set", score

    addr["spatial_relation"] = [relation(t, s) for t, s in zip(addr.clean_text_lower, spans)]
    addr["locality_name"] = [
        resolve_locality(t, tid, p, loc) if tid != "OUT" else np.nan
        for t, tid, p in zip(addr.clean_text_lower, addr.town_id, addr.pincode)
    ]

    # Granular Indian street keys
    raw_texts = addr.address_text.fillna("").tolist()
    street_key_records = [
        extract_street_keys(t, raw) for t, raw in zip(addr.clean_text_lower, raw_texts)
    ]
    sk_df = pd.DataFrame(street_key_records)
    for col in sk_df.columns:
        addr[col] = sk_df[col]

    cols = [
        "address_id", "account_id", "town_id", "address_text", "door_no", "street_info",
        "cross_no", "main_no", "cross_main", "block", "road_no", "block_road", "gali_no", "gali", "street_key",
        "landmark_type", "spatial_relation", "landmark_source", "landmark_match_score",
        "locality_name", "pincode"
    ]
    out = addr[cols]
    out.to_csv(OUT / "structured_addresses.csv", index=False, encoding="utf-8-sig")

    intown = out[out.town_id != "OUT"]
    audit = intown[intown.landmark_type.isna() | intown.locality_name.isna() | intown.pincode.isna()]
    audit.to_csv(OUT / "phase1_audit.csv", index=False, encoding="utf-8-sig")
    return out



def main():
    addr = pd.read_csv(SHARED / "addresses.csv")
    loc = pd.read_csv(GEO / "localities.csv")
    lm = pd.read_csv(GEO / "landmarks_poi.csv")
    out = parse_addresses(addr, pois=lm, loc=loc)

    intown = out[out.town_id != "OUT"]
    audit = intown[intown.landmark_type.isna() | intown.locality_name.isna() | intown.pincode.isna()]
    print(f"rows: {len(out)} (in-town {len(intown)}, OUT {len(out) - len(intown)})")
    print("coverage on in-town rows:")
    for c in ["door_no", "street_info", "landmark_type", "spatial_relation", "locality_name", "pincode"]:
        print(f"  {c:17s} {intown[c].notna().mean():.1%}")
    print("audit rows:", len(audit))


if __name__ == "__main__":
    main()
