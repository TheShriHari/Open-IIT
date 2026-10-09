"""Steps 3-4: clean and parse address text (also reused for agent remarks).

For each address we extract:
  locality (matched to the official list for THAT town), pincode (validated),
  ALL landmarks (exact landmarks_poi types), the relation to each, building name,
  gali / cross / main / block / road / ward numbers and door number.

Key design choices (each fixes a problem found in the earlier version):
  * typos are corrected token-by-token against a known vocabulary (RapidFuzz);
  * the locality and building names are found FIRST and masked, so words inside
    them ("Anjaneya Badavane", "Green Park Layout", "Royal Gardens") are never
    mistaken for landmarks;
  * Ganesh vs Hanuman temples are kept apart, because the map keeps them apart.
"""
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from lexicon import (ALIASES, BUILDINGS, LANDMARK_INDIC, LANDMARK_LATIN, RELATION_INDIC,
                     RELATION_WORDS, correction_vocabulary)

TOWN_WORDS = ["kaveripura", "devgarh nagar", "devgarh", "navanagara east", "navanagara"]


class AddressParser:
    def __init__(self, localities: pd.DataFrame):
        self.loc = localities.copy()
        self.loc["name_l"] = self.loc.locality_name.str.lower()
        self.vocab = correction_vocabulary(self.loc.locality_name)
        self.vocab_set = set(self.vocab)
        self._cache = {}
        self.lm_latin = [(t, re.compile(r"\b" + p + r"\b")) for t, p in LANDMARK_LATIN]
        rel_pairs = [(r, w) for r, ws in RELATION_WORDS.items() for w in ws]
        rel_pairs.sort(key=lambda p: -len(p[1]))
        self.rel_latin = [(r, re.compile(r"\b" + re.escape(w) + r"\b")) for r, w in rel_pairs]

    # ------------------------------------------------------------ cleaning
    def _fix_token(self, tok: str) -> str:
        if tok in ALIASES:
            return ALIASES[tok]
        if tok in self.vocab_set or len(tok) < 4 or not tok.isalpha():
            return tok
        if tok not in self._cache:
            cutoff = 75 if len(tok) <= 5 else 80
            hit = process.extractOne(tok, self.vocab, scorer=fuzz.ratio, score_cutoff=cutoff)
            self._cache[tok] = hit[0] if hit else tok
        return self._cache[tok]

    def clean(self, text: str):
        """Lowercase, normalise scripts and punctuation, fix typos. Returns (clean, n_fixes)."""
        t = unicodedata.normalize("NFC", str(text)).lower()
        t = t.replace("b/h", " behind ").replace("h.no", " house ").replace("#", " house ")
        t = re.sub(r"\b(?:no|noo|nno|nu|n)\s*[-.]?\s*(\d)", r"no \1", t)  # "gali noo-7" -> "gali no 7"
        t = re.sub(r"[,;:()\-]", " ", t)
        t = t.replace(".", " ")                                    # "blk." "nr." "h.no." -> words
        t = re.sub(r"(\d)[a-z](\d)", r"\1\2", t)                   # "1t2h" -> "12h"
        t = re.sub(r"\b(\d+)(?:st|nd|rd|th|ht|h|t)\b",               # "10ht" "11h" -> ordinal
                   lambda m: m.group(1) + {"1": "st", "2": "nd", "3": "rd"}.get(
                       m.group(1)[-1] if not m.group(1).endswith(("11", "12", "13")) else "", "th"), t)
        out, fixes = [], 0
        for tok in t.split():
            new = self._fix_token(tok)
            fixes += new != tok
            out.append(new)
        return " ".join(out), fixes

    # ------------------------------------------------------------ pieces
    def _locality(self, text, town, pincode):
        cand = self.loc[self.loc.town_id == town]
        if cand.empty:
            return None, "none", 0
        # 1) exact phrase after typo correction
        for _, r in cand.sort_values("name_l", key=lambda s: -s.str.len()).iterrows():
            if re.search(r"\b" + re.escape(r.name_l) + r"\b", text):
                return r, "exact", 100
        # 2) fuzzy: best window of the same word length
        toks = text.split()
        best, best_score = None, 0
        for _, r in cand.iterrows():
            n = len(r.name_l.split())
            for i in range(max(1, len(toks) - n + 1)):
                s = fuzz.ratio(" ".join(toks[i:i + n]), r.name_l)
                if s > best_score:
                    best, best_score = r, s
        if best_score >= 85:
            return best, "fuzzy", best_score
        # 3) pincode, only if it points to exactly one locality in this town
        if pincode is not None:
            p = cand[cand.pincode == pincode]
            if len(p) == 1:
                return p.iloc[0], "pincode_unique", 0
        return None, "none", best_score

    def _pincode(self, raw, town):
        valid = set(self.loc.loc[self.loc.town_id == town, "pincode"])
        for m in re.findall(r"\b(\d{5,7})\b", raw):
            if int(m) in valid:
                return int(m), "valid"
            fixes = set()
            if len(m) == 7:  # one stray digit, e.g. 9700203 -> 970203
                fixes |= {int(m[:i] + m[i + 1:]) for i in range(7)}
            if len(m) == 6:  # two neighbouring digits swapped, e.g. 970230 -> 970203
                fixes |= {int(m[:i] + m[i + 1] + m[i] + m[i + 2:]) for i in range(5)}
            if len(m) == 5:  # one digit dropped, e.g. 97002 -> 970202 (only if unique)
                fixes |= {int(m[:i] + d + m[i:]) for i in range(6) for d in "0123456789"}
            hits = fixes & valid
            if len(hits) == 1:
                return hits.pop(), "repaired"
        m = re.search(r"\b(\d{6})\b", raw)
        return (int(m.group(1)), "not_in_town") if m else (None, "missing")

    def _relation_near(self, text, start, end, indic=False):
        window_after = text[end:end + 25]
        window_before = text[max(0, start - 25):start]
        if indic:
            for rel, words in RELATION_INDIC.items():
                if any(w in window_after for w in words):
                    return rel
        for rel, pat in self.rel_latin:
            if pat.search(window_after) or pat.search(window_before):
                return rel
        return None

    def landmarks(self, text, raw_norm):
        """All landmarks in (masked, cleaned) text plus Indic-script ones in the raw text."""
        found, work = [], text
        for typ, pat in self.lm_latin:
            for m in pat.finditer(work):
                rel = self._relation_near(text, m.start(), m.end())
                found.append({"type": typ, "relation": rel, "pos": m.start(), "match": m.group(0)})
            work = pat.sub(lambda m: "#" * len(m.group(0)), work)
        work_i = raw_norm
        for typ, phrases in LANDMARK_INDIC:
            for ph in sorted(phrases, key=len, reverse=True):
                i = work_i.find(ph)
                while i >= 0:
                    rel = self._relation_near(work_i, i, i + len(ph), indic=True)
                    found.append({"type": typ, "relation": rel, "pos": 1000 + i, "match": ph})
                    work_i = work_i[:i] + "#" * len(ph) + work_i[i + len(ph):]
                    i = work_i.find(ph)
        found.sort(key=lambda d: d["pos"])
        return found

    @staticmethod
    def _street(text, raw_norm):
        def num(p, src=text):
            m = re.search(p, src)
            return m.group(1) if m else None

        return {
            "gali_no": num(r"\bgali (?:no |n )?(\d+)") or num(r"गली नं\.?\s*(\d+)", raw_norm),
            "cross_no": num(r"\b(\d+)(?:st|nd|rd|th)? cross\b"),
            "main_no": num(r"\b(\d+)(?:st|nd|rd|th)? main\b"),
            "block": num(r"\bblock ([a-f])\b") or num(r"\b([a-f]) block\b"),
            "road_no": num(r"\broad (\d+)\b") or num(r"\b(\d+)(?:st|nd|rd|th) road\b"),
            "ward_no": num(r"\bward (\d+)\b"),
            "door_no": num(r"^(?:house |no )?(\d+(?:/\d+)?)\b")
                       or num(r"(?<!gali )\b(?:house|no) (\d+(?:/\d+)?)\b"),
        }

    # ------------------------------------------------------------ main
    def parse(self, text, town):
        raw_norm = unicodedata.normalize("NFC", str(text)).lower()
        clean, fixes = self.clean(text)
        out = {"clean_text": clean, "typo_fixes": fixes}
        if town == "OUT":
            out.update({"locality_method": "out_of_territory", "landmarks": []})
            return out
        pincode, pin_status = self._pincode(raw_norm, town)
        loc, method, score = self._locality(clean, town, pincode)
        out.update({
            "pincode": pincode, "pincode_status": pin_status,
            "locality_id": None if loc is None else loc.locality_id,
            "locality_name": None if loc is None else loc.locality_name,
            "locality_method": method, "locality_score": score,
        })
        # mask locality, building and town names before looking for landmarks
        masked = clean
        building = next((b for b in BUILDINGS if b in clean), None)
        out["building"] = building
        for phrase in ([loc.name_l] if loc is not None else []) + ([building] if building else []) + TOWN_WORDS:
            masked = re.sub(r"\b" + re.escape(phrase) + r"\b", lambda m: "_" * len(m.group(0)), masked)
        lms = self.landmarks(masked, raw_norm)
        out["landmarks"] = [{k: d[k] for k in ("type", "relation", "match")} for d in lms]
        out.update(self._street(clean, raw_norm))
        return out


def parse_all(addr: pd.DataFrame, localities: pd.DataFrame) -> pd.DataFrame:
    p = AddressParser(localities)
    rows = [p.parse(t, town) for t, town in zip(addr.address_text, addr.town_id)]
    out = pd.DataFrame(rows, index=addr.index)
    out.insert(0, "address_id", addr.address_id.values)
    out["n_landmarks"] = out.landmarks.str.len()
    out["lm1_type"] = out.landmarks.apply(lambda l: l[0]["type"] if l else None)
    out["lm1_relation"] = out.landmarks.apply(lambda l: l[0]["relation"] if l else None)
    out["lm2_type"] = out.landmarks.apply(lambda l: l[1]["type"] if len(l) > 1 else None)
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from load import load_addresses, map_tables

    a = load_addresses()
    _, loc, _ = map_tables()
    parsed = parse_all(a, loc)
    parsed.to_json(Path(__file__).parent.parent / "outputs" / "parsed_addresses.json",
                   orient="records", force_ascii=False, indent=1)
    print(parsed.locality_method.value_counts())
