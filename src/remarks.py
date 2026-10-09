"""Step 8: agent remark corrections.

Some remarks say where the house really is, in the clause after the status report:
  "lock laga hai; asli ghar Paani ki Tanki ke peeche hai, 2 gali aage"
  "avara appa jothe maataadide; nija mane Milk Dairy hinde ide, 2 cross munde"
For each positive, non-fake visit with such a clause:
  1. the landmark type + relation are read with textparse.AddressParser (same lexicon as
     the address text); 'temple_any' is skipped (not a map type)
  2. the landmark is resolved to the POI of that type in the same town nearest the agent's
     check-in, if within MAX_POI_M; otherwise the correction is ignored
  3. it becomes one observation at that POI, sigma = sqrt(base^2 + (k * offset_units)^2),
     where offset_units is the "2 cross munde / 2 gali aage / 2 lanes ahead" count
Repeated corrections at one address that resolve to the same POI are one clue (not independent).

RC_SIGMA is fitted on TRAIN addresses: POI vs the leave-one-out median of the address's other
unflagged reliable visits, sigma = median error / 1.18 (Rayleigh). `python src/remarks.py`
refits and prints; the values are frozen below. The answer key is never used here.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evidence import _loo_consensus  # noqa: E402
from load import map_tables  # noqa: E402
from textparse import AddressParser  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORRECTION = re.compile(
    r"nija mane|asli ghar|actual house|address (?:tappu|thappu|tappagide|galat|wrong)"
    r"|\bmane\b.*\bide\b|\bghar\b.*\bhai\b|house (?:is|near)|wala ghar", re.I)
OFFSET = re.compile(r"(\d+)\s*(?:cross|gali|galli|lanes?)\s*(?:munde|aage|ahead|mundhe)", re.I)
MAX_POI_M = 400
RAYLEIGH_MEDIAN = 1.18
RC_SIGMA = {"base": 122, "offset_per_unit": 39}    # fitted by fit_sigma() on train; see README


def _clauses(remark: str):
    return [c.strip() for c in str(remark).split(";")[1:] if c.strip()]


# ------------------------------------------------------------- door cues (Step 7c)
COLOURS = {"blue": r"blue|neela|neeli|nila", "green": r"green|hara|hari|hasiru",
           "red": r"red|laal|lal|kempu", "yellow": r"yellow|peela|peeli|haladi",
           "white": r"white|safed|bili"}
GATE = re.compile(r"\b(" + "|".join(COLOURS.values()) + r")\s+(?:colou?r\s+|rang\s+(?:ka|ki)\s+)?"
                  r"(?:gate|baagilu|darwaza)\b", re.I)
CORNER = re.compile(r"\bcorner\s+(?:house|mane|ghar|wala)|\bkone\s+(?:wala|ka|ki)\s+ghar", re.I)
FLOOR = re.compile(r"\b(\d+)(?:st|nd|rd|th|ne)?\s*(?:floor|manzil|maadi)\b|\b(ground)\s+floor\b", re.I)


def remark_cues(remark: str) -> list[str]:
    """Visual door cues in one remark: "blue gate", "corner house", "2 floor"."""
    s, cues = str(remark), []
    for m in GATE.finditer(s):
        word = m.group(1).lower()
        cues.append(next(c for c, pat in COLOURS.items() if re.fullmatch(pat, word)) + " gate")
    if CORNER.search(s):
        cues.append("corner house")
    m = FLOOR.search(s)
    if m:
        cues.append("ground floor" if m.group(2) else f"floor {m.group(1)}")
    return cues


def door_cues(ev: pd.DataFrame) -> pd.Series:
    """Per address: the door cues from non-fake positive visits, most often reported first."""
    ok = ev[(ev.role == "positive") & ~ev.flag_fake & ev.remark.notna()]
    rows = [(a, c) for a, rm in zip(ok.address_id, ok.remark) for c in remark_cues(rm)]
    if not rows:
        return pd.Series(dtype=str, name="door_cues")
    c = pd.DataFrame(rows, columns=["address_id", "cue"]).value_counts().reset_index()
    return c.groupby("address_id").cue.agg(", ".join).rename("door_cues")


def parse_corrections(ev: pd.DataFrame, parser: AddressParser) -> pd.DataFrame:
    """One row per visit with a correction clause that names a map landmark."""
    rows = []
    cand = ev[(ev.role == "positive") & ~ev.flag_fake & ev.remark.notna()]
    for r in cand.itertuples():
        for cl in _clauses(r.remark):
            if not CORRECTION.search(cl):
                continue
            lms = [l for l in parser.landmarks(parser.clean(cl)[0], cl.lower()) if l["type"] != "temple_any"]
            if not lms:
                continue
            m = OFFSET.search(cl)
            rows.append(dict(visit_id=r.visit_id, address_id=r.address_id, town_id=r.town_id, split=r.split,
                             pos_x=r.pos_x, pos_y=r.pos_y, clause=cl,
                             rc_type=lms[0]["type"], relation=lms[0]["relation"] or ("RIGHT_SIDE" if "right side" in cl.lower() else "NONE"),
                             offset_units=int(m.group(1)) if m else 0))
            break
    return pd.DataFrame(rows)


def resolve(corr: pd.DataFrame, poi: pd.DataFrame) -> pd.DataFrame:
    """Nearest POI of the named type in the same town; poi_id NaN if none within MAX_POI_M."""
    out = []
    for r in corr.itertuples():
        q = poi[(poi.town_id == r.town_id) & (poi.landmark_type == r.rc_type)]
        if q.empty:
            out.append((None, np.nan, np.nan, np.nan))
            continue
        d = np.hypot(q.x.values - r.pos_x, q.y.values - r.pos_y)
        i = int(np.argmin(d))
        ok = d[i] <= MAX_POI_M
        out.append((q.poi_id.iloc[i] if ok else None, q.x.iloc[i] if ok else np.nan,
                    q.y.iloc[i] if ok else np.nan, d[i]))
    corr = corr.copy()
    corr[["poi_id", "rc_x", "rc_y", "dist_checkin"]] = pd.DataFrame(out, index=corr.index)
    return corr


def rc_sigma(units, table=RC_SIGMA):
    return np.sqrt(table["base"] ** 2 + (table["offset_per_unit"] * np.asarray(units, float)) ** 2)


def remark_clues(ev: pd.DataFrame, table=RC_SIGMA, corr: pd.DataFrame | None = None) -> pd.DataFrame:
    """address_id, rc_x, rc_y, rc_sigma, rc_type, rc_n: one row per distinct POI per address."""
    if corr is None:
        corr = all_corrections(ev)
    c = corr[corr.poi_id.notna()]
    g = c.groupby(["address_id", "poi_id"]).agg(
        rc_x=("rc_x", "first"), rc_y=("rc_y", "first"), rc_type=("rc_type", "first"),
        units=("offset_units", "median"), rc_n=("visit_id", "size")).reset_index()
    g["rc_sigma"] = rc_sigma(g.units, table)
    # several POIs at one address (rare): keep the one named most often
    g = g.sort_values(["address_id", "rc_n"], ascending=[True, False]).drop_duplicates("address_id")
    return g[["address_id", "rc_x", "rc_y", "rc_sigma", "rc_type", "rc_n"]].reset_index(drop=True)


def remark_hints(corr: pd.DataFrame) -> pd.DataFrame:
    """Field-direction hint per address: address_id, rc_type, rc_relation, rc_poi (empty if
    unresolved). The most often named (type, relation) wins; resolved corrections first."""
    c = corr.assign(resolved=corr.poi_id.notna())
    n = c.groupby(["address_id", "rc_type", "relation"]).visit_id.transform("size")
    c = c.assign(n=n).sort_values(["address_id", "resolved", "n"], ascending=[True, False, False])
    h = c.drop_duplicates("address_id")[["address_id", "rc_type", "relation", "poi_id"]]
    return h.rename(columns={"relation": "rc_relation", "poi_id": "rc_poi"}).reset_index(drop=True)


def all_corrections(ev: pd.DataFrame) -> pd.DataFrame:
    _, loc, poi = map_tables()
    return resolve(parse_corrections(ev, AddressParser(loc)), poi)


# ------------------------------------------------------------------ tuning
def loo_table(ev: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    """Train corrections resolved to a POI vs LOO median of the other reliable visits."""
    t = _loo_consensus(ev, other_agents_only=False, min_others=1)[["visit_id", "tx", "ty", "err"]]
    t = corr[corr.poi_id.notna()].merge(t.rename(columns={"err": "checkin_err"}), on="visit_id")
    t["err"] = np.hypot(t.rc_x - t.tx, t.rc_y - t.ty)
    return t


def fit_sigma(t: pd.DataFrame) -> dict:
    base = t[t.offset_units == 0].err.median() / RAYLEIGH_MEDIAN
    off = t[t.offset_units > 0]
    k = 0.0
    if len(off):
        s_off = off.err.median() / RAYLEIGH_MEDIAN
        k = np.sqrt(max(0.0, s_off ** 2 - base ** 2)) / off.offset_units.mean()
    return {"base": round(float(base)), "offset_per_unit": round(float(k))}


if __name__ == "__main__":
    from evidence import build_evidence

    ev = build_evidence()
    corr = all_corrections(ev)
    n_pos = ev[(ev.role == "positive") & ~ev.flag_fake].address_id.nunique()
    print(f"corrections found: {len(corr)} visits, {corr.address_id.nunique()} addresses "
          f"({100 * corr.address_id.nunique() / n_pos:.0f}% of addresses with positive visits)")
    print(f"resolved to a POI within {MAX_POI_M} m: {corr.poi_id.notna().mean():.0%} "
          f"({corr.poi_id.notna().sum()} visits)\n")
    print("by landmark type (n, % resolved):")
    print(corr.groupby("rc_type").poi_id.agg(n="size", resolved=lambda s: s.notna().mean())
          .round(2).to_string(), "\n")
    print("10 examples:")
    ex = corr.sample(10, random_state=0)
    for r in ex.itertuples():
        print(f"  {r.clause[:60]:60s} -> {r.rc_type}/{r.relation}/+{r.offset_units} -> "
              f"{r.poi_id or '-'} ({r.dist_checkin:.0f} m from check-in)")
    t = loo_table(ev, corr)
    print(f"\ntrain LOO ({len(t)} resolved corrections): POI vs other reliable visits")
    print(t.groupby(["relation", t.offset_units > 0]).agg(
        n=("err", "size"), poi_median=("err", "median"), poi_p90=("err", lambda e: e.quantile(0.9)),
        checkin_median=("checkin_err", "median")).round(0).to_string())
    fitted = fit_sigma(t)
    print(f"\nfitted RC_SIGMA: {fitted}\nfrozen RC_SIGMA: {RC_SIGMA}")
    sg = rc_sigma(t.offset_units, fitted)
    print(f"coverage of 2.15*sigma (should be ~0.90): {(t.err <= 2.15 * sg).mean():.2f}")
    rc = remark_clues(ev, corr=corr)
    print(f"\nclues: {len(rc)} addresses (rc_n: {rc.rc_n.value_counts().sort_index().to_dict()})")
