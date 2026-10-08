"""Phase 3 - Spatial fallback hierarchy (blueprint section 5).

Tier 1 visit pin > Tier 2 baseline street/rooftop > Tier 3 landmark (within 800 m of locality
centroid, baseline precision == locality) > Tier 4 coarse baseline.
Output: output/master_pins.csv
"""
import numpy as np
import pandas as pd

from config import GEO, OUT, SHARED
from fuzzy_remarks import get_address_level_remark_evidence

LANDMARK_MAX_FROM_CENTROID = 800.0
HINT_MAX_DIST = 200.0
AT_LANDMARK_M = 25.0


def centroid(row, loc, town_xy):
    c = loc[(loc.town_id == row.town_id) & (loc.locality_name == row.locality_name)]
    if c.empty and isinstance(row.pincode, str):
        c = loc[(loc.town_id == row.town_id) & (loc.pincode.astype(str) == row.pincode)]
    if not c.empty:
        return c.centroid_x.mean(), c.centroid_y.mean()
    return town_xy.get(row.town_id, (np.nan, np.nan))


def nearest_poi(pois, landmark_type, town_id, x, y):
    c = pois[(pois.town_id == town_id) & pois.landmark_type.str.contains(landmark_type, regex=False)]
    if c.empty:
        return None
    d = np.hypot(c.x.to_numpy() - x, c.y.to_numpy() - y)
    i = int(np.argmin(d))
    return c.iloc[i], float(d[i])


def main():
    st = pd.read_csv(OUT / "structured_addresses.csv", dtype={"pincode": str})
    base = pd.read_csv(GEO / "baseline_geocodes.csv")
    vis = pd.read_csv(OUT / "visit_derived_pins.csv")
    loc = pd.read_csv(GEO / "localities.csv")
    pois = pd.read_csv(GEO / "landmarks_poi.csv")
    town_xy = {t: (g.centroid_x.mean(), g.centroid_y.mean()) for t, g in loc.groupby("town_id")}

    # Load audit trail and aggregate address-level remark evidence
    audit_file = OUT / "remark_extracted_corrections.csv"
    if audit_file.exists():
        audit_df = pd.read_csv(audit_file)
        addr_rem = get_address_level_remark_evidence(audit_df)
    else:
        addr_rem = pd.DataFrame(columns=["address_id", "best_remark_conf", "best_remark_status",
                                         "remark_landmark_name", "remark_relation", "has_strong_correction"])

    df = st.merge(base, on="address_id", how="left").merge(vis, on="address_id", how="left")
    df = df.merge(addr_rem, on="address_id", how="left")
    df = df[df.geocoder_x.notna() | df.visit_px.notna()].copy()

    out = []
    for r in df.itertuples(index=False):
        rem_conf = float(r.remark_confidence) if hasattr(r, "remark_confidence") and pd.notna(r.remark_confidence) else (
            float(r.best_remark_conf) if hasattr(r, "best_remark_conf") and pd.notna(r.best_remark_conf) else 0.0
        )

        if pd.notna(r.visit_px):
            px, py, tier = r.visit_px, r.visit_py, r.visit_tier
            n = int(r.n_good_visits)
        else:
            n = 0
            if r.precision in ("street", "rooftop"):
                px, py, tier = r.geocoder_x, r.geocoder_y, "baseline_street"
            else:
                px, py, tier = r.geocoder_x, r.geocoder_y, "baseline_coarse"
                # Check address text landmark first
                lm_to_search = r.landmark_type if isinstance(r.landmark_type, str) else None
                # If address text had no landmark, but agent remark gave strong correction landmark
                if lm_to_search is None and hasattr(r, "best_remark_status") and r.best_remark_status == "STRONG_CORRECTION" and isinstance(r.remark_landmark_name, str):
                    lm_to_search = r.remark_landmark_name.lower().replace(" ", "_")

                if lm_to_search:
                    cx, cy = centroid(r, loc, town_xy)
                    hit = nearest_poi(pois, lm_to_search, r.town_id, cx, cy)
                    if hit and hit[1] <= LANDMARK_MAX_FROM_CENTROID:
                        px, py, tier = hit[0].x, hit[0].y, "landmark"

        # Generate landmark hint (combining address text landmark and field remark evidence)
        hint = ""
        if isinstance(r.landmark_type, str):
            hit = nearest_poi(pois, r.landmark_type, r.town_id, px, py)
            if hit and hit[1] <= HINT_MAX_DIST:
                rel = r.spatial_relation.title() if isinstance(r.spatial_relation, str) else "Near"
                name, d = hit[0]["name"], hit[1]
                if tier == "landmark":
                    hint = f"{rel} {name} (pin is a guess near this landmark)"
                elif d < AT_LANDMARK_M:
                    hint = f"{rel} {name} (pin is at this landmark)"
                else:
                    hint = f"{rel} {name} (about {round(d, -1):.0f} m from pin)"

        # Enrich or override with high-confidence field agent remark evidence if available
        if hasattr(r, "best_remark_conf") and pd.notna(r.best_remark_conf) and r.best_remark_conf >= 0.60:
            rem_rel = str(r.remark_relation).title() if pd.notna(r.remark_relation) else "Near"
            rem_lm = str(r.remark_landmark_name) if pd.notna(r.remark_landmark_name) else ""
            if rem_lm:
                if hint:
                    hint += f" | Remark Cue: {rem_rel} {rem_lm} (conf={r.best_remark_conf:.2f})"
                else:
                    hint = f"Field Remark: {rem_rel} {rem_lm} (conf={r.best_remark_conf:.2f})"

        out.append((r.address_id, r.account_id, r.town_id, px, py, tier, n, hint, round(rem_conf, 3)))

    mp = pd.DataFrame(out, columns=["address_id", "account_id", "town_id", "px", "py", "tier",
                                    "n_good_visits", "landmark_hint", "remark_confidence"])
    mp.to_csv(OUT / "master_pins.csv", index=False, encoding="utf-8-sig")

    print(f"master pins: {len(mp)}  (addresses without any pin: {len(st) - len(mp)})")
    print(mp.tier.value_counts().to_string())
    print("hints:", (mp.landmark_hint != "").sum())
    print("pins with remark_confidence > 0:", (mp.remark_confidence > 0).sum())

    # evaluation vs survey (all 100 rows)
    truth = pd.read_csv(GEO / "surveyed_addresses.csv")
    d = truth.merge(mp, on="address_id").merge(base, on="address_id")
    d["err"] = np.hypot(d.px - d.surveyed_x, d.py - d.surveyed_y)
    d["base_err"] = np.hypot(d.geocoder_x - d.surveyed_x, d.geocoder_y - d.surveyed_y)
    g = d.groupby("tier").agg(n=("err", "size"), model_med=("err", "median"),
                              model_p90=("err", lambda s: s.quantile(.9)), base_med=("base_err", "median"))
    print("\nvs survey:\n", g.round(1).to_string())
    print("overall n=%d model med %.1f p90 %.1f | baseline med %.1f p90 %.1f" % (
        len(d), d.err.median(), d.err.quantile(.9), d.base_err.median(), d.base_err.quantile(.9)))


if __name__ == "__main__":
    main()
