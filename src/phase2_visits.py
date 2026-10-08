"""Phase 2 - Visit ingestion, GPS filtering, weighted median (blueprint section 4).

Output: output/visit_derived_pins.csv
"""
import numpy as np
import pandas as pd
from typing import Optional

from config import GEO, OUT, SHARED
from fuzzy_remarks import parse_agent_remark, extract_and_audit_all_remarks

ACCEPT_OUTCOMES = ["locked_premises", "met_borrower", "met_family", "neighbour_says_shifted"]
MAX_GPS_ACC = 30.0
FAKE_PHOTO_MIN_ACCOUNTS = 3
AGREE_SPREAD_M = 100.0


def weighted_median(vals, weights):
    v, w = np.asarray(vals, float), np.asarray(weights, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return v[np.searchsorted(c, 0.5 * c[-1])]


def build_pins(visits: pd.DataFrame, pois: Optional[pd.DataFrame] = None, addr: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    v = visits[visits.outcome.isin(ACCEPT_OUTCOMES) & (visits.gps_accuracy_m <= MAX_GPS_ACC)]
    n_acc = visits.groupby("photo_hash").account_id.nunique()
    v = v[~v.photo_hash.isin(n_acc[n_acc >= FAKE_PHOTO_MIN_ACCOUNTS].index)].copy()

    # Join town_id if addr is provided
    if addr is not None and "town_id" in addr.columns and "town_id" not in v.columns:
        v = v.merge(addr[["address_id", "town_id"]], on="address_id", how="left")
    elif "town_id" not in v.columns:
        v["town_id"] = None

    # Parse remarks using fuzzy logic engine
    rem_conf = []
    rem_status = []
    for row in v.itertuples(index=False):
        p = parse_agent_remark(row.remark, row.town_id, pois, row.checkin_x, row.checkin_y)
        rem_conf.append(p["remark_confidence"])
        rem_status.append(p["remark_status"])
    v["remark_confidence"] = rem_conf
    v["remark_status"] = rem_status

    # Evidence weighting: visit_weight x GPS_quality x remark_confidence factor
    gps_quality = 1.0 / v.gps_accuracy_m.clip(lower=1.0) ** 2
    rem_factor = np.ones(len(v))
    for i, (st, c) in enumerate(zip(v.remark_status, v.remark_confidence)):
        if st == "STRONG_CORRECTION":
            rem_factor[i] = 1.0 + 1.5 * c
        elif st == "USEFUL_SPATIAL_CUE":
            rem_factor[i] = 1.0 + 0.8 * c
        elif st == "CONFLICTING":
            rem_factor[i] = 0.25
    v["w"] = gps_quality * rem_factor

    rows = []
    for aid, g in v.groupby("address_id"):
        wx, wy = weighted_median(g.checkin_x, g.w), weighted_median(g.checkin_y, g.w)
        n = len(g)
        spread = float(np.hypot(g.checkin_x - wx, g.checkin_y - wy).max()) if n >= 2 else 0.0
        tier = "visit_1" if n == 1 else "visits_agree" if spread <= AGREE_SPREAD_M else "visits_disagree"
        max_rem_c = float(g.remark_confidence.max()) if len(g) > 0 else 0.0
        rows.append((aid, wx, wy, n, tier, round(max_rem_c, 3)))
    return pd.DataFrame(rows, columns=["address_id", "visit_px", "visit_py", "n_good_visits", "visit_tier", "remark_confidence"])


def main():
    # Generate/refresh audit trail
    extract_and_audit_all_remarks()

    visits = pd.read_csv(SHARED / "field_visits.csv")
    addr = pd.read_csv(SHARED / "addresses.csv")
    pois = pd.read_csv(GEO / "landmarks_poi.csv")

    pins = build_pins(visits, pois=pois, addr=addr)
    pins.to_csv(OUT / "visit_derived_pins.csv", index=False)

    print(f"visits in: {len(visits)}, addresses with a visit pin: {len(pins)}")
    print(pins.visit_tier.value_counts().to_string())

    # sanity check vs ground truth (evaluation only)
    truth = pd.read_csv(GEO / "surveyed_addresses.csv")
    splits = pd.read_csv(SHARED / "splits.csv")
    addr = pd.read_csv(SHARED / "addresses.csv")[["address_id", "account_id"]]
    d = truth.merge(addr, on="address_id").merge(splits, on="account_id").merge(pins, on="address_id")
    d["err"] = np.hypot(d.visit_px - d.surveyed_x, d.visit_py - d.surveyed_y)
    print("\nerror vs survey (m), all splits:")
    print(d.groupby("visit_tier").err.agg(["count", "median", lambda s: s.quantile(.9)])
          .rename(columns={"<lambda_0>": "p90"}).round(1).to_string())
    print("overall: n=%d median %.1f p90 %.1f" % (len(d), d.err.median(), d.err.quantile(.9)))


if __name__ == "__main__":
    main()
