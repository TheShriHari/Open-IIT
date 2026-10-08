"""Phase 2 - Visit ingestion, GPS filtering, weighted median (blueprint section 4).

Output: output/visit_derived_pins.csv
"""
import numpy as np
import pandas as pd

from config import GEO, OUT, SHARED

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


def build_pins(visits: pd.DataFrame) -> pd.DataFrame:
    v = visits[visits.outcome.isin(ACCEPT_OUTCOMES) & (visits.gps_accuracy_m <= MAX_GPS_ACC)]
    n_acc = visits.groupby("photo_hash").account_id.nunique()
    v = v[~v.photo_hash.isin(n_acc[n_acc >= FAKE_PHOTO_MIN_ACCOUNTS].index)].copy()
    v["w"] = 1.0 / v.gps_accuracy_m.clip(lower=1.0) ** 2

    rows = []
    for aid, g in v.groupby("address_id"):
        wx, wy = weighted_median(g.checkin_x, g.w), weighted_median(g.checkin_y, g.w)
        n = len(g)
        spread = float(np.hypot(g.checkin_x - wx, g.checkin_y - wy).max()) if n >= 2 else 0.0
        tier = "visit_1" if n == 1 else "visits_agree" if spread <= AGREE_SPREAD_M else "visits_disagree"
        rows.append((aid, wx, wy, n, tier))
    return pd.DataFrame(rows, columns=["address_id", "visit_px", "visit_py", "n_good_visits", "visit_tier"])


def main():
    visits = pd.read_csv(SHARED / "field_visits.csv")
    pins = build_pins(visits)
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
