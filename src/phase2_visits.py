"""Phase 2 - Visit ingestion, canonical dwell correction, learned agent reliability.

Output: output/visit_derived_pins.csv
"""
import numpy as np
import pandas as pd
from typing import Optional

from config import GEO, OUT, SHARED
from pin_synthesis import synthesize_visit_pins, weighted_median


def build_pins(
    visits: pd.DataFrame,
    pois: Optional[pd.DataFrame] = None,
    addr: Optional[pd.DataFrame] = None,
    gps: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Builds visit-derived pins using canonical dwell coordinates and learned reliability."""
    if gps is None and (GEO / "visit_gps_points.csv").exists():
        gps = pd.read_csv(GEO / "visit_gps_points.csv")
    pins_df, _ = synthesize_visit_pins(
        visits_df=visits,
        gps_df=gps,
        pois_df=pois,
        addr_df=addr,
    )
    return pins_df


def main():
    visits = pd.read_csv(SHARED / "field_visits.csv")
    addr = pd.read_csv(SHARED / "addresses.csv")
    pois = pd.read_csv(GEO / "landmarks_poi.csv")
    gps = pd.read_csv(GEO / "visit_gps_points.csv")

    pins = build_pins(visits, pois=pois, addr=addr, gps=gps)

    print(f"Visits in: {len(visits)}, addresses with a visit pin: {len(pins)}")
    print(pins["visit_tier"].value_counts().to_string())

    # Sanity check vs ground truth
    truth = pd.read_csv(GEO / "surveyed_addresses.csv")
    splits = pd.read_csv(SHARED / "splits.csv")
    addr_split = pd.read_csv(SHARED / "addresses.csv")[["address_id", "account_id"]]
    d = truth.merge(addr_split, on="address_id").merge(splits, on="account_id").merge(pins, on="address_id")
    d["err"] = np.hypot(d["visit_px"] - d["surveyed_x"], d["visit_py"] - d["surveyed_y"])
    print("\nError vs survey (m), all splits:")
    print(d.groupby("visit_tier")["err"].agg(["count", "median", lambda s: s.quantile(0.9)])
          .rename(columns={"<lambda_0>": "p90"}).round(1).to_string())
    print("Overall: n=%d median %.1f p90 %.1f" % (len(d), d["err"].median(), d["err"].quantile(0.9)))


if __name__ == "__main__":
    main()
