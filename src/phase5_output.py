"""Phase 5 - Dispatch actions and final output (blueprint section 7).

Output: output/geocoder_output.csv
"""
import numpy as np
import pandas as pd

from config import GEO, OUT, SHARED

NEAR_M, MID_M = 100.0, 500.0


def main():
    mp = pd.read_csv(OUT / "master_pins.csv")
    R = pd.read_csv(OUT / "calibration_radii.csv")[["tier", "R90_final"]].rename(columns={"R90_final": "R90_meters"})
    df = mp.merge(R, on="tier", how="left")
    df["action"] = np.select([df.R90_meters <= NEAR_M, df.R90_meters <= MID_M],
                             ["DIRECT_VISIT", "VISIT_WITH_HINT"], default="VERIFY_FIRST")
    cols = ["address_id", "account_id", "town_id", "px", "py", "tier", "R90_meters", "action", "landmark_hint", "remark_confidence"]
    df[cols].round({"px": 1, "py": 1, "R90_meters": 1, "remark_confidence": 3}).to_csv(OUT / "geocoder_output.csv", index=False, encoding="utf-8-sig")

    print(f"rows: {len(df)}")
    print(pd.crosstab(df.tier, df.action, margins=True).to_string())
    print((df.action.value_counts(normalize=True) * 100).round(1).to_string())

    truth = pd.read_csv(GEO / "surveyed_addresses.csv")
    d = truth.merge(df, on="address_id")
    d["err"] = np.hypot(d.px - d.surveyed_x, d.py - d.surveyed_y)
    print("\nerror by action on the 100 surveyed rows:")
    print(d.groupby("action").err.agg(["count", "median", lambda s: s.quantile(.9)])
          .rename(columns={"<lambda_0>": "p90"}).round(1).to_string())


if __name__ == "__main__":
    main()
