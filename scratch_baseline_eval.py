import pandas as pd
import numpy as np

truth = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\surveyed_addresses.csv")
mp = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\output\master_pins.csv")
base = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\baseline_geocodes.csv")

d = truth.merge(mp, on="address_id").merge(base, on="address_id")
d["err"] = np.hypot(d.px - d.surveyed_x, d.py - d.surveyed_y)
d["base_err"] = np.hypot(d.geocoder_x - d.surveyed_x, d.geocoder_y - d.surveyed_y)

print("Baseline Geocoder alone:")
print(f"  median: {d.base_err.median():.1f}m, p90: {d.base_err.quantile(0.9):.1f}m")
print("\nCurrent Geocoder (Phases 1-3):")
print(f"  median: {d.err.median():.1f}m, p90: {d.err.quantile(0.9):.1f}m")
print(d.groupby("tier").err.agg(["count", "median", lambda s: s.quantile(0.9)]).rename(columns={"<lambda_0>": "p90"}).round(1))
