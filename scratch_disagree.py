import pandas as pd
import numpy as np

truth = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\surveyed_addresses.csv")
visits = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\field_visits.csv")
addr = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\addresses.csv")
base = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\baseline_geocodes.csv")
mp = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\output\master_pins.csv")

d = truth.merge(mp, on="address_id").merge(base, on="address_id")
disagree = d[d.tier == "visits_disagree"].copy()
disagree["err"] = np.hypot(disagree.px - disagree.surveyed_x, disagree.py - disagree.surveyed_y)
print(f"Total visits_disagree in surveyed: {len(disagree)}")

for _, r in disagree.sort_values("err", ascending=False).iterrows():
    aid = r.address_id
    print(f"\n--- Address {aid} | Err={r.err:.1f}m | Ground Truth=({r.surveyed_x:.1f}, {r.surveyed_y:.1f}) | Master Pin=({r.px:.1f}, {r.py:.1f})")
    v_a = visits[visits.address_id == aid]
    for _, v in v_a.iterrows():
        v_err = np.hypot(v.checkin_x - r.surveyed_x, v.checkin_y - r.surveyed_y)
        print(f"   Visit {v.visit_id}: outcome={v.outcome}, acc={v.gps_accuracy_m}m, checkin=({v.checkin_x:.1f}, {v.checkin_y:.1f}), err={v_err:.1f}m | remark: {v.remark}")
