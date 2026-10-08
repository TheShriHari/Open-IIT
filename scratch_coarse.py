import pandas as pd
import numpy as np

truth = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\surveyed_addresses.csv")
visits = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\field_visits.csv")
addr = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\addresses.csv")
base = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\baseline_geocodes.csv")
mp = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\output\master_pins.csv")

d = truth.merge(mp, on="address_id").merge(base, on="address_id").merge(addr, on="address_id")
coarse = d[d.tier == "baseline_coarse"].copy()
coarse["err"] = np.hypot(coarse.px - coarse.surveyed_x, coarse.py - coarse.surveyed_y)
print(f"Total baseline_coarse in surveyed: {len(coarse)}")

for _, r in coarse.iterrows():
    aid = r.address_id
    v_a = visits[visits.address_id == aid]
    txt = r.address_text.encode("ascii", "replace").decode()
    print(f"\n{aid} | town={r.town_id_x} | base_err={r.err:.1f}m | text={txt}")
    print(f"   visits: {len(v_a)} total")
    for _, v in v_a.iterrows():
        rem = str(v.remark).encode("ascii", "replace").decode()
        print(f"     outcome={v.outcome}, acc={v.gps_accuracy_m}m, remark={rem}")
