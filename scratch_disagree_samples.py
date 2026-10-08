import pandas as pd
import numpy as np

visits = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\field_visits.csv")
addr = pd.read_csv(r"c:\Users\toshr\Downloads\shared-20261007T174707Z-1-001\shared\addresses.csv")
pois = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\ps3_geocoder\landmarks_poi.csv")
mp = pd.read_csv(r"c:\Users\toshr\Downloads\ps3_geocoder-20261007T174650Z-1-001\output\master_pins.csv")

semi = visits[visits.remark.str.contains(";", na=False)]
semi_aids = semi.address_id.unique()

mp_semi = mp[mp.address_id.isin(semi_aids)]
disagree_aids = mp_semi[mp_semi.tier == "visits_disagree"].address_id.tolist()

print(f"Sample disagree addresses with cues ({len(disagree_aids)} total):")
for aid in disagree_aids[:10]:
    v_a = visits[visits.address_id == aid]
    print(f"\nAddress {aid}:")
    for _, v in v_a.iterrows():
        print(f"  {v.visit_id} | {v.outcome} | {v.gps_accuracy_m}m | ({v.checkin_x:.1f}, {v.checkin_y:.1f}) | {v.remark}")
